"""Train the transaction GRU: choose the epoch count on validation, then refit for production.

--stage validate  fit on the fit snapshots (Jan–Apr 2022), score Jul 2022 after every epoch and keep
                  the best epoch's validation scores for scripts/validate_propensity.py. Stops after
                  PATIENCE epochs without a better validation AUC.
--stage final     refit on all 7 training snapshots for the chosen number of epochs.
Run after `python -m lead_scoring.data`. Builds data/processed/events on first use; delete that folder
to rebuild it after changing the amount cap or vocabulary.
"""

from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from lead_scoring.config import PipelineConfig
from lead_scoring.data import LABELS, amount_cap, category_vocab
from lead_scoring.gru import SIDE_COLUMNS, EventGRU, device, predict, save, train_epoch
from lead_scoring.metrics import lift_at_fraction
from lead_scoring.paths import CUSTOMER_FEATURES_FILE, EVENT_STORE_DIR, GRU_MODEL_FILE, GRU_VAL_SCORES_FILE, TABLES_DIR
from lead_scoring.sequences import VOCAB_SIZE, EventStore, build_event_store
from lead_scoring.temporal import assert_plausible_auc, assert_selection_snapshots

PATIENCE = 2  # epochs without a better validation AUC before stopping; it peaked at epoch 3 of 9 before


def _store(cfg: PipelineConfig) -> EventStore:
    if not any(EVENT_STORE_DIR.glob("fold=*.npz")):
        print("Building the event store (one fold at a time)...", flush=True)
        build_event_store(EVENT_STORE_DIR, cap=amount_cap(before=cfg.test_snapshot),
                          vocab=category_vocab(before=cfg.fit_snapshots[-1], top=VOCAB_SIZE - 2))
    return EventStore(EVENT_STORE_DIR)


def _fit(cfg, store, samples, epochs, args, on_epoch=None) -> EventGRU:
    torch.manual_seed(cfg.random_state)
    dev = device()
    model = EventGRU(hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    labels = samples[LABELS].to_numpy(np.float32)
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train_epoch(model, opt, store, samples, labels, max_len=args.max_len, batch_size=args.batch_size,
                           seed=cfg.random_state + epoch, dev=dev)
        if on_epoch and on_epoch(epoch, model, loss, time.time() - start):
            break
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["validate", "final"], required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="train on a random subset of this many rows (smoke runs)")
    args = parser.parse_args()
    cfg = PipelineConfig()
    table = pd.read_parquet(CUSTOMER_FEATURES_FILE, columns=["customer_id", "snapshot_date", *LABELS, *SIDE_COLUMNS])
    store = _store(cfg)

    if args.stage == "validate":
        assert_selection_snapshots([*cfg.fit_snapshots, cfg.validation_snapshot], cfg.fit_snapshots,
                                   cfg.validation_snapshot, cfg.horizon_months)
        fit = table[table["snapshot_date"].isin(cfg.fit_snapshots)]
        if args.limit:
            fit = fit.sample(n=min(args.limit, len(fit)), random_state=cfg.random_state)
        fit = fit.reset_index(drop=True)
        val = table[table["snapshot_date"] == cfg.validation_snapshot].reset_index(drop=True)
        rows, best = [], {"val_auc": -1.0, "epoch": 0}

        def on_epoch(epoch, model, loss, seconds):
            nonlocal best
            scores = predict(model, store, val, max_len=args.max_len, batch_size=512, dev=device())[:, 0]
            row = {"epoch": epoch, "train_loss": loss, "val_auc": float(roc_auc_score(val["converted"], scores)),
                   "val_lift_at_10pct": float(lift_at_fraction(val["converted"], scores, 0.10)), "epoch_seconds": seconds}
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(TABLES_DIR / "gru_validation.csv", index=False)
            if row["val_auc"] > best["val_auc"]:
                best = row
                val[["customer_id", "snapshot_date"]].assign(converted=scores).to_parquet(GRU_VAL_SCORES_FILE, index=False)
            return epoch - best["epoch"] >= PATIENCE

        _fit(cfg, store, fit, args.epochs, args, on_epoch)
        assert_plausible_auc(best["val_auc"])
    else:
        history = pd.read_csv(TABLES_DIR / "gru_validation.csv")
        epochs = int(history.loc[history["val_auc"].idxmax(), "epoch"])
        train = table[table["snapshot_date"].isin(cfg.train_snapshots)].reset_index(drop=True)
        model = _fit(cfg, store, train, epochs, args)
        save(model.cpu(), GRU_MODEL_FILE, max_len=args.max_len, epochs=epochs, snapshots=list(cfg.train_snapshots))
        print(f"Saved {GRU_MODEL_FILE} after {epochs} epochs on {len(train):,} sequences")


if __name__ == "__main__":
    main()
