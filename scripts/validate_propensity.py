"""Choose the propensity model on the validation snapshot, never on the test months.

Fits LightGBM on the fit snapshots (Jan–Apr 2022) and validates on Jul 2022. When
`scripts/train_gru.py --stage validate` has run, it also picks the GRU blend weight there.
Writes reports/tables/propensity_validation.csv, reports/tables/single_feature_auc.csv and
models/propensity_choice.json. Run after: python -m lead_scoring.data && python -m lead_scoring.features
"""

from pathlib import Path
import itertools
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lead_scoring.blend import blend, choose_weight, fit_calibrator
from lead_scoring.config import PipelineConfig
from lead_scoring.metrics import lift_at_fraction
from lead_scoring.paths import GRU_VAL_SCORES_FILE, PREPROCESSED_DATA_DIR, PROPENSITY_CHOICE_FILE, TABLES_DIR
from lead_scoring.temporal import (
    assert_no_single_feature_leak,
    assert_plausible_auc,
    assert_selection_snapshots,
    single_feature_auc,
)

BASE = dict(objective="binary", metric="auc", learning_rate=0.05, feature_fraction=0.5,
            deterministic=True, force_row_wise=True, verbose=-1, seed=42)
GRID = [dict(num_leaves=leaves, min_child_samples=leaf_rows) for leaves, leaf_rows in itertools.product((31, 63), (100, 400))]


def _row(model: str, y, score, **extra) -> dict:
    return {"model": model, **extra, "val_auc": float(roc_auc_score(y, score)),
            "val_lift_at_10pct": float(lift_at_fraction(y, score, 0.10))}


def main() -> None:
    cfg = PipelineConfig()
    X = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(PREPROCESSED_DATA_DIR / "y_clf_train.parquet")["converted"].to_numpy()
    ids = pd.read_csv(PREPROCESSED_DATA_DIR / "train_ids.csv")
    fit = ids["snapshot_date"].isin(cfg.fit_snapshots).to_numpy()
    val = (ids["snapshot_date"] == cfg.validation_snapshot).to_numpy()
    assert_selection_snapshots(ids.loc[fit | val, "snapshot_date"].unique(), cfg.fit_snapshots,
                               cfg.validation_snapshot, cfg.horizon_months)

    leaks = single_feature_auc(X[val], y[val])
    leaks.to_csv(TABLES_DIR / "single_feature_auc.csv", index=False)
    assert_no_single_feature_leak(leaks)

    train_set = lgb.Dataset(X[fit], y[fit], free_raw_data=False)
    val_set = lgb.Dataset(X[val], y[val], reference=train_set)
    rows, best = [], None
    for params in GRID:
        booster = lgb.train({**BASE, **params}, train_set, num_boost_round=3000, valid_sets=[val_set],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        score = booster.predict(X[val], num_iteration=booster.best_iteration)
        row = _row("lightgbm", y[val], score, **params, best_iteration=int(booster.best_iteration))
        rows.append(row)
        print(row, flush=True)
        if best is None or row["val_auc"] > best[0]["val_auc"]:
            best = (row, score)
    chosen, lgbm_val = best
    assert_plausible_auc(chosen["val_auc"])

    gru_val = None
    if GRU_VAL_SCORES_FILE.exists():
        scores = pd.read_parquet(GRU_VAL_SCORES_FILE)
        gru_val = ids.loc[val, ["customer_id", "snapshot_date"]].merge(
            scores, on=["customer_id", "snapshot_date"], how="left")["converted"].to_numpy()
        if np.isnan(gru_val).any():
            raise ValueError("GRU validation scores miss some validation clients; rerun train_gru.py --stage validate")
        rows.append(_row("gru", y[val], gru_val))
        assert_plausible_auc(rows[-1]["val_auc"])
    choice = choose_weight(y[val], lgbm_val, gru_val)
    blended = blend(lgbm_val, gru_val, choice["weight"])
    rows.append(_row("production_blend", y[val], blended, gru_weight=choice["weight"]))
    assert_plausible_auc(choice["auc"])

    table = pd.DataFrame(rows).assign(validation_snapshot=cfg.validation_snapshot)
    table.to_csv(TABLES_DIR / "propensity_validation.csv", index=False)
    PROPENSITY_CHOICE_FILE.write_text(json.dumps({
        "lgbm_params": {**BASE, "num_leaves": chosen["num_leaves"], "min_child_samples": chosen["min_child_samples"]},
        # More rows in the final refit (7 snapshots vs 4), so proportionally more rounds.
        "lgbm_rounds": int(round(chosen["best_iteration"] * len(cfg.train_snapshots) / len(cfg.fit_snapshots))),
        "blend": choice,
        "calibrator": fit_calibrator(y[val], blended),
        "validation_snapshot": cfg.validation_snapshot,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()
