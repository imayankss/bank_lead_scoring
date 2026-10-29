"""GRU over a client's transaction sequence, with one logit per prediction target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from lead_scoring.data import CATEGORY_COLUMNS, HISTORY_MONTHS, LABELS, TARGETS
from lead_scoring.paths import EVENT_STORE_DIR, GRU_MODEL_FILE
from lead_scoring.sequences import N_CONTINUOUS, VOCAB_SIZE, EventStore, collate, cutoff_ms

HEADS = LABELS  # any product in 3 months (blended), each product in 3 months, each product next month
HISTORY = [f"hist_{name}_{m}" for name in (*TARGETS, "any") for m in HISTORY_MONTHS]
RECENCY = [f"hist_{name}_months_since" for name in (*TARGETS, "any")]
SIDE_COLUMNS = HISTORY + RECENCY  # read from the customer feature table, keyed by client and snapshot
N_SIDE = len(HISTORY) + len(HISTORY_MONTHS) + len(RECENCY)


def side_features(samples: pd.DataFrame) -> np.ndarray:
    """Purchase history known at the snapshot, shaped for a network: no NaN, all in [0, 1].

    Monthly flags are 0 where the month is unknown, with one known/unknown flag per month alongside.
    Recency is 1 / (1 + months since the last purchase), and 0 for clients who never bought.
    """
    known = np.stack([samples[f"hist_any_{m}"].notna() for m in HISTORY_MONTHS], axis=1)
    recency = 1 / (1 + samples[RECENCY].to_numpy(np.float32))
    return np.concatenate([samples[HISTORY].fillna(0).to_numpy(np.float32), known,
                           np.nan_to_num(recency)], axis=1).astype(np.float32)


class EventGRU(nn.Module):
    def __init__(self, hidden: int = 64, emb_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=0) for _ in CATEGORY_COLUMNS)
        self.gru = nn.GRU(len(CATEGORY_COLUMNS) * emb_dim + N_CONTINUOUS, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + N_SIDE + 1, len(HEADS))  # + 1: the no-transactions flag

    def forward(self, codes: torch.Tensor, cont: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb(codes[..., i]) for i, emb in enumerate(self.embeddings)] + [cont], dim=-1)
        # Real events always have codes >= 1, so an all-padding sequence means no events before the cutoff.
        empty = (codes == 0).flatten(1).all(dim=1, keepdim=True).float()
        return self.head(torch.cat([self.dropout(self.gru(x)[1][-1]), side, empty], dim=1))


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _batches(store: EventStore, samples: pd.DataFrame, max_len: int, batch_size: int, shuffle: bool, seed: int):
    clients = samples["customer_id"].astype(str).to_numpy()
    cutoffs = samples["snapshot_date"].map({s: cutoff_ms(s) for s in samples["snapshot_date"].unique()}).to_numpy()
    side = side_features(samples)
    order = np.random.default_rng(seed).permutation(len(samples)) if shuffle else np.arange(len(samples))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        codes, cont = collate([store.window(clients[j], int(cutoffs[j]), max_len) for j in idx], max_len)
        yield torch.from_numpy(codes), torch.from_numpy(cont), torch.from_numpy(side[idx]), idx


def train_epoch(model, opt, store, samples, labels: np.ndarray, *, max_len: int, batch_size: int,
                seed: int, dev: torch.device) -> float:
    model.train()
    loss_fn, total = nn.BCEWithLogitsLoss(), 0.0
    for codes, cont, side, idx in _batches(store, samples, max_len, batch_size, True, seed):
        opt.zero_grad()
        loss = loss_fn(model(codes.to(dev), cont.to(dev), side.to(dev)), torch.from_numpy(labels[idx]).to(dev))
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / len(samples)


@torch.no_grad()
def predict(model, store, samples, *, max_len: int, batch_size: int, dev: torch.device) -> np.ndarray:
    model.eval()
    out = np.zeros((len(samples), len(HEADS)), np.float32)
    for codes, cont, side, idx in _batches(store, samples, max_len, batch_size, False, 0):
        out[idx] = torch.sigmoid(model(codes.to(dev), cont.to(dev), side.to(dev))).cpu().numpy()
    return out


def save(model: EventGRU, path: Path, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": {"hidden": model.gru.hidden_size, **meta}}, path)


def load(path: Path) -> tuple[EventGRU, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model = EventGRU(hidden=ckpt["meta"]["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["meta"]


def score_snapshot(features: pd.DataFrame, *, model_path: Path = GRU_MODEL_FILE,
                   store_dir: Path = EVENT_STORE_DIR) -> np.ndarray:
    """All heads' probabilities for each row of a feature table (customer_id, snapshot_date and the
    SIDE_COLUMNS), on CPU so reruns are identical."""
    model, meta = load(model_path)
    samples = features[["customer_id", "snapshot_date", *SIDE_COLUMNS]].astype({"customer_id": str})
    return predict(model, EventStore(store_dir), samples, max_len=meta["max_len"], batch_size=512,
                   dev=torch.device("cpu"))
