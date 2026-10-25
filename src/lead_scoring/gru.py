"""GRU over a client's transaction sequence, with one logit per prediction target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from lead_scoring.data import CATEGORY_COLUMNS, LABELS
from lead_scoring.paths import EVENT_STORE_DIR, GRU_MODEL_FILE
from lead_scoring.sequences import N_CONTINUOUS, VOCAB_SIZE, EventStore, collate, cutoff_ms

HEADS = LABELS  # any product in 3 months (blended), each product in 3 months, each product next month


class EventGRU(nn.Module):
    def __init__(self, hidden: int = 128, emb_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=0) for _ in CATEGORY_COLUMNS)
        self.gru = nn.GRU(len(CATEGORY_COLUMNS) * emb_dim + N_CONTINUOUS, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, len(HEADS)))

    def forward(self, codes: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb(codes[..., i]) for i, emb in enumerate(self.embeddings)] + [cont], dim=-1)
        return self.head(self.gru(x)[1][-1])


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _batches(store: EventStore, samples: pd.DataFrame, max_len: int, batch_size: int, shuffle: bool, seed: int):
    clients = samples["customer_id"].astype(str).to_numpy()
    cutoffs = samples["snapshot_date"].map({s: cutoff_ms(s) for s in samples["snapshot_date"].unique()}).to_numpy()
    order = np.random.default_rng(seed).permutation(len(samples)) if shuffle else np.arange(len(samples))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        codes, cont = collate([store.window(clients[j], int(cutoffs[j]), max_len) for j in idx], max_len)
        yield torch.from_numpy(codes), torch.from_numpy(cont), idx


def train_epoch(model, opt, store, samples, labels: np.ndarray, *, max_len: int, batch_size: int,
                seed: int, dev: torch.device) -> float:
    model.train()
    loss_fn, total = nn.BCEWithLogitsLoss(), 0.0
    for codes, cont, idx in _batches(store, samples, max_len, batch_size, True, seed):
        opt.zero_grad()
        loss = loss_fn(model(codes.to(dev), cont.to(dev)), torch.from_numpy(labels[idx]).to(dev))
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / len(samples)


@torch.no_grad()
def predict(model, store, samples, *, max_len: int, batch_size: int, dev: torch.device) -> np.ndarray:
    model.eval()
    out = np.zeros((len(samples), len(HEADS)), np.float32)
    for codes, cont, idx in _batches(store, samples, max_len, batch_size, False, 0):
        out[idx] = torch.sigmoid(model(codes.to(dev), cont.to(dev))).cpu().numpy()
    return out


def save(model: EventGRU, path: Path, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": {"hidden": model.gru.hidden_size, **meta}}, path)


def load(path: Path) -> tuple[EventGRU, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model = EventGRU(hidden=ckpt["meta"]["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["meta"]


def score_snapshot(client_ids: pd.Series, snapshot: str, *, model_path: Path = GRU_MODEL_FILE,
                   store_dir: Path = EVENT_STORE_DIR) -> np.ndarray:
    """All heads' probabilities for each client at a snapshot, on CPU so reruns are identical."""
    model, meta = load(model_path)
    samples = pd.DataFrame({"customer_id": client_ids.astype(str).to_numpy(), "snapshot_date": snapshot})
    return predict(model, EventStore(store_dir), samples, max_len=meta["max_len"], batch_size=512,
                   dev=torch.device("cpu"))
