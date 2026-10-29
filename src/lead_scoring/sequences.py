"""Per-client transaction sequences for the GRU, cached as numpy arrays.

Events are sorted by client and time once, one fold at a time to stay inside 8 GB
of RAM. A sample is the last `max_len` events strictly before the end of the
snapshot day (Moscow time), found by binary search on time.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from lead_scoring.data import CATEGORY_COLUMNS, LOCAL_TS, list_folds
from lead_scoring.paths import MBD_DIR

VOCAB_SIZE = 1024  # 0 = padding, 1 = rare or unseen value, 2.. = the 1,022 most frequent values
N_CONTINUOUS = 3  # log amount, log hours since the previous event, log days before the snapshot


def cutoff_ms(snapshot: str) -> int:
    """End of the snapshot day in local-time epoch milliseconds; events must be strictly earlier."""
    return int((pd.Timestamp(snapshot).normalize() + pd.Timedelta(days=1)).value // 1_000_000)


def _encode(values: np.ndarray, vocab: list[int]) -> np.ndarray:
    if not vocab:
        return np.ones(len(values), np.int16)
    known = np.asarray(vocab, dtype=np.int64)
    order = np.argsort(known)
    pos = np.searchsorted(known[order], values).clip(max=len(known) - 1)
    hit = known[order][pos] == values
    return np.where(hit, 2 + order[pos], 1).astype(np.int16)


def build_event_store(out_dir: Path, *, mbd_dir: Path = MBD_DIR, cap: float, vocab: dict[str, list[int]]) -> Path:
    """Write one .npz per client fold: codes, time, log amount, clients and their offsets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in list_folds(mbd_dir):
        trx = str(mbd_dir / "detail" / "trx" / f"fold={fold}" / "*.parquet")
        con = duckdb.connect()
        con.execute("SET enable_progress_bar = false")
        counts = con.execute("SELECT client_id, count(*) AS n FROM read_parquet($trx) GROUP BY 1 ORDER BY 1",
                             {"trx": trx}).df()
        events = con.execute(f"""
            SELECT epoch_ms({LOCAL_TS}) AS t, ln(1 + least(amount, $cap)) AS log_amount,
                   {", ".join(f"coalesce({c}, -1) AS {c}" for c in CATEGORY_COLUMNS)}
            FROM read_parquet($trx) ORDER BY client_id, t""", {"trx": trx, "cap": cap}).fetchnumpy()
        con.close()
        np.savez(
            out_dir / f"fold={fold}.npz",
            codes=np.stack([_encode(events[c], vocab[c]) for c in CATEGORY_COLUMNS], axis=1),
            time=events["t"].astype(np.int64),
            log_amount=events["log_amount"].astype(np.float32),
            clients=counts["client_id"].astype(str).to_numpy().astype("U"),  # unicode, not object: no pickle
            offsets=np.concatenate([[0], np.cumsum(counts["n"].to_numpy())]),
        )
    return out_dir


class EventStore:
    """All folds' events in memory (about 1.2 GB for MBD-mini) with a client → row-range index."""

    def __init__(self, store_dir: Path):
        parts = [np.load(p) for p in sorted(Path(store_dir).glob("fold=*.npz"))]
        if not parts:
            raise FileNotFoundError(f"no event store in {store_dir}; run scripts/train_gru.py to build it")
        self.codes = np.concatenate([p["codes"] for p in parts])
        self.time = np.concatenate([p["time"] for p in parts])
        self.log_amount = np.concatenate([p["log_amount"] for p in parts])
        self.index: dict[str, tuple[int, int]] = {}
        base = 0
        for p in parts:
            offsets = p["offsets"]
            for client, start, end in zip(p["clients"], offsets[:-1], offsets[1:]):
                self.index[str(client)] = (base + int(start), base + int(end))
            base += len(p["time"])

    def window(self, client: str, cutoff: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = self.index.get(client, (0, 0))
        stop = start + int(np.searchsorted(self.time[start:end], cutoff, side="left"))
        begin = max(start, stop - max_len)
        if stop == begin:
            return np.zeros((1, len(CATEGORY_COLUMNS)), np.int64), np.zeros((1, N_CONTINUOUS), np.float32)
        t = self.time[begin:stop]
        gap_hours = np.diff(t, prepend=t[0]) / 3.6e6
        age_days = (cutoff - t) / 8.64e7
        cont = np.stack([self.log_amount[begin:stop], np.log1p(gap_hours), np.log1p(age_days)], axis=1)
        return self.codes[begin:stop].astype(np.int64), cont.astype(np.float32)


def collate(windows: list[tuple[np.ndarray, np.ndarray]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad every window to exactly max_len, so a score never depends on its batch-mates."""
    codes = np.zeros((len(windows), max_len, len(CATEGORY_COLUMNS)), np.int64)
    cont = np.zeros((len(windows), max_len, N_CONTINUOUS), np.float32)
    for i, (c, x) in enumerate(windows):
        codes[i, max_len - len(c):] = c
        cont[i, max_len - len(x):] = x
    return codes, cont
