"""Tiny MBD-shaped dataset so tests never need the real 2 GB download."""

from pathlib import Path

import pandas as pd
import pytest

MONTHS = pd.date_range("2022-02-28", "2023-01-31", freq="ME")


def _write(df: pd.DataFrame, root: Path, table: str) -> None:
    for fold, part in df.groupby("fold"):
        out = root / table / f"fold={fold}"
        out.mkdir(parents=True, exist_ok=True)
        part.drop(columns="fold").to_parquet(out / "part-0.parquet", index=False)


def _tx(client, fold, utc_time, amount):
    return {"client_id": client, "fold": fold, "event_time": pd.Timestamp(utc_time), "amount": float(amount), "event_type": 1}


@pytest.fixture
def mbd_dir(tmp_path: Path) -> Path:
    """Snapshot of interest in the tests is 2022-01-31 (Moscow time, UTC+3)."""
    tx = [
        _tx("a", 0, "2021-06-01 10:00:00", 100),
        _tx("a", 0, "2022-01-31 20:00:00", 50),   # 23:00 Moscow on snapshot day: counted
        _tx("a", 0, "2022-01-31 22:00:00", 999),  # 01:00 Moscow on Feb 1: after snapshot
        _tx("a", 0, "2022-03-15 10:00:00", 70),   # inside the 3-month target window
        _tx("a", 0, "2022-06-15 10:00:00", 500),  # after the target window
        _tx("b", 1, "2021-12-20 10:00:00", 10_000_000),  # extreme amount, capped
        _tx("c", 1, "2020-12-31 21:30:00", 5),    # 00:30 Moscow on 1 Jan 2021: older than 365 days
    ]
    targets = []
    for client, fold in [("a", 0), ("b", 1), ("c", 1), ("d", 0)]:  # "d" has no transactions at all
        for mon in MONTHS:
            targets.append({"client_id": client, "fold": fold, "mon": mon.date().isoformat(),
                            "target_1": 0, "target_2": 0, "target_3": 0, "target_4": 0,
                            "trans_count": 7, "diff_trans_date": 1.0})
    t = pd.DataFrame(targets)
    t.loc[(t.client_id == "a") & (t.mon == "2022-03-31"), "target_2"] = 1  # inside (Jan 31, Apr 30]
    t.loc[(t.client_id == "b") & (t.mon == "2022-05-31"), "target_1"] = 1  # after the window
    t.loc[(t.client_id == "c") & (t.mon == "2022-02-28"), "target_4"] = 1  # first label month
    _write(pd.DataFrame(tx), tmp_path / "detail", "trx")
    _write(t, tmp_path, "targets")
    return tmp_path
