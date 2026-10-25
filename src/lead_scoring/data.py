"""Customer-level feature and target tables built from the MBD parquet files.

Transactions are aggregated in DuckDB straight from parquet, one MBD client
fold per worker process. Folds share no clients, so each is an independent
partition: the same code runs MBD-mini locally or full MBD fold by fold.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from lead_scoring.config import PipelineConfig
from lead_scoring.paths import CUSTOMER_FEATURES_FILE, MBD_DIR, SCORING_FEATURES_FILE, ensure_directories
from lead_scoring.temporal import assert_events_before, month_end

TARGETS = ["target_1", "target_2", "target_3", "target_4"]
CATEGORY_COLUMNS = ["event_type", "event_subtype", "currency", "src_type11", "src_type12", "dst_type11",
                    "dst_type12", "src_type21", "src_type22", "src_type31", "src_type32"]
TOP_CATEGORIES = 10  # top 20 would hold ~3 GB of int64 during the build on an 8 GB machine
# Any product within the horizon (headline), each product within the horizon, each product next month.
LABELS = ["converted", *(f"y3_{t}" for t in TARGETS), *(f"y1_{t}" for t in TARGETS)]
# MBD stores event_time in UTC; the bank's day boundaries are Moscow time.
LOCAL_TS = "event_time + INTERVAL 3 HOUR"


def list_folds(mbd_dir: Path = MBD_DIR) -> list[int]:
    """Client folds present in the MBD targets table."""
    folds = sorted(int(p.name.split("=")[1]) for p in (mbd_dir / "targets").glob("fold=*"))
    if not folds:
        raise FileNotFoundError(f"No MBD folds under {mbd_dir / 'targets'}; run scripts/fetch_mbd.py")
    return folds


def amount_cap(mbd_dir: Path = MBD_DIR, before: str = PipelineConfig.test_snapshot) -> float:
    """Upper bound applied to every transaction amount before aggregation.

    MBD adds noise to amounts (max is about 1.4e12), so a few values dominate any sum.
    """
    # 0.999 trims only the noise tail; 0.99 would also flatten genuinely large clients, who matter most for lead value.
    # Only amounts up to the end of the `before` day count, so the cap never sees the test months.
    trx = str(mbd_dir / "detail" / "trx" / "fold=*" / "*.parquet")
    cutoff = pd.Timestamp(before).normalize() + pd.Timedelta(days=1)
    query = f"SELECT quantile_cont(amount, 0.999) FROM read_parquet($trx) WHERE {LOCAL_TS} < $cutoff"
    return float(duckdb.execute(query, {"trx": trx, "cutoff": cutoff}).fetchone()[0])


def category_vocab(mbd_dir: Path = MBD_DIR, before: str = PipelineConfig.fit_snapshots[-1],
                   top: int = TOP_CATEGORIES) -> dict[str, list[int]]:
    """Most frequent values of each categorical column, counting events up to the end of the `before` day."""
    trx = str(mbd_dir / "detail" / "trx" / "fold=*" / "*.parquet")
    cutoff = pd.Timestamp(before).normalize() + pd.Timedelta(days=1)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    vocab = {}
    for col in CATEGORY_COLUMNS:
        rows = con.execute(f"""SELECT {col} FROM read_parquet($trx)
                               WHERE {LOCAL_TS} < $cutoff AND {col} IS NOT NULL
                               GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {int(top)}""",
                           {"trx": trx, "cutoff": cutoff}).fetchall()
        vocab[col] = [int(r[0]) for r in rows]
    con.close()
    return vocab


def _block(i: int) -> str:
    """Count and amount in the i-th 30-day block before the snapshot (1 = the latest)."""
    upper = "$cutoff" if i == 1 else f"$d30_{i - 1}"
    window = f"ts >= $d30_{i} AND ts < {upper}"
    return f"count(*) FILTER (WHERE {window}) AS txn_count_d30_{i}, sum(amount) FILTER (WHERE {window}) AS txn_amt_d30_{i}"


def _fold_features(fold: int, *, mbd_dir: Path, snapshot: str, include_target: bool,
                   config: PipelineConfig, cap: float, vocab: dict[str, list[int]]) -> pd.DataFrame:
    snap = pd.Timestamp(snapshot)
    day_after = snap + pd.Timedelta(days=1)
    horizon_end = month_end(snap, config.horizon_months)
    trx = str(mbd_dir / "detail" / "trx" / f"fold={fold}" / "*.parquet")
    targets = str(mbd_dir / "targets" / f"fold={fold}" / "*.parquet")
    params = {
        "trx": trx, "targets": targets, "cap": cap,
        "start": day_after - pd.Timedelta(days=config.lookback_days), "start90": day_after - pd.Timedelta(days=90),
        "cutoff": day_after, "horizon_cutoff": horizon_end + pd.Timedelta(days=1),
        "snap_mon": snap.date().isoformat(), "horizon_mon": horizon_end.date().isoformat(),
        "next_mon": month_end(snap, 1).date().isoformat(),
        # The two label months before the snapshot month; the snapshot month itself is never used.
        "m1": month_end(snap, -1).date().isoformat(), "m2": month_end(snap, -2).date().isoformat(),
        **{f"d30_{i}": day_after - pd.Timedelta(days=30 * i) for i in range(1, 7)},
    }
    any_product = "greatest(" + ", ".join(TARGETS) + ")"
    history = [*((t, t) for t in TARGETS), ("any", any_product)]
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    first_label = str(con.execute("SELECT min(mon) FROM read_parquet($targets)", {"targets": targets}).fetchone()[0])
    clients = con.execute(f"""
        SELECT client_id,
               sum({any_product}) FILTER (WHERE mon < $snap_mon) AS products_taken_before,
               {", ".join(f"sum({t}) FILTER (WHERE mon < $snap_mon) AS {t}_before" for t in TARGETS)},
               {", ".join(f"max({col}) FILTER (WHERE mon = ${m}) AS hist_{name}_{m}" for name, col in history for m in ("m1", "m2"))},
               max({any_product}) FILTER (WHERE mon > $snap_mon AND mon <= $horizon_mon) AS converted,
               {", ".join(f"max({t}) FILTER (WHERE mon > $snap_mon AND mon <= $horizon_mon) AS y3_{t}" for t in TARGETS)},
               {", ".join(f"max({t}) FILTER (WHERE mon = $next_mon) AS y1_{t}" for t in TARGETS)}
        FROM read_parquet($targets) GROUP BY client_id""",
        {k: params[k] for k in ("targets", "snap_mon", "horizon_mon", "next_mon", "m1", "m2")}).df()
    aggs = [
        "sum(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS total_txn_amt_365",
        "count(*) FILTER (WHERE ts >= $start AND ts < $cutoff) AS txn_count_365",
        "avg(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS avg_txn_amt_365",
        "max(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS max_txn_amt_365",
        "count(DISTINCT CAST(ts AS DATE)) FILTER (WHERE ts >= $start AND ts < $cutoff) AS active_days_365",
        "count(*) FILTER (WHERE ts >= $start90 AND ts < $cutoff) AS txn_count_90",
        "sum(amount) FILTER (WHERE ts >= $start90 AND ts < $cutoff) AS total_txn_amt_90",
        "min(ts) FILTER (WHERE ts < $cutoff) AS first_txn",
        "max(ts) FILTER (WHERE ts < $cutoff) AS last_txn",
        "sum(amount) FILTER (WHERE ts >= $cutoff AND ts < $horizon_cutoff) AS future_value",
        *(_block(i) for i in range(1, 7)),
        *(f"count(*) FILTER (WHERE ts >= ${w} AND ts < $cutoff AND {col} = {int(code)}) AS cat_{col}_{int(code)}_{days}"
          for col, codes in vocab.items() for code in codes for w, days in (("start", 365), ("start90", 90))),
        *(f"max(ts) FILTER (WHERE ts < $cutoff AND event_type = {int(code)}) AS last_evt_{int(code)}"
          for code in vocab["event_type"]),
    ]
    tx = con.execute(f"""
        WITH tx AS (SELECT client_id, {LOCAL_TS} AS ts, least(amount, $cap) AS amount, {", ".join(CATEGORY_COLUMNS)}
                    FROM read_parquet($trx))
        SELECT client_id, {", ".join(aggs)} FROM tx GROUP BY client_id""",
        {k: params[k] for k in ("trx", "cap", "start", "start90", "cutoff", "horizon_cutoff", *(f"d30_{i}" for i in range(1, 7)))}).df()
    con.close()

    df = clients.merge(tx, on="client_id", how="left").rename(columns={"client_id": "customer_id"})
    assert_events_before(df["last_txn"], snap)
    filled = [c for c in df.columns if c.endswith(("_365", "_90", "_before"))
              or c.startswith(("cat_", "txn_count_d30_", "txn_amt_d30_")) or c in (*LABELS, "future_value")]
    df[filled] = df[filled].fillna(0)
    for m in ("m1", "m2"):
        cols = [c for c in df.columns if c.startswith("hist_") and c.endswith(f"_{m}")]
        # A month before the first label month has no labels at all: unknown, not "bought nothing".
        df[cols] = np.nan if params[m] < first_label else df[cols].fillna(0)
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    df[cat_cols] = df[cat_cols].astype("int32")
    df["recency_days"] = (snap - df["last_txn"]).dt.days.fillna(config.lookback_days + 1).clip(lower=0)
    df["tenure_days"] = (snap - df["first_txn"]).dt.days.fillna(0).clip(lower=0)
    df["ratio_30_90"] = df["txn_count_d30_1"] / df["txn_count_90"].clip(lower=1)
    df["ratio_90_365"] = df["txn_count_90"] / df["txn_count_365"].clip(lower=1)
    for code in vocab["event_type"]:
        last = pd.to_datetime(df.pop(f"last_evt_{int(code)}"))
        df[f"days_since_evt_{int(code)}"] = (snap - last).dt.days.fillna(config.lookback_days + 1).clip(lower=0)
    df["fold"] = fold
    if include_target:
        df[LABELS] = df[LABELS].astype(int)
    else:
        df = df.drop(columns=["future_value", *LABELS])
    return df


def build_customer_features(
    snapshot_date: str,
    *,
    include_target: bool = True,
    config: PipelineConfig = PipelineConfig(),
    mbd_dir: Path = MBD_DIR,
    amount_cap: float,
    vocab: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """One row per client: features known at the snapshot and, optionally, the next-horizon targets."""
    vocab = vocab if vocab is not None else category_vocab(mbd_dir, before=config.fit_snapshots[-1])
    work = partial(_fold_features, mbd_dir=mbd_dir, snapshot=snapshot_date, include_target=include_target,
                   config=config, cap=amount_cap, vocab=vocab)
    folds = list_folds(mbd_dir)
    if config.workers > 1:
        with ProcessPoolExecutor(max_workers=min(config.workers, len(folds))) as pool:
            parts = list(pool.map(work, folds))
    else:
        parts = [work(f) for f in folds]
    df = pd.concat(parts, ignore_index=True)
    df["snapshot_date"] = pd.Timestamp(snapshot_date).date().isoformat()
    return df.sort_values("customer_id").reset_index(drop=True)


def save_customer_features(config: PipelineConfig = PipelineConfig(), mbd_dir: Path = MBD_DIR) -> tuple[Path, Path]:
    """Build stacked training/test snapshots and the current scoring table."""
    ensure_directories()
    cap = amount_cap(mbd_dir, before=config.test_snapshot)
    vocab = category_vocab(mbd_dir, before=config.fit_snapshots[-1])
    snapshots = [*config.train_snapshots, config.test_snapshot]
    training = pd.concat(
        [build_customer_features(s, config=config, mbd_dir=mbd_dir, amount_cap=cap, vocab=vocab) for s in snapshots],
        ignore_index=True,
    )
    scoring = build_customer_features(config.score_snapshot, include_target=False, config=config,
                                      mbd_dir=mbd_dir, amount_cap=cap, vocab=vocab)
    training.to_parquet(CUSTOMER_FEATURES_FILE, index=False)
    scoring.to_parquet(SCORING_FEATURES_FILE, index=False)

    metadata = {
        "training_rows": len(training),
        "scoring_rows": len(scoring),
        "positive_training_rows": int(training["converted"].sum()),
        "amount_cap": cap,
        "category_vocab": json.dumps(vocab),
        **{k: str(v) for k, v in asdict(config).items()},
    }
    pd.DataFrame([metadata]).to_csv(CUSTOMER_FEATURES_FILE.with_suffix(".metadata.csv"), index=False)
    return CUSTOMER_FEATURES_FILE, SCORING_FEATURES_FILE


def main() -> None:
    paths = save_customer_features()
    print(f"Saved training features to {paths[0]}")
    print(f"Saved scoring features to {paths[1]}")


if __name__ == "__main__":
    main()
