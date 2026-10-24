"""Customer-level feature and target tables built from the MBD parquet files.

Transactions are aggregated in DuckDB straight from parquet, one MBD client
fold per worker process. Folds share no clients, so each is an independent
partition: the same code runs MBD-mini locally or full MBD fold by fold.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

import duckdb
import pandas as pd

from lead_scoring.config import PipelineConfig
from lead_scoring.paths import CUSTOMER_FEATURES_FILE, MBD_DIR, SCORING_FEATURES_FILE, ensure_directories
from lead_scoring.temporal import assert_events_before, month_end

TARGETS = ["target_1", "target_2", "target_3", "target_4"]
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


def _fold_features(fold: int, *, mbd_dir: Path, snapshot: str, include_target: bool,
                   config: PipelineConfig, cap: float) -> pd.DataFrame:
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
    }
    any_product = "greatest(" + ", ".join(TARGETS) + ")"
    con = duckdb.connect(config={"enable_progress_bar": False})
    clients = con.execute(f"""
        SELECT client_id,
               sum({any_product}) FILTER (WHERE mon < $snap_mon) AS products_taken_before,
               {", ".join(f"sum({t}) FILTER (WHERE mon < $snap_mon) AS {t}_before" for t in TARGETS)},
               max({any_product}) FILTER (WHERE mon > $snap_mon AND mon <= $horizon_mon) AS converted
        FROM read_parquet($targets) GROUP BY client_id""",
        {k: params[k] for k in ("targets", "snap_mon", "horizon_mon")}).df()
    tx = con.execute(f"""
        WITH tx AS (SELECT client_id, {LOCAL_TS} AS ts, least(amount, $cap) AS amount FROM read_parquet($trx))
        SELECT client_id,
               sum(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS total_txn_amt_365,
               count(*) FILTER (WHERE ts >= $start AND ts < $cutoff) AS txn_count_365,
               avg(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS avg_txn_amt_365,
               max(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS max_txn_amt_365,
               count(DISTINCT CAST(ts AS DATE)) FILTER (WHERE ts >= $start AND ts < $cutoff) AS active_days_365,
               count(*) FILTER (WHERE ts >= $start90 AND ts < $cutoff) AS txn_count_90,
               sum(amount) FILTER (WHERE ts >= $start90 AND ts < $cutoff) AS total_txn_amt_90,
               min(ts) FILTER (WHERE ts < $cutoff) AS first_txn,
               max(ts) FILTER (WHERE ts < $cutoff) AS last_txn,
               sum(amount) FILTER (WHERE ts >= $cutoff AND ts < $horizon_cutoff) AS future_value
        FROM tx GROUP BY client_id""",
        {k: params[k] for k in ("trx", "cap", "start", "start90", "cutoff", "horizon_cutoff")}).df()
    con.close()

    df = clients.merge(tx, on="client_id", how="left").rename(columns={"client_id": "customer_id"})
    assert_events_before(df["last_txn"], snap)
    counts = [c for c in df.columns if c.endswith(("_365", "_90", "_before")) or c in ("converted", "future_value")]
    df[counts] = df[counts].fillna(0)
    df["recency_days"] = (snap - df["last_txn"]).dt.days.fillna(config.lookback_days + 1).clip(lower=0)
    df["tenure_days"] = (snap - df["first_txn"]).dt.days.fillna(0).clip(lower=0)
    df["fold"] = fold
    if include_target:
        df["converted"] = df["converted"].astype(int)
    else:
        df = df.drop(columns=["converted", "future_value"])
    return df


def build_customer_features(
    snapshot_date: str,
    *,
    include_target: bool = True,
    config: PipelineConfig = PipelineConfig(),
    mbd_dir: Path = MBD_DIR,
    amount_cap: float,
) -> pd.DataFrame:
    """One row per client: features known at the snapshot and, optionally, the next-horizon targets."""
    work = partial(_fold_features, mbd_dir=mbd_dir, snapshot=snapshot_date, include_target=include_target,
                   config=config, cap=amount_cap)
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
    snapshots = [*config.train_snapshots, config.test_snapshot]
    training = pd.concat(
        [build_customer_features(s, config=config, mbd_dir=mbd_dir, amount_cap=cap) for s in snapshots],
        ignore_index=True,
    )
    scoring = build_customer_features(config.score_snapshot, include_target=False, config=config,
                                      mbd_dir=mbd_dir, amount_cap=cap)
    training.to_parquet(CUSTOMER_FEATURES_FILE, index=False)
    scoring.to_parquet(SCORING_FEATURES_FILE, index=False)

    metadata = {
        "training_rows": len(training),
        "scoring_rows": len(scoring),
        "positive_training_rows": int(training["converted"].sum()),
        "amount_cap": cap,
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
