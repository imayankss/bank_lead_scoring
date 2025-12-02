from __future__ import annotations

from typing import Tuple, List

import duckdb
import pandas as pd

from src.common.config import settings

# Use central config instead of hard-coded string
DB_PATH = str(settings.project.db_path)

# Core feature list used by all models (current V2 feature set)
FEATURES_V2: List[str] = [
    "acct_cnt", "sa_cnt", "ca_cnt",
    "txn_cnt_12m", "txn_months_12m", "txn_amt_12m", "avg_txn_amt_12m",
    "std_txn_amt_12m", "med_txn_amt_12m", "debit_cnt_12m", "credit_cnt_12m",
    "recency_days", "chrg_amt_12m", "chrg_months_12m",
    "avg_bal_12m_savings", "avg_bal_12m_current_ac", "avg_bal_12m_td",
]


def _resolve_db_path(db_path: str | None = None) -> str:
    """
    Helper to resolve DuckDB path from config or override.
    """
    if db_path is None:
        return DB_PATH
    return str(db_path)


def load_classification_dataset(
    db_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns X, y for lead-scoring classification (label = top-20% by CLTV).
    Uses table ans.modeling_dataset_v2 built earlier.
    """
    db_path = _resolve_db_path(db_path)
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute("SELECT * FROM ans.modeling_dataset_v2").fetchdf()
    finally:
        con.close()

    X = df[FEATURES_V2].fillna(0)
    y = df["y_high_cltv"].astype(int)
    return X, y


def load_regression_dataset(
    db_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns X, y for CLTV regression.
    Uses table ans.modeling_dataset_v2 built earlier.
    """
    db_path = _resolve_db_path(db_path)
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute("SELECT * FROM ans.modeling_dataset_v2").fetchdf()
    finally:
        con.close()

    X = df[FEATURES_V2].fillna(0)
    y_reg = df["cltv_profit"].astype(float)
    return X, y_reg


def write_scores(
    table_name: str,
    scores: pd.DataFrame,
    db_path: str | None = None,
) -> None:
    """
    Writes model output scores back into the DuckDB database.
    """
    db_path = _resolve_db_path(db_path)
    con = duckdb.connect(db_path)
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("__scores", scores)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM __scores")
        con.unregister("__scores")
    finally:
        con.close()
