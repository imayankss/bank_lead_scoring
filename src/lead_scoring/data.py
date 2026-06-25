"""Raw data loading and customer-level feature assembly."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from lead_scoring.config import PipelineConfig
from lead_scoring.paths import (
    CUSTOMER_FEATURES_FILE,
    RAW_ACCOUNTS_FILE,
    RAW_CUSTOMERS_FILE,
    RAW_TRANSACTIONS_HISTORY_FILE,
    RAW_TRANSACTIONS_RECENT_FILE,
    SCORING_FEATURES_FILE,
    ensure_directories,
)


DATE_COL = "transaction_date"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent snake_case columns."""
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


def load_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load customers, accounts, and combined transactions."""
    customers = normalize_columns(pd.read_csv(RAW_CUSTOMERS_FILE, low_memory=False))
    accounts = normalize_columns(pd.read_csv(RAW_ACCOUNTS_FILE, low_memory=False))
    tx_hist = normalize_columns(pd.read_csv(RAW_TRANSACTIONS_HISTORY_FILE, low_memory=False))
    tx_recent = normalize_columns(pd.read_csv(RAW_TRANSACTIONS_RECENT_FILE, low_memory=False))

    transactions = pd.concat([tx_hist, tx_recent], ignore_index=True, sort=False)
    return customers, accounts, prepare_transactions(transactions)


def prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Clean transaction dates, amounts, and statuses."""
    tx = normalize_columns(transactions)
    if DATE_COL not in tx.columns:
        candidates = [c for c in tx.columns if "date" in c]
        if not candidates:
            raise ValueError("No transaction date column found.")
        tx = tx.rename(columns={candidates[0]: DATE_COL})

    required = {"customer_id", "amount", DATE_COL}
    missing = sorted(required - set(tx.columns))
    if missing:
        raise ValueError(f"Missing required transaction columns: {missing}")

    tx[DATE_COL] = pd.to_datetime(tx[DATE_COL], errors="coerce")
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    tx = tx.dropna(subset=["customer_id", DATE_COL])

    if "transaction_status" in tx.columns:
        completed = tx["transaction_status"].astype(str).str.lower().eq("completed")
        tx = tx[completed].copy()

    return tx.sort_values(["customer_id", DATE_COL]).reset_index(drop=True)


def _empty_tx_agg() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "customer_id",
            "total_txn_amt_365",
            "txn_count_365",
            "avg_txn_amt_365",
            "max_txn_amt_365",
            "credit_txn_count_365",
            "debit_txn_count_365",
            "active_days_365",
            "first_txn",
            "last_txn",
        ]
    )


def aggregate_transactions(transactions: pd.DataFrame, snapshot_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    """Aggregate customer activity in the lookback window ending at snapshot_date."""
    feature_start = snapshot_date - pd.Timedelta(days=lookback_days)
    window = transactions[
        (transactions[DATE_COL] > feature_start)
        & (transactions[DATE_COL] <= snapshot_date)
    ].copy()
    if window.empty:
        return _empty_tx_agg()

    transaction_type = (
        window["transaction_type"].astype(str).str.lower()
        if "transaction_type" in window.columns
        else pd.Series("", index=window.index)
    )
    window["is_credit"] = transaction_type.eq("credit").astype(int)
    window["is_debit"] = transaction_type.eq("debit").astype(int)

    agg = (
        window.groupby("customer_id")
        .agg(
            total_txn_amt_365=("amount", "sum"),
            txn_count_365=("amount", "count"),
            avg_txn_amt_365=("amount", "mean"),
            max_txn_amt_365=("amount", "max"),
            credit_txn_count_365=("is_credit", "sum"),
            debit_txn_count_365=("is_debit", "sum"),
            active_days_365=(DATE_COL, lambda x: x.dt.date.nunique()),
            first_txn=(DATE_COL, "min"),
            last_txn=(DATE_COL, "max"),
        )
        .reset_index()
    )
    return agg


def aggregate_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate account-level customer signals."""
    accounts = normalize_columns(accounts)
    if accounts.empty or "customer_id" not in accounts.columns:
        return pd.DataFrame(columns=["customer_id"])

    accounts["balance"] = pd.to_numeric(accounts.get("balance", 0), errors="coerce").fillna(0.0)
    status = accounts.get("account_status", pd.Series("", index=accounts.index)).astype(str).str.lower()
    accounts["is_active_account"] = status.eq("active").astype(int)
    accounts["is_dormant_account"] = status.eq("dormant").astype(int)
    accounts["is_closed_account"] = status.eq("closed").astype(int)

    return (
        accounts.groupby("customer_id")
        .agg(
            account_count=("account_id", "count"),
            active_account_count=("is_active_account", "sum"),
            dormant_account_count=("is_dormant_account", "sum"),
            closed_account_count=("is_closed_account", "sum"),
            total_balance=("balance", "sum"),
            avg_balance=("balance", "mean"),
        )
        .reset_index()
    )


def build_target(transactions: pd.DataFrame, snapshot_date: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    """Build future revenue and conversion labels after the snapshot date."""
    target_end = snapshot_date + pd.Timedelta(days=horizon_days)
    target_tx = transactions[
        (transactions[DATE_COL] > snapshot_date)
        & (transactions[DATE_COL] <= target_end)
    ]
    if target_tx.empty:
        return pd.DataFrame(columns=["customer_id", "future_revenue_12m", "converted_12m"])

    targets = (
        target_tx.groupby("customer_id")
        .agg(future_revenue_12m=("amount", "sum"))
        .reset_index()
    )
    targets["converted_12m"] = (targets["future_revenue_12m"] > 0).astype(int)
    return targets


def build_customer_features(
    *,
    snapshot_date: pd.Timestamp | None = None,
    include_target: bool = True,
    config: PipelineConfig = PipelineConfig(),
) -> pd.DataFrame:
    """Create a customer-level modeling or scoring table."""
    customers, accounts, transactions = load_raw_tables()
    max_txn_date = transactions[DATE_COL].max().normalize()
    if snapshot_date is None:
        snapshot_date = max_txn_date - pd.Timedelta(days=config.target_horizon_days) if include_target else max_txn_date
    snapshot_date = pd.Timestamp(snapshot_date).normalize()

    base = customers.copy()
    tx_agg = aggregate_transactions(transactions, snapshot_date, config.lookback_days)
    account_agg = aggregate_accounts(accounts)

    dataset = base.merge(tx_agg, on="customer_id", how="left").merge(account_agg, on="customer_id", how="left")

    numeric_defaults = {
        "total_txn_amt_365": 0.0,
        "txn_count_365": 0,
        "avg_txn_amt_365": 0.0,
        "max_txn_amt_365": 0.0,
        "credit_txn_count_365": 0,
        "debit_txn_count_365": 0,
        "active_days_365": 0,
        "account_count": 0,
        "active_account_count": 0,
        "dormant_account_count": 0,
        "closed_account_count": 0,
        "total_balance": 0.0,
        "avg_balance": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in dataset.columns:
            dataset[col] = default
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce").fillna(default)

    dataset["last_txn"] = pd.to_datetime(dataset["last_txn"], errors="coerce")
    dataset["first_txn"] = pd.to_datetime(dataset["first_txn"], errors="coerce")
    dataset["recency_days"] = (snapshot_date - dataset["last_txn"]).dt.days
    dataset["recency_days"] = dataset["recency_days"].fillna(config.lookback_days + 1).clip(lower=0)
    dataset["tenure_days"] = (snapshot_date - dataset["first_txn"]).dt.days
    dataset["tenure_days"] = dataset["tenure_days"].fillna(0).clip(lower=0)

    dataset["dob"] = pd.to_datetime(dataset.get("dob"), errors="coerce")
    dataset["age"] = ((snapshot_date - dataset["dob"]).dt.days // 365).astype("float")
    dataset["date_of_lead"] = pd.to_datetime(dataset.get("date_of_lead"), errors="coerce")
    dataset["lead_age_days"] = (snapshot_date - dataset["date_of_lead"]).dt.days
    dataset["lead_age_days"] = dataset["lead_age_days"].clip(lower=0)

    if include_target:
        targets = build_target(transactions, snapshot_date, config.target_horizon_days)
        dataset = dataset.merge(targets, on="customer_id", how="left")
        dataset["future_revenue_12m"] = pd.to_numeric(dataset["future_revenue_12m"], errors="coerce").fillna(0.0)
        dataset["converted_12m"] = pd.to_numeric(dataset["converted_12m"], errors="coerce").fillna(0).astype(int)

    dataset["snapshot_date"] = snapshot_date.date().isoformat()
    dataset["target_window_days"] = config.target_horizon_days if include_target else 0
    return dataset


def save_customer_features(config: PipelineConfig = PipelineConfig()) -> tuple[Path, Path]:
    """Build and save training and current scoring feature tables."""
    ensure_directories()
    training_features = build_customer_features(include_target=True, config=config)
    scoring_features = build_customer_features(include_target=False, config=config)

    training_features.to_parquet(CUSTOMER_FEATURES_FILE, index=False)
    scoring_features.to_parquet(SCORING_FEATURES_FILE, index=False)

    metadata = pd.DataFrame(
        [
            {
                "training_rows": len(training_features),
                "scoring_rows": len(scoring_features),
                "positive_training_customers": int(training_features["converted_12m"].sum()),
                **asdict(config),
            }
        ]
    )
    metadata.to_csv(CUSTOMER_FEATURES_FILE.with_suffix(".metadata.csv"), index=False)
    return CUSTOMER_FEATURES_FILE, SCORING_FEATURES_FILE


def main() -> None:
    paths = save_customer_features()
    print(f"Saved training features to {paths[0]}")
    print(f"Saved scoring features to {paths[1]}")


if __name__ == "__main__":
    main()
