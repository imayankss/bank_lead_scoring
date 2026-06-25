"""Transaction trend and seasonality artifacts."""

from __future__ import annotations

import pandas as pd

from lead_scoring.data import load_raw_tables
from lead_scoring.paths import TABLES_DIR, ensure_directories


def build_seasonality_artifacts() -> pd.DataFrame:
    """Aggregate transaction history by month for dashboard trend visuals."""
    ensure_directories()
    _, _, transactions = load_raw_tables()
    tx = transactions.copy()
    tx["period"] = tx["transaction_date"].dt.to_period("M").astype(str)
    tx["month"] = tx["transaction_date"].dt.month
    tx["year"] = tx["transaction_date"].dt.year

    monthly = (
        tx.groupby(["period", "year", "month"])
        .agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
            active_customers=("customer_id", "nunique"),
        )
        .reset_index()
        .sort_values("period")
    )
    monthly["rolling_3m_amount"] = monthly["total_amount"].rolling(3, min_periods=1).mean()
    monthly.to_csv(TABLES_DIR / "seasonality_monthly.csv", index=False)

    month_pattern = (
        monthly.groupby("month")
        .agg(avg_total_amount=("total_amount", "mean"), avg_transaction_count=("transaction_count", "mean"))
        .reset_index()
    )
    month_pattern.to_csv(TABLES_DIR / "seasonality_pattern.csv", index=False)
    return monthly


def main() -> None:
    output = build_seasonality_artifacts()
    print(f"Saved {len(output)} monthly seasonality rows.")


if __name__ == "__main__":
    main()
