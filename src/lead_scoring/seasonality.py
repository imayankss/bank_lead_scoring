"""Transaction trend and seasonality artifacts."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from lead_scoring.data import LOCAL_TS, amount_cap
from lead_scoring.paths import MBD_DIR, TABLES_DIR, ensure_directories


def build_seasonality_artifacts(mbd_dir: Path = MBD_DIR) -> pd.DataFrame:
    """Aggregate transactions by Moscow-time month in DuckDB, without loading them into memory."""
    ensure_directories()
    trx = str(mbd_dir / "detail" / "trx" / "fold=*" / "*.parquet")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    monthly = con.execute(f"""
        WITH tx AS (SELECT client_id, {LOCAL_TS} AS ts, least(amount, $cap) AS amount FROM read_parquet($trx))
        SELECT strftime(ts, '%Y-%m') AS period, year(ts) AS year, month(ts) AS month,
               count(*) AS transaction_count, sum(amount) AS total_amount, avg(amount) AS average_amount,
               count(DISTINCT client_id) AS active_customers
        FROM tx GROUP BY ALL ORDER BY period""", {"trx": trx, "cap": amount_cap(mbd_dir)}).df()
    con.close()
    monthly["rolling_3m_amount"] = monthly["total_amount"].rolling(3, min_periods=1).mean()
    # DuckDB sums in parallel, so the last float digits vary run to run; cents keep the table stable.
    money = ["total_amount", "average_amount", "rolling_3m_amount"]
    monthly[money] = monthly[money].round(2)
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
