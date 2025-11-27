from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


# ---------------------------------------------------------------------
# CONFIGURE THIS TO MATCH YOUR POSTGRES
# ---------------------------------------------------------------------
# Example for local Postgres:
#   user: bls_user
#   password: bls_pass
#   host: localhost
#   port: 5432
#   database: bls_db
PG_CONN_STR = "postgresql://postgres:postgres@localhost:5432/bank_dw"
SCHEMA = "public"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LEADS_CSV = PROCESSED_DIR / "modeling_dataset_leads_unified.csv"
CUSTOMERS_CSV = PROCESSED_DIR / "modeling_dataset_customers_universe.csv"


def load_csv_to_table(csv_path: Path, table_name: str, if_exists: str = "replace") -> None:
    if not csv_path.exists():
        print(f"[WARN] File not found, skipping: {csv_path}")
        return

    print(f"[LOAD] {csv_path} -> {SCHEMA}.{table_name}")
    df = pd.read_csv(csv_path)

    engine = create_engine(PG_CONN_STR)
    with engine.begin() as conn:
        df.to_sql(
            table_name,
            con=conn,
            schema=SCHEMA,
            if_exists=if_exists,
            index=False,
        )
    print(f"[OK] Loaded {len(df)} rows into {SCHEMA}.{table_name}")


def main() -> None:
    load_csv_to_table(LEADS_CSV, "mart_leads_unified", if_exists="replace")
    load_csv_to_table(CUSTOMERS_CSV, "mart_customers_universe", if_exists="replace")


if __name__ == "__main__":
    main()
