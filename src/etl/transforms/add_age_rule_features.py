# src/etl/transform/add_age_rule_features.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ml.nbp_age_rules import assign_age_based_families



def load_frame(path: Path) -> pd.DataFrame:
    """
    Load a DataFrame from CSV or Parquet based on file suffix.
    """
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    elif path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type for {path}")


def save_frame(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame to CSV or Parquet based on file suffix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".csv"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file type for {path}")


def main(input_path: str, output_path: str, age_col: str = "age") -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)

    print(f"[INFO] Loading input dataset from {in_path}")
    df = load_frame(in_path)

    print(f"[INFO] Adding age-based NBP families using column '{age_col}'")
    df_enriched = assign_age_based_families(df, age_col=age_col)

    print(f"[INFO] Saving enriched dataset to {out_path}")
    save_frame(df_enriched, out_path)

    print("[INFO] Done. New columns added:")
    print("  - age_rule_nbp1_family")
    print("  - age_rule_nbp2_family")
    print("  - age_rule_nbp3_family")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attach age-based NBP family features to a dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input dataset (CSV or Parquet), e.g. data/warehouse/mart_customer_360.parquet",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output dataset (CSV or Parquet), e.g. data/warehouse/mart_customer_360_age_rules.parquet",
    )
    parser.add_argument(
        "--age-col",
        default="age",
        help="Name of the age column in the input dataset (default: age)",
    )

    args = parser.parse_args()
    main(args.input, args.output, age_col=args.age_col)
