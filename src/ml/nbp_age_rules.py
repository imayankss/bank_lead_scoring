# src/rules/nbp_age_rules.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.config import PROJECT_ROOT

# Default path to the age-based NBP rules table
DEFAULT_AGE_RULES_PATH = PROJECT_ROOT / "data" / "raw" / "dim_nbp_age_rules.csv"


def load_age_rules(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the age-based NBP rules from CSV and validate the schema.
    """
    if path is None:
        path = DEFAULT_AGE_RULES_PATH

    df = pd.read_csv(path)

    expected_cols = {"age_min", "age_max", "nbp1_family", "nbp2_family", "nbp3_family"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"dim_nbp_age_rules.csv is missing required columns: {missing}")

    df = df.copy()
    df["age_min"] = df["age_min"].astype(int)
    df["age_max"] = df["age_max"].astype(int)

    return df


def assign_age_based_families(
    df: pd.DataFrame,
    age_col: str = "age",
    prefix: str = "age_rule_",
    rules: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Given a DataFrame with an 'age' column, attach age-based NBP families
    from dim_nbp_age_rules as new columns:

        age_rule_nbp1_family
        age_rule_nbp2_family
        age_rule_nbp3_family

    This is purely rule-based, no ML yet.
    """
    if age_col not in df.columns:
        raise KeyError(f"Input DataFrame is missing required age column '{age_col}'")

    if rules is None:
        rules = load_age_rules()
    rules = rules.copy()

    out = df.copy()

    # Initialise columns with NA
    nbp1_col = f"{prefix}nbp1_family"
    nbp2_col = f"{prefix}nbp2_family"
    nbp3_col = f"{prefix}nbp3_family"

    out[nbp1_col] = pd.NA
    out[nbp2_col] = pd.NA
    out[nbp3_col] = pd.NA

    # For each rule row, fill the matching age band
    for _, r in rules.iterrows():
        age_min = r["age_min"]
        age_max = r["age_max"]

        mask = (out[age_col] >= age_min) & (out[age_col] <= age_max)

        out.loc[mask, nbp1_col] = r["nbp1_family"]
        out.loc[mask, nbp2_col] = r["nbp2_family"]
        out.loc[mask, nbp3_col] = r["nbp3_family"]

    # Optional: basic sanity check – warn if some rows did not match any rule
    unmatched = out[nbp1_col].isna().sum()
    if unmatched > 0:
        print(
            f"[WARN] assign_age_based_families: {unmatched} rows did not match any age band "
            f"in dim_nbp_age_rules.csv"
        )

    return out

