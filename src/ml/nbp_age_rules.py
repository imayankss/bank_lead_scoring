# src/ml/nbp_age_rules.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

from src.common.config import PROJECT_ROOT


# Path to your age rules CSV
AGE_RULES_PATH = PROJECT_ROOT / "data" / "raw" / "dim_nbp_age_rules.csv"


def load_age_rules(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load age → NBP family rules from dim_nbp_age_rules.csv.

    Expected columns:
        age_min, age_max, nbp1_family, nbp2_family, nbp3_family
    """
    rules_path = Path(path) if path is not None else AGE_RULES_PATH
    rules = pd.read_csv(rules_path)

    # Basic sanity checks
    required_cols = {
        "age_min",
        "age_max",
        "nbp1_family",
        "nbp2_family",
        "nbp3_family",
    }
    missing = required_cols.difference(rules.columns)
    if missing:
        raise ValueError(f"dim_nbp_age_rules.csv is missing columns: {missing}")

    return rules


def _lookup_age_row(age: float, rules: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return the rule row where age_min <= age <= age_max.
    If no match, return None.
    """
    if pd.isna(age):
        return None

    mask = (rules["age_min"] <= age) & (age <= rules["age_max"])
    matched = rules[mask]

    if matched.empty:
        return None

    # In case of overlaps, take the first
    return matched.iloc[0]


def map_age_to_nbp_families(
    age: float,
    rules: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Given an age and the rules DataFrame, return
    (nbp1_family, nbp2_family, nbp3_family).
    """
    row = _lookup_age_row(age, rules)
    if row is None:
        return None, None, None

    return (
        row["nbp1_family"],
        row["nbp2_family"],
        row["nbp3_family"],
    )


def attach_age_nbp_families(
    df: pd.DataFrame,
    rules: Optional[pd.DataFrame] = None,
    age_col: str = "age",
    prefix: str = "age_",
) -> pd.DataFrame:
    """
    Given a DataFrame with an `age` column, add:
        prefix + "nbp1_family"
        prefix + "nbp2_family"
        prefix + "nbp3_family"

    Returns a copy of df with new columns.

    Typical usage:
        df_enriched = attach_age_nbp_families(df_leads)
    """
    if rules is None:
        rules = load_age_rules()

    if age_col not in df.columns:
        raise KeyError(
            f"Expected column '{age_col}' in DataFrame, "
            f"available columns: {list(df.columns)}"
        )

    df_out = df.copy()

    def _row_mapper(a: float):
        return map_age_to_nbp_families(a, rules)

    mapped = df_out[age_col].apply(_row_mapper)
    df_out[f"{prefix}nbp1_family"] = mapped.apply(lambda t: t[0])
    df_out[f"{prefix}nbp2_family"] = mapped.apply(lambda t: t[1])
    df_out[f"{prefix}nbp3_family"] = mapped.apply(lambda t: t[2])

    return df_out
