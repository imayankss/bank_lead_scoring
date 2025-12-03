from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.common.config import settings, PROJECT_ROOT  # PROJECT_ROOT is defined in config.py


# Path to the unified leads modeling dataset
UNIFIED_LEADS_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_leads_unified_age_rules.csv"

# Main classification target for the unified lead-scoring model
TARGET_COL = "label_conv"


def load_unified_leads_dataset(
    path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the unified leads modeling dataset and return (X, y).

    - y = label_conv (0/1: whether the lead eventually converted)
    - X = numeric feature columns, excluding:
        * label columns
        * obvious ID / personally-identifying numeric fields
    """
    csv_path = Path(path) if path is not None else UNIFIED_LEADS_PATH
    df = pd.read_csv(csv_path)

    # Encode age-rule NBP families as categorical codes for ML
    age_rule_cols = [
    "age_rule_nbp1_family",
    "age_rule_nbp2_family",
    "age_rule_nbp3_family",
    ]

    for col in age_rule_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes


    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL!r} not found in {csv_path}")

    # Target: main conversion label
    y = df[TARGET_COL].astype(int)

    # All numeric columns
    num_cols = df.select_dtypes(include="number").columns

    # Columns we do NOT want as features (IDs / PII / labels)
    EXCLUDE_NUMERIC = {
        "Phone",
        "aadhaar_no",
        "lead_id",
        "source_id",
        "chosen_product_id",
        "best_first_product_id",
        "label_conv",
        "label_conv_90d",
    }

    feature_cols = [c for c in num_cols if c not in EXCLUDE_NUMERIC]

    X = df[feature_cols].copy().fillna(0)

    return X, y
