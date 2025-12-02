import os
from pathlib import Path

import pandas as pd
import pytest

from src.etl.schemas import validate_modeling_dataset_leads_unified

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_leads_unified.csv"


@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="modeling_dataset_leads_unified.csv not generated yet; run ETL first.",
)
def test_modeling_dataset_leads_unified_schema_valid():
    df = pd.read_csv(DATA_PATH)

    validated = validate_modeling_dataset_leads_unified(df)

    # IDs and labels should not be null
    for col in ["Customer_ID", "lead_id", "label_conv", "label_conv_90d"]:
        assert validated[col].notna().all(), f"Null values found in {col}"

    # Labels must be 0 or 1
    assert set(validated["label_conv"].unique()) <= {0, 1}
    assert set(validated["label_conv_90d"].unique()) <= {0, 1}
