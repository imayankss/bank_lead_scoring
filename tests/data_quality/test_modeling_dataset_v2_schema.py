import os
from pathlib import Path

import pandas as pd
import pytest

from src.etl.schemas import validate_modeling_dataset_v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_v2.csv"


@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="modeling_dataset_v2.csv not generated yet; run ETL first.",
)
def test_modeling_dataset_v2_schema_valid():
    df = pd.read_csv(DATA_PATH)

    # Pandera validation (types + ranges)
    validated = validate_modeling_dataset_v2(df)

    # Explicit non-null target check
    assert validated["y_high_cltv"].notna().all()
    assert validated["cltv_profit"].notna().all()

    # CLTV should be non-negative (extra safety)
    assert (validated["cltv_profit"] >= 0).all()

    # Lead score should be within 0–100
    assert ((validated["lead_score_0_100"] >= 0) & (validated["lead_score_0_100"] <= 100)).all()

