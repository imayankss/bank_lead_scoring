"""Central project paths used by the pipeline and dashboard."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREPROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "preprocessed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# MBD-mini (Sber AI Lab, CC BY 4.0), downloaded by scripts/fetch_mbd.py and never committed.
MBD_DIR = RAW_DATA_DIR / "mbd"

CUSTOMER_FEATURES_FILE = PROCESSED_DATA_DIR / "customer_features.parquet"
SCORING_FEATURES_FILE = PROCESSED_DATA_DIR / "scoring_features.parquet"
LEAD_SCORES_FILE = PROCESSED_DATA_DIR / "lead_scores.csv"
FEATURE_METADATA_FILE = PREPROCESSED_DATA_DIR / "feature_metadata.json"

CLTV_MODEL_FILE = MODEL_DIR / "cltv_model.pkl"
PROPENSITY_MODEL_FILE = MODEL_DIR / "propensity_model.pkl"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"


def ensure_directories() -> None:
    """Create pipeline output directories."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PREPROCESSED_DATA_DIR,
        MODEL_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        DOCS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
