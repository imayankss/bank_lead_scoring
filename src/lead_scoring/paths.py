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

RAW_CUSTOMERS_FILE = RAW_DATA_DIR / "customers.csv"
RAW_ACCOUNTS_FILE = RAW_DATA_DIR / "account_master.csv"
RAW_TRANSACTIONS_HISTORY_FILE = RAW_DATA_DIR / "transactions_history.csv"
RAW_TRANSACTIONS_RECENT_FILE = RAW_DATA_DIR / "transactions_recent.csv"
RAW_TRANSACTION_CHANGE_LOG_FILE = RAW_DATA_DIR / "transaction_change_log.csv"

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
