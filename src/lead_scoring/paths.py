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
MBD_REVISION = "4b0d5b4c0b014e02af6c2e6b7b27851e6fb0d5c3"  # pinned Hugging Face commit of ai-lab/MBD-mini
MBD_ARCHIVES = {  # archive name -> (SHA-256, members extracted)
    "targets.tar.gz": ("6e9130a14ea0adeb4924f76a77430f0d57a376030410166679eb9f31864ccb0b", ("targets/",)),
    "detail.tar.gz": ("3f5cba0deb0ace86a7c5dd76ffc39f3320c0920b01ee2e034bdf92c1835f7e11", ("detail/trx/", "detail/dialog/")),
}
RUN_MANIFEST_FILE = PROJECT_ROOT / "reports" / "run_manifest.json"

CUSTOMER_FEATURES_FILE = PROCESSED_DATA_DIR / "customer_features.parquet"
SCORING_FEATURES_FILE = PROCESSED_DATA_DIR / "scoring_features.parquet"
LEAD_SCORES_FILE = PROCESSED_DATA_DIR / "lead_scores.csv"
FEATURE_METADATA_FILE = PREPROCESSED_DATA_DIR / "feature_metadata.json"

CLTV_MODEL_FILE = MODEL_DIR / "cltv_model.pkl"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"
PROPENSITY_LGBM_FILE = MODEL_DIR / "propensity_lgbm.txt"
PROPENSITY_CHOICE_FILE = MODEL_DIR / "propensity_choice.json"  # committed: the choices made on validation
GRU_MODEL_FILE = MODEL_DIR / "gru_final.pt"
EVENT_STORE_DIR = PROCESSED_DATA_DIR / "events"
GRU_VAL_SCORES_FILE = PROCESSED_DATA_DIR / "gru_val_scores.parquet"


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
