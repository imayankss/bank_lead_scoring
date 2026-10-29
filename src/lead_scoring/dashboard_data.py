"""Data access helpers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd

from lead_scoring.paths import LEAD_SCORES_FILE, SCORING_FEATURES_FILE, TABLES_DIR

ACTIVITY_COLUMNS = [
    "customer_id", "recency_days", "txn_count_365", "txn_count_90", "total_txn_amt_365",
    "products_taken_before", "target_1_before", "target_2_before", "target_3_before", "target_4_before",
]


def load_dashboard_data() -> pd.DataFrame:
    """Load scored leads with each client's activity and product history at the scoring snapshot."""
    scores = pd.read_csv(LEAD_SCORES_FILE)
    activity = pd.read_parquet(SCORING_FEATURES_FILE, columns=ACTIVITY_COLUMNS)
    data = scores.merge(activity, on="customer_id", how="left")
    fallbacks = TABLES_DIR / "fallback_explanations.csv"
    if fallbacks.exists():
        data = data.merge(pd.read_csv(fallbacks), on="customer_id", how="left")
    return data


def load_metrics() -> pd.DataFrame:
    path = TABLES_DIR / "metrics_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_decile_lift() -> pd.DataFrame:
    path = TABLES_DIR / "decile_lift.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_feature_importance() -> pd.DataFrame:
    path = TABLES_DIR / "feature_importance.csv"
    if not path.exists():
        path = TABLES_DIR / "shap_top_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_explanations() -> pd.DataFrame:
    path = TABLES_DIR / "per_lead_explanations.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
