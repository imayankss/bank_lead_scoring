"""Data access helpers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd

from lead_scoring.paths import LEAD_SCORES_FILE, RAW_CUSTOMERS_FILE, TABLES_DIR


def load_dashboard_data() -> pd.DataFrame:
    """Load scored leads merged with customer profile fields."""
    scores = pd.read_csv(LEAD_SCORES_FILE)
    customers = pd.read_csv(RAW_CUSTOMERS_FILE)
    customers.columns = customers.columns.str.strip()

    if "lead_category" not in scores.columns:
        scores["lead_category"] = pd.cut(
            scores["lead_score"],
            bins=[-1, 29, 69, 100],
            labels=["Cold", "Medium", "Hot"],
        ).astype(str)

    data = scores.merge(customers, left_on="customer_id", right_on="Customer_ID", how="left")
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
