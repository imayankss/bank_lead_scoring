"""Lightweight model explanation artifacts."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from lead_scoring.paths import (
    CLTV_MODEL_FILE,
    FEATURE_METADATA_FILE,
    FIGURES_DIR,
    LEAD_SCORES_FILE,
    PREPROCESSED_DATA_DIR,
    SCORING_FEATURES_FILE,
    TABLES_DIR,
    ensure_directories,
)


def build_explainability_artifacts() -> dict[str, str]:
    """Save global feature importance and per-lead top driver summaries."""
    ensure_directories()
    X_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "y_reg_test.parquet")["future_revenue_12m"]
    X_scoring = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_scoring.parquet")
    scoring_ids = pd.read_csv(PREPROCESSED_DATA_DIR / "scoring_ids.csv")
    scoring_features = pd.read_parquet(SCORING_FEATURES_FILE)
    scores = pd.read_csv(LEAD_SCORES_FILE)

    model = joblib.load(CLTV_MODEL_FILE)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)
    # Compatibility for older dashboard/docs references. This is permutation importance, not SHAP.
    importance.rename(columns={"importance_mean": "mean_abs_shap"}).to_csv(
        TABLES_DIR / "shap_top_features.csv",
        index=False,
    )
    _plot_feature_importance(importance.head(15))

    baseline = X_scoring.median(numeric_only=True)
    weights = importance.set_index("feature")["importance_mean"].clip(lower=0)
    rows = []
    for idx, row in X_scoring.iterrows():
        deviations = ((row - baseline).abs() * weights.reindex(X_scoring.columns).fillna(0)).sort_values(ascending=False)
        top_features = [feature for feature, value in deviations.head(3).items() if value > 0]
        if not top_features:
            top_features = list(importance["feature"].head(3))
        rows.append(
            {
                "customer_id": scoring_ids.loc[idx, "customer_id"],
                "top_drivers": "; ".join(top_features),
            }
        )

    explanations = pd.DataFrame(rows).merge(scores[["customer_id", "lead_score", "lead_category"]], on="customer_id", how="left")
    explanations.to_csv(TABLES_DIR / "per_lead_explanations.csv", index=False)

    summary = _build_explanation_summary(scoring_features, scores, importance)
    (TABLES_DIR / "explainability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "feature_importance": str(TABLES_DIR / "feature_importance.csv"),
        "per_lead_explanations": str(TABLES_DIR / "per_lead_explanations.csv"),
    }


def _plot_feature_importance(importance: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plot_df = importance.sort_values("importance_mean", ascending=True)
    plt.barh(plot_df["feature"], plot_df["importance_mean"], color="#2563eb")
    plt.xlabel("Permutation importance")
    plt.title("Top CLTV Model Drivers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
    plt.close()


def _build_explanation_summary(
    scoring_features: pd.DataFrame,
    scores: pd.DataFrame,
    importance: pd.DataFrame,
) -> dict[str, object]:
    return {
        "scored_customers": int(len(scores)),
        "hot_leads": int((scores["lead_category"] == "Hot").sum()),
        "medium_leads": int((scores["lead_category"] == "Medium").sum()),
        "cold_leads": int((scores["lead_category"] == "Cold").sum()),
        "top_features": importance["feature"].head(5).tolist(),
        "avg_recency_days": float(scoring_features["recency_days"].mean()),
    }


def main() -> None:
    paths = build_explainability_artifacts()
    print(paths)


if __name__ == "__main__":
    main()
