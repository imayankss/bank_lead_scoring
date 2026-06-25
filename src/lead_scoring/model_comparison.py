"""Model and baseline comparison artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lead_scoring.metrics import classification_metrics, precision_at_fraction, regression_metrics
from lead_scoring.paths import TABLES_DIR, ensure_directories


def build_model_comparison() -> pd.DataFrame:
    """Compare the trained model outputs against simple baselines."""
    ensure_directories()
    predictions = pd.read_csv(TABLES_DIR / "test_predictions.csv")
    actual_revenue = predictions["actual_future_revenue_12m"]
    actual_conversion = predictions["actual_converted_12m"]

    mean_revenue = float(actual_revenue.mean())
    baseline_propensity = float(actual_conversion.mean())
    candidates = [
        {
            "model_id": "expected_value_ranker",
            "model_name": "Expected Value Ranker",
            "revenue_prediction": predictions["predicted_cltv"],
            "ranking_score": predictions["expected_value"],
        },
        {
            "model_id": "propensity_rank_baseline",
            "model_name": "Propensity Rank Baseline",
            "revenue_prediction": pd.Series(mean_revenue, index=predictions.index),
            "ranking_score": predictions["predicted_propensity"],
        },
        {
            "model_id": "cltv_rank_baseline",
            "model_name": "CLTV Rank Baseline",
            "revenue_prediction": predictions["predicted_cltv"],
            "ranking_score": predictions["predicted_cltv"],
        },
        {
            "model_id": "mean_value_baseline",
            "model_name": "Mean Value Baseline",
            "revenue_prediction": pd.Series(mean_revenue, index=predictions.index),
            "ranking_score": pd.Series(baseline_propensity * mean_revenue, index=predictions.index),
        },
    ]

    rows = []
    for candidate in candidates:
        reg = regression_metrics(actual_revenue, candidate["revenue_prediction"])
        cls = classification_metrics(actual_conversion, candidate["ranking_score"])
        rows.append(
            {
                "model_id": candidate["model_id"],
                "model_name": candidate["model_name"],
                "mae": reg["mae"],
                "rmse": reg["rmse"],
                "mape": reg["mape"],
                "smape": reg["smape"],
                "r2": reg["r2"],
                "forecast_bias": reg["forecast_bias"],
                "auc": cls["auc"],
                "directional_accuracy": cls["directional_accuracy"],
                "precision_at_10pct": precision_at_fraction(actual_conversion, candidate["ranking_score"], 0.10),
                "precision_at_20pct": precision_at_fraction(actual_conversion, candidate["ranking_score"], 0.20),
            }
        )

    output = pd.DataFrame(rows)
    output["production_model"] = output["model_id"].eq("expected_value_ranker")
    output["business_score"] = (
        output["precision_at_10pct"].fillna(0) * 0.35
        + output["precision_at_20pct"].fillna(0) * 0.25
        + output["auc"].fillna(0.5) * 0.25
        + output["directional_accuracy"].fillna(0) * 0.10
        + output["production_model"].astype(float) * 0.05
    )
    output["rank"] = output["business_score"].rank(method="first", ascending=False).astype(int)
    output["is_best"] = output["rank"] == output["rank"].min()
    output = output.sort_values(["rank", "rmse"]).reset_index(drop=True)
    output.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    return output


def main() -> None:
    output = build_model_comparison()
    print(f"Saved model comparison with {len(output)} rows.")


if __name__ == "__main__":
    main()
