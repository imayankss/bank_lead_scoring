"""Residual diagnostics for the CLTV model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lead_scoring.paths import TABLES_DIR, ensure_directories


def build_residual_diagnostics() -> dict[str, object]:
    """Save residual records and summary diagnostics."""
    ensure_directories()
    predictions = pd.read_csv(TABLES_DIR / "test_predictions.csv")
    residuals = predictions["actual_future_value"] - predictions["predicted_cltv"]
    absolute_error = residuals.abs()

    records = predictions[["customer_id", "actual_future_value", "predicted_cltv"]].copy()
    records["residual"] = residuals
    records["absolute_error"] = absolute_error
    # Errors are judged against what clients actually transact: below a median client's
    # 3-month value is low, above a top-1% client's is extreme.
    q50, q90, q99 = predictions["actual_future_value"].quantile([0.5, 0.9, 0.99])
    records["error_bucket"] = pd.cut(
        absolute_error,
        bins=[-1, q50, q90, q99, np.inf],
        labels=["low", "moderate", "high", "extreme"],
    ).astype(str)
    records.to_csv(TABLES_DIR / "residual_records.csv", index=False)

    summary = {
        "mean_residual": float(residuals.mean()),
        "median_residual": float(residuals.median()),
        "mean_absolute_error": float(absolute_error.mean()),
        "residual_std": float(residuals.std()),
        "positive_bias_rate": float((residuals > 0).mean()),
        "extreme_error_count": int((records["error_bucket"] == "extreme").sum()),
    }
    pd.DataFrame([summary]).to_csv(TABLES_DIR / "residual_diagnostics.csv", index=False)

    distribution = (
        records["error_bucket"]
        .value_counts()
        .rename_axis("bucket")
        .reset_index(name="count")
        .sort_values("bucket")
    )
    distribution.to_csv(TABLES_DIR / "error_distribution.csv", index=False)
    return {"summary": summary, "records": records.to_dict(orient="records")}


def main() -> None:
    summary = build_residual_diagnostics()["summary"]
    print(summary)


if __name__ == "__main__":
    main()
