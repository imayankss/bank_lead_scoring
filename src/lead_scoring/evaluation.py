"""Evaluation artifacts for lead scoring models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from lead_scoring.paths import (
    CLTV_MODEL_FILE,
    FIGURES_DIR,
    MODEL_METADATA_FILE,
    PREPROCESSED_DATA_DIR,
    PROPENSITY_MODEL_FILE,
    TABLES_DIR,
    ensure_directories,
)


def _read_target(path: str, column: str) -> pd.Series:
    return pd.read_parquet(PREPROCESSED_DATA_DIR / path)[column]


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def precision_at_k(y_true: pd.Series, scores: np.ndarray, fraction: float) -> float:
    """Calculate precision among the top scoring fraction."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    n_top = max(1, int(len(scores) * fraction))
    top_idx = np.argsort(scores)[-n_top:]
    return float(y_true.reset_index(drop=True).iloc[top_idx].mean())


def evaluate_models() -> dict[str, float]:
    """Evaluate persisted models and save metrics, lift, and calibration artifacts."""
    ensure_directories()
    X_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet")
    y_reg_test = _read_target("y_reg_test.parquet", "future_revenue_12m")
    y_clf_test = _read_target("y_clf_test.parquet", "converted_12m").astype(int)

    cltv_model = joblib.load(CLTV_MODEL_FILE)
    propensity_model = joblib.load(PROPENSITY_MODEL_FILE)

    pred_cltv = np.clip(cltv_model.predict(X_test), 0, None)
    pred_propensity = propensity_model.predict_proba(X_test)[:, 1]
    expected_value = pred_cltv * pred_propensity

    metrics = {
        "rows": float(len(X_test)),
        "positive_rate": float(y_clf_test.mean()),
        "propensity_auc": _safe_auc(y_clf_test, pred_propensity),
        "propensity_brier": float(brier_score_loss(y_clf_test, pred_propensity)),
        "cltv_rmse": float(mean_squared_error(y_reg_test, pred_cltv) ** 0.5),
        "cltv_mae": float(mean_absolute_error(y_reg_test, pred_cltv)),
        "precision_at_10pct": precision_at_k(y_clf_test, expected_value, 0.10),
        "precision_at_20pct": precision_at_k(y_clf_test, expected_value, 0.20),
    }
    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "metrics_summary.csv", index=False)

    test_ids = pd.read_csv(PREPROCESSED_DATA_DIR / "test_ids.csv")
    predictions = pd.DataFrame(
        {
            "customer_id": test_ids["customer_id"],
            "actual_future_revenue_12m": y_reg_test.reset_index(drop=True),
            "actual_converted_12m": y_clf_test.reset_index(drop=True),
            "predicted_cltv": pred_cltv,
            "predicted_propensity": pred_propensity,
            "expected_value": expected_value,
        }
    )
    predictions.to_csv(TABLES_DIR / "test_predictions.csv", index=False)

    decile_count = min(10, len(predictions))
    predictions["decile"] = pd.qcut(
        predictions["expected_value"].rank(method="first"),
        q=decile_count,
        labels=False,
        duplicates="drop",
    )
    lift = (
        predictions.groupby("decile")
        .agg(
            n=("expected_value", "size"),
            mean_expected_value=("expected_value", "mean"),
            conversion_rate=("actual_converted_12m", "mean"),
            actual_revenue=("actual_future_revenue_12m", "sum"),
        )
        .reset_index()
        .sort_values("decile", ascending=False)
    )
    lift.to_csv(TABLES_DIR / "decile_lift.csv", index=False)

    _save_calibration_plot(y_clf_test, pred_propensity, FIGURES_DIR / "calibration.png")

    metadata = {}
    if MODEL_METADATA_FILE.exists():
        metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    metadata["evaluation"] = metrics
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _save_calibration_plot(y_true: pd.Series, pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    if y_true.nunique() > 1:
        prob_true, prob_pred = calibration_curve(y_true, pred, n_bins=10, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed conversion rate")
    plt.title("Propensity Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    metrics = evaluate_models()
    print(f"Saved evaluation artifacts to {TABLES_DIR} and {FIGURES_DIR}")
    print(metrics)


if __name__ == "__main__":
    main()
