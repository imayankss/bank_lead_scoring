"""Evaluation artifacts for lead scoring models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from lead_scoring.metrics import bootstrap_ci, decile_table, lift_at_fraction, precision_at_fraction
from lead_scoring.paths import (
    CLTV_MODEL_FILE,
    FIGURES_DIR,
    MODEL_METADATA_FILE,
    PREPROCESSED_DATA_DIR,
    TABLES_DIR,
    ensure_directories,
)
from lead_scoring.propensity import score_propensity


def _read_target(path: str, column: str) -> pd.Series:
    return pd.read_parquet(PREPROCESSED_DATA_DIR / path)[column]


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


RANKING_METRICS = {
    "propensity_auc": lambda y, s: roc_auc_score(y, s),
    "pr_auc": lambda y, s: average_precision_score(y, s),
    "lift_at_10pct": lambda y, s: lift_at_fraction(y, s, 0.10),
}


def evaluate_models() -> dict[str, float]:
    """Evaluate persisted models and save metrics, lift, and calibration artifacts."""
    ensure_directories()
    X_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet")
    y_reg_test = _read_target("y_reg_test.parquet", "future_value")
    y_clf_test = _read_target("y_clf_test.parquet", "converted").astype(int)

    cltv_model = joblib.load(CLTV_MODEL_FILE)
    pred_cltv = np.clip(cltv_model.predict(X_test), 0, None)
    scores = score_propensity(X_test)
    pred_propensity = scores["predicted_propensity"].to_numpy()
    rank_score = scores["lgbm_score"].to_numpy()  # finer than the calibrated steps, so ties don't blur the ranking
    expected_value = pred_cltv * pred_propensity

    metrics = {
        "rows": float(len(X_test)),
        "positive_rate": float(y_clf_test.mean()),
        "propensity_auc": _safe_auc(y_clf_test, rank_score),
        "propensity_brier": float(brier_score_loss(y_clf_test, pred_propensity)),
        "cltv_rmse": float(mean_squared_error(y_reg_test, pred_cltv) ** 0.5),
        "cltv_mae": float(mean_absolute_error(y_reg_test, pred_cltv)),
        "precision_at_10pct": precision_at_fraction(y_clf_test, expected_value, 0.10),
        "precision_at_20pct": precision_at_fraction(y_clf_test, expected_value, 0.20),
    }
    intervals = bootstrap_ci(y_clf_test, rank_score, RANKING_METRICS)
    for name, ci in intervals.items():
        metrics[name] = ci["point"]
        metrics[f"{name}_ci_lower"] = ci["lower"]
        metrics[f"{name}_ci_upper"] = ci["upper"]
    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "metrics_summary.csv", index=False)

    test_ids = pd.read_csv(PREPROCESSED_DATA_DIR / "test_ids.csv")
    predictions = pd.DataFrame(
        {
            "customer_id": test_ids["customer_id"],
            "actual_future_value": y_reg_test.reset_index(drop=True),
            "actual_converted": y_clf_test.reset_index(drop=True),
            "predicted_cltv": pred_cltv,
            "lgbm_score": rank_score,
            "predicted_propensity": pred_propensity,
            "expected_value": expected_value,
        }
    )
    predictions.to_csv(TABLES_DIR / "test_predictions.csv", index=False)

    # Deciles of the operational lead score (expected value), as relationship managers would work them.
    lift = decile_table(predictions["actual_converted"], predictions["expected_value"], predictions["actual_future_value"])
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
    top = 1.0
    if y_true.nunique() > 1:
        # Quantile bins: almost every prediction is near the 2% base rate, so uniform bins would hold one point.
        prob_true, prob_pred = calibration_curve(y_true, pred, n_bins=10, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        top = float(max(prob_pred.max(), prob_true.max())) * 1.1
    plt.plot([0, top], [0, top], linestyle="--", color="gray", label="Perfect calibration")
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
