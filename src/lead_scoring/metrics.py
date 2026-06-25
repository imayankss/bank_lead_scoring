"""Reusable metric helpers for model comparison and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def regression_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Return common regression metrics with zero-safe percentage errors."""
    actual = pd.Series(y_true).astype(float).reset_index(drop=True)
    pred = pd.Series(y_pred).astype(float).reset_index(drop=True)
    error = actual - pred
    nonzero = actual.abs() > 1e-9

    mae = float(mean_absolute_error(actual, pred))
    rmse = float(mean_squared_error(actual, pred) ** 0.5)
    mape = float((error[nonzero].abs() / actual[nonzero].abs()).mean() * 100) if nonzero.any() else float("nan")
    smape_denominator = (actual.abs() + pred.abs()).replace(0, np.nan)
    smape = float((2 * error.abs() / smape_denominator).mean(skipna=True) * 100)
    bias = float(error.mean())
    r2 = float(r2_score(actual, pred)) if actual.nunique() > 1 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "forecast_bias": bias,
    }


def classification_metrics(y_true: pd.Series, y_score: pd.Series | np.ndarray) -> dict[str, float]:
    """Return propensity-ranking metrics."""
    actual = pd.Series(y_true).astype(int).reset_index(drop=True)
    score = pd.Series(y_score).astype(float).reset_index(drop=True)
    auc = float(roc_auc_score(actual, score)) if actual.nunique() > 1 else float("nan")

    return {
        "auc": auc,
        "positive_rate": float(actual.mean()),
        "directional_accuracy": directional_accuracy(actual, score),
        "precision_at_10pct": precision_at_fraction(actual, score, 0.10),
        "precision_at_20pct": precision_at_fraction(actual, score, 0.20),
    }


def directional_accuracy(y_true: pd.Series, y_score: pd.Series | np.ndarray) -> float:
    """Approximate whether scores rank above/below median outcomes correctly."""
    actual = pd.Series(y_true).astype(float).reset_index(drop=True)
    score = pd.Series(y_score).astype(float).reset_index(drop=True)
    if actual.nunique() < 2 or score.nunique() < 2:
        return float("nan")
    actual_direction = actual >= actual.median()
    score_direction = score >= score.median()
    return float((actual_direction == score_direction).mean())


def precision_at_fraction(y_true: pd.Series, scores: pd.Series | np.ndarray, fraction: float) -> float:
    """Precision among the top ranked rows."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    actual = pd.Series(y_true).astype(int).reset_index(drop=True)
    score = pd.Series(scores).astype(float).reset_index(drop=True)
    n_top = max(1, int(len(score) * fraction))
    top_idx = score.sort_values(ascending=False).head(n_top).index
    return float(actual.loc[top_idx].mean())
