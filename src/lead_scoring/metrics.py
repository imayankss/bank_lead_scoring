"""Reusable metric helpers for model comparison and diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


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
    pr_auc = float(average_precision_score(actual, score)) if actual.nunique() > 1 else float("nan")

    return {
        "auc": auc,
        "pr_auc": pr_auc,
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
    actual = np.asarray(y_true, dtype=float)
    score = np.asarray(scores, dtype=float)
    n_top = max(1, int(len(score) * fraction))
    cutoff = np.sort(score)[::-1][n_top - 1]
    above = score > cutoff
    tied = score == cutoff
    # Rows tied at the cutoff fill the remaining slots in random order, so they count at their mean rate.
    hits = actual[above].sum() + actual[tied].mean() * (n_top - above.sum())
    return float(hits / n_top)


def lift_at_fraction(y_true: pd.Series, scores: pd.Series | np.ndarray, fraction: float) -> float:
    """Conversion rate in the top-ranked fraction divided by the overall rate."""
    base = float(pd.Series(y_true).mean())
    return precision_at_fraction(y_true, scores, fraction) / base if base > 0 else float("nan")


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(
    y_true: pd.Series,
    scores: pd.Series | np.ndarray,
    metrics: dict[str, MetricFn],
    *,
    n_boot: int = 300,
    seed: int = 42,
    level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap over rows. A fixed seed gives every model the same resamples."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue  # a resample with one class has no ranking metric
        for name, fn in metrics.items():
            draws[name].append(fn(y[idx], s[idx]))
    tail = (1 - level) / 2 * 100
    return {
        name: {
            "point": float(fn(y, s)),
            "lower": float(np.percentile(draws[name], tail)),
            "upper": float(np.percentile(draws[name], 100 - tail)),
        }
        for name, fn in metrics.items()
    }


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; stays inside [0, 1] for small counts."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # The bounds are exactly 0 and 1 at the extremes; set them so float error can't nudge them inward.
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == n else min(1.0, centre + half)
    return float(lower), float(upper)


def decile_table(y_true: pd.Series, scores: pd.Series, value: pd.Series) -> pd.DataFrame:
    """Score deciles (10 = highest) with conversion rate, 95% Wilson interval, lift and cumulative reach."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "score": np.asarray(scores, dtype=float),
                       "value": np.asarray(value, dtype=float)})
    df["decile"] = pd.qcut(df["score"].rank(method="first"), 10, labels=False) + 1
    table = (
        df.groupby("decile")
        .agg(n=("y", "size"), converters=("y", "sum"), mean_score=("score", "mean"), actual_value=("value", "sum"))
        .reset_index()
        .sort_values("decile", ascending=False)
        .reset_index(drop=True)
    )
    table["conversion_rate"] = table["converters"] / table["n"]
    bounds = [wilson_interval(int(k), int(n)) for k, n in zip(table["converters"], table["n"])]
    table["conversion_ci_lower"] = [lo for lo, _ in bounds]
    table["conversion_ci_upper"] = [hi for _, hi in bounds]
    table["lift"] = table["conversion_rate"] / df["y"].mean()
    table["cumulative_share_of_converters"] = table["converters"].cumsum() / df["y"].sum()
    return table
