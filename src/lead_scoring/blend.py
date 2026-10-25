"""Blend two propensity rankers and turn the blend into a calibrated probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

WEIGHT_GRID = tuple(float(w) for w in np.round(np.arange(0, 1.0001, 0.05), 2))
MIN_GAIN = 0.005  # validation AUC the GRU must add to earn a place in production
CALIBRATION_BINS = 50  # equal-count bins (~2,000 validation clients each), so one lucky top client can't set the top rate


def rank_pct(scores) -> np.ndarray:
    """Percentile ranks, so models on different scales blend fairly."""
    return pd.Series(np.asarray(scores, dtype=float)).rank(method="average", pct=True).to_numpy()


def blend(lgbm, gru, weight: float) -> np.ndarray:
    if weight == 0:
        return rank_pct(lgbm)
    return weight * rank_pct(gru) + (1 - weight) * rank_pct(lgbm)


def choose_weight(y, lgbm, gru=None, grid=WEIGHT_GRID, min_gain: float = MIN_GAIN) -> dict:
    """GRU weight with the best validation AUC; 0 unless the GRU adds at least `min_gain`."""
    lgbm_auc = float(roc_auc_score(y, lgbm))
    if gru is None:
        return {"weight": 0.0, "auc": lgbm_auc, "lgbm_auc": lgbm_auc, "gru_auc": None}
    aucs = {w: float(roc_auc_score(y, blend(lgbm, gru, w))) for w in grid}
    best = max(aucs, key=lambda w: (aucs[w], -w))  # ties go to the smaller GRU weight
    if aucs[best] - lgbm_auc < min_gain:
        return {"weight": 0.0, "auc": lgbm_auc, "lgbm_auc": lgbm_auc, "gru_auc": aucs[1.0]}
    return {"weight": best, "auc": aucs[best], "lgbm_auc": lgbm_auc, "gru_auc": aucs[1.0]}


def fit_calibrator(y, blended, bins: int = CALIBRATION_BINS) -> dict[str, list[float]]:
    """Isotonic map from blend score to conversion rate, fit on equal-count bin means, as JSON-friendly breakpoints."""
    df = pd.DataFrame({"s": np.asarray(blended, dtype=float), "y": np.asarray(y, dtype=float)})
    df["bin"] = pd.qcut(df["s"].rank(method="first"), q=bins, labels=False)
    g = df.groupby("bin").agg(s=("s", "mean"), y=("y", "mean"), n=("y", "size"))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(g["s"], g["y"], sample_weight=g["n"])
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def calibrate(blended, calibrator: dict[str, list[float]]) -> np.ndarray:
    return np.interp(np.asarray(blended, dtype=float), calibrator["x"], calibrator["y"])
