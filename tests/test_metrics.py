import math

import pandas as pd

from lead_scoring.metrics import classification_metrics, regression_metrics


def test_regression_metrics_returns_expected_keys():
    metrics = regression_metrics(pd.Series([100, 200, 300]), pd.Series([110, 190, 330]))

    assert {"mae", "rmse", "mape", "smape", "r2", "forecast_bias"}.issubset(metrics)
    assert metrics["mae"] > 0
    assert not math.isnan(metrics["rmse"])


def test_classification_metrics_handles_rank_scores():
    metrics = classification_metrics(pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]))

    assert metrics["auc"] == 1.0
    assert metrics["precision_at_20pct"] == 1.0
