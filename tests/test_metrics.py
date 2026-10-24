import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lead_scoring.metrics import bootstrap_ci, decile_table, classification_metrics, lift_at_fraction, regression_metrics, wilson_interval


def test_regression_metrics_returns_expected_keys():
    metrics = regression_metrics(pd.Series([100, 200, 300]), pd.Series([110, 190, 330]))

    assert {"mae", "rmse", "mape", "smape", "r2", "forecast_bias"}.issubset(metrics)
    assert metrics["mae"] > 0
    assert not math.isnan(metrics["rmse"])


def test_classification_metrics_handles_rank_scores():
    metrics = classification_metrics(pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]))

    assert metrics["auc"] == 1.0
    assert metrics["precision_at_20pct"] == 1.0


def test_lift_is_top_precision_over_base_rate():
    y = pd.Series([1] + [0] * 9)
    score = pd.Series([0.9] + [0.1] * 9)
    assert lift_at_fraction(y, score, 0.10) == 10.0


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 400))
    score = pd.Series(y + rng.normal(0, 1, 400))
    fns = {"auc": lambda yy, ss: roc_auc_score(yy, ss)}
    first = bootstrap_ci(y, score, fns, n_boot=200, seed=1)["auc"]
    assert first["lower"] < first["point"] < first["upper"]
    assert first == bootstrap_ci(y, score, fns, n_boot=200, seed=1)["auc"]


def test_bootstrap_ci_collapses_for_perfect_separation():
    y = pd.Series([0] * 20 + [1] * 20)
    ci = bootstrap_ci(y, y.astype(float), {"auc": lambda yy, ss: roc_auc_score(yy, ss)}, n_boot=50, seed=0)["auc"]
    assert ci == {"point": 1.0, "lower": 1.0, "upper": 1.0}


def test_wilson_interval_matches_reference_values():
    lower, upper = wilson_interval(5, 10)
    assert (round(lower, 4), round(upper, 4)) == (0.2366, 0.7634)
    lower, upper = wilson_interval(0, 10)
    assert lower == 0.0 and round(upper, 4) == 0.2775


def test_classification_metrics_include_pr_auc():
    metrics = classification_metrics(pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]))
    assert metrics["pr_auc"] == 1.0


def test_decile_table_orders_top_decile_first_with_lift_and_interval():
    score = pd.Series(np.arange(100, dtype=float))
    y = pd.Series((score >= 90).astype(int))  # all positives in the top decile
    table = decile_table(y, score, value=pd.Series(1.0, index=score.index))
    assert len(table) == 10 and table["n"].sum() == 100
    top = table.iloc[0]
    assert top["decile"] == 10 and top["conversion_rate"] == 1.0 and top["lift"] == 10.0
    assert top["conversion_ci_lower"] < 1.0 <= top["conversion_ci_upper"]
    assert table["cumulative_share_of_converters"].iloc[0] == 1.0
