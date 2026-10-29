import numpy as np
import pandas as pd

from lead_scoring.model_comparison import compare_rankers, compare_value_models


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < 0.2).astype(int))
    return y, {
        "perfect": y.astype(float) + rng.random(n) * 0.1,
        "noisy": y + rng.normal(0, 1.5, n),
        "prior_baseline": pd.Series(0.0, index=y.index),
    }


def test_rankers_are_ordered_by_lift_with_one_best_and_no_bonus():
    y, scores = _data()
    table = compare_rankers(y, scores, n_boot=50)
    assert table["model_id"].tolist() == ["perfect", "noisy", "prior_baseline"]
    assert table["rank"].tolist() == [1, 2, 3]
    assert table["is_best"].sum() == 1 and table.loc[0, "is_best"]
    assert "business_score" not in table.columns


def test_ranker_rows_carry_confidence_intervals():
    y, scores = _data()
    row = compare_rankers(y, scores, n_boot=50).set_index("model_id").loc["noisy"]
    for metric in ("auc", "pr_auc", "lift_at_10pct"):
        assert row[f"{metric}_ci_lower"] <= row[metric] <= row[f"{metric}_ci_upper"]


def test_prior_baseline_has_no_ranking_skill():
    y, scores = _data()
    row = compare_rankers(y, scores, n_boot=50).set_index("model_id").loc["prior_baseline"]
    assert row["auc"] == 0.5


def test_value_models_rank_by_rmse():
    actual = pd.Series([10.0, 20.0, 30.0, 40.0])
    table = compare_value_models(actual, {"exact": actual, "mean_baseline": pd.Series(25.0, index=actual.index)})
    assert table["model_id"].tolist() == ["exact", "mean_baseline"]
    assert table.loc[0, "rmse"] == 0.0 and table.loc[0, "is_best"]
