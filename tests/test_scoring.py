import pandas as pd

from lead_scoring.scoring import _score_to_100


def test_score_to_100_uses_ranked_distribution():
    expected_value = pd.Series([10.0, 20.0, 30.0, 40.0])
    scores = _score_to_100(expected_value)

    assert scores.tolist() == [25, 50, 75, 100]


def test_score_to_100_handles_constant_scores():
    expected_value = pd.Series([5.0, 5.0, 5.0])
    scores = _score_to_100(expected_value)

    assert scores.tolist() == [50, 50, 50]
