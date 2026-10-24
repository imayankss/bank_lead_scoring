import pandas as pd

from lead_scoring.risk import _risk_category, risk_composite


def _client(recency_days, txn_count_90, txn_count_365, lead_score):
    return pd.DataFrame([{"recency_days": recency_days, "txn_count_90": txn_count_90,
                          "txn_count_365": txn_count_365, "lead_score": lead_score}])


def test_stale_cooling_cold_client_is_high_risk():
    score = risk_composite(_client(recency_days=300, txn_count_90=0, txn_count_365=40, lead_score=10)).iloc[0]
    assert _risk_category(score) == "High"


def test_recent_steady_hot_client_is_low_risk():
    score = risk_composite(_client(recency_days=1, txn_count_90=30, txn_count_365=100, lead_score=90)).iloc[0]
    assert _risk_category(score) == "Low"
