import pandas as pd

from lead_scoring.data import build_customer_features
from lead_scoring.features import DATE_COLUMNS, IDENTIFIER_COLUMNS, PII_COLUMNS, TARGET_COLUMNS, fit_preprocessor


def test_training_snapshot_has_meaningful_target_window():
    features = build_customer_features(include_target=True)

    assert len(features) == 500
    assert features["converted_12m"].sum() > 100
    assert features["future_revenue_12m"].gt(0).sum() == features["converted_12m"].sum()


def test_model_feature_spec_excludes_leakage_and_pii():
    features = build_customer_features(include_target=True)
    spec = fit_preprocessor(features)

    forbidden = TARGET_COLUMNS | IDENTIFIER_COLUMNS | PII_COLUMNS | DATE_COLUMNS
    assert not forbidden.intersection(spec["feature_cols"])
    assert "future_revenue_12m" not in spec["feature_cols"]
    assert "converted_12m" not in spec["feature_cols"]


def test_current_scoring_features_cover_all_customers_without_targets():
    scoring_features = build_customer_features(include_target=False)

    assert len(scoring_features) == 500
    assert "future_revenue_12m" not in scoring_features.columns
    assert "converted_12m" not in scoring_features.columns
    assert scoring_features["customer_id"].is_unique
