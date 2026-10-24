import pandas as pd
import pytest

from lead_scoring.config import PipelineConfig
from lead_scoring.features import FORBIDDEN_COLUMNS, fit_preprocessor, split_training_data
from lead_scoring.temporal import LeakageError


def _stacked(snapshots):
    return pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(len(snapshots))],
        "snapshot_date": snapshots,
        "txn_count_365": range(len(snapshots)),
        "trans_count": 7,
        "converted": [0, 1] * (len(snapshots) // 2) + [0] * (len(snapshots) % 2),
        "future_value": 1.0,
    })


def test_split_is_by_snapshot_not_random():
    cfg = PipelineConfig(train_snapshots=("2022-01-31", "2022-04-30"), test_snapshot="2022-10-31")
    df = _stacked(["2022-01-31", "2022-04-30", "2022-10-31", "2022-10-31"])
    train, test = split_training_data(df, cfg)
    assert set(train["snapshot_date"]) == {"2022-01-31", "2022-04-30"}
    assert set(test["snapshot_date"]) == {"2022-10-31"}


def test_split_raises_when_training_target_window_overlaps_test():
    cfg = PipelineConfig(train_snapshots=("2022-08-31",), test_snapshot="2022-10-31", horizon_months=3)
    with pytest.raises(LeakageError):
        split_training_data(_stacked(["2022-08-31", "2022-10-31"]), cfg)


def test_split_raises_when_a_side_is_empty():
    cfg = PipelineConfig(train_snapshots=("2022-01-31",), test_snapshot="2022-10-31")
    with pytest.raises(ValueError, match="test"):
        split_training_data(_stacked(["2022-01-31", "2022-01-31"]), cfg)


def test_unexplained_mbd_columns_are_forbidden_and_never_features():
    assert {"trans_count", "diff_trans_date"} <= FORBIDDEN_COLUMNS
    spec = fit_preprocessor(_stacked(["2022-01-31", "2022-04-30"]))
    assert not FORBIDDEN_COLUMNS & set(spec["feature_cols"])
    assert "txn_count_365" in spec["feature_cols"]
