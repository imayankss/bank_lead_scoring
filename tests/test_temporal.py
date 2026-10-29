import numpy as np
import pandas as pd
import pytest

from lead_scoring.config import PipelineConfig
from lead_scoring.temporal import (
    LeakageError,
    assert_events_before,
    assert_no_forbidden_columns,
    assert_no_single_feature_leak,
    assert_plausible_auc,
    assert_selection_snapshots,
    assert_target_windows_before,
    single_feature_auc,
)


def test_target_windows_ending_on_test_snapshot_pass():
    assert_target_windows_before(["2022-01-31", "2022-07-31"], horizon_months=3, test_snapshot="2022-10-31")


def test_target_window_reaching_past_test_snapshot_raises():
    with pytest.raises(LeakageError, match="2022-08-31"):
        assert_target_windows_before(["2022-08-31"], horizon_months=3, test_snapshot="2022-10-31")


def test_events_on_snapshot_pass():
    assert_events_before(pd.Series(pd.to_datetime(["2022-01-01 00:00:00", "2022-01-31 23:59:00"])), "2022-01-31")


def test_event_after_snapshot_raises():
    with pytest.raises(LeakageError, match="2022-02-01"):
        assert_events_before(pd.Series(pd.to_datetime(["2022-01-10 00:00:00", "2022-02-01 00:00:01"])), "2022-01-31")


def test_forbidden_columns_raise_with_names():
    with pytest.raises(LeakageError, match="trans_count"):
        assert_no_forbidden_columns(["recency_days", "trans_count"], {"trans_count", "converted"})


def test_allowed_columns_pass():
    assert_no_forbidden_columns(["recency_days"], {"trans_count", "converted"})


def test_selection_accepts_fit_and_validation_only():
    assert_selection_snapshots(["2022-01-31", "2022-07-31"], ("2022-01-31", "2022-04-30"), "2022-07-31", 3)


def test_selection_raises_on_test_snapshot_rows():
    with pytest.raises(LeakageError, match="2022-10-31"):
        assert_selection_snapshots(["2022-04-30", "2022-10-31"], ("2022-04-30",), "2022-07-31", 3)


def test_selection_raises_when_fit_labels_reach_validation():
    with pytest.raises(LeakageError, match="2022-05-31"):
        assert_selection_snapshots(["2022-05-31"], ("2022-05-31",), "2022-07-31", 3)


def test_default_config_windows_are_leak_free():
    cfg = PipelineConfig()
    assert_selection_snapshots([*cfg.fit_snapshots, cfg.validation_snapshot], cfg.fit_snapshots,
                               cfg.validation_snapshot, cfg.horizon_months)
    assert_target_windows_before(cfg.train_snapshots, cfg.horizon_months, cfg.test_snapshot)
    assert set(cfg.fit_snapshots) < set(cfg.train_snapshots) and cfg.validation_snapshot in cfg.train_snapshots
    assert len(cfg.train_snapshots) == 7


def test_single_feature_leak_raises_on_a_copy_of_the_target():
    y = pd.Series([0, 1] * 50)
    X = pd.DataFrame({"leak": y * 3.0, "noise": np.arange(100) % 7})
    table = single_feature_auc(X, y)
    assert table.iloc[0]["feature"] == "leak" and table.iloc[0]["auc"] == 1.0
    with pytest.raises(LeakageError, match="leak"):
        assert_no_single_feature_leak(table)


def test_a_reversed_feature_counts_as_signal():
    y = pd.Series([0, 1] * 50)
    assert single_feature_auc(pd.DataFrame({"neg": -y}), y).iloc[0]["auc"] == 1.0


def test_auc_above_the_ceiling_raises():
    assert_plausible_auc(0.82)
    with pytest.raises(LeakageError, match="0.910"):
        assert_plausible_auc(0.91)
