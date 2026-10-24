import pandas as pd
import pytest

from lead_scoring.temporal import (
    LeakageError,
    assert_events_before,
    assert_no_forbidden_columns,
    assert_target_windows_before,
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
