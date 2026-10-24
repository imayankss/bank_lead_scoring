"""Leakage guards for time-ordered training and evaluation.

Every reported metric must come from a later period than the data the model
was trained on. These guards raise instead of warning so a leak cannot pass silently.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class LeakageError(AssertionError):
    """Raised when training or feature data reaches into the evaluation period."""


def month_end(snapshot, months: int = 0) -> pd.Timestamp:
    """Month-end `months` after the month-end snapshot."""
    return (pd.Timestamp(snapshot) + pd.offsets.MonthEnd(months)).normalize()


def assert_target_windows_before(train_snapshots: Iterable, horizon_months: int, test_snapshot) -> None:
    """Fail if any training target window ends after the test snapshot."""
    test = pd.Timestamp(test_snapshot)
    late = [str(pd.Timestamp(s).date()) for s in train_snapshots if month_end(s, horizon_months) > test]
    if late:
        raise LeakageError(f"training snapshots {late} have target windows ending after test snapshot {test.date()}")


def assert_events_before(event_times: pd.Series, snapshot) -> None:
    """Fail if any event used for features happens after the end of the snapshot day."""
    cutoff = pd.Timestamp(snapshot).normalize() + pd.Timedelta(days=1)
    latest = event_times.max()
    if pd.notna(latest) and latest >= cutoff:
        raise LeakageError(f"feature event at {latest} is after snapshot {pd.Timestamp(snapshot).date()}")


def assert_no_forbidden_columns(columns: Iterable[str], forbidden: set[str]) -> None:
    """Fail if any target, identifier or unexplained column reaches the model matrix."""
    leaked = sorted(forbidden.intersection(columns))
    if leaked:
        raise LeakageError(f"forbidden columns in model features: {leaked}")
