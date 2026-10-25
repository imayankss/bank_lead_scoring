"""Leakage guards for time-ordered training and evaluation.

Every reported metric must come from a later period than the data the model
was trained on. These guards raise instead of warning so a leak cannot pass silently.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.metrics import roc_auc_score


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


def assert_selection_snapshots(
    snapshots: Iterable, fit_snapshots: Iterable, validation_snapshot, horizon_months: int
) -> None:
    """Fail if model selection sees anything but fit and validation rows, or fit labels reach validation."""
    fit_snapshots = list(fit_snapshots)
    allowed = {str(pd.Timestamp(s).date()) for s in (*fit_snapshots, validation_snapshot)}
    extra = sorted({str(pd.Timestamp(s).date()) for s in snapshots} - allowed)
    if extra:
        raise LeakageError(f"model selection saw snapshots {extra}; only fit and validation snapshots are allowed")
    assert_target_windows_before(fit_snapshots, horizon_months, validation_snapshot)


def single_feature_auc(X: pd.DataFrame, y) -> pd.DataFrame:
    """AUC of each feature used alone, folded so 0.5 is no signal and 1.0 a perfect ranker either way."""
    rows = []
    for col in X.columns:
        auc = float(roc_auc_score(y, X[col])) if X[col].nunique() > 1 else 0.5
        rows.append({"feature": col, "auc": max(auc, 1.0 - auc)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False, kind="stable").reset_index(drop=True)


def assert_no_single_feature_leak(table: pd.DataFrame, threshold: float = 0.80) -> None:
    """Fail if any one feature ranks validation almost on its own: likely a label in disguise."""
    leaked = table.loc[table["auc"] > threshold, "feature"].tolist()
    if leaked:
        raise LeakageError(f"features {leaked} reach validation AUC above {threshold} alone; audit them first")


def assert_plausible_auc(auc: float, ceiling: float = 0.85) -> None:
    """Fail above the ceiling: the published best on full MBD is 0.824, so higher means audit first."""
    if auc > ceiling:
        raise LeakageError(f"validation AUC {auc:.3f} is above {ceiling}; audit for leakage before reporting it")
