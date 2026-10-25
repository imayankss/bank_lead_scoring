"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for feature building and model training.

    Snapshots are month-ends. Each training snapshot's target window must end
    on or before the test snapshot (enforced by lead_scoring.temporal).
    """

    lookback_days: int = 365
    horizon_months: int = 3
    # Model selection: fit on Jan–Apr 2022, validate on Jul 2022 (its labels end on the test snapshot).
    fit_snapshots: tuple[str, ...] = ("2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30")
    validation_snapshot: str = "2022-07-31"
    train_snapshots: tuple[str, ...] = (
        "2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31", "2022-06-30", "2022-07-31",
    )
    test_snapshot: str = "2022-10-31"
    score_snapshot: str = "2022-12-31"
    random_state: int = 42
    workers: int = 5  # one process per MBD client fold
