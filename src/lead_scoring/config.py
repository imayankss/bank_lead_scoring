"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for feature building and model training."""

    lookback_days: int = 365
    target_horizon_days: int = 365
    test_size: float = 0.2
    random_state: int = 42
