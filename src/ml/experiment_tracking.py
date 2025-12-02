from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import mlflow

from src.common.config import PROJECT_ROOT

TRACKING_DIR = PROJECT_ROOT / "mlruns"


def init_mlflow(experiment_name: str) -> None:
    """
    Initialize MLflow tracking for this project.

    - Uses a local file-based backend under <PROJECT_ROOT>/mlruns
    - Sets the active experiment to `experiment_name`
    """
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(TRACKING_DIR))
    mlflow.set_experiment(experiment_name)


def log_params_from_dict(d: Dict[str, Any], prefix: str | None = None) -> None:
    """
    Log a dictionary of parameters to MLflow.

    If prefix is given, keys are logged as "<prefix>.<key>".
    """
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        flat[key] = v
    mlflow.log_params(flat)


def log_feature_list(feature_cols: Iterable[str]) -> None:
    """
    Log feature list as:
    - param: n_features
    - artifact: feature_columns.json
    """
    cols = list(feature_cols)
    mlflow.log_param("n_features", len(cols))
    mlflow.log_text(json.dumps(cols, indent=2), "feature_columns.json")

