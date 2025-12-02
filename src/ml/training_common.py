from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import json
import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from joblib import dump as joblib_dump

from src.common.config import settings
from src.ml.features import (
    FEATURE_SETS_V2,
    # load_classification_dataset, load_regression_dataset  # if you already have them
)
from src.ml.unified_features import load_unified_leads_dataset


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    meta: Dict


def load_cltv_regression_dataset(feature_set: str = "V2_CORE") -> Dataset:
    db_path = settings.project.db_path
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("SELECT * FROM ans.modeling_dataset_v2").fetchdf()
    finally:
        con.close()

    feature_cols = FEATURE_SETS_V2[feature_set]
    target_col = "cltv_profit"

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "n_samples": int(len(df)),
    }
    return Dataset(X=X, y=y, meta=meta)


def load_leads_v2_classification_dataset(feature_set: str = "V2_CORE") -> Dataset:
    db_path = settings.project.db_path
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("SELECT * FROM ans.modeling_dataset_v2").fetchdf()
    finally:
        con.close()

    feature_cols = FEATURE_SETS_V2[feature_set]
    target_col = "y_high_cltv"

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "n_samples": int(len(df)),
    }
    return Dataset(X=X, y=y, meta=meta)


def load_unified_leads_dataset_wrapped(feature_set: str = "UNIFIED_BASE") -> Dataset:
    X_df, y_ser = load_unified_leads_dataset(feature_set=feature_set)
    meta = {
        "feature_cols": list(X_df.columns),
        "target_col": "label_conv",
        "n_samples": int(len(X_df)),
    }
    return Dataset(
        X=X_df.to_numpy(),
        y=y_ser.to_numpy(),
        meta=meta,
    )


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib_dump(model, path)


def save_metrics(metrics: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)


def compute_classification_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> Dict:
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
    }


def compute_regression_metrics(y_true, y_pred) -> Dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
