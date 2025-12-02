from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class TimeSplit:
    """Indices for a single time-based split."""
    train_idx: np.ndarray
    test_idx: np.ndarray
    cutoff_date: pd.Timestamp


def simple_time_based_split(
    df: pd.DataFrame,
    date_col: str,
    test_fraction: float = 0.2,
) -> TimeSplit:
    """
    One-shot time-based split:
    - Sort by date_col ascending
    - Use earliest (1 - test_fraction) as train, latest test_fraction as test
    """
    df_sorted = df.sort_values(date_col)
    n = len(df_sorted)
    split_point = int(n * (1 - test_fraction))

    train_idx = df_sorted.index[:split_point].to_numpy()
    test_idx = df_sorted.index[split_point:].to_numpy()
    cutoff_date = pd.to_datetime(df_sorted.iloc[split_point - 1][date_col])

    return TimeSplit(train_idx=train_idx, test_idx=test_idx, cutoff_date=cutoff_date)


def time_series_folds(
    df: pd.DataFrame,
    date_col: str,
    n_folds: int = 3,
    min_train_fraction: float = 0.4,
) -> List[TimeSplit]:
    """
    Rolling-origin time-series cross-validation.

    - Sorts df by date_col
    - Creates n_folds splits, each with:
      - train up to a cutoff
      - test after that cutoff (equally sized chunks)

    Intended for CLTV evaluation:
    - Each fold simulates training on older data and testing on newer data.
    """
    df_sorted = df.sort_values(date_col)
    n = len(df_sorted)
    fold_size = n // (n_folds + 1)

    splits: List[TimeSplit] = []

    for k in range(1, n_folds + 1):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_folds else n

        if test_start <= int(n * min_train_fraction):
            # ensure we always have a reasonable training window
            continue

        train_idx = df_sorted.index[:test_start].to_numpy()
        test_idx = df_sorted.index[test_start:test_end].to_numpy()
        cutoff_date = pd.to_datetime(df_sorted.iloc[test_start - 1][date_col])

        splits.append(TimeSplit(train_idx=train_idx, test_idx=test_idx, cutoff_date=cutoff_date))

    return splits


def evaluate_regression_splits(
    model_ctor,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    splits: Iterable[TimeSplit],
) -> Dict:
    """
    Train and evaluate a regression model on one or more time-based splits.

    Parameters
    ----------
    model_ctor:
        A callable that returns a new (unfitted) regressor instance each time.
    df:
        Full modeling dataframe.
    feature_cols:
        Feature column names.
    target_col:
        Regression target column, e.g. 'cltv_profit'.

    Returns
    -------
    dict with per-fold RMSE, MAE, R2 and global averages.
    """
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    rmse_list = []
    mae_list = []
    r2_list = []
    fold_info = []

    for i, split in enumerate(splits, start=1):
        model = model_ctor()

        X_train, y_train = X[split.train_idx], y[split.train_idx]
        X_test, y_test = X[split.test_idx], y[split.test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Your sklearn version does not support 'squared' kwarg, so:
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

        fold_info.append(
            {
                "fold": i,
                "cutoff_date": str(split.cutoff_date.date()),
                "n_train": int(len(split.train_idx)),
                "n_test": int(len(split.test_idx)),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

    summary = {
        "folds": fold_info,
        "rmse_mean": float(np.mean(rmse_list)) if rmse_list else None,
        "rmse_std": float(np.std(rmse_list)) if rmse_list else None,
        "mae_mean": float(np.mean(mae_list)) if mae_list else None,
        "mae_std": float(np.std(mae_list)) if mae_list else None,
        "r2_mean": float(np.mean(r2_list)) if r2_list else None,
        "r2_std": float(np.std(r2_list)) if r2_list else None,
    }

    return summary


def save_eval_report(report: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2)

