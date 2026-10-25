"""Feature preprocessing for model training and batch scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lead_scoring.config import PipelineConfig
from lead_scoring.data import LABELS
from lead_scoring.paths import (
    CUSTOMER_FEATURES_FILE,
    FEATURE_METADATA_FILE,
    PREPROCESSED_DATA_DIR,
    SCORING_FEATURES_FILE,
    ensure_directories,
)
from lead_scoring.temporal import assert_no_forbidden_columns, assert_target_windows_before


TARGET_COLUMNS = {"future_value", *LABELS}
IDENTIFIER_COLUMNS = {"customer_id", "fold"}
PII_COLUMNS = {"first_name", "last_name", "email", "phone", "dob", "postal_code"}
DATE_COLUMNS = {"first_txn", "last_txn", "snapshot_date"}
# MBD columns no transaction window reproduces (best match 11% of rows), so their timing is unknown.
UNEXPLAINED_COLUMNS = {"trans_count", "diff_trans_date"}
# Cumulative history spans 0 months at the Jan-2022 snapshot and 8 at the test one (labels start 2022-02),
# so the same value means different things in training and test. Kept for display and the rule baseline only.
CUMULATIVE_HISTORY = {"products_taken_before", "target_1_before", "target_2_before", "target_3_before", "target_4_before"}
FORBIDDEN_COLUMNS = TARGET_COLUMNS | IDENTIFIER_COLUMNS | PII_COLUMNS | DATE_COLUMNS | UNEXPLAINED_COLUMNS | CUMULATIVE_HISTORY

NUMERIC_FEATURE_CANDIDATES = [
    "total_txn_amt_365", "txn_count_365", "avg_txn_amt_365", "max_txn_amt_365", "active_days_365",
    "txn_count_90", "total_txn_amt_90", "recency_days", "tenure_days", "ratio_30_90", "ratio_90_365",
]
# Built per vocabulary and window by lead_scoring.data, so they are matched by prefix.
GENERATED_FEATURE_PREFIXES = ("hist_", "cat_", "txn_count_d30_", "txn_amt_d30_", "days_since_evt_")

# MBD publishes no client attributes; frequency encoding stays for data sources that do.
CATEGORICAL_FEATURE_CANDIDATES: list[str] = []


def _available(columns: pd.Index, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in columns]


def split_training_data(
    df: pd.DataFrame, config: PipelineConfig = PipelineConfig()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earlier snapshots, test on the later one. Raises if target windows overlap the test."""
    assert_target_windows_before(config.train_snapshots, config.horizon_months, config.test_snapshot)
    train_df = df[df["snapshot_date"].isin(config.train_snapshots)]
    test_df = df[df["snapshot_date"] == config.test_snapshot]
    for name, part in (("train", train_df), ("test", test_df)):
        if part.empty:
            raise ValueError(f"no {name} rows for snapshots in config")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_preprocessor(train_df: pd.DataFrame) -> dict[str, Any]:
    """Fit simple numeric imputation and train-only frequency encoders."""
    numeric_cols = _available(train_df.columns, NUMERIC_FEATURE_CANDIDATES)
    numeric_cols += [c for c in train_df.columns if c.startswith(GENERATED_FEATURE_PREFIXES)]
    categorical_cols = _available(train_df.columns, CATEGORICAL_FEATURE_CANDIDATES)

    medians = {}
    missing_indicator_cols = []
    for col in numeric_cols:
        values = pd.to_numeric(train_df[col], errors="coerce")
        medians[col] = float(values.median()) if values.notna().any() else 0.0
        if values.isna().any():
            missing_indicator_cols.append(col)

    frequency_maps: dict[str, dict[str, float]] = {}
    for col in categorical_cols:
        values = train_df[col].fillna("missing").astype(str)
        frequency_maps[col] = values.value_counts(normalize=True).to_dict()

    feature_cols = numeric_cols + [f"{c}_missing" for c in missing_indicator_cols]
    feature_cols += [f"{c}_freq" for c in categorical_cols]

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_medians": medians,
        "missing_indicator_cols": missing_indicator_cols,
        "frequency_maps": frequency_maps,
        "feature_cols": feature_cols,
        "excluded_columns": sorted(FORBIDDEN_COLUMNS),
    }


def transform_features(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Transform a customer table into a model matrix using a fitted spec."""
    out: dict[str, pd.Series] = {}

    for col in spec["numeric_cols"]:
        values = pd.to_numeric(df.get(col), errors="coerce")
        out[col] = values.fillna(spec["numeric_medians"].get(col, 0.0)).astype("float32")

    for col in spec["missing_indicator_cols"]:
        values = pd.to_numeric(df.get(col), errors="coerce")
        out[f"{col}_missing"] = values.isna().astype("float32")

    for col in spec["categorical_cols"]:
        values = df.get(col, pd.Series("missing", index=df.index)).fillna("missing").astype(str)
        mapping = spec["frequency_maps"].get(col, {})
        out[f"{col}_freq"] = values.map(mapping).fillna(0.0).astype("float32")

    return pd.DataFrame(out, index=df.index).reindex(columns=spec["feature_cols"], fill_value=0.0).astype("float32")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_feature_sets(config: PipelineConfig = PipelineConfig()) -> dict[str, Path]:
    """Create train/test/scoring model matrices with no target leakage."""
    ensure_directories()
    training_df = pd.read_parquet(CUSTOMER_FEATURES_FILE)
    scoring_df = pd.read_parquet(SCORING_FEATURES_FILE)

    missing_targets = sorted(TARGET_COLUMNS - set(training_df.columns))
    if missing_targets:
        raise ValueError(f"Missing target columns in training features: {missing_targets}")

    train_df, test_df = split_training_data(training_df, config)
    spec = fit_preprocessor(train_df)

    assert_no_forbidden_columns(spec["feature_cols"], FORBIDDEN_COLUMNS)

    X_train = transform_features(train_df, spec)
    X_test = transform_features(test_df, spec)
    X_scoring = transform_features(scoring_df, spec)

    y_reg_train = pd.to_numeric(train_df["future_value"], errors="coerce").fillna(0.0)
    y_reg_test = pd.to_numeric(test_df["future_value"], errors="coerce").fillna(0.0)
    y_clf_train = train_df["converted"].astype(int)
    y_clf_test = test_df["converted"].astype(int)

    paths = {
        "X_train": PREPROCESSED_DATA_DIR / "X_train.parquet",
        "X_test": PREPROCESSED_DATA_DIR / "X_test.parquet",
        "X_scoring": PREPROCESSED_DATA_DIR / "X_scoring.parquet",
        "y_reg_train": PREPROCESSED_DATA_DIR / "y_reg_train.parquet",
        "y_reg_test": PREPROCESSED_DATA_DIR / "y_reg_test.parquet",
        "y_clf_train": PREPROCESSED_DATA_DIR / "y_clf_train.parquet",
        "y_clf_test": PREPROCESSED_DATA_DIR / "y_clf_test.parquet",
        "y_labels_train": PREPROCESSED_DATA_DIR / "y_labels_train.parquet",
        "y_labels_test": PREPROCESSED_DATA_DIR / "y_labels_test.parquet",
        "train_ids": PREPROCESSED_DATA_DIR / "train_ids.csv",
        "test_ids": PREPROCESSED_DATA_DIR / "test_ids.csv",
        "scoring_ids": PREPROCESSED_DATA_DIR / "scoring_ids.csv",
        "metadata": FEATURE_METADATA_FILE,
    }

    X_train.to_parquet(paths["X_train"], index=False)
    X_test.to_parquet(paths["X_test"], index=False)
    X_scoring.to_parquet(paths["X_scoring"], index=False)
    y_reg_train.to_frame("future_value").to_parquet(paths["y_reg_train"], index=False)
    y_reg_test.to_frame("future_value").to_parquet(paths["y_reg_test"], index=False)
    y_clf_train.to_frame("converted").to_parquet(paths["y_clf_train"], index=False)
    y_clf_test.to_frame("converted").to_parquet(paths["y_clf_test"], index=False)
    train_df[LABELS].astype("int8").to_parquet(paths["y_labels_train"], index=False)
    test_df[LABELS].astype("int8").to_parquet(paths["y_labels_test"], index=False)
    train_df[["customer_id", "snapshot_date"]].to_csv(paths["train_ids"], index=False)
    test_df[["customer_id", "snapshot_date"]].to_csv(paths["test_ids"], index=False)
    scoring_df[["customer_id"]].to_csv(paths["scoring_ids"], index=False)

    spec["train_rows"] = len(train_df)
    spec["test_rows"] = len(test_df)
    spec["scoring_rows"] = len(scoring_df)
    spec["train_positive_rate"] = float(y_clf_train.mean())
    spec["test_positive_rate"] = float(y_clf_test.mean())
    spec["train_snapshots"] = list(config.train_snapshots)
    spec["test_snapshot"] = config.test_snapshot
    _write_json(paths["metadata"], spec)
    return paths


def main() -> None:
    paths = prepare_feature_sets()
    print(f"Saved train matrix to {paths['X_train']}")
    print(f"Saved test matrix to {paths['X_test']}")
    print(f"Saved scoring matrix to {paths['X_scoring']}")


if __name__ == "__main__":
    main()
