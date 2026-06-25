"""Feature preprocessing for model training and batch scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lead_scoring.config import PipelineConfig
from lead_scoring.paths import (
    CUSTOMER_FEATURES_FILE,
    FEATURE_METADATA_FILE,
    PREPROCESSED_DATA_DIR,
    SCORING_FEATURES_FILE,
    ensure_directories,
)


TARGET_COLUMNS = {"future_revenue_12m", "converted_12m", "converted_30d"}
IDENTIFIER_COLUMNS = {"customer_id"}
PII_COLUMNS = {"first_name", "last_name", "email", "phone", "dob", "postal_code"}
DATE_COLUMNS = {"first_txn", "last_txn", "date_of_lead", "snapshot_date"}

NUMERIC_FEATURE_CANDIDATES = [
    "total_txn_amt_365",
    "txn_count_365",
    "avg_txn_amt_365",
    "max_txn_amt_365",
    "credit_txn_count_365",
    "debit_txn_count_365",
    "active_days_365",
    "recency_days",
    "tenure_days",
    "account_count",
    "active_account_count",
    "dormant_account_count",
    "closed_account_count",
    "total_balance",
    "avg_balance",
    "annual_income",
    "risk_score",
    "age",
    "lead_age_days",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "gender",
    "city",
    "state",
    "lead_source",
    "occupation",
    "marital_status",
    "customer_segment",
    "best_first_option",
    "best_second_option",
    "chosen_product",
]


def _available(columns: pd.Index, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in columns]


def split_training_data(
    df: pd.DataFrame, config: PipelineConfig = PipelineConfig()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training rows while preserving target balance when possible."""
    y = df["converted_12m"].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_preprocessor(train_df: pd.DataFrame) -> dict[str, Any]:
    """Fit simple numeric imputation and train-only frequency encoders."""
    numeric_cols = _available(train_df.columns, NUMERIC_FEATURE_CANDIDATES)
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
        "excluded_columns": sorted(TARGET_COLUMNS | IDENTIFIER_COLUMNS | PII_COLUMNS | DATE_COLUMNS),
    }


def transform_features(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Transform a customer table into a model matrix using a fitted spec."""
    out = pd.DataFrame(index=df.index)

    for col in spec["numeric_cols"]:
        values = pd.to_numeric(df.get(col), errors="coerce")
        out[col] = values.fillna(spec["numeric_medians"].get(col, 0.0)).astype(float)

    for col in spec["missing_indicator_cols"]:
        values = pd.to_numeric(df.get(col), errors="coerce")
        out[f"{col}_missing"] = values.isna().astype(int)

    for col in spec["categorical_cols"]:
        values = df.get(col, pd.Series("missing", index=df.index)).fillna("missing").astype(str)
        mapping = spec["frequency_maps"].get(col, {})
        out[f"{col}_freq"] = values.map(mapping).fillna(0.0).astype(float)

    return out.reindex(columns=spec["feature_cols"], fill_value=0.0)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_feature_sets(config: PipelineConfig = PipelineConfig()) -> dict[str, Path]:
    """Create train/test/scoring model matrices with no target leakage."""
    ensure_directories()
    training_df = pd.read_parquet(CUSTOMER_FEATURES_FILE)
    scoring_df = pd.read_parquet(SCORING_FEATURES_FILE)

    missing_targets = sorted({"future_revenue_12m", "converted_12m"} - set(training_df.columns))
    if missing_targets:
        raise ValueError(f"Missing target columns in training features: {missing_targets}")

    train_df, test_df = split_training_data(training_df, config)
    spec = fit_preprocessor(train_df)

    forbidden = TARGET_COLUMNS | IDENTIFIER_COLUMNS | PII_COLUMNS | DATE_COLUMNS
    leakage = sorted(forbidden.intersection(spec["feature_cols"]))
    if leakage:
        raise ValueError(f"Leaky or non-model columns found in feature columns: {leakage}")

    X_train = transform_features(train_df, spec)
    X_test = transform_features(test_df, spec)
    X_scoring = transform_features(scoring_df, spec)

    y_reg_train = pd.to_numeric(train_df["future_revenue_12m"], errors="coerce").fillna(0.0)
    y_reg_test = pd.to_numeric(test_df["future_revenue_12m"], errors="coerce").fillna(0.0)
    y_clf_train = train_df["converted_12m"].astype(int)
    y_clf_test = test_df["converted_12m"].astype(int)

    paths = {
        "X_train": PREPROCESSED_DATA_DIR / "X_train.parquet",
        "X_test": PREPROCESSED_DATA_DIR / "X_test.parquet",
        "X_scoring": PREPROCESSED_DATA_DIR / "X_scoring.parquet",
        "y_reg_train": PREPROCESSED_DATA_DIR / "y_reg_train.parquet",
        "y_reg_test": PREPROCESSED_DATA_DIR / "y_reg_test.parquet",
        "y_clf_train": PREPROCESSED_DATA_DIR / "y_clf_train.parquet",
        "y_clf_test": PREPROCESSED_DATA_DIR / "y_clf_test.parquet",
        "train_ids": PREPROCESSED_DATA_DIR / "train_ids.csv",
        "test_ids": PREPROCESSED_DATA_DIR / "test_ids.csv",
        "scoring_ids": PREPROCESSED_DATA_DIR / "scoring_ids.csv",
        "metadata": FEATURE_METADATA_FILE,
    }

    X_train.to_parquet(paths["X_train"], index=False)
    X_test.to_parquet(paths["X_test"], index=False)
    X_scoring.to_parquet(paths["X_scoring"], index=False)
    y_reg_train.to_frame("future_revenue_12m").to_parquet(paths["y_reg_train"], index=False)
    y_reg_test.to_frame("future_revenue_12m").to_parquet(paths["y_reg_test"], index=False)
    y_clf_train.to_frame("converted_12m").to_parquet(paths["y_clf_train"], index=False)
    y_clf_test.to_frame("converted_12m").to_parquet(paths["y_clf_test"], index=False)
    train_df[["customer_id"]].to_csv(paths["train_ids"], index=False)
    test_df[["customer_id"]].to_csv(paths["test_ids"], index=False)
    scoring_df[["customer_id"]].to_csv(paths["scoring_ids"], index=False)

    spec["train_rows"] = len(train_df)
    spec["test_rows"] = len(test_df)
    spec["scoring_rows"] = len(scoring_df)
    spec["train_positive_rate"] = float(y_clf_train.mean())
    spec["test_positive_rate"] = float(y_clf_test.mean())
    _write_json(paths["metadata"], spec)
    return paths


def main() -> None:
    paths = prepare_feature_sets()
    print(f"Saved train matrix to {paths['X_train']}")
    print(f"Saved test matrix to {paths['X_test']}")
    print(f"Saved scoring matrix to {paths['X_scoring']}")


if __name__ == "__main__":
    main()
