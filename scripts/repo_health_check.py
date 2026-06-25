"""Lightweight project health check for generated artifacts and data contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lead_scoring.features import DATE_COLUMNS, IDENTIFIER_COLUMNS, PII_COLUMNS, TARGET_COLUMNS
from lead_scoring.paths import (
    CUSTOMER_FEATURES_FILE,
    FEATURE_METADATA_FILE,
    LEAD_SCORES_FILE,
    MODEL_METADATA_FILE,
    PREPROCESSED_DATA_DIR,
    RAW_ACCOUNTS_FILE,
    RAW_CUSTOMERS_FILE,
    RAW_TRANSACTIONS_HISTORY_FILE,
    RAW_TRANSACTIONS_RECENT_FILE,
    SCORING_FEATURES_FILE,
    TABLES_DIR,
)


REQUIRED_FILES = [
    RAW_CUSTOMERS_FILE,
    RAW_ACCOUNTS_FILE,
    RAW_TRANSACTIONS_HISTORY_FILE,
    RAW_TRANSACTIONS_RECENT_FILE,
    CUSTOMER_FEATURES_FILE,
    SCORING_FEATURES_FILE,
    PREPROCESSED_DATA_DIR / "X_train.parquet",
    PREPROCESSED_DATA_DIR / "X_test.parquet",
    PREPROCESSED_DATA_DIR / "X_scoring.parquet",
    PREPROCESSED_DATA_DIR / "train_ids.csv",
    PREPROCESSED_DATA_DIR / "test_ids.csv",
    PREPROCESSED_DATA_DIR / "scoring_ids.csv",
    LEAD_SCORES_FILE,
    TABLES_DIR / "metrics_summary.csv",
    TABLES_DIR / "decile_lift.csv",
    FEATURE_METADATA_FILE,
    MODEL_METADATA_FILE,
]


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    for path in REQUIRED_FILES:
        if not path.exists():
            failures.append(f"missing required artifact: {path.relative_to(ROOT)}")

    if not failures:
        customers = pd.read_csv(RAW_CUSTOMERS_FILE)
        training = pd.read_parquet(CUSTOMER_FEATURES_FILE)
        scoring = pd.read_parquet(SCORING_FEATURES_FILE)
        scores = pd.read_csv(LEAD_SCORES_FILE)
        X_train = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet")

        if len(training) != len(customers):
            failures.append(f"training features cover {len(training)} customers; expected {len(customers)}")
        if len(scoring) != len(customers):
            failures.append(f"scoring features cover {len(scoring)} customers; expected {len(customers)}")
        if len(scores) != len(customers):
            failures.append(f"lead scores cover {len(scores)} customers; expected {len(customers)}")
        if scores["customer_id"].duplicated().any():
            failures.append("lead_scores.csv contains duplicate customer_id values")

        forbidden = TARGET_COLUMNS | IDENTIFIER_COLUMNS | PII_COLUMNS | DATE_COLUMNS
        leakage = sorted(forbidden.intersection(X_train.columns))
        if leakage:
            failures.append(f"model matrix contains forbidden columns: {leakage}")

        positive_rate = float(training["converted_12m"].mean())
        if positive_rate < 0.05:
            warnings.append(f"low training conversion rate: {positive_rate:.2%}")
        if scores["lead_score"].nunique() < 5:
            warnings.append("lead scores have fewer than 5 unique values")

    if FEATURE_METADATA_FILE.exists():
        metadata = json.loads(FEATURE_METADATA_FILE.read_text(encoding="utf-8"))
        print(f"Feature columns: {len(metadata.get('feature_cols', []))}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        return 1
    print("Health check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
