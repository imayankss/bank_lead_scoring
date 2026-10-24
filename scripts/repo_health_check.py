"""Lightweight project health check for generated artifacts and data contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lead_scoring.features import FORBIDDEN_COLUMNS
from lead_scoring.paths import (
    CUSTOMER_FEATURES_FILE,
    FEATURE_METADATA_FILE,
    LEAD_SCORES_FILE,
    MBD_DIR,
    MODEL_METADATA_FILE,
    PREPROCESSED_DATA_DIR,
    SCORING_FEATURES_FILE,
    TABLES_DIR,
)


REQUIRED_FILES = [
    MBD_DIR / "targets",
    MBD_DIR / "detail" / "trx",
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
        clients = pd.read_parquet(MBD_DIR / "targets", columns=["client_id"])["client_id"].nunique()
        training = pd.read_parquet(CUSTOMER_FEATURES_FILE)
        scoring = pd.read_parquet(SCORING_FEATURES_FILE)
        scores = pd.read_csv(LEAD_SCORES_FILE)
        X_train = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet")

        for snapshot, rows in training.groupby("snapshot_date").size().items():
            if rows != clients:
                failures.append(f"snapshot {snapshot} covers {rows} clients; expected {clients}")
        if len(scoring) != clients:
            failures.append(f"scoring features cover {len(scoring)} clients; expected {clients}")
        if len(scores) != clients:
            failures.append(f"lead scores cover {len(scores)} clients; expected {clients}")
        if scores["customer_id"].duplicated().any():
            failures.append("lead_scores.csv contains duplicate customer_id values")

        leakage = sorted(FORBIDDEN_COLUMNS.intersection(X_train.columns))
        if leakage:
            failures.append(f"model matrix contains forbidden columns: {leakage}")

        positive_rate = float(training["converted"].mean())
        if positive_rate < 0.005:
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
