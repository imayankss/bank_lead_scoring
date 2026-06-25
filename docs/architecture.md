# Architecture

## Pipeline Flow

```text
data/raw/*.csv
  -> src/lead_scoring/data.py
  -> data/processed/customer_features.parquet
  -> data/processed/scoring_features.parquet
  -> src/lead_scoring/features.py
  -> data/processed/preprocessed/*.parquet
  -> src/lead_scoring/training.py
  -> models/*.pkl
  -> src/lead_scoring/evaluation.py
  -> reports/tables/*.csv and reports/figures/*.png
  -> src/lead_scoring/scoring.py
  -> data/processed/lead_scores.csv
  -> app/streamlit_app.py
```

## Key Modules

- `src/lead_scoring/paths.py`: central file paths.
- `src/lead_scoring/config.py`: pipeline settings such as lookback and target horizon.
- `src/lead_scoring/data.py`: raw data loading, transaction aggregation, account aggregation, target creation.
- `src/lead_scoring/features.py`: train/test split, preprocessing, leakage-safe feature matrix creation.
- `src/lead_scoring/training.py`: CLTV and propensity model training.
- `src/lead_scoring/evaluation.py`: metrics, calibration, predictions, and decile lift.
- `src/lead_scoring/scoring.py`: current customer scoring and fallback explanations.
- `src/lead_scoring/explainability.py`: permutation feature importance and per-lead drivers.
- `src/lead_scoring/pipeline.py`: end-to-end orchestration.
- `app/streamlit_app.py`: dashboard UI.

## Correctness Fixes

- Training features use a historical snapshot at `max_transaction_date - 365 days`.
- Targets use transactions after the training snapshot.
- Current scoring uses a separate latest-snapshot feature table with no target columns.
- Customer IDs are saved separately through train, test, and scoring matrices.
- Target columns, identifiers, dates, and direct PII are excluded from model features.
