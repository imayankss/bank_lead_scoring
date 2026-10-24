# Architecture

## Pipeline flow

```text
scripts/fetch_mbd.py -> data/raw/mbd/{targets,detail/trx}     (pinned revision, SHA-256 checked; not committed)
  -> src/lead_scoring/data.py         DuckDB per client fold, in parallel processes
  -> data/processed/customer_features.parquet   (train snapshots + test snapshot, stacked)
  -> data/processed/scoring_features.parquet    (current snapshot, no targets)
  -> src/lead_scoring/features.py     snapshot time split, leakage guards, model matrices
  -> src/lead_scoring/training.py     CLTV and propensity models -> models/*.pkl
  -> src/lead_scoring/evaluation.py   metrics with bootstrap CIs, decile table, calibration
  -> src/lead_scoring/scoring.py      lead scores and follow-up flags
  -> model_comparison, diagnostics, risk, scenario, seasonality, explainability
  -> src/lead_scoring/web_export.py   web/public/data/*.json (strict JSON)
  -> src/lead_scoring/provenance.py   reports/run_manifest.json
  -> web/ (Next.js) and app/streamlit_app.py
```

## Key modules

- `paths.py`: file locations and the pinned MBD revision and hashes.
- `config.py`: lookback, horizon, train/test/scoring snapshots, worker count.
- `temporal.py`: leakage guards that raise `LeakageError`.
- `data.py`: client-level features and targets per snapshot, amount cap.
- `features.py`: time split, preprocessing, forbidden-column list.
- `metrics.py`: lift, tie-aware top-k precision, bootstrap CIs, Wilson intervals, decile table.
- `model_comparison.py`: models against rule, linear and prior baselines.
- `risk.py`, `scoring.py`: heuristic risk score and follow-up flags (rules, not fitted models).
- `provenance.py`: run manifest.

## Scaling

MBD's client folds share no clients, so `build_customer_features` runs one fold per process with `ProcessPoolExecutor`, and each process aggregates its fold's parquet files in DuckDB without loading raw transactions into pandas. The same code runs MBD-mini on a laptop or full MBD fold by fold. `tests/test_data_contracts.py` checks that parallel and serial output are identical.
