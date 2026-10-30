# Architecture

## Pipeline flow

```text
scripts/fetch_mbd.py -> data/raw/mbd/{targets,detail/trx,detail/dialog}  (pinned revision, SHA-256 checked; not committed)
  -> src/lead_scoring/data.py         DuckDB per client fold, in parallel processes
  -> data/processed/customer_features.parquet   (train snapshots + test snapshot, stacked)
  -> data/processed/scoring_features.parquet    (current snapshot, no targets)
  -> src/lead_scoring/features.py     snapshot time split, leakage guards, model matrices
  -> scripts/train_gru.py             optional GRU validation scores (data/processed/events cache)
  -> scripts/validate_propensity.py   LightGBM settings, blend weight, calibration -> models/propensity_choice.json
  -> src/lead_scoring/training.py     CLTV model -> models/cltv_model.pkl; LightGBM refit -> models/propensity_lgbm.txt
  -> src/lead_scoring/evaluation.py   metrics with bootstrap CIs, decile table, calibration
  -> src/lead_scoring/scoring.py      lead scores and follow-up flags
  -> src/lead_scoring/next_product.py one 3-month LightGBM per product -> next best product; per-client reasons (SHAP by feature family)
  -> model_comparison, paper_protocol, diagnostics, risk, scenario, seasonality, explainability
  -> src/lead_scoring/web_export.py   web/public/data/*.json (strict JSON)
  -> src/lead_scoring/provenance.py   reports/run_manifest.json
  -> web/ (Next.js): Lead Desk at / (crm.json), model results at /results (results.json); Vercel, root web
  -> app/streamlit_app.py (legacy)
```

## Key modules

- `paths.py`: file locations and the pinned MBD revision and hashes.
- `config.py`: lookback, horizon, train/test/scoring snapshots, worker count.
- `temporal.py`: leakage guards that raise `LeakageError`.
- `data.py`: client-level features and targets per snapshot (transactions, purchase history, dialogs), amount cap.
- `propensity.py`: production propensity: the chosen LightGBM, ranked and calibrated with the validation map.
- `blend.py`: rank blend, blend-weight choice and the isotonic calibrator.
- `sequences.py`, `gru.py`: per-client transaction sequences and the GRU (validated, weight 0 in production).
- `paper_protocol.py`: next-month per-product AUC next to the MBD paper.
- `features.py`: time split, preprocessing, forbidden-column list.
- `metrics.py`: lift, tie-aware top-k precision, bootstrap CIs, Wilson intervals, decile table.
- `model_comparison.py`: models against rule, linear and prior baselines.
- `risk.py`, `scoring.py`: heuristic risk score and follow-up flags (rules, not fitted models).
- `provenance.py`: run manifest.

## Scaling

MBD's client folds share no clients, so `build_customer_features` runs one fold per process with `ProcessPoolExecutor`, and each process aggregates its fold's parquet files in DuckDB without loading raw transactions into pandas. The same code runs MBD-mini on a laptop or full MBD fold by fold. `tests/test_data_contracts.py` checks that parallel and serial output are identical.
