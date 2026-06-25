# Lead Scoring and CLTV Prediction System

This project builds a structured lead scoring workflow for bank customers. It combines customer profiles, account history, and transaction activity to estimate 12 month customer value, conversion propensity, expected value, and a 0-100 lead score.

The current version is a portfolio-ready ML analytics product scaffold with a reproducible Python pipeline, generated evaluation artifacts, tests, static dashboard exports, a legacy Streamlit app, and a modern Next.js dashboard.

## What It Does

- Builds customer-level features from raw customer, account, and transaction CSV files.
- Uses a historical training snapshot so labels come from future transactions rather than the same feature window.
- Trains separate models for 12 month revenue and conversion propensity.
- Scores the current customer universe with stable customer ID mapping.
- Generates model metrics, decile lift, feature importance, fallback reasons, and per-lead explanations.
- Exports frontend-ready JSON files for a modern dashboard.
- Provides a Next.js command-center dashboard and a legacy Streamlit fallback.

## Project Structure

```text
.
  app/                      Streamlit dashboard
  configs/                  Project, model, dashboard, and scenario config
  data/raw/                 Source CSV files
  data/processed/           Rebuilt feature matrices and lead scores
  docs/                     Architecture, methodology, and results notes
  models/                   Trained model artifacts
  reports/figures/          Evaluation and explainability figures
  reports/tables/           Metrics, lift, explanations, and predictions
  scripts/                  Pipeline and health-check entrypoints
  src/lead_scoring/         Reusable pipeline package
  tests/                    Contract and scoring tests
  web/                      Next.js dashboard reading static JSON exports
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/run_export_web_data.py
python scripts/repo_health_check.py
python -m pytest -q
cd web && npm install && npm run dev
```

## Current Results

The latest rebuilt pipeline scores all 500 customers.

| Metric | Value |
| --- | ---: |
| Scored customers | 500 |
| Feature columns | 29 |
| Test rows | 100 |
| Test positive rate | 66.0% |
| Propensity AUC | 0.913 |
| Propensity Brier score | 0.126 |
| CLTV RMSE | INR 145,270 |
| CLTV MAE | INR 82,076 |

Additional product-intelligence artifacts include:

- `reports/tables/model_comparison.csv`
- `reports/tables/residual_diagnostics.csv`
- `reports/tables/risk_signals.csv`
- `reports/tables/scenario_forecasts.csv`
- `reports/tables/seasonality_monthly.csv`
- `web/public/data/*.json`

Lead score distribution after the corrected scoring pass:

| Category | Customers |
| --- | ---: |
| Hot | 153 |
| Medium | 200 |
| Cold | 147 |

## Dashboard Preview

Screenshots can be added after the dashboard is reviewed locally:

- `reports/figures/dashboard_overview.png`
- `reports/figures/dashboard_models.png`
- `reports/figures/dashboard_risk.png`

## Important Notes

This is a synthetic, small-sample portfolio project. The corrected pipeline fixes the earlier target leakage, target-window issue, and shuffled customer ID mapping problem, but the reported metrics should still be treated as prototype validation rather than production evidence.

## Main Commands

Run the full pipeline:

```bash
python scripts/run_pipeline.py
```

Run the health check:

```bash
python scripts/repo_health_check.py
```

Run tests:

```bash
python -m pytest -q
```

Launch the dashboard:

```bash
cd web
npm install
npm run dev
```

Launch the legacy Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Makefile Shortcuts

```bash
make train
make export-web
make health
make test
make dashboard
```
