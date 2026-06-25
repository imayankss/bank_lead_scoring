# Project Audit

Generated during the product upgrade pass.

## Current Purpose

This repository is a lead scoring and CLTV analytics project for banking customers. It combines customer profile, account, and transaction data to estimate 12 month customer value, conversion propensity, expected value, and a ranked 0-100 lead score.

The project is not a time-series forecasting repo in its current data shape. The upgraded dashboard should therefore present it as a lead scoring intelligence platform with portfolio value scenarios, model quality, risk signals, and customer prioritization.

## Current Folder Structure

- `src/lead_scoring/`: reusable Python package for data processing, feature preparation, model training, evaluation, scoring, explainability, and dashboard loading.
- `scripts/`: executable entrypoints for pipeline and health checks.
- `data/raw/`: source CSV files.
- `data/processed/`: generated feature tables, model matrices, IDs, and lead scores.
- `models/`: trained CLTV and propensity model artifacts.
- `reports/figures/`: generated charts.
- `reports/tables/`: metrics, decile lift, feature importance, test predictions, fallback reasons, and per-lead explanations.
- `docs/`: architecture, methodology, installation, results, and GitHub checklist.
- `app/`: legacy Streamlit dashboard.
- `tests/`: backend contract tests.

## Existing Models

- CLTV model: `HistGradientBoostingRegressor` wrapped in a transformed target regressor.
- Propensity model: `HistGradientBoostingClassifier`.
- Current models are trained through `src/lead_scoring/training.py`.

## Existing Scripts

- `scripts/run_pipeline.py`: runs feature generation, preprocessing, training, evaluation, scoring, fallback explanations, and explainability artifacts.
- `scripts/repo_health_check.py`: validates key artifacts and leakage protections.
- Root wrappers such as `train_models.py`, `score_batch.py`, and `dashboard.py` preserve old commands while routing to the structured package.

## Existing Outputs

Dashboard-ready backend outputs already exist:

- `data/processed/lead_scores.csv`
- `reports/tables/metrics_summary.csv`
- `reports/tables/decile_lift.csv`
- `reports/tables/feature_importance.csv`
- `reports/tables/test_predictions.csv`
- `reports/tables/fallback_explanations.csv`
- `reports/tables/per_lead_explanations.csv`
- `reports/figures/calibration.png`
- `reports/figures/feature_importance.png`

## Duplicate Logic

- Legacy root scripts are intentionally retained as thin wrappers.
- Dashboard logic exists in Streamlit and will be treated as legacy once the Next.js dashboard is added.
- `reports/tables/shap_top_features.csv` is a compatibility artifact; it now mirrors permutation importance, not true SHAP.

## Broken or Risky Files

- `.local/Miniforge3-MacOSX-arm64.sh` is intentionally ignored and should not be committed.
- The Streamlit dashboard is functional but not the primary product presentation layer.
- The project folder is still not initialized as a Git repository.

## Missing Pieces Before This Upgrade

- Static JSON export contract for frontend.
- Modern `web/` dashboard.
- Model registry and model comparison table.
- Residual diagnostics.
- Risk intelligence artifacts.
- Scenario analysis artifacts.
- Dashboard-specific README and Makefile commands.

## Recommended Improvements

1. Add model registry metadata and comparison outputs.
2. Add residual diagnostics and model-quality summaries.
3. Add risk scoring and alert rules.
4. Add portfolio scenario analysis.
5. Export stable JSON files into `web/public/data/`.
6. Build a modern Next.js dashboard that reads only JSON artifacts.
7. Update README with backend and dashboard commands.
8. Add tests for metric calculations and web export contracts.
