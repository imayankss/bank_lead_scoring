# Lead Scoring Dashboard

The dashboard is a Streamlit app for exploring corrected lead scores, customer segments, product recommendations, model quality, and feature importance.

## Run

```bash
streamlit run app/streamlit_app.py
```

The legacy command also works:

```bash
streamlit run dashboard.py
```

## Required Artifacts

Run the pipeline first if these files are missing or stale:

```bash
python scripts/run_pipeline.py
```

The dashboard reads:

- `data/processed/lead_scores.csv`
- `data/raw/customers.csv`
- `reports/tables/metrics_summary.csv`
- `reports/tables/decile_lift.csv`
- `reports/tables/feature_importance.csv`
- `reports/tables/per_lead_explanations.csv`

## Views

- Overview: scored customer counts, category distribution, score distribution, and top product recommendations.
- Leads: filtered lead table with export.
- Customer Detail: selected customer profile with masked contact fields and product alignment.
- Model Quality: test-set metrics and decile lift.
- Explainability: global feature importance table and chart.
