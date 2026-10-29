# Streamlit Dashboard (legacy)

The main dashboard is the Next.js app in `web/`. The Streamlit app is kept for interactive exploration of lead scores.

```bash
streamlit run app/streamlit_app.py
```

Run `python scripts/run_pipeline.py` first. The app reads `data/processed/lead_scores.csv`, `data/processed/scoring_features.parquet` and the tables in `reports/tables/`.

## Views

- **Overview:** lead tiers, score distribution, clients who already took each product.
- **Leads:** filterable table (top 1,000 shown; the download has every client).
- **Customer Detail:** activity at the scoring date, product history and follow-up flags for the top 500 leads.
- **Model Quality:** test-month metrics with 95% intervals and the decile table.
- **Explainability:** permutation feature importance.

Data is Sber AI Lab's public MBD-mini (business clients, CC BY 4.0), not Bank of India customers. Amounts are anonymised units.
