# Results dashboard

A static Next.js app with two pages. Live at https://bank-lead-scoring-one.vercel.app.

- `/` **Lead Desk:** the RM call list from `public/data/crm.json`: the top 1,000 clients of the current book, next best product, reasons and flags.
- `/results`: the measured model results from `public/data/results.json`.

Nothing is recomputed in the browser. The other JSON files in `public/data/` are the exporter's older data contract, kept for its tests and the Streamlit app.

## Run

```bash
PYTHONPATH=src .venv/bin/python -m lead_scoring.web_export   # from the repo root: rebuild the JSON from reports/tables/
cd web
npm install
npm run dev
```

## Check

```bash
npm run lint
npm run build
```

`tests/test_web_exporter.py` checks the committed JSON against `reports/tables/`.

## Deploy

Vercel project `bank-lead-scoring` with root directory `web`; every push to `main` redeploys.
