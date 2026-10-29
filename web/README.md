# Results dashboard

A single static page (Next.js) that shows the committed results of the lead-scoring rebuild on MBD-mini. Live at https://bank-lead-scoring-one.vercel.app.

Every control switches between measured results read from `public/data/results.json`; nothing is recomputed in the browser. The other JSON files in `public/data/` are the exporter's older data contract, kept for its tests and the Streamlit app.

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
