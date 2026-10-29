# Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # pinned versions
.venv/bin/python scripts/fetch_mbd.py         # MBD-mini, ~2 GB, SHA-256 checked
.venv/bin/python -m lead_scoring.data         # feature build, about an hour
.venv/bin/python -m lead_scoring.features
.venv/bin/python scripts/validate_propensity.py  # choose settings on Jul 2022
.venv/bin/python scripts/run_pipeline.py      # refit, test, reports; rebuilds features first
.venv/bin/python scripts/cltv_validation.py   # optional: CLTV model choice on validation
.venv/bin/python scripts/repo_health_check.py
.venv/bin/python -m pytest -q                 # fixtures only; no download needed
```

Dashboards:

```bash
cd web && npm install && npm run dev           # Next.js, http://localhost:3000
streamlit run app/streamlit_app.py             # legacy Streamlit app
```
