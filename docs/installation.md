# Installation

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild Artifacts

```bash
python scripts/run_pipeline.py
```

## Verify

```bash
python scripts/repo_health_check.py
python -m pytest -q
```

## Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

If port 8501 is busy:

```bash
streamlit run app/streamlit_app.py --server.port 8502
```
