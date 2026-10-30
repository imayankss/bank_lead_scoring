# GitHub Checklist

Before publishing:

- Run `python scripts/fetch_mbd.py` once, then `python scripts/run_pipeline.py` and `python scripts/cltv_validation.py`.
- Run `python scripts/repo_health_check.py`.
- Run `python -m pytest -q`.
- `tests/test_readme_numbers.py` checks README numbers against `reports/tables/`; keep it passing.
- Run `cd web && npm run lint && npm run build`; Vercel redeploys https://bank-lead-scoring-one.vercel.app on every push to `main`.
- Do not commit `.local/`, virtual environments, cache files, or OS metadata.
- Keep the data source visible: Sber AI Lab MBD-mini, business clients, not Bank of India customers.
- Never commit `data/raw/mbd/`, `data/processed/` or `models/*.pkl`.
