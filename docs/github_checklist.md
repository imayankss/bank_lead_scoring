# GitHub Checklist

Before publishing:

- Run `python scripts/run_pipeline.py`.
- Run `python scripts/repo_health_check.py`.
- Run `python -m pytest -q`.
- Confirm `README.md` metrics match `reports/tables/metrics_summary.csv`.
- Confirm dashboard opens with `streamlit run app/streamlit_app.py`.
- Do not commit `.local/`, virtual environments, cache files, or OS metadata.
- Keep the synthetic data limitation visible in the README and project page.
