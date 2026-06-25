.PHONY: audit train forecast export-web test dashboard health

audit:
	python3 scripts/run_project_audit.py

train:
	python3 scripts/run_pipeline.py

forecast:
	python3 scripts/run_pipeline.py

export-web:
	python3 scripts/run_export_web_data.py

test:
	python3 -m pytest -q

health:
	python3 scripts/repo_health_check.py

dashboard:
	streamlit run app/streamlit_app.py
