import os, pytest
if os.environ.get("WITH_AIRFLOW") != "1":
    pytest.skip("Airflow not enabled", allow_module_level=True)
try:
    from airflow.models import DagBag
except Exception as e:
    pytest.skip(f"Airflow not available: {e}", allow_module_level=True)

def test_dags_load_without_errors():
    bag = DagBag()
    assert not bag.import_errors, f"DAG import errors: {bag.import_errors}"
