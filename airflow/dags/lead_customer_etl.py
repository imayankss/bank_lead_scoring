from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# Hard-coded project root; change if your path differs
PROJECT_ROOT = "/Users/mayanksuryavanshi/bank_lead_scoring_project"
VENV_ACTIVATE = ". .venv/bin/activate"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
}

with DAG(
    dag_id="lead_customer_etl_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # trigger manually for now
    catchup=False,
    default_args=default_args,
    tags=["bank_lead_scoring"],
) as dag:

    build_leads = BashOperator(
        task_id="build_modeling_dataset_leads_unified",
        bash_command=(
            f"cd {PROJECT_ROOT} && {VENV_ACTIVATE} && "
            "python3 src/etl/build_modeling_dataset_leads_unified.py"
        ),
    )

    build_customers = BashOperator(
        task_id="build_modeling_dataset_customers_universe",
        bash_command=(
            f"cd {PROJECT_ROOT} && {VENV_ACTIVATE} && "
            "python3 src/etl/build_modeling_dataset_customers_universe.py"
        ),
    )

    load_to_postgres = BashOperator(
        task_id="load_processed_to_postgres",
        bash_command=(
            f"cd {PROJECT_ROOT} && {VENV_ACTIVATE} && "
            "python3 src/etl/load_processed_to_postgres.py"
        ),
    )

    build_leads >> build_customers >> load_to_postgres
