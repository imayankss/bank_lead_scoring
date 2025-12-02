import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_module(mod: str) -> None:
    """Run `python -m <mod>` from project root and fail if non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", mod],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise AssertionError(f"Module {mod} failed with code {result.returncode}")


def test_unified_pipeline_end_to_end():
    """
    Smoke test: run unified ETL + ML + scoring pipeline and
    verify key outputs exist.
    """
    # 1. Build modeling datasets (customers + leads)
    run_module("src.etl.build_modeling_dataset_customers_universe")
    run_module("src.etl.build_modeling_dataset_leads_unified")

    # 2. Train unified lead model
    run_module("src.ml.train_unified_lgbm")

    # 3. Score customers and leads
    run_module("src.ml.score_unified_customers")
    run_module("src.scoring.lead_scoring")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    exports_dir = PROJECT_ROOT / "data" / "exports"

    # Key files we expect:
    assert (processed_dir / "unified_lead_scores.csv").exists()
    assert (processed_dir / "unified_customer_scores.csv").exists()

    # CRM-facing exports:
    # adjust names if your actual CSVs differ slightly
    crm_files = list(exports_dir.glob("customer_360_dashboard_summary*.csv")) + list(
        exports_dir.glob("customer_profiles_nbp*.csv")
    )
    assert crm_files, "Expected CRM export CSVs to be generated"
