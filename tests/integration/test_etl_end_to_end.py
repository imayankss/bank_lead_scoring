import shutil
from pathlib import Path

import duckdb

from src.etl.ETL import run_etl


def _copy_fixtures(raw_src: Path, raw_dst: Path) -> None:
    raw_dst.mkdir(parents=True, exist_ok=True)
    for csv_path in raw_src.glob("*.csv"):
        shutil.copy(csv_path, raw_dst / csv_path.name)


def test_etl_end_to_end(tmp_path: Path) -> None:
    """
    End-to-end smoke test: run ETL on tiny fixture data and ensure
    core tables are populated and summary/cache artifacts are created.
    """
    project_root = Path(__file__).resolve().parents[2]
    fixtures_raw = project_root / "tests" / "fixtures" / "raw"

    db_path = tmp_path / "cltv_test.duckdb"
    cache_dir = tmp_path / "cache"
    raw_dir = tmp_path / "raw"

    # Copy sample raw CSVs into a temporary raw directory
    _copy_fixtures(fixtures_raw, raw_dir)

    # Run ETL staging on the tiny sample
    run_etl(
        src_dir=raw_dir,
        db_path=db_path,
        cache_dir=cache_dir,
        # keep defaults for threads/memory/tempdir/strict/force/indexes
    )

    # Verify DuckDB tables exist and contain rows
    con = duckdb.connect(str(db_path))

    # Adjust these table names to match what ETL.py actually creates
    tables_to_check = [
        "schema.general_acct_mast_table",
        "schema.htd",
        "schema.chrg_tran_log_tbl",
    ]

    for t in tables_to_check:
        count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        assert count > 0, f"Table {t} is empty in test ETL run"

    # Optionally check that summary.json exists in the cache dir
    summary = cache_dir / "summary.json"
    assert summary.exists(), "summary.json not created in cache_dir"

