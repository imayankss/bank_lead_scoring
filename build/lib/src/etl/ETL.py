"""
stage_duckdb.py
Improvements:
- Schema validation (--strict) with required columns per table.
- Deterministic types: load CSV as varchar then CAST explicitly.
- Change detection via SHA-256 checksums; manifest under cache/manifest.json.
- Atomic refresh: build __table_new then swap in a single transaction.
- Tunables: --threads, --memory, --tempdir, --force, --indexes.
- Observability: emit per-table row counts, date min/max, and null-rates to cache/summary.json.

Usage:
  python stage_duckdb.py --src "/path/to/Synthetic bank data set" \
                         --db "./cltv.duckdb" --cache "./cache" \
                         --threads 8 --memory "2GB" --strict
Requires: duckdb (>=0.9), pyarrow
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List
from src.common.config import settings

import duckdb

# ---- Logical → file mapping ----
TABLE_MAP: Dict[str, str] = {
    "schema.general_acct_mast_table": "prod_ods_cbsind_general_acct_mast_table.csv",
    "schema.htd": "schema_htd.csv",
    "schema.chrg_tran_log_tbl": "schema_chrg_tran_log_tbl.csv",
    "schema.gam": "schema_gam.csv",
    "ans.acc_car": "ans_acc_car.csv",
}

# ---- Required columns per table ----
REQUIRED_COLS: Dict[str, List[str]] = {
    "schema.general_acct_mast_table": [
        "sol_id","cust_id","acid","foracid","acct_opn_date","acct_cls_date",
        "acct_cls_flg","acct_crncy_code","schm_code","schm_type","sanct_lim","clr_bal_amt","del_flg"
    ],
    "schema.htd": [
        "txn_id","cust_id","acid","tran_date","tran_amt","part_tran_type",
        "tran_sub_type","tran_particular","del_flg","pstd_flg"
    ],
    "schema.chrg_tran_log_tbl": [
        "charge_id","target_acid","actual_amt_coll","chrg_tran_date","tran_particular",
        "part_tran_type","reversal_flg","entity_cre_flg","del_flg"
    ],
    "schema.gam": ["cust_id","cif_id"],
    "ans.acc_car": ["cust_id","avg_bal_12m_td","avg_bal_12m_savings","avg_bal_12m_current_ac"],
}

# ---- Explicit type map for deterministic casting (DuckDB types) ----
TYPE_MAP: Dict[str, Dict[str, str]] = {
    "schema.general_acct_mast_table": {
        "sol_id":"BIGINT","cust_id":"VARCHAR","acid":"VARCHAR","foracid":"VARCHAR",
        "acct_opn_date":"DATE","acct_cls_date":"DATE","acct_cls_flg":"VARCHAR",
        "acct_crncy_code":"VARCHAR","schm_code":"VARCHAR","schm_type":"VARCHAR",
        "sanct_lim":"DOUBLE","clr_bal_amt":"DOUBLE","del_flg":"VARCHAR"
    },
    "schema.htd": {
        "txn_id":"VARCHAR","cust_id":"VARCHAR","acid":"VARCHAR","tran_date":"DATE",
        "tran_amt":"DOUBLE","part_tran_type":"VARCHAR","tran_sub_type":"VARCHAR",
        "tran_particular":"VARCHAR","del_flg":"VARCHAR","pstd_flg":"VARCHAR"
    },
    "schema.chrg_tran_log_tbl": {
        "charge_id":"VARCHAR","target_acid":"VARCHAR","actual_amt_coll":"DOUBLE",
        "chrg_tran_date":"DATE","tran_particular":"VARCHAR","part_tran_type":"VARCHAR",
        "reversal_flg":"VARCHAR","entity_cre_flg":"VARCHAR","del_flg":"VARCHAR"
    },
    "schema.gam": {"cust_id":"VARCHAR","cif_id":"VARCHAR"},
    "ans.acc_car": {
        "cust_id":"VARCHAR","avg_bal_12m_td":"DOUBLE",
        "avg_bal_12m_savings":"DOUBLE","avg_bal_12m_current_ac":"DOUBLE"
    },
}

DATE_COLS = {
    "schema.general_acct_mast_table": ["acct_opn_date","acct_cls_date"],
    "schema.htd": ["tran_date"],
    "schema.chrg_tran_log_tbl": ["chrg_tran_date"],
}

def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_header(csv_path: Path):
    import csv
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)

def validate_columns(table: str, csv_path: Path, strict: bool) -> None:
    required = set(REQUIRED_COLS[table])
    header = set(read_header(csv_path))
    missing = sorted(list(required - header))
    if missing:
        msg = f"[ERROR] {table}: missing required columns: {missing} in {csv_path.name}"
        if strict:
            raise ValueError(msg)
        else:
            print(msg)

def ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def cast_select_clause(table: str, cols) -> str:
    tm = TYPE_MAP[table]
    select_parts = []
    for c in cols:
        t = tm.get(c, "VARCHAR")
        # Use quoted identifier to handle mixed-case safely
        ident = '"' + c.replace('"','""') + '"'
        select_parts.append(f"TRY_CAST({ident} AS {t}) AS {ident}")
    return ", ".join(select_parts) if select_parts else "*"

def atomic_swap(con: duckdb.DuckDBPyConnection, schema: str, table: str, new_table: str):
    con.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
    con.execute(f"ALTER TABLE {schema}.{new_table} RENAME TO {table}")

def summarize_table(con: duckdb.DuckDBPyConnection, full_name: str) -> dict:
    info = {"table": full_name}
    rows = con.execute(f"SELECT COUNT(*) FROM {full_name}").fetchone()[0]
    info["row_count"] = int(rows)
    for dcol in DATE_COLS.get(full_name, []):
        cols = [r[0] for r in con.execute(f"PRAGMA table_info('{full_name}')").fetchall()]
        if dcol in cols:
            mn, mx = con.execute(f"SELECT MIN({dcol}), MAX({dcol}) FROM {full_name}").fetchone()
            info[f"{dcol}_min"] = str(mn) if mn is not None else None
            info[f"{dcol}_max"] = str(mx) if mx is not None else None
    req = REQUIRED_COLS.get(full_name, [])
    if req:
        expr = ", ".join([
            f"avg(CASE WHEN \"{c}\" IS NULL THEN 1.0 ELSE 0.0 END) AS null_rate_{c}"
            for c in req
        ])
        row = con.execute(f"SELECT {expr} FROM {full_name}").fetchdf().to_dict(orient='records')[0]
        info.update({k: float(v) for k, v in row.items()})
    return info

def stage(src_dir: str, db_path: str, cache_dir: str, threads: int, memory: str,
          tempdir: str, strict: bool, force: bool, indexes: bool):
    src = Path(src_dir)
    dbp = Path(db_path)
    cache = Path(cache_dir)
    ensure_dirs(cache)
    manifest_path = cache / "manifest.json"
    summary_path = cache / "summary.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    con = duckdb.connect(str(dbp))
    if threads:
        con.execute(f"PRAGMA threads={threads}")
    if memory:
        con.execute(f"PRAGMA memory_limit='{memory}'")
    if tempdir:
        con.execute(f"PRAGMA temp_directory='{tempdir}'")
    con.execute("PRAGMA enable_object_cache=true")

    for sch in {"schema", "ans"}:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {sch}")

    summaries = []

    for full_name, csv_file in TABLE_MAP.items():
        schema, table = full_name.split(".")
        csv_path = src / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file {csv_path} for {full_name}")

        validate_columns(full_name, csv_path, strict)

        pq_path = cache / f"{table}.parquet"
        checksum = sha256_file(csv_path)
        needs_build = force or (full_name not in manifest) or (manifest[full_name].get("sha256") != checksum) or (not pq_path.exists())

        if needs_build:
            print(f"[BUILD] {full_name}: CSV→Parquet")
            con.execute(f"""
                COPY (
                  SELECT * FROM read_csv_auto('{csv_path.as_posix()}', header=true, all_varchar=true)
                ) TO '{pq_path.as_posix()}' (FORMAT PARQUET);
            """)
            manifest[full_name] = {"sha256": checksum, "parquet": str(pq_path)}
        else:
            print(f"[SKIP]  {full_name}: using cached Parquet")

        # Determine available columns in Parquet
        cols = [r[0] for r in con.execute(f"SELECT * FROM read_parquet('{pq_path.as_posix()}') LIMIT 1").description]
        target_cols = [c for c in TYPE_MAP[full_name].keys() if c in cols]
        select_clause = cast_select_clause(full_name, target_cols)
        new_tbl = f"__{table}_new"

        con.execute("BEGIN TRANSACTION")
        con.execute(f"DROP TABLE IF EXISTS {schema}.{new_tbl}")
        con.execute(f"CREATE TABLE {schema}.{new_tbl} AS SELECT {select_clause} FROM read_parquet('{pq_path.as_posix()}')")
        atomic_swap(con, schema, table, new_tbl)
        con.execute("COMMIT")

        if indexes and full_name in ("schema.htd","schema.chrg_tran_log_tbl","schema.general_acct_mast_table"):
            existing_cols = [r[0] for r in con.execute(f"PRAGMA table_info('{full_name}')").fetchall()]
            for key in ("acid","cust_id"):
                if key in existing_cols:
                    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{key} ON {full_name}({key})")

        summaries.append(summarize_table(con, full_name))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"[OK] Manifest → {manifest_path}")
    print(f"[OK] Summary  → {summary_path}")
    con.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        required=True,
        help="Directory with CSVs (unzipped)",
    )
    ap.add_argument(
        "--db",
        default=str(settings.project.db_path),
        help="DuckDB file path",
    )
    ap.add_argument(
        "--cache",
        default=str(settings.project.cache_dir),
        help="Parquet cache dir",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 4,
    )
    ap.add_argument(
        "--memory",
        default="2GB",
        help="DuckDB PRAGMA memory_limit (e.g., 1GB, 2GB)",
    )
    ap.add_argument(
        "--tempdir",
        default="",
        help="DuckDB temp directory",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if required columns are missing",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if checksum unchanged",
    )
    ap.add_argument(
        "--indexes",
        action="store_true",
        help="Create optional indexes for join keys",
    )
    args = ap.parse_args()

    stage(
        src_dir=args.src,
        db_path=args.db,
        cache_dir=args.cache,
        threads=args.threads,
        memory=args.memory,
        tempdir=args.tempdir,
        strict=args.strict,
        force=args.force,
        indexes=args.indexes,
    )

