from src.common.config import settings
import os
import duckdb, sys, os

DB = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BLS_DUCKDB", str(settings.project.db_path))
os.makedirs(os.path.dirname(DB), exist_ok=True)
con = duckdb.connect(DB)

def exists(tbl):
    return con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE lower(table_name)=lower(?)", [tbl]).fetchone()[0] > 0