import duckdb, sys, os

DB = sys.argv[1] if len(sys.argv) > 1 else "data/warehouse/cltv.duckdb"
os.makedirs(os.path.dirname(DB), exist_ok=True)
con = duckdb.connect(DB)

def exists(tbl):
    return con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE lower(table_name)=lower(?)", [tbl]).fetchone()[0] > 0

