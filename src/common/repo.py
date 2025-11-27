import duckdb, pandas as pd

def _exists(con, name: str) -> bool:
    return con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE lower(table_name)=lower(?)",
        [name]
    ).fetchone()[0] > 0

def get_customer_bundle(db_path: str, cust_id: int) -> dict:
    con = duckdb.connect(db_path, read_only=True)
    try:
        prof = con.execute(
            "SELECT * FROM customer_profile_v WHERE cust_id = ?", [cust_id]
        ).df()
        last_tx = con.execute(
            "SELECT tx_id, tx_date, amount, channel FROM fact_transactions WHERE cust_id = ? ORDER BY tx_date DESC LIMIT 5",
            [cust_id],
        ).df() if _exists(con, "fact_transactions") else pd.DataFrame()
        risk = con.execute(
            "SELECT flag, severity, detected_at FROM risk_flags WHERE cust_id = ? ORDER BY detected_at DESC LIMIT 5",
            [cust_id],
        ).df() if _exists(con, "risk_flags") else pd.DataFrame()
        return {
            "profile": prof.to_dict("records")[0] if len(prof) else None,
            "last_transactions": last_tx.to_dict("records"),
            "risk_flags": risk.to_dict("records"),
        }
    finally:
        con.close()
