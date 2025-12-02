from __future__ import annotations

import os
import json
from typing import Dict, Any

import duckdb
import pandas as pd

from src.common.config import settings, PROJECT_ROOT


def build_crm_export_unified(
    con: "duckdb.DuckDBPyConnection",
    rw: float,
    mw: float,
) -> None:
    """
    Build ans.crm_export_final using the unified LGBM probabilities
    from ans.unified_lead_scores (ml_unified_proba), for ALL rows in that table.

    - ans.unified_lead_scores is the primary source (new unified dataset).
    - ans.customer_lead_scores is joined optionally (old rules-based scores).
    """

    con.execute(
        f"""
        CREATE OR REPLACE TABLE ans.crm_export_final AS
        SELECT
          u.cust_id,
          u.Customer_ID,
          u.lead_id,
          -- rules-based lead score (0 if not available from old table)
          COALESCE(l.lead_score_0_100, 0) AS lead_score_0_100,
          l.cltv_decile,
          l.cltv_category_q,
          l.recommended_action,
          l.priority_flag,
          u.ml_unified_proba AS ml_proba,
          ROUND(
            {rw} * COALESCE(l.lead_score_0_100, 0)
            + {mw} * (100.0 * COALESCE(u.ml_unified_proba, 0))
          ) AS hybrid_score_0_100,
          'unified_lgbm'::VARCHAR AS champion_model
        FROM ans.unified_lead_scores u
        LEFT JOIN ans.customer_lead_scores l
          USING (cust_id)
        """
    )

    export_path = str(settings.exports.crm_hybrid)
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM ans.crm_export_final
            ORDER BY hybrid_score_0_100 DESC, lead_score_0_100 DESC
        )
        TO '{export_path}' (HEADER, DELIMITER ',')
        """
    )

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_select_unified.json", "w") as f:
        json.dump(
            {
                "weights": {
                    "rules_pct": rw,
                    "ml_pct": mw,
                },
                "champion": "unified_lgbm",
            },
            f,
            indent=2,
        )

    print(f"[OK] Unified CRM export written -> {export_path}")


def build_crm_export(
    con: "duckdb.DuckDBPyConnection",
    rw: float,
    mw: float,
) -> None:
    """
    Backwards-compatible wrapper that delegates to the unified CRM export builder.

    Older scripts may import `build_crm_export`. In the current design,
    we standardize on the unified LGBM-based pipeline, so this simply
    calls `build_crm_export_unified` with the same arguments.
    """
    return build_crm_export_unified(con=con, rw=rw, mw=mw)

UNIFIED_CUSTOMER_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "unified_customer_scores.csv"


def load_unified_customer_scores_to_duckdb() -> None:
    """
    Load the unified customer-level scores CSV into DuckDB as ans.unified_customer_scores.
    One row per customer (cust_id), scored by the unified model.
    """
    if not UNIFIED_CUSTOMER_SCORES_PATH.exists():
        raise FileNotFoundError(f"Unified customer scores CSV not found at {UNIFIED_CUSTOMER_SCORES_PATH}")

    df = pd.read_csv(UNIFIED_CUSTOMER_SCORES_PATH)

    con = duckdb.connect(str(settings.project.db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS ans")
        con.execute("DROP TABLE IF EXISTS ans.unified_customer_scores")
        con.register("__unified_customer_scores", df)
        con.execute(
            """
            CREATE TABLE ans.unified_customer_scores AS
            SELECT *
            FROM __unified_customer_scores
            """
        )
        con.unregister("__unified_customer_scores")
        print("[OK] Created ans.unified_customer_scores in DuckDB")
    finally:
        con.close()
