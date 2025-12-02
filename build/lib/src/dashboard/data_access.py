from __future__ import annotations

from contextlib import contextmanager

import duckdb
import pandas as pd

from src.common.config import settings


@contextmanager
def duck_conn():
    """
    Context manager for DuckDB connection using the configured DB path.
    """
    con = duckdb.connect(str(settings.project.db_path))
    try:
        yield con
    finally:
        con.close()


def load_customer_360_dashboard(limit: int | None = 500) -> pd.DataFrame:
    """
    Load per-customer 360 view for the dashboard from ans.customer_360_dashboard_v2.
    Optionally limit the number of rows for the UI.
    """
    query = """
        SELECT
          cust_id,
          full_name,
          gender,
          age,
          income_annual_inr,
          income_segment,
          credit_score,
          risk_bucket,
          region,
          zone,
          urban_rural_flag,
          num_products,
          cbs_total_clr_bal_amt,
          aa_avg_monthly_inflows,
          aa_avg_monthly_outflows,
          label_has_any_product,
          label_has_loan_product,
          label_has_any_lead,
          ml_unified_customer_proba,
          cltv_profit_final,
          cltv_decile_final,
          hybrid_score_0_100_final,
          priority_segment,
          action_bucket,
          lead_cnt,
          last_lead_date,
          lead_recency_days,
          lead_recency_bucket,
          last_lead_source,
          last_prod_product_id
        FROM ans.customer_360_dashboard_v2
        ORDER BY hybrid_score_0_100_final DESC
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    with duck_conn() as con:
        return con.execute(query).fetchdf()


def load_customer_360_summary() -> pd.DataFrame:
    """
    Load aggregated summary for charts from ans.customer_360_dashboard_summary.
    """
    query = """
        SELECT
          priority_segment,
          action_bucket,
          lead_recency_bucket,
          last_lead_source,
          n_customers,
          avg_hybrid_score,
          avg_ml_proba,
          avg_cltv
        FROM ans.customer_360_dashboard_summary
        ORDER BY
          priority_segment,
          action_bucket,
          lead_recency_bucket,
          last_lead_source
    """
    with duck_conn() as con:
        return con.execute(query).fetchdf()


def load_hero_slice() -> pd.DataFrame:
    """
    Load the hero outreach slice (HIGH, A1/A2, recent, digital sources).
    """
    query = """
        SELECT
          cust_id,
          full_name,
          gender,
          age,
          income_annual_inr,
          income_segment,
          credit_score,
          risk_bucket,
          region,
          zone,
          urban_rural_flag,
          num_products,
          cbs_total_clr_bal_amt,
          aa_avg_monthly_inflows,
          aa_avg_monthly_outflows,
          ml_unified_customer_proba,
          cltv_profit_final,
          cltv_decile_final,
          hybrid_score_0_100_final,
          priority_segment,
          action_bucket,
          lead_cnt,
          last_lead_date,
          lead_recency_days,
          lead_recency_bucket,
          last_lead_source,
          last_prod_product_id
        FROM ans.customer_360_hero_slice
        ORDER BY hybrid_score_0_100_final DESC
    """
    with duck_conn() as con:
        return con.execute(query).fetchdf()

