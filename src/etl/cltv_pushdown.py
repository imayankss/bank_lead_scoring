from __future__ import annotations
import argparse, os
from datetime import datetime, timedelta
import duckdb
from src.common.config import settings


def run(db_path: str, end_date_str: str, export_parquet: str, export_csv: str):
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start_date = (end_date - timedelta(days=365)) + timedelta(days=1)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=" + str(os.cpu_count() or 4))

    # Parameters (CAST instead of ::DATE)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE __params AS "
        "SELECT CAST(? AS DATE) AS start_d, CAST(? AS DATE) AS end_d",
        [start_date, end_date],
    )

    sql = r"""
    WITH
    live_accounts AS (
        SELECT
            gam.sol_id, gam.cust_id, gam.acid, gam.foracid,
            gam.acct_opn_date, gam.acct_cls_date, gam.acct_cls_flg,
            gam.acct_crncy_code, gam.schm_code, gam.schm_type, gam.sanct_lim, gam.clr_bal_amt, gam.del_flg
        FROM schema.general_acct_mast_table AS gam, __params p
        WHERE gam.acct_opn_date <= p.end_d
          AND (gam.acct_cls_date > p.end_d OR gam.acct_cls_date IS NULL)
          AND gam.del_flg = 'N'
          AND gam.schm_type NOT IN ('HOC','OAB','OAP','OSP','FBA','PCA','TDA')
    ),
    chrg_summ AS (
        SELECT
            target_acid,
            COALESCE(SUM(actual_amt_coll),0) AS chrg_amt,
            COUNT(DISTINCT strftime(chrg_tran_date,'%Y-%m')) AS chrg_mnth_cnt
        FROM schema.chrg_tran_log_tbl, __params p
        WHERE entity_cre_flg = 'Y'
          AND del_flg = 'N'
          AND chrg_tran_date BETWEEN p.start_d AND p.end_d
          AND part_tran_type = 'D'
          AND COALESCE(reversal_flg,'N') <> 'Y'
          AND UPPER(COALESCE(tran_particular,'')) NOT LIKE '%GST%'
          AND UPPER(COALESCE(tran_particular,'')) NOT LIKE '%TCS%'
          AND UPPER(COALESCE(tran_particular,'')) NOT LIKE '%TAX%'
        GROUP BY target_acid
    ),
    ic_summ AS (
        SELECT
            acid,
            COUNT(DISTINCT strftime(tran_date,'%Y-%m')) AS ic_mnth_cnt,
            COALESCE(SUM(tran_amt),0) AS ic_tran_amt
        FROM schema.htd, __params p
        WHERE tran_date BETWEEN p.start_d AND p.end_d
          AND part_tran_type = 'D'
          AND tran_sub_type = 'IC'
          AND del_flg = 'N'
          AND pstd_flg = 'Y'
          AND UPPER(COALESCE(tran_particular,'')) NOT LIKE '%GST%'
          AND UPPER(COALESCE(tran_particular,'')) NOT LIKE '%TCS%'
          AND UPPER(COALESCE(tran_particular,'')) NOT LIKE '%TAX%'
        GROUP BY acid
    ),
    join1 AS (
        SELECT
            la.*,
            date_diff('month', date_trunc('month', la.acct_opn_date), date_trunc('month', (SELECT end_d FROM __params))) + 1 AS mob,
            cs.chrg_amt,
            cs.chrg_mnth_cnt,
            ic.ic_tran_amt,
            ic.ic_mnth_cnt
        FROM live_accounts la
        LEFT JOIN chrg_summ cs ON cs.target_acid = la.acid
        LEFT JOIN ic_summ   ic ON ic.acid = la.acid
    ),
    with_financials AS (
        SELECT
            *,
            CASE
              WHEN schm_type IN ('SBA','CAA','ODA','CCA','LAA') AND mob > 12
                THEN round((COALESCE(chrg_amt,0)/12.0) * mob, 2)
              WHEN schm_type IN ('SBA','CAA','ODA','CCA','LAA') AND mob <= 12
                THEN round(COALESCE(chrg_amt,0), 2)
              ELSE 0.0
            END AS chrg_amt_fin,
            round(COALESCE(ic_tran_amt,0) / NULLIF(ic_mnth_cnt,0) * mob, 2) AS ic_amt_fin
        FROM join1
    ),
    cust_agg AS (
        SELECT
            cust_id,
            CAST(round(SUM(chrg_amt_fin)) AS INTEGER) AS fin_chg,
            round(SUM(chrg_amt_fin), 2) AS chrg_amt_fin,
            round(SUM(ic_amt_fin), 2)   AS ic_amt_fin
        FROM with_financials
        GROUP BY cust_id
    ),
    casa_base AS (
        SELECT
            gam.cust_id, gam.acid, gam.foracid, gam.acct_opn_date, gam.acct_cls_date, gam.acct_cls_flg,
            gam.acct_crncy_code, gam.clr_bal_amt, gam.schm_code, gam.schm_type,
            date_diff('month', date_trunc('month', gam.acct_opn_date), date_trunc('month', (SELECT end_d FROM __params))) + 1 AS mob
        FROM schema.general_acct_mast_table gam, __params p
        WHERE gam.acct_opn_date <= p.end_d
          AND (gam.acct_cls_date > p.end_d OR gam.acct_cls_date IS NULL)
          AND gam.del_flg = 'N'
          AND gam.schm_type IN ('SBA','CAA')
          AND NOT (gam.schm_type = 'CAA' AND gam.schm_code = 'CD260')
    ),
    casa_group AS (
        SELECT
            cust_id,
            SUM(CASE WHEN schm_type='SBA' THEN 1 ELSE 0 END) AS sa_cnt,
            SUM(CASE WHEN schm_type='CAA' THEN 1 ELSE 0 END) AS ca_cnt,
            SUM(CASE WHEN schm_type='SBA' THEN COALESCE(clr_bal_amt,0) ELSE 0 END) AS sa_bal_amt,
            SUM(CASE WHEN schm_type='CAA' THEN COALESCE(clr_bal_amt,0) ELSE 0 END) AS ca_bal_amt,
            AVG(CASE WHEN schm_type='SBA' THEN mob ELSE NULL END) AS sa_mob,
            AVG(CASE WHEN schm_type='CAA' THEN mob ELSE NULL END) AS ca_mob
        FROM casa_base
        GROUP BY cust_id
    ),
    merged_casa AS (
        SELECT
            cg.*,
            car.avg_bal_12m_td,
            car.avg_bal_12m_savings,
            car.avg_bal_12m_current_ac,
            CASE
              WHEN car.avg_bal_12m_savings IS NOT NULL AND car.avg_bal_12m_savings >= 0 THEN car.avg_bal_12m_savings
              WHEN cg.sa_bal_amt >= 0 THEN cg.sa_bal_amt
              ELSE 0
            END AS sa_bal_fin,
            CASE
              WHEN car.avg_bal_12m_current_ac IS NOT NULL AND car.avg_bal_12m_current_ac >= 0 THEN car.avg_bal_12m_current_ac
              WHEN cg.ca_bal_amt >= 0 THEN cg.ca_bal_amt
              ELSE 0
            END AS ca_bal_fin,
            CASE
              WHEN car.avg_bal_12m_td IS NOT NULL AND car.avg_bal_12m_td >= 0 THEN car.avg_bal_12m_td
              ELSE 0
            END AS td_bal_fin
        FROM casa_group cg
        LEFT JOIN ans.acc_car car USING(cust_id)
    ),
    final_calc AS (
        SELECT
            ca.cust_id,
            GREATEST(COALESCE(ca.fin_chg,0), 0) AS non_interest_income,
            CAST(
                ROUND(
                    COALESCE(mc.sa_bal_fin,0) * COALESCE(mc.sa_mob,0) * 0.05 / 12.0
                  + COALESCE(mc.ca_bal_fin,0) * COALESCE(mc.ca_mob,0) * 0.08 / 12.0
                  + COALESCE(mc.td_bal_fin,0) * 0.015
                )
            AS BIGINT) AS interest_income,
            CAST(
                ROUND(
                    GREATEST(COALESCE(ca.fin_chg,0), 0)
                  + (
                        COALESCE(mc.sa_bal_fin,0) * COALESCE(mc.sa_mob,0) * 0.05 / 12.0
                      + COALESCE(mc.ca_bal_fin,0) * COALESCE(mc.ca_mob,0) * 0.08 / 12.0
                      + COALESCE(mc.td_bal_fin,0) * 0.015
                    )
                )
            AS BIGINT) AS cltv_profit
        FROM cust_agg ca
        LEFT JOIN merged_casa mc USING(cust_id)
    ),
    labeled AS (
        SELECT
            f.*,
            CASE
              WHEN cltv_profit <= 0 THEN 'Low'
              WHEN cltv_profit <= 5412 THEN 'Low'
              WHEN cltv_profit <= 19974 THEN 'Medium'
              WHEN cltv_profit <= 89370 THEN 'High'
              ELSE 'Very High'
            END AS cltv_category
        FROM final_calc f
    )
    SELECT * FROM labeled
    """

    df = con.execute(sql).fetchdf()

    con.execute("CREATE SCHEMA IF NOT EXISTS ans")
    con.execute("DROP TABLE IF EXISTS ans.acc_cltv_final_py")
    con.register("__tmp_df", df)
    con.execute("CREATE TABLE ans.acc_cltv_final_py AS SELECT * FROM __tmp_df")
    con.unregister("__tmp_df")

    if export_parquet:
        os.makedirs(os.path.dirname(export_parquet), exist_ok=True)
        con.execute(f"COPY (SELECT * FROM ans.acc_cltv_final_py) TO '{export_parquet}' (FORMAT PARQUET)")
    if export_csv:
        os.makedirs(os.path.dirname(export_csv), exist_ok=True)
        con.execute(f"COPY (SELECT * FROM ans.acc_cltv_final_py) TO '{export_csv}' (HEADER, DELIMITER ',')")

    cnt = con.execute("SELECT COUNT(*) FROM ans.acc_cltv_final_py").fetchone()[0]
    print(f"[OK] ans.acc_cltv_final_py rows: {cnt}")
    print(f"[OK] wrote: {export_parquet} and {export_csv}")

    con.close()

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to DuckDB file")
    ap.add_argument("--end-date", default="2025-09-30", help="YYYY-MM-DD")
    ap.add_argument("--export-parquet", default="data/processed/customer_features.parquet")
    ap.add_argument("--export-csv", default="data/processed/customer_features.csv")
    args = ap.parse_args()
    run(args.db, args.end_date, args.export_parquet, args.export_csv)

if __name__ == "__main__":
    cli()


