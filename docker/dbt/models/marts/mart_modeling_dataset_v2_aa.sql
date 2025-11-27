{{ config(materialized='table') }}

WITH mdl AS (
    SELECT *
    FROM {{ source('processed', 'modeling_dataset_v2') }}
),

aa_map AS (
    SELECT
        aa_customer_id,
        cust_id
    FROM {{ source('raw', 'aa_customer_map') }}
    WHERE bank_code = 'ANCHOR'
),

c360 AS (
    SELECT *
    FROM {{ source('warehouse', 'mart_customer_360_aa') }}
)

SELECT
    m.*,
    c360.num_accounts,
    c360.num_bank_relationships,
    c360.num_anchor_accounts,
    c360.num_comp_accounts,
    c360.total_balance_all_banks,
    c360.anchor_balance,
    c360.competitor_balance,
    c360.total_credit_limit,
    c360.txn_count_total,
    c360.total_inflows,
    c360.total_outflows,
    c360.avg_txn_amount,
    c360.txns_last_90d,
    c360.digital_txn_count,
    c360.digital_usage_ratio,
    c360.avg_monthly_inflows,
    c360.avg_monthly_outflows
FROM mdl AS m
LEFT JOIN aa_map AS map
    ON m.cust_id = map.cust_id
LEFT JOIN c360
    ON map.aa_customer_id = c360.aa_customer_id;
