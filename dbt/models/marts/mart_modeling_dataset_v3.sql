{{ config(
    materialized='table',
    schema='processed',
    alias='modeling_dataset_v3'
) }}

WITH core AS (
    SELECT *
    FROM {{ source('processed', 'modeling_dataset_v2') }}
),

aa_map AS (
    SELECT
        cust_id,
        aa_customer_id
    FROM {{ source('raw', 'aa_customer_map') }}
    WHERE bank_code = 'ANCHOR'
),

aa_c360 AS (
    SELECT *
    FROM {{ source('warehouse', 'mart_customer_360_aa') }}
)

SELECT
    core.*,

    -- AA features – adapt to your actual columns
    aa_c360.num_accounts,
    aa_c360.num_bank_relationships,
    aa_c360.num_anchor_accounts,
    aa_c360.num_comp_accounts,
    aa_c360.total_balance_all_banks,
    aa_c360.anchor_balance,
    aa_c360.competitor_balance,
    aa_c360.total_credit_limit,
    aa_c360.txn_count_total,
    aa_c360.total_inflows,
    aa_c360.total_outflows,
    aa_c360.avg_txn_amount,
    aa_c360.txns_last_90d,
    aa_c360.digital_txn_count,
    aa_c360.digital_usage_ratio,
    aa_c360.avg_monthly_inflows,
    aa_c360.avg_monthly_outflows
FROM core
LEFT JOIN aa_map
    ON core.cust_id = aa_map.cust_id
LEFT JOIN aa_c360
    ON aa_map.aa_customer_id = aa_c360.aa_customer_id;

