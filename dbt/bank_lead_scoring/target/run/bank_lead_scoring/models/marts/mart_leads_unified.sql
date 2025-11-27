
  create view "bank_dw"."public"."mart_leads_unified__dbt_tmp"
    
    
  as (
    

-- Simple passthrough view over the Postgres table
select
    *
from public.mart_leads_unified
  );