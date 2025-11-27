
  create view "bank_dw"."public"."mart_customers_universe__dbt_tmp"
    
    
  as (
    

-- Simple passthrough view over the Postgres table
select
    *
from public.mart_customers_universe
  );