PYTHON ?= python
CONF   := conf/settings.yaml

# Pull settings from YAML
END_DATE := $(shell $(PYTHON) - <<'PY'\nimport yaml;print(yaml.safe_load(open("conf/settings.yaml"))["project"]["end_date"])\nPY)
DB_PATH  := $(shell $(PYTHON) - <<'PY'\nimport yaml;print(yaml.safe_load(open("conf/settings.yaml"))["project"]["db_path"])\nPY)
SRC_DIR  := $(shell $(PYTHON) - <<'PY'\nimport yaml;print(yaml.safe_load(open("conf/settings.yaml"))["project"]["src_dir"])\nPY)
CACHE_DIR:= $(shell $(PYTHON) - <<'PY'\nimport yaml;print(yaml.safe_load(open("conf/settings.yaml"))["project"]["cache_dir"])\nPY)
CRM_CSV  := $(shell $(PYTHON) - <<'PY'\nimport yaml;print(yaml.safe_load(open("conf/settings.yaml"))["exports"]["crm"])\nPY)
CRM_HYB  := $(shell $(PYTHON) - <<'PY'\nimport yaml;print(yaml.safe_load(open("conf/settings.yaml"))["exports"]["crm_hybrid"])\nPY)
FPARQ    := $(shell $(PYTHON) - <<'PY'\nimport yaml;print(yaml.safe_load(open("conf/settings.yaml"))["exports"]["features_parquet"])\nPY)
FCSV     := $(shell $(PYTHON) - <<'PY'\nimport yaml;print(yaml.safe_load(open("conf/settings.yaml"))["exports"]["features_csv"])\nPY)

.PHONY: stage cltv score train_rf train_hgb export_crm export_hybrid qc all

stage:
	$(PYTHON) src/etl/ETL.py --src $(SRC_DIR) --db $(DB_PATH) --cache $(CACHE_DIR) --threads 8 --memory "2GB" --strict --indexes

cltv:
	$(PYTHON) src/etl/cltv_pushdown.py --db $(DB_PATH) --end-date $(END_DATE) --export-parquet $(FPARQ) --export-csv $(FCSV)

# Build quantile bins + 0–100 score table (idempotent)
score:
	$(PYTHON) - <<'PY'
import os, duckdb
con=duckdb.connect("$(DB_PATH)")
minv,maxv = con.execute("SELECT MIN(cltv_profit), MAX(cltv_profit) FROM ans.acc_cltv_final_py").fetchone()
q20,q40,q70,q90 = con.execute("SELECT quantile(cltv_profit,0.2), quantile(cltv_profit,0.4), quantile(cltv_profit,0.7), quantile(cltv_profit,0.9) FROM ans.acc_cltv_final_py").fetchone()
t1,t2,t3,t4 = 200,700,2000,3550  # from config
con.execute("CREATE SCHEMA IF NOT EXISTS ans")
con.execute("DROP TABLE IF EXISTS ans.acc_cltv_final_scored")
con.execute(f"""
CREATE TABLE ans.acc_cltv_final_scored AS
SELECT *,
  CASE WHEN cltv_profit <= {t1} THEN 'Very Low'
       WHEN cltv_profit <= {t2} THEN 'Low'
       WHEN cltv_profit <= {t3} THEN 'Medium'
       WHEN cltv_profit <= {t4} THEN 'High'
       ELSE 'Very High' END AS cltv_category_q,
  CAST(ROUND(CASE WHEN {maxv}-{minv}=0 THEN 0 ELSE 100.0*(cltv_profit-{minv})/({maxv}-{minv}) END)) AS INTEGER AS cltv_score_0_100
FROM ans.acc_cltv_final_py
""")
# Build deciles + lead table
con.execute("DROP TABLE IF EXISTS ans.customer_lead_scores")
con.execute("""
CREATE TABLE ans.customer_lead_scores AS
WITH base AS (
  SELECT cust_id, cltv_profit, cltv_category_q, cltv_score_0_100,
         NTILE(10) OVER (ORDER BY cltv_profit DESC) AS cltv_decile
  FROM ans.acc_cltv_final_scored)
SELECT cust_id, cltv_profit, cltv_category_q, cltv_score_0_100 AS lead_score_0_100,
       cltv_decile,
       CASE WHEN cltv_decile=1 THEN 'Call + RM follow-up'
            WHEN cltv_decile<=3 THEN 'Call within 48h'
            WHEN cltv_decile<=6 THEN 'Nurture email/SMS'
            ELSE 'Low-touch cohort' END AS recommended_action,
       CASE WHEN cltv_decile<=2 THEN 1 ELSE 0 END AS priority_flag
FROM base
""")
con.close()
PY

train_rf:
	$(PYTHON) src/ml/train.py

# Faster, stronger calibrated model (HGB)
train_hgb:
	$(PYTHON) src/ml/train_hgb.py

export_crm:
	$(PYTHON) - <<'PY'
import os, duckdb
os.makedirs("data/exports", exist_ok=True)
con=duckdb.connect("$(DB_PATH)")
con.execute("""
COPY (
  SELECT cust_id, lead_score_0_100, cltv_decile, cltv_category_q, recommended_action, priority_flag
  FROM ans.customer_lead_scores
  ORDER BY lead_score_0_100 DESC, cust_id
) TO '$(CRM_CSV)' (HEADER, DELIMITER ',')
""")
con.close()
print("Exported -> $(CRM_CSV)")
PY

export_hybrid:
	$(PYTHON) - <<'PY'
import os, yaml, duckdb
w=yaml.safe_load(open("conf/settings.yaml"))
rw=w["scoring"]["hybrid_weights"]["rules_pct"]; mw=w["scoring"]["hybrid_weights"]["ml_pct"]
con=duckdb.connect("$(DB_PATH)")
con.execute("""
CREATE OR REPLACE TABLE ans.crm_export AS
SELECT
  l.cust_id, l.lead_score_0_100, l.cltv_decile, l.cltv_category_q, l.recommended_action, l.priority_flag,
  s.ml_high_cltv_proba, COALESCE(c.ml_high_cltv_proba_cal, s.ml_high_cltv_proba) AS ml_proba_cal,
  ROUND({rw}*l.lead_score_0_100 + {mw}*(COALESCE(c.ml_high_cltv_proba_cal, s.ml_high_cltv_proba)*100)) AS hybrid_score_0_100
FROM ans.customer_lead_scores l
LEFT JOIN ans.customer_ml_scores s USING(cust_id)
LEFT JOIN ans.customer_ml_scores_cal c USING(cust_id)
""".format(rw=rw, mw=mw))
os.makedirs("data/exports", exist_ok=True)
con.execute("COPY (SELECT * FROM ans.crm_export ORDER BY hybrid_score_0_100 DESC, lead_score_0_100 DESC) TO '$(CRM_HYB)' (HEADER, DELIMITER ',')")
print("Exported -> $(CRM_HYB)")
con.close()
PY

qc:
	$(PYTHON) - <<'PY'
import os, json, datetime, duckdb
db="$(DB_PATH)"; os.makedirs("reports", exist_ok=True)
stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); out=f"reports/qc_summary_{stamp}.json"
con=duckdb.connect(db)
def one(q): return con.execute(q).fetchone()
summary={}
summary["row_counts"]={t:one(f"SELECT COUNT(*) FROM {t}")[0] for t in [
  "schema.general_acct_mast_table","schema.htd","schema.chrg_tran_log_tbl",
  "ans.acc_car","ans.acc_cltv_final_py","ans.customer_lead_scores","ans.crm_export"
] if con.execute(f"SELECT 1 FROM information_schema.tables WHERE table_schema||'.'||table_name='{t}'").fetchone()}
summary["lead_bucket_counts"]=[dict(zip(["cltv_category_q","n"],r)) for r in con.execute("SELECT cltv_category_q,COUNT(*) FROM ans.acc_cltv_final_scored GROUP BY 1 ORDER BY 2 DESC").fetchall()]
with open(out,"w") as f: json.dump(summary,f,indent=2)
print("QC ->", out); con.close()
PY

all: stage cltv score train_hgb export_hybrid qc
