PYTHON ?= python
CONF   ?= conf/settings.yaml

# --------------------------------------------------------------------
# Config values from conf/settings.yaml
# --------------------------------------------------------------------

END_DATE := $(shell $(PYTHON) - <<'PY'
import yaml
cfg = yaml.safe_load(open("$(CONF)", "r"))
print(cfg["project"]["end_date"])
PY
)

DB_PATH := $(shell $(PYTHON) - <<'PY'
import yaml
cfg = yaml.safe_load(open("$(CONF)", "r"))
print(cfg["project"]["db_path"])
PY
)

SRC_DIR := $(shell $(PYTHON) - <<'PY'
import yaml
cfg = yaml.safe_load(open("$(CONF)", "r"))
print(cfg["project"]["src_dir"])
PY
)

CACHE_DIR := $(shell $(PYTHON) - <<'PY'
import yaml
cfg = yaml.safe_load(open("$(CONF)", "r"))
print(cfg["project"]["cache_dir"])
PY
)

CRM_CSV := $(shell $(PYTHON) - <<'PY'
import yaml
cfg = yaml.safe_load(open("$(CONF)", "r"))
print(cfg["exports"]["crm"])
PY
)

CRM_HYB := $(shell $(PYTHON) - <<'PY'
import yaml
cfg = yaml.safe_load(open("$(CONF)", "r"))
print(cfg["exports"]["crm_hybrid"])
PY
)

FPARQ := $(shell $(PYTHON) - <<'PY'
import yaml
cfg = yaml.safe_load(open("$(CONF)", "r"))
print(cfg["exports"]["features_parquet"])
PY
)

FCSV := $(shell $(PYTHON) - <<'PY'
import yaml
cfg = yaml.safe_load(open("$(CONF)", "r"))
print(cfg["exports"]["features_csv"])
PY
)

# --------------------------------------------------------------------
# Phony targets
# --------------------------------------------------------------------

.PHONY: help stage cltv score train_rf train_hgb export_crm export_hybrid qc all pipeline

# --------------------------------------------------------------------
# Help
# --------------------------------------------------------------------

help:
	@echo "Local DuckDB CLTV + lead scoring pipeline"
	@echo
	@echo "make stage          - Run local ETL into DuckDB (uses src/etl/*)"
	@echo "make cltv           - Compute CLTV features into DuckDB (cltv_pushdown)"
	@echo "make score          - Build CLTV/lead scores (0–100, deciles)"
	@echo "make train_rf       - Train RF model (legacy/experimental)"
	@echo "make train_hgb      - Train main HGB model"
	@echo "make export_crm     - Export rules-based lead scores to CSV"
	@echo "make export_hybrid  - Export hybrid (rules + ML) scores to CSV"
	@echo "make qc             - Basic QC summary JSON under reports/"
	@echo "make all / pipeline - Full local pipeline: stage → cltv → score → train_hgb → export_hybrid → qc"

# --------------------------------------------------------------------
# Local ETL / CLTV (DuckDB-based pipeline)
# --------------------------------------------------------------------

# NOTE: This is the legacy/local ETL path into DuckDB, independent of the
# Docker/Postgres/dbt stack described in the README. Keep src/etl/ETL.py
# or update this command if your ETL entrypoint has been renamed.

stage:
	$(PYTHON) src/etl/ETL.py \
		--src $(SRC_DIR) \
		--db $(DB_PATH) \
		--cache $(CACHE_DIR) \
		--threads 8 \
		--memory "2GB" \
		--strict \
		--indexes

cltv:
	$(PYTHON) src/etl/cltv_pushdown.py \
		--db $(DB_PATH) \
		--end-date $(END_DATE) \
		--export-parquet $(FPARQ) \
		--export-csv $(FCSV)

# --------------------------------------------------------------------
# Scoring (CLTV → lead scores in DuckDB)
# --------------------------------------------------------------------

score:
	$(PYTHON) - <<'PY'
import duckdb, os
db_path = "$(DB_PATH)"
con = duckdb.connect(db_path)

# Range of CLTV for normalization
minv, maxv = con.execute("""
    SELECT MIN(cltv_profit), MAX(cltv_profit)
    FROM ans.acc_cltv_final_py
""").fetchone()

# Quantiles (currently not used in labels, but kept for reference/plots)
q20, q40, q70, q90 = con.execute("""
    SELECT
      quantile(cltv_profit, 0.2),
      quantile(cltv_profit, 0.4),
      quantile(cltv_profit, 0.7),
      quantile(cltv_profit, 0.9)
    FROM ans.acc_cltv_final_py
""").fetchone()

# Hard-coded thresholds for now; move to settings.yaml if you want fully config-driven bins.
t1, t2, t3, t4 = 200, 700, 2000, 3550

con.execute("CREATE SCHEMA IF NOT EXISTS ans")
con.execute("DROP TABLE IF EXISTS ans.acc_cltv_final_scored")

con.execute(f"""
CREATE TABLE ans.acc_cltv_final_scored AS
SELECT
  *,
  CASE
    WHEN cltv_profit <= {t1} THEN 'Very Low'
    WHEN cltv_profit <= {t2} THEN 'Low'
    WHEN cltv_profit <= {t3} THEN 'Medium'
    WHEN cltv_profit <= {t4} THEN 'High'
    ELSE 'Very High'
  END AS cltv_category_q,
  CAST(ROUND(
    CASE
      WHEN {maxv} - {minv} = 0 THEN 0
      ELSE 100.0 * (cltv_profit - {minv}) / ({maxv} - {minv})
    END
  ) AS INTEGER) AS cltv_score_0_100
FROM ans.acc_cltv_final_py
""")

# Lead table with deciles + recommended actions
con.execute("DROP TABLE IF EXISTS ans.customer_lead_scores")

con.execute("""
CREATE TABLE ans.customer_lead_scores AS
WITH base AS (
  SELECT
    cust_id,
    cltv_profit,
    cltv_category_q,
    cltv_score_0_100,
    NTILE(10) OVER (ORDER BY cltv_profit DESC) AS cltv_decile
  FROM ans.acc_cltv_final_scored
)
SELECT
  cust_id,
  cltv_profit,
  cltv_category_q,
  cltv_score_0_100 AS lead_score_0_100,
  cltv_decile,
  CASE
    WHEN cltv_decile = 1 THEN 'Call + RM follow-up'
    WHEN cltv_decile <= 3 THEN 'Call within 48h'
    WHEN cltv_decile <= 6 THEN 'Nurture email/SMS'
    ELSE 'Low-touch cohort'
  END AS recommended_action,
  CASE WHEN cltv_decile <= 2 THEN 1 ELSE 0 END AS priority_flag
FROM base
""")

con.close()
print("Built ans.acc_cltv_final_scored and ans.customer_lead_scores in", db_path)
PY

# --------------------------------------------------------------------
# ML training (local)
# --------------------------------------------------------------------

train_rf:
	$(PYTHON) src/ml/train.py

# Main calibrated model (Histogram Gradient Boosting or similar)
train_hgb:
	$(PYTHON) src/ml/train_hgb.py

# --------------------------------------------------------------------
# Exports (CRM CSVs)
# --------------------------------------------------------------------

export_crm:
	$(PYTHON) - <<'PY'
import os, duckdb
db_path = "$(DB_PATH)"
out_csv = "$(CRM_CSV)"

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
con = duckdb.connect(db_path)

con.execute(f"""
COPY (
  SELECT
    cust_id,
    lead_score_0_100,
    cltv_decile,
    cltv_category_q,
    recommended_action,
    priority_flag
  FROM ans.customer_lead_scores
  ORDER BY lead_score_0_100 DESC, cust_id
) TO '{out_csv}' (HEADER, DELIMITER ',')
""")

con.close()
print("Exported CRM lead scores ->", out_csv)
PY

export_hybrid:
	$(PYTHON) - <<'PY'
import os, yaml, duckdb

cfg = yaml.safe_load(open("$(CONF)", "r"))
rw = cfg["scoring"]["hybrid_weights"]["rules_pct"]
mw = cfg["scoring"]["hybrid_weights"]["ml_pct"]

db_path = "$(DB_PATH)"
out_csv = "$(CRM_HYB)"

con = duckdb.connect(db_path)

con.execute(f"""
CREATE OR REPLACE TABLE ans.crm_export AS
SELECT
  l.cust_id,
  l.lead_score_0_100,
  l.cltv_decile,
  l.cltv_category_q,
  l.recommended_action,
  l.priority_flag,
  s.ml_high_cltv_proba,
  COALESCE(c.ml_high_cltv_proba_cal, s.ml_high_cltv_proba) AS ml_proba_cal,
  ROUND(
    {rw} * l.lead_score_0_100 +
    {mw} * (COALESCE(c.ml_high_cltv_proba_cal, s.ml_high_cltv_proba) * 100)
  ) AS hybrid_score_0_100
FROM ans.customer_lead_scores l
LEFT JOIN ans.customer_ml_scores     s USING (cust_id)
LEFT JOIN ans.customer_ml_scores_cal c USING (cust_id)
""")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)

con.execute(f"""
COPY (
  SELECT *
  FROM ans.crm_export
  ORDER BY hybrid_score_0_100 DESC, lead_score_0_100 DESC
) TO '{out_csv}' (HEADER, DELIMITER ',')
""")

con.close()
print("Exported hybrid CRM lead scores ->", out_csv)
PY

# --------------------------------------------------------------------
# QC
# --------------------------------------------------------------------

qc:
	$(PYTHON) - <<'PY'
import os, json, datetime, duckdb

db_path = "$(DB_PATH)"
os.makedirs("reports", exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out   = f"reports/qc_summary_{stamp}.json"

con = duckdb.connect(db_path)

def table_exists(qualified_name: str) -> bool:
    schema, table = qualified_name.split(".")
    return bool(con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?",
        [schema, table]
    ).fetchone())

tables_to_check = [
    "schema.general_acct_mast_table",
    "schema.htd",
    "schema.chrg_tran_log_tbl",
    "ans.acc_car",
    "ans.acc_cltv_final_py",
    "ans.customer_lead_scores",
    "ans.crm_export",
]

summary = {}
summary["row_counts"] = {}

for t in tables_to_check:
    if table_exists(t):
        count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        summary["row_counts"][t] = count

if table_exists("ans.acc_cltv_final_scored"):
    rows = con.execute("""
        SELECT cltv_category_q, COUNT(*) AS n
        FROM ans.acc_cltv_final_scored
        GROUP BY 1
        ORDER BY n DESC
    """).fetchall()
    summary["lead_bucket_counts"] = [
        {"cltv_category_q": r[0], "n": r[1]} for r in rows
    ]

with open(out, "w") as f:
    json.dump(summary, f, indent=2)

con.close()
print("QC summary ->", out)
PY

# --------------------------------------------------------------------
# Full local pipeline
# --------------------------------------------------------------------

pipeline: all

all: stage cltv score train_hgb export_hybrid qc
