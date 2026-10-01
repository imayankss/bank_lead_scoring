````markdown
# Bank Lead Scoring, CLTV & Next-Best-Product Platform

Local / containerized platform for:

- ELT / ETL on synthetic retail banking data
- Customer 360 + Customer Lifetime Value (CLTV) marts
- ML-based lead scoring (unified XGBoost model)
- Rule- and ML-driven Next-Best-Product (NBP1 / NBP2 / NBP3)
- FastAPI scoring API
- Streamlit CRM dashboard (customer 360, CLTV, lead scores, NBP)

---

## 1. Stack at a Glance

**Services**

- Postgres 16 (data warehouse)
- Airflow 2.10 (LocalExecutor)
- dbt (Postgres)
- FastAPI (ML scoring API)
- Streamlit (dashboard)
- pgAdmin (DB UI)

**Ports**

| Service   | URL                        |
| --------- | -------------------------- |
| pgAdmin   | http://localhost:5050      |
| Airflow   | http://localhost:8080      |
| API       | http://localhost:8000/docs |
| Dashboard | http://localhost:8501      |
| Postgres  | `localhost:5432`           |

For quick local dev (without Docker), the code also supports:

- `python -m src.etl.…` for transforms
- `python -m src.ml.…` for model scoring
- `streamlit run src/dashboard/app.py` for the dashboard

---

## 2. Project Structure (Key Paths)

```text
docker/
  docker-compose.yml          # stack definition
  .env                        # copy of root .env (for compose)

airflow/
  dags/
    lead_scoring_pipeline_dag.py   # ELT + dbt + scoring DAG

dbt/
  dbt_project.yml             # dbt project
  profiles.yml                # Postgres target
  models/                     # staging / marts / CLTV / lead scoring
  seeds/                      # optional dims / mappings

src/
  api/
    main.py                   # FastAPI entrypoint
  dashboard/
    app.py                    # Streamlit dashboard entry
    data_access.py            # DuckDB / Postgres reads for dashboard
  etl/
    __init__.py
    transforms/
      add_age_rule_features.py    # attach age-rule NBP families to leads
      rfm.py                      # CLTV / RFM utilities
  ml/
    recommendation.py         # generic NBP scoring / ranking logic
    nbp_age_rules.py          # age-band → NBP families mapping helper
    score_unified_leads_nbp.py    # build NBP1–3 per customer from model + rules

etl/
  __init__.py
  transforms/
    rfm.py                    # (older ETL helpers)

data/
  raw/
    dim_product_boi_2025.csv      # product dimension (id, name, family, risk,…)
    dim_nbp_age_rules.csv         # age-band → product families for NBP1–3
    ...                           # synthetic customer / account / txn CSVs
  processed/
    modeling_dataset_customers_universe.csv
    modeling_dataset_leads_unified.csv
    modeling_dataset_leads_unified_age_rules.csv  # leads + age-rule NBP families
    unified_lead_scores.csv       # per-lead unified ML probabilities
    unified_nbp_customers.csv     # per-customer NBP1–3 for dashboard
  warehouse/
    cltv.duckdb                   # DuckDB marts (`mart_customer_360_*`, etc.)
  outputs/
    ...                           # exports, evaluation CSVs

models/                           # ML training scripts + artifacts
notebooks/                        # EDA / prototype CLTV, scoring ideas
reports/                          # notebooks / HTML / reports
tests/                            # unit / integration / data-quality tests
````

### Legacy / Experimental Code

Kept for reference, **not** part of the main v1 pipeline:

* `arch/`, `legacy/`, `archive/` (e.g. `arch/tests/data_quality/`, `arch/tests/unit/`)
* Early notebooks under `notebooks/` (EDA, prototype CLTV formulas)
* Older scripts at the repo root or under `src/`:

  * `*_draft.py`, `*_experiment.py`
  * early `cltv_pipeline_*.py`, `expand_dataset_*.py`, etc.

For day-to-day development, prefer:

* Airflow DAGs in `airflow/dags/`
* dbt models in `dbt/models/`
* ETL modules under `src/etl/transforms`
* ML utilities under `src/ml`
* API and dashboard under `src/api` and `src/dashboard`

---

## 3. Prerequisites

* macOS (Apple Silicon or Intel) or Linux
* Docker Desktop (or Colima) with Docker Compose v2
* `python3` + `pip` (for local runs / development)
* `curl` and `jq` (optional, for health checks)

---

## 4. Environment Setup

Create `.env` at the **repo root**:

```ini
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=bank_dw

PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=adminadmin

AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin
AIRFLOW_EMAIL=admin@example.com
AIRFLOW_FIRSTNAME=Admin
AIRFLOW_LASTNAME=User
AIRFLOW_UID=50000

# keys: generate once
AIRFLOW_FERNET_KEY=REPLACE_ME
AIRFLOW_SECRET_KEY=REPLACE_ME
```

Generate keys:

```bash
python3 - <<'PY'
import os, base64, secrets
print("AIRFLOW_FERNET_KEY="+base64.urlsafe_b64encode(os.urandom(32)).decode())
print("AIRFLOW_SECRET_KEY="+secrets.token_urlsafe(32))
PY
```

Copy `.env` next to the compose file:

```bash
cp -f .env docker/.env
```

If `docker/docker-compose.yml` still has a top-level `version:` key, remove it.

---

## 5. Start the Stack (Docker)

```bash
# Postgres first
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d postgres

# wait for health
until docker ps --filter name=bls_postgres --format '{{.Status}}' | grep -qi healthy; do
  sleep 1
done

# init Airflow + pgAdmin
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d pgadmin airflow-init

# main services
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d \
  airflow-webserver airflow-scheduler dbt api dashboard
```

Quick health checks:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
curl -sSf http://localhost:8080/health >/dev/null
curl -sSf http://localhost:8000/docs  >/dev/null
curl -sI  http://localhost:8501 | head -n1
```

---

## 6. Data Pipeline & dbt

**Data layout**

* Place raw CSVs (customer, account, transactions, products, etc.) in `data/raw/`.

**Run dbt (inside dbt container)**

```bash
docker compose -f docker/docker-compose.yml exec dbt bash -lc '
  dbt --version && dbt debug && dbt deps && dbt run && dbt test
'
```

Requirements:

* `DBT_PROFILES_DIR=/usr/app` inside the `dbt` container
* `profile:` in `dbt/dbt_project.yml` matches a profile key in `dbt/profiles.yml`

Airflow DAG `lead_scoring_pipeline_dag.py` orchestrates:

* Load from `data/raw/` → Postgres staging
* dbt transforms → CLTV / feature marts
* Optional ML / scoring refresh jobs

Trigger via the Airflow UI.

---

## 7. ML, NBP & Dashboard

### 7.1 Lead-scoring model

* Unified binary classifier (e.g. XGBoost) trained on lead-level features.
* Outputs per-lead probability `ml_unified_proba`, stored in:

  ```text
  data/processed/unified_lead_scores.csv
  ```

### 7.2 Age-rule NBP families

Rules for age bands → product families live in:

* `data/raw/dim_nbp_age_rules.csv` (e.g. 18–23 = Education Loan first, 30–38 = Home Loan, 45+ = Insurance, etc.)

They are applied to the leads modeling dataset via:

```bash
python -m src.etl.transforms.add_age_rule_features \
  --input data/processed/modeling_dataset_leads_unified.csv \
  --output data/processed/modeling_dataset_leads_unified_age_rules.csv \
  --age-col age
```

This adds:

* `age_rule_nbp1_family`
* `age_rule_nbp2_family`
* `age_rule_nbp3_family`

### 7.3 NBP1–3 per customer

NBP computation is implemented in:

* `src/ml/recommendation.py` – generic ranking + eligibility helpers
* `src/ml/nbp_age_rules.py` – age-rule lookup utilities
* `src/ml/score_unified_leads_nbp.py` – glue script that produces per-customer NBPs

To build NBP1–3 for all **lead customers**:

```bash
python -m src.ml.score_unified_leads_nbp
```

This script:

1. Loads `modeling_dataset_leads_unified_age_rules.csv`.
2. Joins `unified_lead_scores.csv` to get `ml_unified_proba` per lead.
3. Joins `dim_product_boi_2025.csv` for product metadata.
4. Uses `add_nbp_recommendations(...)` to rank products per customer.
5. Merges in age-rule families per customer.
6. Writes:

   ```text
   data/processed/unified_nbp_customers.csv
   ```

`unified_nbp_customers.csv` includes, among others:

* `cust_id`, `Customer_ID`
* `nbp1`, `nbp1_score` (product ID + score from the ML model, where available)
* `age_rule_nbp1_family`, `age_rule_nbp2_family`, `age_rule_nbp3_family`

### 7.4 Dashboard integration (NBP1 / NBP2 / NBP3)

The Streamlit app:

* Loads the main customer 360 dashboard table via `load_customer_360_dashboard()`.

* Merges in NBP suggestions from `unified_nbp_customers.csv` inside `load_dashboard_data()` by mapping:

  ```text
  age_rule_nbp1_family → nbp1_name
  age_rule_nbp2_family → nbp2_name
  age_rule_nbp3_family → nbp3_name
  ```

* Exposes the three “Next Best Product” columns in the **All leads** grid:

  * `NBP1` – top suggested product family (e.g. “Education Loan”, “Home Loan”)
  * `NBP2` – second priority (e.g. “Credit Card”, “Vehicle Loan”)
  * `NBP3` – third priority (e.g. “Deposit”, “Gold Loan”)

This gives the RM a simple, age-aware, rule-driven NBP view that can later be upgraded to a full hybrid rules + ML recommender using the same architecture (`src/ml/recommendation.py` + `unified_nbp_customers.csv` + dashboard merge).

### 7.5 Local dashboard run

For local development without Docker:

```bash
# create & activate venv, install requirements (once)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# make sure NBP file exists
python -m src.ml.score_unified_leads_nbp

# run dashboard
streamlit run src/dashboard/app.py
```

---

## 8. API (FastAPI)

* Entry: `src/api/main.py`
* Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Loads the latest trained model artifact and exposes scoring endpoints (lead-level features in, probabilities and NBP fields out – depending on implementation).

Example container command:

```bash
docker compose -f docker/docker-compose.yml exec api bash -lc '
  export PYTHONPATH=/app:/app/src
  python -m models.train_lead_scoring
'
```

---

## 9. Tests

Run tests with a correct `PYTHONPATH`:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e PYTHONPATH=/app:/app/src \
  api bash -lc '
    [ -f requirements-dev.txt ] && pip install -q -r requirements-dev.txt || pip install -q pytest==9.0.0;
    pytest -vv
  '
```

If you see `ModuleNotFoundError: etl` or `src.*`:

* Ensure `etl/__init__.py` and `src/__init__.py` exist.
* Set `PYTHONPATH=/app:/app/src` for the relevant service.

---

## 10. Common Issues

* **Docker not running**
  Start Docker Desktop (or Colima) before any compose commands.

* **Environment not interpolated**
  Keep `docker/.env` in sync with root `.env` or always pass `--env-file docker/.env`.

* **Postgres unhealthy**
  Usually missing/empty `POSTGRES_*` env vars. Fix `.env`, copy to `docker/.env`, then:

  ```bash
  docker compose --env-file docker/.env -f docker/docker-compose.yml down -v
  docker compose --env-file docker/.env -f docker/docker-compose.yml up -d postgres
  ```

* **API down / `uvicorn: command not found`**
  Ensure `fastapi` and `uvicorn` are in `requirements.txt`, then:

  ```bash
  docker compose --env-file docker/.env -f docker/docker-compose.yml up -d --force-recreate api
  ```

* **NBP columns blank in dashboard**
  Check:

  1. `data/processed/unified_nbp_customers.csv` exists.
  2. `python -m src.ml.score_unified_leads_nbp` has been run after any dataset or model changes.
  3. `load_dashboard_data()` is merging `nbp1_name/nbp2_name/nbp3_name` from the age-rule family columns.

---

## 11. Stop & Clean Up

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml down -v
```

Do **not** commit `.env` or any secrets to version control.

```
```



