Here is a more minimal, clean, and professional version you can use as your `README.md`:

````markdown
# Bank Lead Scoring & CLTV Platform

Local, containerized platform for:

- ELT / ETL on synthetic banking data
- Customer Lifetime Value (CLTV) feature marts
- ML-based lead scoring
- FastAPI scoring API
- Streamlit customer 360 / lead scoring dashboard

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

| Service   | URL                                                      |
| --------- | -------------------------------------------------------- |
| pgAdmin   | http://localhost:5050                                    |
| Airflow   | http://localhost:8080                                    |
| API       | http://localhost:8000/docs                               |
| Dashboard | http://localhost:8501                                    |
| Postgres  | `localhost:5432`                                         |

---

## 2. Project Structure (Key Paths)

```text
docker/
  docker-compose.yml        # stack definition
  .env                      # copy of root .env (for compose)

airflow/
  dags/
    lead_scoring_pipeline_dag.py   # ELT pipeline DAG

dbt/
  dbt_project.yml           # dbt project
  profiles.yml              # Postgres target
  models/                   # staging / marts / CLTV / lead scoring
  seeds/                    # optional dims / mappings

src/
  api/
    main.py                 # FastAPI entrypoint
  dashboard/
    app.py                  # Streamlit entry
  etl/ (if present)
    __init__.py
    extract.py              # raw → staging
    transform.py            # extra Python transforms
    load.py                 # load into Postgres

etl/
  __init__.py
  transforms/
    rfm.py                  # CLTV / RFM utilities

data/
  raw/                      # raw synthetic CSVs
  intermediate/             # pre-processed / expanded data
  warehouse/                # e.g. DuckDB snapshots, exports
  outputs/                  # scores, evaluation CSVs

models/                     # ML training scripts + artifacts
reports/                    # notebooks / HTML / reports
tests/                      # unit / integration / data-quality tests
````

### Legacy / Experimental Code

The following are kept for reference and are **not** part of the primary v1 pipeline:

* `arch/`, `legacy/`, `archive/` (e.g. `arch/tests/data_quality/`, `arch/tests/unit/`)
* Early notebooks under `notebooks/` (EDA, prototype CLTV formulas)
* Older scripts at the repo root or under `src/`:

  * `*_draft.py`, `*_experiment.py`
  * early `cltv_pipeline_*.py`, `expand_dataset_*.py`, etc.

For day-to-day development, prefer:

* DAGs in `airflow/dags/`
* dbt models in `dbt/models/`
* ETL modules under `src/etl` / `etl`
* API and dashboard under `src/api` and `src/dashboard`

---

## 3. Prerequisites

* macOS (Apple Silicon or Intel) or Linux
* Docker Desktop (or Colima) with Docker Compose v2
* `python3` (for generating keys / quick checks)
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

## 5. Start the Stack

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

**Run dbt**

```bash
docker compose -f docker/docker-compose.yml exec dbt bash -lc '
  dbt --version && dbt debug && dbt deps && dbt run && dbt test
'
```

Requirements:

* `DBT_PROFILES_DIR=/usr/app` inside the `dbt` container
* `profile:` in `dbt/dbt_project.yml` matches a profile key in `dbt/profiles.yml`

Airflow DAG `lead_scoring_pipeline_dag.py` (in `airflow/dags/`) orchestrates:

* load from `data/raw/` → Postgres staging
* dbt transforms → CLTV / feature marts
* optional ML / scoring refresh jobs

Trigger via the Airflow UI.

---

## 7. ML, API & Dashboard

**ML**

* ML scripts and models live under `models/` (and/or `src/`).
* Typical pattern (inside the `api` container):

```bash
docker compose -f docker/docker-compose.yml exec api bash -lc '
  export PYTHONPATH=/app:/app/src
  python -m models.train_lead_scoring
'
```

**API (FastAPI)**

* Entry: `src/api/main.py`
* Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Loads the latest model artifact and exposes scoring endpoints (for lead/CLTV inputs).

**Dashboard (Streamlit)**

* Entry: `src/dashboard/app.py`
* UI: [http://localhost:8501](http://localhost:8501)
* Focus: customer 360 view, CLTV, lead scores, and basic segmentation.

For development:

```bash
docker logs -f bls_api
docker logs -f bls_dashboard

docker compose --env-file docker/.env -f docker/docker-compose.yml up -d \
  --force-recreate api dashboard
```

Runtime essentials:

```txt
# api/requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.2
pyyaml==6.0.2

# dashboard/requirements.txt
streamlit==1.39.0
pandas==2.2.2
pyarrow==17.0.0
```

---

## 8. Tests

Run tests with a correct `PYTHONPATH`:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e PYTHONPATH=/app:/app/src \
  api bash -lc '
    [ -f requirements-dev.txt ] && pip install -q -r requirements-dev.txt || pip install -q pytest==9.0.0;
    pytest -vv
  '
```

If you see `ModuleNotFoundError: etl`, ensure:

```text
etl/__init__.py
etl/transforms/rfm.py
```

exist, or add:

```yaml
environment:
  - PYTHONPATH=/app:/app/src
```

to `api:` and `dashboard:` services.

---

## 9. Common Issues

* **Docker not running**
  Start Docker Desktop (or Colima) before any commands.

* **Environment not interpolated**
  Keep `docker/.env` in sync with root `.env` or pass `--env-file docker/.env` explicitly.

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

---

## 10. Stop & Clean Up

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml down -v
```

Do **not** commit `.env` or any secrets to version control.

```
```


