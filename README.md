# Bank Lead Scoring — Developer Quick-Start

Containerized stack for ELT/ML experimentation:

* Postgres 16
* Airflow 2.10 (LocalExecutor)
* dbt (Postgres)
* FastAPI API
* Streamlit dashboard
* pgAdmin

## Ports

| Service   | URL                                                      |
| --------- | -------------------------------------------------------- |
| pgAdmin   | [http://localhost:5050](http://localhost:5050)           |
| Airflow   | [http://localhost:8080](http://localhost:8080)           |
| API       | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Dashboard | [http://localhost:8501](http://localhost:8501)           |
| Postgres  | localhost:5432                                           |

## Repo layout (key paths)

```
airflow/dags/                 # your DAGs
dbt/                          # dbt project and profiles.yml
src/api/main.py               # FastAPI entry
src/dashboard/app.py          # Streamlit app
docker/docker-compose.yml     # compose file
data/                         # datasets and outputs
models/, reports/, tests/     # ML artifacts and tests
```

## Prerequisites

* macOS with Apple Silicon or Intel
* Docker Desktop (or Colima) with Docker Compose v2
* `curl` and `jq` optional for checks

## 1) Environment setup

Create `.env` at repo root:

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

Generate keys with stdlib:

```bash
python3 - <<'PY'
import os, base64, secrets
print("AIRFLOW_FERNET_KEY="+base64.urlsafe_b64encode(os.urandom(32)).decode())
print("AIRFLOW_SECRET_KEY="+secrets.token_urlsafe(32))
PY
```

Compose file lives in `docker/`, so copy the env for parse-time interpolation:

```bash
cp -f .env docker/.env
```

If your compose contains `version:`, remove it:

```bash
sed -i '' '/^version:/d' docker/docker-compose.yml 2>/dev/null || true
```

## 2) Start the stack

Bring up Postgres first, then the rest:

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d postgres
# wait until health
until docker ps --filter name=bls_postgres --format '{{.Status}}' | grep -qi healthy; do sleep 1; done

docker compose --env-file docker/.env -f docker/docker-compose.yml up -d pgadmin airflow-init
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d airflow-webserver airflow-scheduler dbt api dashboard
```

## 3) Health checks

```bash
# merged config should show filled values (no blanks)
docker compose --env-file docker/.env -f docker/docker-compose.yml config \
| grep -E 'AIRFLOW__DATABASE__SQL_ALCHEMY_CONN|POSTGRES_USER'

docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Airflow health JSON
curl -sSf http://localhost:8080/health | jq .

# API and Dashboard
curl -sSf http://localhost:8000/docs >/dev/null && echo "API OK" || echo "API DOWN"
curl -sI  http://localhost:8501 | head -n1
```

## 4) dbt usage

Apple Silicon uses a Python base image that installs dbt at container start.

```bash
# debug connectivity and profiles
docker compose -f docker/docker-compose.yml exec dbt bash -lc 'dbt --version && dbt debug'

# run models and tests
docker compose -f docker/docker-compose.yml exec dbt bash -lc 'dbt deps && dbt run && dbt test'
```

Ensure `DBT_PROFILES_DIR=/usr/app` and `dbt/profiles.yml` key matches `profile:` in `dbt/dbt_project.yml`.

## 5) API and Dashboard dev loop

Live code mounts are enabled (`..:/app`). Recreate only when changing Python deps.

```bash
# tail logs
docker logs -f bls_api
docker logs -f bls_dashboard

# rebuild just these containers if needed
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d --force-recreate api dashboard
```

Minimal runtime deps the API must have in `requirements.txt`:

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.2
pyyaml==6.0.2
```

For the dashboard:

```txt
streamlit==1.39.0
pandas==2.2.2
pyarrow==17.0.0
```

## 6) Tests

Expose packages on `PYTHONPATH` then run a one-off test:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e PYTHONPATH=/app:/app/src \
  api bash -lc '
    [ -f requirements-dev.txt ] && pip install -q -r requirements-dev.txt || pip install -q pytest==9.0.0;
    pytest -vv
  '
```

If imports fail with `ModuleNotFoundError: etl`, ensure:

```
etl/__init__.py
etl/transforms/rfm.py
```

exist at repo root, or add `environment: ["PYTHONPATH=/app:/app/src"]` under `api:` and `dashboard:` services.

## 7) Airflow notes

* The DAG file `airflow/dags/lead_scoring_pipeline_dag.py` must contain a valid DAG. A zero-byte file will load nothing.
* Admin user is created by `airflow-init` from your `.env`.
* Trigger DAGs from the UI at `http://localhost:8080`.

## 8) Common issues and fixes

* **Docker daemon not running**
  Start Docker Desktop: `open -a Docker`.

* **Env not interpolated**
  Compose reads `.env` next to the compose file. Keep `docker/.env` or run with `--env-file docker/.env`.

* **Postgres unhealthy**
  Means blank `POSTGRES_*`. Refill `.env`, copy to `docker/.env`, re-`up`.

* **API DOWN / `uvicorn: command not found`**
  Ensure `uvicorn` and `fastapi` are in `requirements.txt` (see section 5), then `up -d --force-recreate api`.

* **dbt arm64 image error**
  Use the Python base (already configured) or force x86 with `platform: linux/amd64` at higher CPU cost.

* **Airflow 3.0 warning**
  The UI adds `airflow.api.auth.backend.session`. Safe to ignore for now; update config later.

## 9) One-command smoke test

```bash
bash -lc '
set -e
ok(){ printf "\033[32m[OK]\033[0m %s\n" "$1"; }
fail(){ printf "\033[31m[FAIL]\033[0m %s\n" "$1"; exit 1; }

docker compose --env-file docker/.env -f docker/docker-compose.yml config >/dev/null || fail "compose config"
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d >/dev/null || fail "compose up"

docker ps --filter name=bls_postgres --format "{{.Status}}" | grep -qi healthy || fail "postgres not healthy"; ok "postgres healthy"
curl -sSf http://localhost:8080/health >/dev/null || fail "airflow web not healthy"; ok "airflow web"
curl -sSf http://localhost:8000/docs  >/dev/null || fail "api not responding"; ok "api responding"
curl -sI  http://localhost:8501 | head -n1 | grep -q "200" || fail "dashboard not responding"; ok "dashboard responding"

docker compose -f docker/docker-compose.yml exec dbt bash -lc "dbt debug -q" >/dev/null && ok "dbt debug passed" || fail "dbt debug failed"
'
```

## 10) Stop and clean

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml down -v
```

**Do not commit `.env` or secrets.**
