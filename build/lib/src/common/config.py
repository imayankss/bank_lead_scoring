from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel
import yaml


# ---------------------------------------------------------------------------
# Project root: env override + fallback to repo structure
# ---------------------------------------------------------------------------

# If BLS_PROJECT_ROOT is set (e.g. in Airflow/CI), use that.
# Otherwise, assume this file lives under <ROOT>/src/common/config.py
PROJECT_ROOT = Path(
    os.environ.get("BLS_PROJECT_ROOT", "")
) or Path(__file__).resolve().parents[2]

PROJECT_ROOT = PROJECT_ROOT.resolve()

CONF_DIR = PROJECT_ROOT / "conf"
DATA_DIR = PROJECT_ROOT / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class ProjectSettings(BaseModel):
    db_path: Path  # DuckDB path
    src_dir: Path
    cache_dir: Path


class Settings(BaseModel):
    project: ProjectSettings
    # ... other sections as you already have ...


def _load_settings() -> Settings:
    settings_path = CONF_DIR / "settings.yaml"
    with open(settings_path, "r") as f:
        raw = yaml.safe_load(f)

    # fill project defaults if missing
    project_raw = raw.get("project", {})
    if "db_path" not in project_raw:
        project_raw["db_path"] = str(WAREHOUSE_DIR / "cltv.duckdb")
    if "src_dir" not in project_raw:
        project_raw["src_dir"] = str(PROJECT_ROOT / "src")
    if "cache_dir" not in project_raw:
        project_raw["cache_dir"] = str(PROJECT_ROOT / ".cache")

    raw["project"] = project_raw
    return Settings(**raw)


settings = _load_settings()

