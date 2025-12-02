from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
import yaml

# ---------------------------------------------------------------------------
# Project root: env override + fallback to repo structure
# ---------------------------------------------------------------------------

_env_root = os.environ.get("BLS_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root)
else:
    # This file lives under <ROOT>/src/common/config.py
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = PROJECT_ROOT.resolve()

CONF_DIR = PROJECT_ROOT / "conf"
DATA_DIR = PROJECT_ROOT / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"


# ---------------------------------------------------------------------------
# Settings models
# ---------------------------------------------------------------------------

class ProjectSettings(BaseModel):
    """Project-level paths (DuckDB, source code, cache)."""
    db_path: Path
    src_dir: Path
    cache_dir: Path

    class Config:
        extra = "allow"  # don't explode if YAML has extra keys


class ExportSettings(BaseModel):
    """Export/output paths used in the project."""
    crm_hybrid: Path

    class Config:
        extra = "allow"


class ScoringConfig(BaseModel):
    """
    Hybrid scoring weights:
    - rules_pct: share for lead_score_0_100 (rules-based)
    - ml_pct:    share for ML probability (scaled 0–100)
    """
    rules_pct: float = 0.4  # 40% rules-based score
    ml_pct: float = 0.6     # 60% ML model score

    class Config:
        extra = "allow"


class Settings(BaseModel):
    project: ProjectSettings
    exports: ExportSettings
    scoring: ScoringConfig = ScoringConfig()

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_settings() -> Settings:
    settings_path = CONF_DIR / "settings.yaml"
    with open(settings_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # ----- project defaults -----
    project_raw = raw.get("project", {}) or {}
    if "db_path" not in project_raw:
        project_raw["db_path"] = str(WAREHOUSE_DIR / "cltv.duckdb")
    if "src_dir" not in project_raw:
        project_raw["src_dir"] = str(PROJECT_ROOT / "src")
    if "cache_dir" not in project_raw:
        project_raw["cache_dir"] = str(PROJECT_ROOT / ".cache")
    raw["project"] = project_raw

    # ----- exports defaults -----
    exports_raw = raw.get("exports", {}) or {}
    if "crm_hybrid" not in exports_raw:
        exports_raw["crm_hybrid"] = str(DATA_DIR / "processed" / "crm_export_hybrid.csv")
    raw["exports"] = exports_raw

    # ----- scoring defaults (optional in YAML) -----
    # If missing, ScoringConfig() defaults (0.4 / 0.6) will be used.
    if "scoring" not in raw:
        raw["scoring"] = {}

    return Settings(**raw)


settings = _load_settings()

