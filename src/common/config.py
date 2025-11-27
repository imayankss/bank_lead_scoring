# src/common/config.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, List

import yaml


# Resolve project root as the repo root (…/bank_lead_scoring_project)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = PROJECT_ROOT / "conf" / "settings.yaml"


@dataclass
class ProjectConfig:
    db_path: Path
    src_dir: Path
    cache_dir: Path
    end_date: date
    top_k_pct: int


@dataclass
class CltvConfig:
    bins: List[float]
    labels: List[str]


@dataclass
class ScoringConfig:
    rules_pct: float
    ml_pct: float


@dataclass
class StageConfig:
    threads: int
    memory: str
    strict: bool
    indexes: bool


@dataclass
class ExportConfig:
    crm: Path
    crm_hybrid: Path
    features_parquet: Path
    features_csv: Path


@dataclass
class Settings:
    project: ProjectConfig
    cltv: CltvConfig
    scoring: ScoringConfig
    stage: StageConfig
    exports: ExportConfig


def _load_raw_settings() -> dict[str, Any]:
    with open(CONF_PATH, "r") as f:
        return yaml.safe_load(f)


def _build_settings(raw: dict[str, Any]) -> Settings:
    project_raw = raw["project"]
    cltv_raw = raw["cltv"]
    scoring_raw = raw["scoring"]["hybrid_weights"]
    stage_raw = raw["stage"]
    exports_raw = raw["exports"]

    # Normalize end_date into a date object
    raw_end = project_raw["end_date"]
    if isinstance(raw_end, date):
        end_dt = raw_end
    else:
        end_dt = date.fromisoformat(str(raw_end))

    return Settings(
        project=ProjectConfig(
            db_path=(PROJECT_ROOT / project_raw["db_path"]).resolve(),
            src_dir=(PROJECT_ROOT / project_raw["src_dir"]).resolve(),
            cache_dir=(PROJECT_ROOT / project_raw["cache_dir"]).resolve(),
            end_date=end_dt,
            top_k_pct=int(project_raw["top_k_pct"]),
        ),
        cltv=CltvConfig(
            bins=list(cltv_raw["bins"]),
            labels=list(cltv_raw["labels"]),
        ),
        scoring=ScoringConfig(
            rules_pct=float(scoring_raw["rules_pct"]),
            ml_pct=float(scoring_raw["ml_pct"]),
        ),
        stage=StageConfig(
            threads=int(stage_raw["threads"]),
            memory=str(stage_raw["memory"]),
            strict=bool(stage_raw["strict"]),
            indexes=bool(stage_raw["indexes"]),
        ),
        exports=ExportConfig(
            crm=(PROJECT_ROOT / exports_raw["crm"]).resolve(),
            crm_hybrid=(PROJECT_ROOT / exports_raw["crm_hybrid"]).resolve(),
            features_parquet=(PROJECT_ROOT / exports_raw["features_parquet"]).resolve(),
            features_csv=(PROJECT_ROOT / exports_raw["features_csv"]).resolve(),
        ),
    )


# Singleton settings object to import elsewhere
_raw = _load_raw_settings()
settings = _build_settings(_raw)
