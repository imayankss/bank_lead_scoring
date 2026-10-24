import re

from lead_scoring.config import PipelineConfig
from lead_scoring.provenance import build_run_manifest


def test_manifest_records_code_data_config_and_versions():
    cfg = PipelineConfig(workers=2)
    manifest = build_run_manifest(cfg)
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", manifest["git_sha"])
    assert isinstance(manifest["git_dirty"], bool)
    assert manifest["config"]["workers"] == 2 and manifest["config"]["test_snapshot"] == cfg.test_snapshot
    assert manifest["data"]["revision"] and len(manifest["data"]["archives"]["detail.tar.gz"]) == 64
    assert {"python", "pandas", "scikit-learn", "duckdb", "numpy"} <= set(manifest["versions"])
