"""Run manifest: what code, data, settings and packages produced the committed results."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from lead_scoring.config import PipelineConfig
from lead_scoring.paths import MBD_ARCHIVES, MBD_REVISION, PROJECT_ROOT, RUN_MANIFEST_FILE

PACKAGES = ["pandas", "numpy", "scikit-learn", "duckdb", "pyarrow", "joblib", "lightgbm", "torch"]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _head_sha() -> str:
    """Commit SHA, read from .git directly when the git binary can't run (e.g. an x86 Python under Rosetta)."""
    sha = _git("rev-parse", "HEAD")
    if sha:
        return sha
    git_dir = PROJECT_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git_dir / ref).exists():
            return (git_dir / ref).read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def build_run_manifest(config: PipelineConfig = PipelineConfig()) -> dict[str, object]:
    """Describe the run; archive hashes are the pinned ones scripts/fetch_mbd.py verifies."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _head_sha(),
        "git_dirty": None if status is None else bool(status),  # None: git couldn't run, so unknown
        "config": asdict(config),
        "data": {
            "source": "huggingface.co/datasets/ai-lab/MBD-mini",
            "licence": "CC BY 4.0",
            "revision": MBD_REVISION,
            "archives": {name: sha for name, (sha, _) in MBD_ARCHIVES.items()},
        },
        "versions": {"python": platform.python_version(), **{p: _version(p) for p in PACKAGES}},
    }


def write_run_manifest(config: PipelineConfig = PipelineConfig()) -> dict[str, object]:
    manifest = build_run_manifest(config)
    RUN_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
