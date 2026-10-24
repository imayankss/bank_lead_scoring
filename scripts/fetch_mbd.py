"""Download MBD-mini (Sber AI Lab, CC BY 4.0) into data/raw/mbd/ and verify it.

Source: https://huggingface.co/datasets/ai-lab/MBD-mini (pinned revision below).
Paper: Mollaev et al., "Multimodal Banking Dataset", arXiv:2409.17587.
Only the targets table and the transaction table are extracted (about 0.7 GB on disk).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lead_scoring.paths import MBD_ARCHIVES as FILES
from lead_scoring.paths import MBD_DIR, MBD_REVISION as REVISION

BASE = f"https://huggingface.co/datasets/ai-lab/MBD-mini/resolve/{REVISION}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, expected: str, archive_dir: Path) -> Path:
    path = archive_dir / name
    if not path.exists() or sha256(path) != expected:
        print(f"Downloading {name} ...")
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(f"{BASE}/{name}") as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(path)
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{name}: SHA-256 {actual} does not match pinned {expected}")
    return path


def main() -> None:
    archive_dir = MBD_DIR / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name, (expected, prefixes) in FILES.items():
        path = fetch(name, expected, archive_dir)
        with tarfile.open(path) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith(prefixes)]
            tar.extractall(MBD_DIR, members=members, filter="data")
        print(f"Extracted {len(members)} entries from {name}")
    print(f"MBD-mini ready in {MBD_DIR}")


if __name__ == "__main__":
    main()
