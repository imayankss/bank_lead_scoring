"""Run the full lead scoring pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lead_scoring.pipeline import run_pipeline


if __name__ == "__main__":
    summary = run_pipeline()
    print("Pipeline completed.")
    print(summary)
