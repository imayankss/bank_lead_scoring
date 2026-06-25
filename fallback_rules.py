"""Compatibility entrypoint for fallback explanation rules."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lead_scoring.scoring import build_fallback_explanations


if __name__ == "__main__":
    output = build_fallback_explanations()
    print(f"Saved {len(output)} fallback explanations.")
