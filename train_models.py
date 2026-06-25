"""Compatibility entrypoint for model training."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lead_scoring.training import main


if __name__ == "__main__":
    main()
