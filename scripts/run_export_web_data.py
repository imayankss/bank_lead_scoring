"""Build dashboard intelligence artifacts and export static web JSON."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lead_scoring.diagnostics import build_residual_diagnostics
from lead_scoring.model_comparison import build_model_comparison
from lead_scoring.risk import build_risk_signals
from lead_scoring.scenario import build_scenarios
from lead_scoring.seasonality import build_seasonality_artifacts
from lead_scoring.web_export import export_web_data


def main() -> None:
    build_model_comparison()
    build_residual_diagnostics()
    build_risk_signals()
    build_scenarios()
    build_seasonality_artifacts()
    paths = export_web_data()
    print("Exported dashboard JSON:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
