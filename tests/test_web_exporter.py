import json
from pathlib import Path

from lead_scoring.web_export import export_web_data


def test_web_export_contract_files_exist_after_export():
    paths = export_web_data()
    names = {path.name for path in paths}

    assert {
        "overview.json",
        "forecast.json",
        "models.json",
        "diagnostics.json",
        "risks.json",
        "seasonality.json",
        "scenarios.json",
        "methodology.json",
    }.issubset(names)

    overview = json.loads(Path("web/public/data/overview.json").read_text())
    assert "project" in overview
    assert "kpis" in overview
    assert overview["kpis"]["customers"] == 500
