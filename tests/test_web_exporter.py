"""The committed dashboard JSON must agree with the committed result tables (no regeneration here)."""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "public" / "data"
TABLES = ROOT / "reports" / "tables"
CONTRACT = ["overview", "forecast", "models", "diagnostics", "risks", "seasonality", "scenarios", "methodology"]


def _reject(constant):
    raise ValueError(f"{constant} is not valid JSON; the browser's JSON.parse rejects the whole file")


def _json(name):
    # Strict like the browser: Python's json.loads would otherwise accept NaN and Infinity.
    return json.loads((WEB / f"{name}.json").read_text(), parse_constant=_reject)


@pytest.mark.parametrize("name", CONTRACT)
def test_every_dashboard_file_exists_and_parses(name):
    assert isinstance(_json(name), dict)


def test_dashboard_metrics_match_results_table():
    metrics = pd.read_csv(TABLES / "metrics_summary.csv").iloc[0]
    shown = _json("models")["metrics"]
    for key in ("propensity_auc", "propensity_auc_ci_lower", "propensity_auc_ci_upper", "lift_at_10pct"):
        assert shown[key] == pytest.approx(metrics[key])


def test_dashboard_leaderboard_matches_comparison_table():
    table = pd.read_csv(TABLES / "model_comparison.csv")
    shown = pd.DataFrame(_json("models")["leaderboard"])
    assert shown["model_id"].tolist() == table["model_id"].tolist()
    assert shown["lift_at_10pct"].fillna(-1).tolist() == pytest.approx(table["lift_at_10pct"].fillna(-1).tolist())


def test_overview_names_the_real_data_source_and_no_synthetic_claims():
    overview = _json("overview")
    assert "MBD" in overview["project"]["description"]
    assert overview["kpis"]["customers"] == 100_224
    assert not any("synthetic" in item.lower() for item in _json("methodology")["limitations"])
