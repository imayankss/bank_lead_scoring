"""The committed dashboard JSON must agree with the committed result tables (no regeneration here)."""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "public" / "data"
TABLES = ROOT / "reports" / "tables"
CONTRACT = ["crm", "results", "overview", "forecast", "models", "diagnostics", "risks", "seasonality", "scenarios", "methodology"]


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


def test_results_page_matches_result_tables():
    results = _json("results")
    table = pd.read_csv(TABLES / "model_comparison.csv").set_index("model_id")
    for ranker in results["rankers"]:
        assert ranker["lift_at_10pct"] == pytest.approx(table.loc[ranker["model_id"], "lift_at_10pct"])
    # The frontier's top decile is the expected-value lead score's measured lift.
    assert results["gains"][0]["lift"] == pytest.approx(table.loc["expected_value_ranker", "lift_at_10pct"], abs=0.01)
    assert results["test"]["test_auc"] == pytest.approx(table.loc["lightgbm", "auc"])


def test_crm_call_list_is_ranked_and_consistent():
    crm = _json("crm")
    clients = crm["clients"]
    assert [c["rank"] for c in clients] == list(range(1, len(clients) + 1))
    assert [c["ev"] for c in clients] == sorted((c["ev"] for c in clients), reverse=True)
    for c in clients:
        assert all(0 <= p <= 1 for p in [c["p_any"], *c["p"]])
        assert c["next"] == c["p"].index(max(c["p"])) + 1
        assert sum(c["p"]) == pytest.approx(c["p_any"], abs=1e-4)
    assert sum(crm["summary"]["tiers"].values()) == crm["summary"]["clients"]
    assert sum(p["next_for"] for p in crm["products"]) == crm["summary"]["clients"]
    assert len(clients) == crm["summary"]["shown"]
    assert sum(p["expected_takers"] for p in crm["products"]) == pytest.approx(crm["summary"]["expected_takers"])


def test_crm_quotes_the_measured_next_product_result():
    table = pd.read_csv(TABLES / "next_product_test.csv")
    assert pd.DataFrame(_json("crm")["test"])["value"].tolist() == pytest.approx(table["value"].tolist())
