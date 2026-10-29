"""Every result quoted in the README must match a committed result file."""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
TABLES = ROOT / "reports" / "tables"
COMPARISON = pd.read_csv(TABLES / "model_comparison.csv")
METRICS = pd.read_csv(TABLES / "metrics_summary.csv").iloc[0]
METHODOLOGY = json.loads((ROOT / "web" / "public" / "data" / "methodology.json").read_text())
CHOICE = json.loads((ROOT / "models" / "propensity_choice.json").read_text())


@pytest.mark.parametrize("row", COMPARISON[COMPARISON["task"] == "propensity"].to_dict("records"), ids=lambda r: r["model_id"])
def test_every_ranker_lift_and_auc_is_quoted_exactly(row):
    assert f"{row['lift_at_10pct']:.2f} ({row['lift_at_10pct_ci_lower']:.2f}–{row['lift_at_10pct_ci_upper']:.2f})" in README
    assert f"{row['auc']:.3f}" in README


@pytest.mark.parametrize("row", COMPARISON[COMPARISON["task"] == "value"].to_dict("records"), ids=lambda r: r["model_id"])
def test_every_value_model_rmse_is_quoted(row):
    assert f"{row['rmse'] / 1e6:.0f}M" in README


def test_headline_auc_interval_matches_metrics():
    assert f"{METRICS['propensity_auc']:.3f} ({METRICS['propensity_auc_ci_lower']:.3f}–{METRICS['propensity_auc_ci_upper']:.3f})" in README


def test_split_sizes_and_base_rates_match():
    data = METHODOLOGY["dataset"]
    assert f"{data['training_rows']:,}" in README and f"{data['test_rows']:,}" in README
    assert f"{data['positive_customers'] / data['training_rows']:.1%}" in README
    assert f"{METRICS['positive_rate']:.1%}" in README


def test_top_decile_reach_matches_decile_table():
    top = pd.read_csv(TABLES / "decile_lift.csv").iloc[0]
    assert f"{top['cumulative_share_of_converters']:.0%}" in README


def test_cltv_validation_claim_matches_validation_table():
    table = pd.read_csv(TABLES / "cltv_validation.csv")
    gbm = table[table["model"].str.startswith("GBM")]
    assert len(gbm) == 4 and table.iloc[0]["model"].startswith("Last 90 days repeat")
    assert "none of four gradient-boosting variants beat that baseline" in README


def test_paper_protocol_rows_are_quoted():
    table = pd.read_csv(TABLES / "paper_protocol.csv")
    for row in table.to_dict("records"):
        assert f"{row['mean_auc']:.3f}" in README


def test_blend_weight_and_validation_auc_are_quoted():
    assert f"GRU weight {CHOICE['blend']['weight']:.2f}" in README
    assert f"validation AUC {CHOICE['blend']['auc']:.3f}" in README


def test_test_month_result_matches_the_one_time_score():
    row = pd.read_csv(TABLES / "propensity_test.csv").iloc[0]
    assert row["test_auc"] == pytest.approx(METRICS["propensity_auc"])
    assert row["test_lift_at_10pct"] == pytest.approx(METRICS["lift_at_10pct"])


def test_next_best_product_results_are_quoted():
    table = pd.read_csv(TABLES / "next_product_test.csv")
    hits = table[table["measure"] == "top1_hit_rate_among_buyers"].set_index("model")["value"]
    assert f"**{hits['lightgbm']:.1%}** | {hits['most_popular_product']:.1%}" in README
    aucs = table[table["measure"] == "auc"]["value"]
    assert ", ".join(f"{v:.3f}" for v in aucs.iloc[:3]) + f" and {aucs.iloc[3]:.3f}" in README
