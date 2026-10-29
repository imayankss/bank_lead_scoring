"""Export backend artifacts into stable JSON files for the Next.js dashboard."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from lead_scoring.model_registry import registry_as_records
from lead_scoring.paths import (
    CUSTOMER_FEATURES_FILE,
    FEATURE_METADATA_FILE,
    MODEL_METADATA_FILE,
    PROJECT_ROOT,
    TABLES_DIR,
    ensure_directories,
)


WEB_DATA_DIR = PROJECT_ROOT / "web" / "public" / "data"
# The dashboard ships static JSON, so row-level arrays are sampled to keep pages light.
SAMPLE_ROWS = 400


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = WEB_DATA_DIR / name
    # NaN/Infinity aren't valid JSON: the browser rejects the whole file, so they become null.
    plain = json.loads(json.dumps(payload, default=_json_default))
    path.write_text(json.dumps(_finite(plain), indent=2, allow_nan=False), encoding="utf-8")
    return path


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(n=min(SAMPLE_ROWS, len(df)), random_state=42).sort_index() if not df.empty else df


def export_web_data() -> list[Path]:
    """Write all frontend JSON data contracts."""
    ensure_directories()
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    customer_features = pd.read_parquet(CUSTOMER_FEATURES_FILE)
    feature_metadata = _read_json(FEATURE_METADATA_FILE)
    model_metadata = _read_json(MODEL_METADATA_FILE)
    lead_scores = _read_csv(PROJECT_ROOT / "data" / "processed" / "lead_scores.csv")
    metrics = _read_csv(TABLES_DIR / "metrics_summary.csv")
    model_comparison = _read_csv(TABLES_DIR / "model_comparison.csv")
    test_predictions = _read_csv(TABLES_DIR / "test_predictions.csv")
    residuals = _read_csv(TABLES_DIR / "residual_records.csv")
    residual_summary = _read_csv(TABLES_DIR / "residual_diagnostics.csv")
    risk_signals = _read_csv(TABLES_DIR / "risk_signals.csv")
    risk_summary = _read_json(TABLES_DIR / "risk_summary.json")
    scenarios = _read_json(TABLES_DIR / "scenario_summary.json")
    seasonality = _read_csv(TABLES_DIR / "seasonality_monthly.csv")
    seasonality_pattern = _read_csv(TABLES_DIR / "seasonality_pattern.csv")
    feature_importance = _read_csv(TABLES_DIR / "feature_importance.csv")
    decile_lift = _read_csv(TABLES_DIR / "decile_lift.csv")

    category_counts = lead_scores["lead_category"].value_counts().to_dict() if not lead_scores.empty else {}
    rankers = model_comparison[model_comparison["task"] == "propensity"] if not model_comparison.empty else model_comparison
    values = model_comparison[model_comparison["task"] == "value"] if not model_comparison.empty else model_comparison
    best_model = rankers.sort_values("rank").iloc[0].to_dict() if not rankers.empty else {}
    best_value_model = values.sort_values("rank").iloc[0].to_dict() if not values.empty else {}
    metric_row = metrics.iloc[0].to_dict() if not metrics.empty else {}

    paths = [
        _write_json(
            "overview.json",
            {
                "project": {
                    "name": "Lead Scoring Intelligence Platform",
                    "description": "Lead prioritization built for a Bank of India-style relationship-manager workflow, "
                    "demonstrated on Sber AI Lab's public MBD-mini data (100,224 business clients, CC BY 4.0).",
                    "domain": "Banking analytics",
                },
                "kpis": {
                    "customers": int(len(lead_scores)),
                    "features": int(len(feature_metadata.get("feature_cols", []))),
                    "models_trained": 2,
                    "best_model": best_model.get("model_name", "Expected Value Ranker"),
                    "best_lift_at_10pct": best_model.get("lift_at_10pct"),
                    "best_lift_ci": [best_model.get("lift_at_10pct_ci_lower"), best_model.get("lift_at_10pct_ci_upper")],
                    "test_auc": metric_row.get("propensity_auc"),
                    "test_auc_ci": [metric_row.get("propensity_auc_ci_lower"), metric_row.get("propensity_auc_ci_upper")],
                    "test_positive_rate": metric_row.get("positive_rate"),
                    "best_value_model": best_value_model.get("model_name"),
                    "best_rmse": best_value_model.get("rmse", metric_row.get("cltv_rmse")),
                    "total_expected_value": float(lead_scores["expected_value"].sum()) if not lead_scores.empty else 0,
                },
                "lead_categories": category_counts,
                "top_leads": lead_scores.sort_values("lead_score", ascending=False).head(12).to_dict(orient="records"),
                "decile_lift": decile_lift.to_dict(orient="records"),
            },
        ),
        _write_json(
            "forecast.json",
            {
                "actual_vs_predicted": _sample(test_predictions).to_dict(orient="records"),
                "forecast": scenarios.get("scenarios", []),
                "latest": {
                    "total_expected_value": float(lead_scores["expected_value"].sum()) if not lead_scores.empty else 0,
                    "average_propensity": float(lead_scores["predicted_propensity"].mean()) if not lead_scores.empty else 0,
                    "average_cltv": float(lead_scores["predicted_cltv"].mean()) if not lead_scores.empty else 0,
                },
            },
        ),
        _write_json(
            "models.json",
            {
                "registry": registry_as_records(),
                "leaderboard": model_comparison.to_dict(orient="records"),
                "metrics": metric_row,
                "model_metadata": model_metadata,
                "feature_importance": feature_importance.to_dict(orient="records"),
            },
        ),
        _write_json(
            "diagnostics.json",
            {
                "summary": residual_summary.iloc[0].to_dict() if not residual_summary.empty else {},
                "residuals": _sample(residuals).to_dict(orient="records"),
                "feature_importance": feature_importance.to_dict(orient="records"),
            },
        ),
        _write_json(
            "risks.json",
            {
                "summary": risk_summary,
                "signals": risk_signals.head(100).to_dict(orient="records"),
            },
        ),
        _write_json(
            "seasonality.json",
            {
                "monthly": seasonality.to_dict(orient="records"),
                "pattern": seasonality_pattern.to_dict(orient="records"),
            },
        ),
        _write_json(
            "scenarios.json",
            scenarios,
        ),
        _write_json(
            "methodology.json",
            {
                "pipeline": [
                    "MBD-mini parquet (pinned, SHA-256 checked)",
                    "DuckDB aggregation, one process per client fold",
                    "Month-end snapshots: 365-day features, 3-month targets",
                    "Time split: train Jan/Apr/Jul 2022, test Oct 2022 snapshot",
                    "Leakage guards that raise",
                    "CLTV and propensity training",
                    "Baselines and bootstrap 95% intervals",
                    "Risk rules and scenario analysis",
                    "Static JSON export",
                    "Next.js dashboard",
                ],
                "leakage_prevention": feature_metadata.get("excluded_columns", []),
                "limitations": [
                    "Public Sber AI Lab data (business clients), not Bank of India customers.",
                    "MBD adds noise to amounts and dates; amounts are capped at the 0.999 quantile.",
                    "The 4 products are unnamed in the source data.",
                    "Risk rules and scenarios are heuristics, not measured models.",
                ],
                "dataset": {
                    # The stacked feature table also holds the test snapshot, so counts come from the split.
                    "training_rows": int(feature_metadata.get("train_rows", 0)),
                    "test_rows": int(feature_metadata.get("test_rows", 0)),
                    "positive_customers": int(customer_features.loc[
                        customer_features["snapshot_date"].isin(feature_metadata.get("train_snapshots", [])), "converted"
                    ].sum()),
                    "feature_columns": feature_metadata.get("feature_cols", []),
                },
            },
        ),
    ]
    return paths


def main() -> None:
    paths = export_web_data()
    print("Exported web data:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
