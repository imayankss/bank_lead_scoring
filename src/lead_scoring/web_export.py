"""Export backend artifacts into stable JSON files for the Next.js dashboard."""

from __future__ import annotations

import json
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


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(name: str, payload: dict[str, Any]) -> Path:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = WEB_DATA_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


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
    best_model = (
        model_comparison.sort_values("rank").iloc[0].to_dict()
        if not model_comparison.empty and "rank" in model_comparison.columns
        else {}
    )
    metric_row = metrics.iloc[0].to_dict() if not metrics.empty else {}
    confidence = max(0.0, min(100.0, 100.0 - float(risk_summary.get("overall_risk_score", 50.0))))

    paths = [
        _write_json(
            "overview.json",
            {
                "project": {
                    "name": "Lead Scoring Intelligence Platform",
                    "description": "AI-powered customer prioritization platform with CLTV, propensity, risk signals, scenario analysis, and model diagnostics.",
                    "domain": "Banking analytics",
                },
                "kpis": {
                    "customers": int(len(lead_scores)),
                    "features": int(len(feature_metadata.get("feature_cols", []))),
                    "models_trained": 2,
                    "best_model": best_model.get("model_name", "Expected Value Ranker"),
                    "best_rmse": best_model.get("rmse", metric_row.get("cltv_rmse")),
                    "forecast_confidence": confidence,
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
                "actual_vs_predicted": test_predictions.to_dict(orient="records"),
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
                "residuals": residuals.to_dict(orient="records"),
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
                    "Raw banking CSVs",
                    "Schema normalization",
                    "Historical feature window",
                    "Leakage-safe preprocessing",
                    "CLTV and propensity training",
                    "Model evaluation and diagnostics",
                    "Risk scoring and scenario analysis",
                    "Static JSON export",
                    "Next.js dashboard",
                ],
                "leakage_prevention": feature_metadata.get("excluded_columns", []),
                "limitations": [
                    "Synthetic and small dataset.",
                    "Metrics are prototype validation, not production proof.",
                    "Scenario outputs are static portfolio simulations.",
                ],
                "dataset": {
                    "training_rows": int(len(customer_features)),
                    "positive_customers": int(customer_features["converted_12m"].sum()),
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
