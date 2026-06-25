"""End-to-end pipeline runner."""

from __future__ import annotations

from lead_scoring.config import PipelineConfig
from lead_scoring.data import save_customer_features
from lead_scoring.diagnostics import build_residual_diagnostics
from lead_scoring.evaluation import evaluate_models
from lead_scoring.explainability import build_explainability_artifacts
from lead_scoring.features import prepare_feature_sets
from lead_scoring.model_comparison import build_model_comparison
from lead_scoring.risk import build_risk_signals
from lead_scoring.scenario import build_scenarios
from lead_scoring.scoring import build_fallback_explanations, score_batch
from lead_scoring.seasonality import build_seasonality_artifacts
from lead_scoring.training import train_models
from lead_scoring.web_export import export_web_data


def run_pipeline(config: PipelineConfig = PipelineConfig()) -> dict[str, object]:
    """Run the full lead scoring workflow."""
    feature_paths = save_customer_features(config)
    prepared_paths = prepare_feature_sets(config)
    training_metrics = train_models(config)
    evaluation_metrics = evaluate_models()
    scores = score_batch()
    fallbacks = build_fallback_explanations()
    model_comparison = build_model_comparison()
    diagnostics = build_residual_diagnostics()
    risks = build_risk_signals()
    scenarios = build_scenarios()
    seasonality = build_seasonality_artifacts()
    explanation_paths = build_explainability_artifacts()
    web_paths = export_web_data()

    return {
        "customer_features": [str(p) for p in feature_paths],
        "prepared": {key: str(value) for key, value in prepared_paths.items()},
        "training_metrics": training_metrics,
        "evaluation_metrics": evaluation_metrics,
        "scored_rows": int(len(scores)),
        "fallback_rows": int(len(fallbacks)),
        "model_comparison_rows": int(len(model_comparison)),
        "residual_diagnostics": diagnostics["summary"],
        "risk_rows": int(len(risks)),
        "scenario_rows": int(len(scenarios.get("scenarios", []))),
        "seasonality_rows": int(len(seasonality)),
        "explainability": explanation_paths,
        "web_exports": [str(path) for path in web_paths],
    }


def main() -> None:
    summary = run_pipeline()
    print(summary)


if __name__ == "__main__":
    main()
