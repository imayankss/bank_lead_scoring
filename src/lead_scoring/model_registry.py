"""Model registry metadata for docs, exports, and dashboards."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    display_name: str
    model_family: str
    target: str
    role: str
    description: str


MODEL_REGISTRY = {
    "cltv_hist_gradient_boosting": ModelRegistryEntry(
        model_id="cltv_hist_gradient_boosting",
        display_name="CLTV Gradient Boosting",
        model_family="HistGradientBoostingRegressor",
        target="future_value",
        role="Revenue forecast",
        description="Predicts the client's transaction value over the next 3 months from activity and product-history features.",
    ),
    "propensity_lightgbm": ModelRegistryEntry(
        model_id="propensity_lightgbm",
        display_name="Propensity LightGBM",
        model_family="LightGBM, isotonic calibration",
        target="converted",
        role="Conversion propensity",
        description="Ranks clients by probability of taking any of the bank's 4 products in the next 3 months, from "
                    "transaction-mix, purchase-history and dialog features; settings and calibration chosen on the July 2022 validation month.",
    ),
    "expected_value_ranker": ModelRegistryEntry(
        model_id="expected_value_ranker",
        display_name="Expected Value Ranker",
        model_family="Composite scorer",
        target="predicted_cltv * predicted_propensity",
        role="Lead prioritization",
        description="Combines value and propensity predictions into a single expected-value lead score.",
    ),
}


def registry_as_records() -> list[dict[str, str]]:
    """Return registry entries as JSON-friendly dictionaries."""
    return [asdict(entry) for entry in MODEL_REGISTRY.values()]
