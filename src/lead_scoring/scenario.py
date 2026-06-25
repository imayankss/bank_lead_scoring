"""Portfolio scenario analysis for lead score outputs."""

from __future__ import annotations

import json

import pandas as pd

from lead_scoring.paths import LEAD_SCORES_FILE, TABLES_DIR, ensure_directories


SCENARIO_MULTIPLIERS = {
    "Pessimistic": 0.78,
    "Base": 1.0,
    "Optimistic": 1.18,
}


def build_scenarios() -> dict[str, object]:
    """Create static 12 month value scenarios for the frontend."""
    ensure_directories()
    scores = pd.read_csv(LEAD_SCORES_FILE)
    base_total = float(scores["expected_value"].sum())
    months = list(range(1, 13))

    scenario_rows = []
    for name, multiplier in SCENARIO_MULTIPLIERS.items():
        cumulative = 0.0
        for month in months:
            monthly_value = base_total * multiplier / 12
            cumulative += monthly_value
            scenario_rows.append(
                {
                    "month": month,
                    "scenario": name,
                    "monthly_expected_value": monthly_value,
                    "cumulative_expected_value": cumulative,
                }
            )

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(TABLES_DIR / "scenario_forecasts.csv", index=False)

    sensitivity = []
    for uplift in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        sensitivity.append(
            {
                "conversion_change": uplift,
                "portfolio_expected_value": base_total * (1 + uplift),
            }
        )
    sensitivity_df = pd.DataFrame(sensitivity)
    sensitivity_df.to_csv(TABLES_DIR / "scenario_sensitivity.csv", index=False)

    payload = {
        "base_expected_value": base_total,
        "scenarios": scenario_df.to_dict(orient="records"),
        "sensitivity": sensitivity,
    }
    (TABLES_DIR / "scenario_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    payload = build_scenarios()
    print(f"Saved scenarios for base expected value {payload['base_expected_value']:.2f}.")


if __name__ == "__main__":
    main()
