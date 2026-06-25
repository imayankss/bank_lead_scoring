"""Risk intelligence and alert rules for lead scoring outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lead_scoring.paths import LEAD_SCORES_FILE, SCORING_FEATURES_FILE, TABLES_DIR, ensure_directories


def _risk_category(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def build_risk_signals() -> pd.DataFrame:
    """Create customer-level risk and confidence signals."""
    ensure_directories()
    scores = pd.read_csv(LEAD_SCORES_FILE)
    features = pd.read_parquet(SCORING_FEATURES_FILE)
    fallbacks_path = TABLES_DIR / "fallback_explanations.csv"
    fallbacks = pd.read_csv(fallbacks_path) if fallbacks_path.exists() else pd.DataFrame(columns=["customer_id"])

    df = scores.merge(
        features[
            [
                "customer_id",
                "recency_days",
                "txn_count_365",
                "active_account_count",
                "risk_score",
                "total_txn_amt_365",
                "avg_balance",
            ]
        ],
        on="customer_id",
        how="left",
    ).merge(fallbacks, on="customer_id", how="left")

    uncertainty = 1 - (df["predicted_propensity"] - 0.5).abs() * 2
    recency_risk = (df["recency_days"].clip(0, 365) / 365) * 25
    account_risk = np.where(df["active_account_count"].fillna(0) <= 0, 20, 0)
    credit_risk = np.where(df["risk_score"].fillna(650) < 500, 20, 0)
    uncertainty_risk = uncertainty.clip(0, 1) * 25
    score_risk = np.where(df["lead_score"] < 30, 10, 0)

    df["risk_score_composite"] = (recency_risk + account_risk + credit_risk + uncertainty_risk + score_risk).clip(0, 100)
    df["risk_category"] = df["risk_score_composite"].map(_risk_category)
    df["confidence_score"] = (100 - df["risk_score_composite"]).clip(0, 100)
    df["alert"] = np.where(
        df["risk_category"].eq("High"),
        "Review before outreach",
        np.where(df["lead_category"].eq("Hot"), "Prioritize outreach", "Monitor"),
    )

    output = df[
        [
            "customer_id",
            "lead_score",
            "lead_category",
            "predicted_propensity",
            "expected_value",
            "risk_score_composite",
            "risk_category",
            "confidence_score",
            "alert",
            "fallback_reasons",
        ]
    ].sort_values(["risk_score_composite", "expected_value"], ascending=[False, False])
    output.to_csv(TABLES_DIR / "risk_signals.csv", index=False)

    summary = {
        "overall_risk_score": float(output["risk_score_composite"].mean()),
        "average_confidence": float(output["confidence_score"].mean()),
        "high_risk_customers": int((output["risk_category"] == "High").sum()),
        "medium_risk_customers": int((output["risk_category"] == "Medium").sum()),
        "low_risk_customers": int((output["risk_category"] == "Low").sum()),
        "top_alerts": output.head(8).to_dict(orient="records"),
    }
    (TABLES_DIR / "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def main() -> None:
    output = build_risk_signals()
    print(f"Saved {len(output)} risk signals.")


if __name__ == "__main__":
    main()
