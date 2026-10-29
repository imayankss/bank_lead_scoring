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


def risk_composite(df: pd.DataFrame) -> pd.Series:
    """0-100 outreach risk from heuristic rules (not a fitted model).

    Stale activity, a cooling client (last 90 days under half their usual quarter) and a
    cold lead score each add risk. A 0.5-centred "uncertainty" term was dropped: with a
    2% base rate every probability sits near 0, so it never contributed.
    """
    recency_risk = df["recency_days"].clip(0, 365) / 365 * 40
    activity_drop_risk = np.where(df["txn_count_90"] < df["txn_count_365"] / 8, 30, 0)
    score_risk = np.where(df["lead_score"] < 30, 30, 0)
    return pd.Series(recency_risk + activity_drop_risk + score_risk, index=df.index).clip(0, 100)


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
                "txn_count_90",
                "total_txn_amt_365",
            ]
        ],
        on="customer_id",
        how="left",
    ).merge(fallbacks, on="customer_id", how="left")

    df["risk_score_composite"] = risk_composite(df)
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
