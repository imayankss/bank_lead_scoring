"""Batch scoring and rules-based fallback explanations."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from lead_scoring.paths import (
    CLTV_MODEL_FILE,
    LEAD_SCORES_FILE,
    PREPROCESSED_DATA_DIR,
    PROPENSITY_MODEL_FILE,
    SCORING_FEATURES_FILE,
    TABLES_DIR,
    ensure_directories,
)


def _score_to_100(expected_value: pd.Series) -> pd.Series:
    """Convert expected value to a stable 0-100 rank score."""
    if expected_value.empty:
        return pd.Series(dtype="int")
    if expected_value.nunique(dropna=False) <= 1:
        return pd.Series(np.full(len(expected_value), 50), index=expected_value.index, dtype="int")
    return (expected_value.rank(method="average", pct=True) * 100).round().clip(0, 100).astype(int)


def score_batch() -> pd.DataFrame:
    """Score the current customer universe and save lead_scores.csv."""
    ensure_directories()
    X_scoring = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_scoring.parquet")
    scoring_ids = pd.read_csv(PREPROCESSED_DATA_DIR / "scoring_ids.csv")

    cltv_model = joblib.load(CLTV_MODEL_FILE)
    propensity_model = joblib.load(PROPENSITY_MODEL_FILE)

    predicted_cltv = np.clip(cltv_model.predict(X_scoring), 0, None)
    predicted_propensity = propensity_model.predict_proba(X_scoring)[:, 1]
    expected_value = pd.Series(predicted_cltv * predicted_propensity)
    lead_score = _score_to_100(expected_value)

    output = pd.DataFrame(
        {
            "customer_id": scoring_ids["customer_id"],
            "predicted_cltv": predicted_cltv,
            "predicted_propensity": predicted_propensity,
            "expected_value": expected_value,
            "lead_score": lead_score,
        }
    )
    output["lead_category"] = pd.cut(
        output["lead_score"],
        bins=[-1, 29, 69, 100],
        labels=["Cold", "Medium", "Hot"],
    ).astype(str)
    output.to_csv(LEAD_SCORES_FILE, index=False)
    return output


def build_fallback_explanations() -> pd.DataFrame:
    """Create simple operational flags for sales follow-up routing."""
    ensure_directories()
    features = pd.read_parquet(SCORING_FEATURES_FILE)

    rows = []
    for _, row in features.iterrows():
        reasons = []
        if row.get("txn_count_365", 0) == 0:
            reasons.append("no_transactions_in_last_365d")
        if row.get("recency_days", 0) > 180:
            reasons.append("stale_transaction_activity")
        if row.get("active_account_count", 0) == 0:
            reasons.append("no_active_account")
        if pd.isna(row.get("annual_income")) or row.get("annual_income", 0) <= 0:
            reasons.append("missing_income")
        if pd.isna(row.get("risk_score")):
            reasons.append("missing_risk_score")
        elif row.get("risk_score", 0) < 500:
            reasons.append("low_risk_score")

        rows.append(
            {
                "customer_id": row["customer_id"],
                "fallback_reasons": "; ".join(reasons) if reasons else "no_issue_detected",
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "fallback_explanations.csv", index=False)
    return output


def main() -> None:
    scores = score_batch()
    fallbacks = build_fallback_explanations()
    print(f"Saved {len(scores)} scores to {LEAD_SCORES_FILE}")
    print(f"Saved {len(fallbacks)} fallback explanations to {TABLES_DIR / 'fallback_explanations.csv'}")


if __name__ == "__main__":
    main()
