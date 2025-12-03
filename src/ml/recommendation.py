# src/ml/recommendation.py
from __future__ import annotations

from typing import Optional
import pandas as pd


def compute_behavioral_boost(row: pd.Series) -> float:
    """
    Behavioural score component.

    Uses (if available):
      - digital_txn_cnt_12m
      - txn_cnt_12m
      - recency_days

    All are treated as optional; missing -> 0 or default.
    """
    digital = row.get("digital_txn_cnt_12m", 0.0) or 0.0
    txn_cnt = row.get("txn_cnt_12m", 0.0) or 0.0
    recency_days = row.get("recency_days", 365.0) or 365.0

    digital_component = min(1.0, digital / 50.0)   # cap at 1
    txn_component = min(1.0, txn_cnt / 100.0)
    recency_penalty = max(0.0, (recency_days - 90.0) / 365.0)

    boost = 0.05 * digital_component + 0.03 * txn_component - 0.02 * recency_penalty
    return float(boost)


def compute_affinity_boost(row: pd.Series) -> float:
    """
    Affinity / CLTV component.

    Uses (if available):
      - product_family
      - top_family_for_customer
      - cltv_product_score_norm

    All are optional; missing -> 0 boost.
    """
    product_family = row.get("product_family")
    top_family = row.get("top_family_for_customer")
    cltv_score = row.get("cltv_product_score_norm", 0.0) or 0.0

    family_boost = 0.05 if (product_family is not None and product_family == top_family) else 0.0
    cltv_boost = 0.05 * float(cltv_score)

    return float(family_boost + cltv_boost)


def apply_eligibility_mask(df: pd.DataFrame) -> pd.Series:
    """
    Basic eligibility filter.

    Uses (if present):
      - already_holds_product (bool)
      - risk_bucket (string, e.g. 'Very High' to exclude)

    If the columns are missing, defaults to all True (everyone eligible).
    """
    if "already_holds_product" in df.columns:
        already_holds = df["already_holds_product"].fillna(False).astype(bool)
    else:
        already_holds = pd.Series(False, index=df.index)

    if "risk_bucket" in df.columns:
        risk_bucket = df["risk_bucket"].fillna("")
    else:
        risk_bucket = pd.Series("", index=df.index)

    mask = (~already_holds) & (risk_bucket != "Very High")
    return mask


def add_nbp_score(
    df: pd.DataFrame,
    base_score_col: str = "lead_score",
    score_col_out: str = "nbp_score",
) -> pd.DataFrame:
    """
    Adds a composite nbp_score column by combining:

      nbp_score = base_score + behavioural + affinity

    Assumes df is customer×product long format (one row per candidate).
    """
    out = df.copy()

    base_score = out.get(base_score_col)
    if base_score is None:
        raise KeyError(f"Base score column '{base_score_col}' not found in DataFrame")

    base_score = base_score.fillna(0.0).astype(float)

    behavioural = out.apply(compute_behavioral_boost, axis=1)
    affinity = out.apply(compute_affinity_boost, axis=1)

    out[score_col_out] = base_score + behavioural + affinity

    return out


def add_nbp_recommendations(
    df: pd.DataFrame,
    customer_col: str = "cust_id",
    product_col: str = "product_id",
    base_score_col: str = "lead_score",
    max_k: int = 3,
) -> pd.DataFrame:
    """
    Given a customer×product scored dataset, compute NBP1..NBP3 per customer.

    Returns a wide (one row per customer) DataFrame with columns:
      nbp1, nbp1_score, nbp2, nbp2_score, nbp3, nbp3_score
    """
    # 1) Eligibility
    eligible_mask = apply_eligibility_mask(df)
    df_eligible = df[eligible_mask].copy()

    # 2) Add composite NBP score
    df_scored = add_nbp_score(
        df_eligible,
        base_score_col=base_score_col,
        score_col_out="nbp_score",
    )

    # 3) Rank per customer
    df_scored.sort_values(
        [customer_col, "nbp_score"],
        ascending=[True, False],
        inplace=True,
    )

    df_scored["nbp_rank"] = df_scored.groupby(customer_col)["nbp_score"].rank(
        method="first",
        ascending=False,
    )

    df_topk = df_scored[df_scored["nbp_rank"] <= max_k].copy()

    # 4) Pivot to wide format
    wide_products = (
        df_topk
        .pivot_table(
            index=customer_col,
            columns="nbp_rank",
            values=product_col,
            aggfunc="first",
        )
        .rename(columns=lambda r: f"nbp{int(r)}")
    )

    wide_scores = (
        df_topk
        .pivot_table(
            index=customer_col,
            columns="nbp_rank",
            values="nbp_score",
            aggfunc="first",
        )
        .rename(columns=lambda r: f"nbp{int(r)}_score")
    )

    wide = wide_products.join(wide_scores, how="outer").reset_index()

    return wide

