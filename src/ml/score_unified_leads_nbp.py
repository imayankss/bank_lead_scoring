# src/ml/score_unified_leads_nbp.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import PROJECT_ROOT
from src.ml.recommendation import add_nbp_recommendations


# Paths (adjust only if your structure is different)
LEADS_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_leads_unified_age_rules.csv"
LEAD_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "unified_lead_scores.csv"
DIM_PRODUCTS_PATH = PROJECT_ROOT / "data" / "raw" / "dim_product_boi_2025.csv"

# Output: one row per customer with nbp1/2/3
NBP_CUSTOMERS_PATH = PROJECT_ROOT / "data" / "processed" / "unified_nbp_customers.csv"


def build_nbp_for_customers() -> None:
    """
    Build NBP1/2/3 per customer using:
      - unified lead scores (ml_unified_proba)
      - lead-level dataset (cust_id, product_id, age_rule_*_family, etc.)
      - product dimension (product_name, product_family, risk_bucket, secured_flag)

    Output: data/processed/unified_nbp_customers.csv with columns like:
      cust_id, Customer_ID,
      nbp1, nbp1_score, nbp1_name,
      nbp2_name, nbp3_name,
      age_rule_nbp1_family, age_rule_nbp2_family, age_rule_nbp3_family
    """
    print(f"[INFO] Loading leads dataset -> {LEADS_PATH}")
    leads = pd.read_csv(LEADS_PATH)

    print(f"[INFO] Loading unified lead scores -> {LEAD_SCORES_PATH}")
    scores = pd.read_csv(LEAD_SCORES_PATH)

    print(f"[INFO] Loading product dimension -> {DIM_PRODUCTS_PATH}")
    dim_prod = pd.read_csv(DIM_PRODUCTS_PATH)

    # 1) Merge lead scores into the leads dataset
    leads_scored = leads.merge(
        scores[["lead_id", "ml_unified_proba"]],
        on="lead_id",
        how="left",
    )

    # 2) Attach product-level info (product_name, product_family, risk_bucket, secured_flag)
    leads_scored = leads_scored.merge(
        dim_prod[["product_id", "product_name", "product_family", "risk_bucket", "secured_flag"]],
        on="product_id",
        how="left",
    )

    # 3) Basic eligibility placeholder
    leads_scored["already_holds_product"] = False

    # 4) Compute NBPs per customer (this will usually only give NBP1 for now)
    nbp_wide = add_nbp_recommendations(
        df=leads_scored,
        customer_col="cust_id",
        product_col="product_id",
        base_score_col="ml_unified_proba",
        max_k=3,
    )

    # 5) Add Customer_ID (human-readable) to the NBP output
    cust_index = (
        leads[["cust_id", "Customer_ID"]]
        .drop_duplicates(subset=["cust_id"])
    )

    nbp_with_ids = cust_index.merge(nbp_wide, on="cust_id", how="right")

    # 6) Bring in age-rule families per customer (from the enriched leads dataset)
    age_rule_cols = [
        "age_rule_nbp1_family",
        "age_rule_nbp2_family",
        "age_rule_nbp3_family",
    ]
    existing_age_rule_cols = [c for c in age_rule_cols if c in leads.columns]

    if existing_age_rule_cols:
        age_rules_per_cust = (
            leads[["cust_id"] + existing_age_rule_cols]
            .drop_duplicates(subset=["cust_id"])
        )
        nbp_with_ids = nbp_with_ids.merge(
            age_rules_per_cust,
            on="cust_id",
            how="left",
        )

    # 7) Map product_id -> product_name for NBP1..3 (where ranks exist)
    id_to_name = dim_prod.set_index("product_id")["product_name"].to_dict()
    for k in [1, 2, 3]:
        id_col = f"nbp{k}"
        name_col = f"nbp{k}_name"
        if id_col in nbp_with_ids.columns:
            nbp_with_ids[name_col] = nbp_with_ids[id_col].map(id_to_name)

    # 8) Fill NBP2 / NBP3 *names* from age rules where we don't have ML-based ranks
    if "nbp2_name" in nbp_with_ids.columns and "age_rule_nbp2_family" in nbp_with_ids.columns:
        nbp_with_ids["nbp2_name"] = nbp_with_ids["nbp2_name"].fillna(
            nbp_with_ids["age_rule_nbp2_family"]
        )
    if "nbp3_name" in nbp_with_ids.columns and "age_rule_nbp3_family" in nbp_with_ids.columns:
        nbp_with_ids["nbp3_name"] = nbp_with_ids["nbp3_name"].fillna(
            nbp_with_ids["age_rule_nbp3_family"]
        )

    # NOTE: NBP2/3 product_ids (nbp2, nbp3) will usually stay NaN,
    # but the dashboard will use the *_name columns for display.

    # 9) Save result
    NBP_CUSTOMERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbp_with_ids.to_csv(NBP_CUSTOMERS_PATH, index=False)

    print(f"[OK] Saved customer NBP recommendations -> {NBP_CUSTOMERS_PATH}")
    print("Rows (customers):", nbp_with_ids.shape[0])


if __name__ == "__main__":
    build_nbp_for_customers()