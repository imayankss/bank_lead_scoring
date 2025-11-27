# src/rules/rule_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set

import yaml
import math


@dataclass
class CustomerSnapshot:
    customer_id: str
    age: int
    monthly_income: float
    credit_score: int
    risk_bucket: str               # e.g. "A"..."E"
    city_tier: int
    total_emi: float               # all existing EMIs
    unsecured_outstanding: float
    current_product_families: List[str]
    relationship_months: int
    max_cc_utilization: Optional[float] = None
    has_fd_lien_50k: bool = False
    recent_delinquency_days: int = 0
    # add other attributes as needed


def load_rules(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _check_range(val: float, min_v: Optional[float], max_v: Optional[float]) -> bool:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return False
    if min_v is not None and val < min_v:
        return False
    if max_v is not None and val > max_v:
        return False
    return True


def is_customer_product_eligible(
    customer: CustomerSnapshot,
    product_code: str,
    rules_cfg: Dict[str, Any],
) -> bool:
    global_cfg = rules_cfg.get("global", {})
    product_cfg = rules_cfg["products"][product_code]
    c = product_cfg["constraints"]

    # 1) Age (global + product-level)
    if not _check_range(customer.age, global_cfg.get("min_age"), global_cfg.get("max_age")):
        return False
    if "age" in c and not _check_range(customer.age, c["age"].get("min"), c["age"].get("max")):
        return False

    # 2) Income
    if "income_monthly" in c:
        if customer.monthly_income < c["income_monthly"]["min"]:
            # unless product has a secured-FD override (for credit cards etc.)
            if not (product_code == "CREDIT_CARD" and
                    customer.has_fd_lien_50k and
                    customer.monthly_income >= 0):
                return False

    # 3) DTI / EMI-to-income
    overall_dti = customer.total_emi / max(customer.monthly_income, 1.0)
    if "max_overall_dti" in global_cfg and overall_dti > global_cfg["max_overall_dti"]:
        return False
    if "dti_ratio_max" in c and overall_dti > c["dti_ratio_max"]:
        return False

    # 4) Unsecured exposure
    if "max_unsecured_exposure_to_income" in c:
        ratio = customer.unsecured_outstanding / max(customer.monthly_income * 12, 1.0)
        if ratio > c["max_unsecured_exposure_to_income"]:
            return False

    # 5) Credit score and risk bucket
    if "credit_score" in c and customer.credit_score < c["credit_score"]["min"]:
        # again allow override for secured FD card
        if not (product_code == "CREDIT_CARD" and customer.has_fd_lien_50k):
            return False

    allowed_rb = c.get("risk_bucket_allowed")
    if allowed_rb and customer.risk_bucket not in allowed_rb:
        return False

    # 6) Relationship vintage
    if customer.relationship_months < c.get("min_relationship_months", 0):
        return False

    # 7) Geography filters
    if "geography_whitelist_tiers" in c and customer.city_tier not in c["geography_whitelist_tiers"]:
        return False
    if "geography_blacklist_tiers" in c and customer.city_tier in c["geography_blacklist_tiers"]:
        return False

    # 8) Product holdings rules
    holdings: Set[str] = set(customer.current_product_families or [])
    must_any = c.get("must_have_products_any_of", [])
    if must_any and not holdings.intersection(must_any):
        return False

    exclude_any = c.get("exclude_if_existing_products_any_of", [])
    if exclude_any and holdings.intersection(exclude_any):
        return False

    # 9) Credit card utilization
    if product_code == "CREDIT_CARD" and customer.max_cc_utilization is not None:
        if "max_cc_utilization" in c and customer.max_cc_utilization > c["max_cc_utilization"]:
            return False

    # 10) Delinquency rules
    if "max_recent_delinquency_days" in c:
        if customer.recent_delinquency_days > c["max_recent_delinquency_days"]:
            return False

    # you can add more specific checks per product family here

    return True


def get_eligible_products_for_customer(
    customer: CustomerSnapshot,
    rules_cfg: Dict[str, Any],
) -> List[str]:
    product_codes = list(rules_cfg["products"].keys())
    eligible = []
    for p in product_codes:
        if is_customer_product_eligible(customer, p, rules_cfg):
            eligible.append(p)

    # Optionally sort by product priority_score
    products_cfg = rules_cfg["products"]
    eligible.sort(key=lambda p: products_cfg[p].get("priority_score", 0), reverse=True)
    return eligible
