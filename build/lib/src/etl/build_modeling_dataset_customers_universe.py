from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.common.config import settings


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def project_root_from_this_file() -> Path:
    """
    This script lives in src/etl/.
    parents[0] = src/etl
    parents[1] = src
    parents[2] = project root
    """
    return Path(__file__).resolve().parents[2]


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def to_cust_id(x: str) -> str:
    s = str(x).strip()
    if s.startswith("CUST"):
        num = s[4:]
        try:
            n = int(num)
            return f"C{n:05d}"
        except ValueError:
            return s
    return s


# ---------------------------------------------------------
# Aggregators
# ---------------------------------------------------------
def aggregate_cbs_accounts(cbs_acct: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate core CBS account metrics per customer.
    Assumes columns: cust_id, acid, sanct_lim, clr_bal_amt, is_loan_account, is_overdue, is_npa, dpd_days
    """
    df = cbs_acct.copy()
    df["cust_id"] = df["cust_id"].astype(str)

    grp = (
        df.groupby("cust_id")
        .agg(
            cbs_num_accounts=("acid", "nunique"),
            cbs_total_sanct_lim=("sanct_lim", "sum"),
            cbs_total_clr_bal_amt=("clr_bal_amt", "sum"),
            cbs_loan_acct_cnt=("is_loan_account", "sum") if "is_loan_account" in df.columns else ("acid", "count"),
            cbs_any_overdue=("is_overdue", "max") if "is_overdue" in df.columns else ("acid", "size"),
            cbs_any_npa=("is_npa", "max") if "is_npa" in df.columns else ("acid", "size"),
            cbs_max_dpd_days=("dpd_days", "max") if "dpd_days" in df.columns else ("acid", "size"),
        )
        .reset_index()
    )
    return grp


def aggregate_cbs_txn(cbs_txn: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CBS transaction behavior per customer.
    Assumes: cust_id, tran_amt, tran_date, tran_particular.
    """
    df = cbs_txn.copy()
    df["cust_id"] = df["cust_id"].astype(str)

    if "tran_date" not in df.columns:
        return pd.DataFrame({"cust_id": df["cust_id"].unique()})

    df["tran_date"] = pd.to_datetime(df["tran_date"], errors="coerce")
    max_date = df["tran_date"].max()
    df["days_from_max"] = (max_date - df["tran_date"]).dt.days

    base = pd.DataFrame({"cust_id": df["cust_id"].unique()}).set_index("cust_id")

    for window in (30, 90, 365):
        mask = df["days_from_max"] <= window
        sub = df[mask]
        grp = sub.groupby("cust_id").agg(
            **{
                f"cbs_txn_cnt_{window}d": ("tran_amt", "count"),
                f"cbs_txn_sum_{window}d": ("tran_amt", "sum"),
                f"cbs_txn_avg_{window}d": ("tran_amt", "mean"),
            }
        )
        base = base.join(grp, how="left")

    last_txn = df.groupby("cust_id")["tran_date"].max().to_frame("cbs_last_txn_date")
    base = base.join(last_txn, how="left")
    base["cbs_txn_recency_days"] = (max_date - base["cbs_last_txn_date"]).dt.days

    if "tran_particular" in df.columns:
        emi_mask = (df["days_from_max"] <= 365) & df["tran_particular"].str.contains("EMI", na=False)
        emi_grp = (
            df[emi_mask]
            .groupby("cust_id")["tran_amt"]
            .count()
            .to_frame("cbs_emi_txn_cnt_365d")
        )
        base = base.join(emi_grp, how="left")

    return base.reset_index()


def aggregate_product_ownership(prod_own: pd.DataFrame, dim_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer product ownership features:
      - num_products, num_distinct_products
      - own_cnt_prod_P001..P005
      - own_cnt_family_DEPOSIT_CURRENT etc.
      - simple labels: has_any_product, has_loan_product
    """
    df = prod_own.copy()
    # Ensure no duplicate column names (e.g., multiple 'product_id' columns)
    df = df.loc[:, ~df.columns.duplicated()]
    df["cust_id"] = df["cust_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)

    # Basic counts
    agg = (
        df.groupby("cust_id")
        .agg(
            num_products=("product_id", "count"),
            num_distinct_products=("product_id", "nunique"),
        )
        .reset_index()
    )

    # Counts by product_id via crosstab (avoids grouper issues)
    pivot_pid = (
        pd.crosstab(df["cust_id"], df["product_id"])
        .add_prefix("own_cnt_prod_")
        .reset_index()
    )

    # Counts by product_family
    dim_min = dim_prod[["product_id", "product_family"]].copy()
    dim_min["product_id"] = dim_min["product_id"].astype(str)

    df2 = df.merge(dim_min, on="product_id", how="left")
    pivot_fam = (
        df2.pivot_table(
            index="cust_id",
            columns="product_family",
            values="product_id",
            aggfunc="count",
            fill_value=0,
        )
        .add_prefix("own_cnt_family_")
        .reset_index()
    )

    out = agg.merge(pivot_pid, on="cust_id", how="left").merge(pivot_fam, on="cust_id", how="left")

    # Simple labels
    out["has_any_product"] = (out["num_products"].fillna(0) > 0).astype(int)
    # Loan = TERM_LOAN or any P003
    loan_cols = [c for c in out.columns if c.startswith("own_cnt_prod_P003")] + \
                [c for c in out.columns if c.startswith("own_cnt_family_TERM_LOAN")]
    if loan_cols:
        out["has_loan_product"] = (out[loan_cols].fillna(0).sum(axis=1) > 0).astype(int)
    else:
        out["has_loan_product"] = 0

    return out


def aggregate_leads_for_customers(leads: pd.DataFrame, lead_prod_map: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer lead features:
      - num_leads, num_distinct_lead_products
      - first/last lead date, lead_recency_days
      - lead_cnt_prod_P001..P005
    """
    df = leads.copy()

    # Ensure cust_id
    if "cust_id" not in df.columns:
        if "Customer_ID" in df.columns:
            df["cust_id"] = df["Customer_ID"].apply(to_cust_id)
        else:
            # No lead-level customer; return empty frame
            return pd.DataFrame(columns=["cust_id"])
    df["cust_id"] = df["cust_id"].astype(str)

    # Lead dates
    if "Date_Of_Lead" in df.columns:
        df["lead_date"] = pd.to_datetime(df["Date_Of_Lead"], errors="coerce")
        max_date = df["lead_date"].max()
    else:
        df["lead_date"] = pd.NaT
        max_date = None

    # Map lead product → product_id using the same mapping dim
    if "Chosen_Product" in df.columns:
        product_choice_col = "Chosen_Product"
    elif "Best_First_Option" in df.columns:
        product_choice_col = "Best_First_Option"
    else:
        product_choice_col = None

    if product_choice_col is not None:
        df["lead_product_name_norm"] = df[product_choice_col].astype(str).str.strip().str.upper()
        if "lead_product_name_norm" not in lead_prod_map.columns:
            lead_prod_map["lead_product_name_norm"] = (
                lead_prod_map["lead_product_name"].astype(str).str.strip().str.upper()
            )
        df = df.merge(
            lead_prod_map[["lead_product_name_norm", "product_id"]],
            on="lead_product_name_norm",
            how="left",
        )
        df["product_id"] = df["product_id"].astype(str)

    # Aggregate per customer
    base = (
        df.groupby("cust_id")
        .agg(
            num_leads=("Customer_ID", "count"),
            num_distinct_lead_products=("product_id", "nunique"),
            first_lead_date=("lead_date", "min"),
            last_lead_date=("lead_date", "max"),
        )
        .reset_index()
    )

    if max_date is not None:
        base["lead_recency_days"] = (max_date - base["last_lead_date"]).dt.days

    # Counts by product_id for leads
    if "product_id" in df.columns:
        piv = (
            df.pivot_table(
                index="cust_id",
                columns="product_id",
                values="Customer_ID",
                aggfunc="count",
                fill_value=0,
            )
            .add_prefix("lead_cnt_prod_")
            .reset_index()
        )
        base = base.merge(piv, on="cust_id", how="left")

    return base


def aggregate_aa_features(aa_mart: pd.DataFrame, aa_map: pd.DataFrame) -> pd.DataFrame:
    """
    AA 360 features per cust_id for ANCHOR bank.
    """
    anchor = aa_map[aa_map["bank_code"] == "ANCHOR"].copy()
    if "cust_id" in anchor.columns:
        anchor["cust_id"] = anchor["cust_id"].astype(str)

    aa = anchor.merge(aa_mart, on="aa_customer_id", how="left")

    keep_cols = [
        "aa_customer_id",
        "cust_id",
        "total_balance_all_banks",
        "anchor_balance",
        "competitor_balance",
        "num_accounts",
        "num_anchor_accounts",
        "num_comp_accounts",
        "txn_count_total",
        "txns_last_90d",
        "avg_txn_amount",
        "digital_usage_ratio",
        "avg_monthly_inflows",
        "avg_monthly_outflows",
    ]
    keep_cols = [c for c in keep_cols if c in aa.columns]
    aa = aa[keep_cols].copy()

    rename_map = {
        "num_accounts": "aa_num_accounts",
        "num_anchor_accounts": "aa_num_anchor_accounts",
        "num_comp_accounts": "aa_num_comp_accounts",
        "txn_count_total": "aa_txn_count_total",
        "txns_last_90d": "aa_txn_count_90d",
        "avg_txn_amount": "aa_avg_txn_amount",
        "digital_usage_ratio": "aa_digital_usage_ratio",
        "avg_monthly_inflows": "aa_avg_monthly_inflows",
        "avg_monthly_outflows": "aa_avg_monthly_outflows",
    }
    aa = aa.rename(columns=rename_map)

    aa_agg = (
        aa.groupby("cust_id")
        .agg({c: "mean" for c in aa.columns if c not in ("cust_id", "aa_customer_id")})
        .reset_index()
    )
    return aa_agg


# ---------------------------------------------------------
# Main builder
# ---------------------------------------------------------
def build_customer_universe_dataset() -> pd.DataFrame:
    root = project_root_from_this_file()
    raw_root = root / "data" / "raw"
    syn_root = raw_root
    aa_root = raw_root

    print("[1/5] Loading raw tables...")

    # Core dims
    cust = load_csv(syn_root / "schema_customer_master.csv")  # 1000 customers
    dim_branch = load_csv(raw_root / "dim_branch.csv")
    dim_prod = load_csv(raw_root / "dim_product_boi_2025.csv")

    # Fact tables
    leads = load_csv(raw_root / "fct_lead.csv")
    prod_own = load_csv(raw_root / "fact_product_ownership.csv")
    acc_car = load_csv(syn_root / "ans_acc_car.csv")
    cbs_acct = load_csv(syn_root / "prod_ods_cbsind_general_acct_mast_table.csv")
    cbs_txn = load_csv(syn_root / "prod_ods_cbsind_hist_tran_dtl_table.csv")

    # Lead mapping dim
    lead_prod_map = load_csv(raw_root / "dim_lead_product_map.csv")

    # AA mart (optional)
    aa_mart_path = aa_root / "mart_customer_360_aa.csv"
    aa_map_path = aa_root / "aa_customer_map.csv"
    aa_mart = load_csv(aa_mart_path) if aa_mart_path.exists() else None
    aa_map = load_csv(aa_map_path) if aa_map_path.exists() else None

    # Ensure IDs are strings
    cust["cust_id"] = cust["cust_id"].astype(str)
    prod_own["cust_id"] = prod_own["cust_id"].astype(str)
    prod_own["product_id"] = prod_own["product_id"].astype(str)
    cbs_acct["cust_id"] = cbs_acct["cust_id"].astype(str)
    cbs_txn["cust_id"] = cbs_txn["cust_id"].astype(str)

    print("[2/5] Aggregating per-customer features...")

    # Product ownership
    prod_agg = aggregate_product_ownership(prod_own, dim_prod)

    # CBS accounts and transactions
    cbs_acct_agg = aggregate_cbs_accounts(cbs_acct)
    cbs_txn_agg = aggregate_cbs_txn(cbs_txn)

    # Lead behavior
    leads_agg = aggregate_leads_for_customers(leads, lead_prod_map)

    # AA features
    if aa_mart is not None and aa_map is not None:
        aa_agg = aggregate_aa_features(aa_mart, aa_map)
    else:
        aa_agg = None
        print("   [WARN] AA mart or aa_customer_map missing; skipping AA features.")

    print("[3/5] Building customer universe base...")

    df = cust.copy()  # 1 row per customer

    # Branch join (keep sol_id for merge, use suffixes for branch columns)
    if "sol_id" in df.columns and "sol_id" in dim_branch.columns:
        df = df.merge(dim_branch, on="sol_id", how="left", suffixes=("", "_branch"))
    else:
        print("   [WARN] No sol_id join to dim_branch; check dim_branch columns.")

    # 12M balances
    df = df.merge(acc_car, on="cust_id", how="left", suffixes=("", "_acc_car"))

    # Product ownership
    df = df.merge(prod_agg, on="cust_id", how="left")

    # CBS accounts
    df = df.merge(cbs_acct_agg, on="cust_id", how="left")

    # CBS transactions
    df = df.merge(cbs_txn_agg, on="cust_id", how="left")

    # Lead behavior (only some customers have leads)
    df = df.merge(leads_agg, on="cust_id", how="left")

    # AA features
    if aa_agg is not None:
        df = df.merge(aa_agg, on="cust_id", how="left")

    print("[4/5] Deriving simple customer-level labels...")

    # Primary labels:
    #  - has_any_product, has_loan_product were created in prod_agg
    df["label_has_any_product"] = df["has_any_product"].fillna(0).astype(int)
    df["label_has_loan_product"] = df["has_loan_product"].fillna(0).astype(int)

    # Has any lead?
    df["label_has_any_lead"] = (df["num_leads"].fillna(0) > 0).astype(int)

    print("[5/5] Finalizing customer universe dataset...")

    # Ensure 1 row per customer
    df = df.drop_duplicates(subset=["cust_id"]).reset_index(drop=True)

    return df


def main() -> None:
    root = project_root_from_this_file()
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = build_customer_universe_dataset()

    out_path = processed_dir / "modeling_dataset_customers_universe.csv"
    df.to_csv(out_path, index=False)

    print("[OK] modeling_dataset_customers_universe.csv written:")
    print(f"     Path : {out_path}")
    print(f"     Rows : {df.shape[0]}")
    print(f"     Cols : {df.shape[1]}")


if __name__ == "__main__":
    main()