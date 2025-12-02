import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    df = pd.read_csv("data/processed/modeling_dataset_v2_aa.csv")

    feature_cols_with_aa = [
        "non_interest_income",
        "interest_income",
        "acct_cnt",
        "txn_cnt_12m",
        "avg_txn_amt_12m",
        "recency_days",
        "avg_bal_12m_savings",
        "avg_bal_12m_current_ac",
        "avg_bal_12m_td",
        "num_accounts",
        "num_bank_relationships",
        "num_anchor_accounts",
        "num_comp_accounts",
        "total_balance_all_banks",
        "anchor_balance",
        "competitor_balance",
        "total_credit_limit",
        "txn_count_total",
        "total_inflows",
        "total_outflows",
        "avg_txn_amount",
        "txns_last_90d",
        "digital_txn_count",
        "digital_usage_ratio",
        "avg_monthly_inflows",
        "avg_monthly_outflows",
    ]

    target = "y_high_cltv"

    X = df[feature_cols_with_aa].fillna(0)
    y = df[target]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    df["pred_high_cltv_prob_aa"] = clf.predict_proba(X)[:, 1]

    out_path = "data/processed/modeling_dataset_v2_aa_scored.csv"
    df.to_csv(out_path, index=False)
    print("Wrote AA-scored dataset to", out_path)

if __name__ == "__main__":
    main()
