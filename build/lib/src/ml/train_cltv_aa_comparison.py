import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_and_auc(df, feature_cols, target):
    X = df[feature_cols].fillna(0)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)

def main():
    df_base = pd.read_csv("data/processed/modeling_dataset_v2.csv")
    df_aa = pd.read_csv("data/processed/modeling_dataset_v2_aa.csv")

    feature_cols_base = [
        "non_interest_income",
        "interest_income",
        "acct_cnt",
        "txn_cnt_12m",
        "avg_txn_amt_12m",
        "recency_days",
        "avg_bal_12m_savings",
        "avg_bal_12m_current_ac",
        "avg_bal_12m_td",
    ]

    feature_cols_with_aa = feature_cols_base + [
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

    auc_base = train_and_auc(df_base, feature_cols_base, target)
    auc_aa = train_and_auc(df_aa, feature_cols_with_aa, target)

    print("AUC_base_without_AA", round(auc_base, 4))
    print("AUC_with_AA_features", round(auc_aa, 4))

if __name__ == "__main__":
    main()
