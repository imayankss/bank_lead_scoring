import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error

# Load unified dataset (core + AA)
df = pd.read_csv("data/processed/modeling_dataset_v3.csv")

# Core features
feature_cols_core = [
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

# AA features
feature_cols_aa = [
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

feature_cols_v3 = feature_cols_core + feature_cols_aa

X = df[feature_cols_v3].fillna(0)
y_cltv = df["cltv_profit"]
y_high_cltv = df["y_high_cltv"]

# ---- CLTV regression ----
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_cltv, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = mse ** 0.5
print("CLTV RMSE (v3):", round(rmse, 2))

# ---- Lead scoring (high CLTV) ----
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_high_cltv, test_size=0.2, random_state=42, stratify=y_high_cltv
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train_c, y_train_c)
y_pred_prob = clf.predict_proba(X_test_c)[:, 1]
auc = roc_auc_score(y_test_c, y_pred_prob)
print("Lead Scoring AUC (v3):", round(auc, 4))
