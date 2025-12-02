import pandas as pd
import sqlalchemy as sa
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

engine = sa.create_engine("postgresql://postgres:postgres@localhost:5432/bank_dw")

# Load unified dataset from Postgres
df = pd.read_sql("SELECT * FROM processed.modeling_dataset_v3;", engine)

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

reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)

reg.fit(X, y_cltv)
clf.fit(X, y_high_cltv)

df["pred_cltv_profit_v3"] = reg.predict(X)
df["pred_high_cltv_prob_v3"] = clf.predict_proba(X)[:, 1]

# Example hybrid master lead score using AA + other components
if {"engagement_score", "intent_score", "risk_score"}.issubset(df.columns):
    df["master_lead_score_v3"] = (
        0.4 * df["pred_high_cltv_prob_v3"].fillna(0)
        + 0.3 * df["engagement_score"].fillna(0)
        + 0.2 * df["intent_score"].fillna(0)
        + 0.1 * (1 - df["risk_score"].fillna(0))
    )
else:
    # Fallback: use only AA-based CLTV probability as the master score
    df["master_lead_score_v3"] = df["pred_high_cltv_prob_v3"].fillna(0)

# Overwrite the unified table with predictions added
df.to_sql(
    "modeling_dataset_v3",
    engine,
    schema="processed",
    if_exists="replace",
    index=False,
)

print("Updated processed.modeling_dataset_v3 with prediction columns.")
