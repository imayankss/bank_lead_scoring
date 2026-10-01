import pandas as pd
from lightgbm import LGBMClassifier

# 1. Load unified modeling dataset
df = pd.read_csv("data/processed/modeling_dataset_leads_unified_age_rules.csv")

# 2. Recreate X, y exactly as in src/ml/unified_features.py
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

EXCLUDE_NUMERIC = {
    "Phone",              # PII / ID
    "aadhaar_no",         # PII / ID
    "lead_id",
    "source_id",
    "chosen_product_id",
    "best_first_product_id",
    "label_conv",
    "label_conv_90d",
}

feature_cols = [c for c in num_cols if c not in EXCLUDE_NUMERIC]
X = df[feature_cols].fillna(0)
y = df["label_conv"].astype(int)

# 3. Train LightGBM with same base params as train_unified_lgbm.py
model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)
model.fit(X, y)

# 4. Gain-based feature importances
booster = model.booster_
gain = booster.feature_importance(importance_type="gain")
feat_names = booster.feature_name()

imp_df = pd.DataFrame({
    "feature": feat_names,
    "gain": gain
}).sort_values("gain", ascending=False)

# Normalize to percentages
imp_df["gain_pct"] = 100 * imp_df["gain"] / imp_df["gain"].sum()

# Save for report
imp_df.to_csv("data/processed/unified_lgbm_feature_importance_gain.csv", index=False)

# Show top 20
print(imp_df.head(20))
