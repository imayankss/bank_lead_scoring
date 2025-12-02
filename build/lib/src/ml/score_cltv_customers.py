from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "cltv_regressor_lgbm.pkl"
FEATURES_JSON_PATH = MODELS_DIR / "cltv_regressor_features.json"

CUSTOMERS_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_customers_universe.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cltv_regression_scores.csv"


def score_cltv_for_all_customers() -> None:
    # 1) Load trained CLTV regressor
    model = joblib.load(MODEL_PATH)

    # 2) Load feature names used at training time
    features = pd.read_json(FEATURES_JSON_PATH, typ="series")
    feature_names = list(features.index)
    print("CLTV model trained with", len(feature_names), "features")
    print("First 10 features:", feature_names[:10])

    # 3) Load full customer universe (1000 customers)
    df = pd.read_csv(CUSTOMERS_PATH)
    print("Customer universe shape:", df.shape)

    # 4) Ensure all expected features exist; if missing, create them with 0
    present = [c for c in feature_names if c in df.columns]
    missing = [c for c in feature_names if c not in df.columns]
    print("Features present in customer_universe:", len(present))
    print("Features missing (will be filled with 0):", len(missing))

    for col in missing:
        df[col] = 0.0

    # 5) Build feature matrix in the exact same order as training
    X = df[feature_names].copy().fillna(0.0)
    X_np = X.to_numpy(float)

    # 6) Predict CLTV for all customers
    cltv_pred = model.predict(X_np)

    # 7) Build output with a stable key + predicted CLTV
    out = df[["cust_id"]].copy()
    out["cltv_profit_ml"] = cltv_pred

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Saved ML CLTV scores -> {OUTPUT_PATH}")
    print("Rows:", out.shape[0])


if __name__ == "__main__":
    score_cltv_for_all_customers()
