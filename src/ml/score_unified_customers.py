from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "unified_lgbm_cls.pkl"
FEATURES_JSON_PATH = MODELS_DIR / "unified_lgbm_features.json"

CUSTOMERS_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_customers_universe.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "unified_customer_scores.csv"


def score_customers_universe() -> None:
    # 1) Load trained unified model
    model = joblib.load(MODEL_PATH)

    # 2) Load the feature names used at training time
    #    We saved them from X.columns in train_unified_lgbm.py
    features = pd.read_json(FEATURES_JSON_PATH, typ="series")
    feature_names = list(features.index)
    print("Model trained with", len(feature_names), "features")
    print("First 10 feature names:", feature_names[:10])

    # 3) Load customer-universe dataset (1000 customers)
    df = pd.read_csv(CUSTOMERS_PATH)
    print("Customer universe shape:", df.shape)

    # 4) Ensure all expected features exist; if missing, create them with 0
    present = [c for c in feature_names if c in df.columns]
    missing = [c for c in feature_names if c not in df.columns]
    print("Features present in customer_universe:", len(present))
    print("Features missing (will be filled with 0):", len(missing))

    for col in missing:
        df[col] = 0.0  # neutral default for lead-only fields

    # 5) Build feature matrix in the exact same column order as training
    X = df[feature_names].copy().fillna(0.0)
    X_np = X.to_numpy(float)

    # 6) Predict probabilities for all customers
    proba = model.predict_proba(X_np)[:, 1]

    # 7) Build output: one row per customer with a stable key
    out = df[["cust_id"]].copy()
    out["ml_unified_customer_proba"] = proba

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Saved unified customer scores -> {OUTPUT_PATH}")
    print("Rows:", out.shape[0])


if __name__ == "__main__":
    score_customers_universe()
