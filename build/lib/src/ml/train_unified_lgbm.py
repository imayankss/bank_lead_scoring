from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.common.config import PROJECT_ROOT
from src.ml.unified_features import (
    load_unified_leads_dataset,
    UNIFIED_LEADS_PATH,
)


MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

OUTPUT_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "unified_lead_scores.csv"
FEATURES_JSON_PATH = MODELS_DIR / "unified_lgbm_features.json"
MODEL_PATH = MODELS_DIR / "unified_lgbm_cls.pkl"


def train_unified_lgbm() -> None:
    # Load features + target
    X, y = load_unified_leads_dataset()
    X_np = X.to_numpy(float)
    y_np = y.to_numpy(int)

    print("Training unified LGBM on:")
    print("  X:", X_np.shape, "y:", y_np.shape)

    # Basic CV to get a feel for performance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_np))

    base = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y_np), start=1):
        m = LGBMClassifier(**base.get_params())
        m.fit(X_np[tr_idx], y_np[tr_idx])
        oof[va_idx] = m.predict_proba(X_np[va_idx])[:, 1]
        print(f"  [fold {fold}] done")

    roc = float(roc_auc_score(y_np, oof))
    pr = float(average_precision_score(y_np, oof))
    print({"cv_roc": roc, "cv_pr": pr})

    # Fit final model on full data
    base.fit(X_np, y_np)
    joblib.dump(base, MODEL_PATH)
    print(f"[OK] Saved unified model -> {MODEL_PATH}")

    # Save feature names
    X.columns.to_series().to_json(FEATURES_JSON_PATH)
    print(f"[OK] Saved feature list -> {FEATURES_JSON_PATH}")

    # Compute scores for all leads and export a CSV
        # Compute scores for all leads and export a CSV
    # Include both Customer_ID and cust_id so we can bridge to existing tables
    df_ids = pd.read_csv(UNIFIED_LEADS_PATH)[["cust_id", "Customer_ID", "lead_id"]]
    proba = base.predict_proba(X_np)[:, 1]

    out = df_ids.copy()
    out["ml_unified_proba"] = proba
    out.to_csv(OUTPUT_SCORES_PATH, index=False)
    print(f"[OK] Saved unified lead scores -> {OUTPUT_SCORES_PATH}")


if __name__ == "__main__":
    train_unified_lgbm()
