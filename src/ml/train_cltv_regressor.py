from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.common.config import PROJECT_ROOT


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_cltv_regression.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "cltv_regressor_lgbm.pkl"
FEATURES_JSON_PATH = MODELS_DIR / "cltv_regressor_features.json"


def train_cltv_regressor() -> None:
    # 1) Load regression dataset
    df = pd.read_csv(DATA_PATH)
    print("CLTV regression dataset shape:", df.shape)

    # y = cltv_profit, X = all other numeric features
    y = df["cltv_profit"].to_numpy(float)
    feature_cols = [c for c in df.columns if c not in ("cust_id", "cltv_profit")]
    X = df[feature_cols].copy().fillna(0.0)
    X_np = X.to_numpy(float)

    print("Number of features:", len(feature_cols))
    print("First 10 features:", feature_cols[:10])

    # 2) Simple train/test split for sanity check
    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # 3) Basic evaluation on validation set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)   # older sklearn: no 'squared' arg
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_val, y_pred))
    print({"val_rmse": rmse, "val_r2": r2})

    # 4) Refit on full data
    model.fit(X_np, y)

    # 5) Save model and feature list
    joblib.dump(model, MODEL_PATH)
    print(f"[OK] Saved CLTV regressor -> {MODEL_PATH}")

    pd.Series(index=feature_cols, data=np.arange(len(feature_cols))).to_json(
        FEATURES_JSON_PATH
    )
    print(f"[OK] Saved CLTV feature list -> {FEATURES_JSON_PATH}")


if __name__ == "__main__":
    train_cltv_regressor()

