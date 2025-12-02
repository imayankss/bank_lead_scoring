from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import mlflow
from src.ml.experiment_tracking import init_mlflow, log_params_from_dict, log_feature_list

from src.common.config import PROJECT_ROOT
from src.common.config import settings  # kept if you later want to swap to DuckDB
from src.ml.features import FEATURE_SETS_V2
from src.ml.validation import (
    time_series_folds,
    evaluate_regression_splits,
    save_eval_report,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_cltv_regression.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "cltv_regressor_lgbm.pkl"
FEATURES_JSON_PATH = MODELS_DIR / "cltv_regressor_features.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
CLTV_EVAL_PATH = REPORTS_DIR / "cltv_eval.json"


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def train_cltv_regressor() -> None:
    # 1) Init MLflow experiment
    init_mlflow("cltv_regression")

    # 2) Load regression dataset
    df = pd.read_csv(DATA_PATH)
    print("CLTV regression dataset shape:", df.shape)

    # y = cltv_profit, X = all other numeric features
    y = df["cltv_profit"].to_numpy(float)
    feature_cols = [c for c in df.columns if c not in ("cust_id", "cltv_profit")]
    X = df[feature_cols].copy().fillna(0.0)
    X_np = X.to_numpy(float)

    print("Number of features:", len(feature_cols))
    print("First 10 features:", feature_cols[:10])

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )

    # 3) Time-invariant simple split (you can later replace with time-based)
    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y, test_size=0.2, random_state=42
    )

    # 4) MLflow run
    with mlflow.start_run(run_name="cltv_lgbm_v1"):
        # Log dataset info + features
        mlflow.log_param("n_samples_total", len(df))
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_val", len(y_val))
        log_feature_list(feature_cols)

        # Log hyperparameters
        log_params_from_dict(model.get_params(), prefix="model")

        # Fit + validation
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_val, y_pred))
        print({"val_rmse": rmse, "val_r2": r2})

        # Log metrics
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_r2", r2)

        # Refit on full data for final model
        model.fit(X_np, y)

        # Save local artifacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"[OK] Saved CLTV regressor -> {MODEL_PATH}")

        pd.Series(index=feature_cols, data=np.arange(len(feature_cols))).to_json(
            FEATURES_JSON_PATH
        )
        print(f"[OK] Saved CLTV feature list -> {FEATURES_JSON_PATH}")

        # Log artifacts to MLflow
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(FEATURES_JSON_PATH), artifact_path="features")



if __name__ == "__main__":
    train_cltv_regressor()




