"""Model training for CLTV and propensity scoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

from lead_scoring.config import PipelineConfig
from lead_scoring.paths import (
    CLTV_MODEL_FILE,
    MODEL_DIR,
    MODEL_METADATA_FILE,
    PREPROCESSED_DATA_DIR,
    PROPENSITY_LGBM_FILE,
    ensure_directories,
)
from lead_scoring.propensity import load_choice


def _read_target(path: str, column: str) -> pd.Series:
    return pd.read_parquet(PREPROCESSED_DATA_DIR / path)[column]


def train_models(config: PipelineConfig = PipelineConfig()) -> dict[str, float]:
    """Train and persist CLTV and propensity models."""
    ensure_directories()
    X_train = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet")
    y_reg_train = _read_target("y_reg_train.parquet", "future_value")
    y_reg_test = _read_target("y_reg_test.parquet", "future_value")
    y_clf_train = _read_target("y_clf_train.parquet", "converted").astype(int)
    y_clf_test = _read_target("y_clf_test.parquet", "converted").astype(int)

    regressor = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=config.random_state,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    regressor.fit(X_train, y_reg_train)

    if y_clf_train.nunique() < 2:
        raise ValueError("propensity training labels have a single class")
    # Parameters and round count were chosen on the validation month by scripts/validate_propensity.py.
    choice = load_choice()
    booster = lgb.train(choice["lgbm_params"], lgb.Dataset(X_train, y_clf_train), num_boost_round=choice["lgbm_rounds"])
    booster.save_model(str(PROPENSITY_LGBM_FILE))

    pred_reg = np.clip(regressor.predict(X_test), 0, None)
    pred_prop = booster.predict(X_test)
    rmse = float(mean_squared_error(y_reg_test, pred_reg) ** 0.5)
    auc = float(roc_auc_score(y_clf_test, pred_prop)) if y_clf_test.nunique() > 1 else float("nan")

    joblib.dump(regressor, CLTV_MODEL_FILE)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "cltv_model": type(regressor.regressor_).__name__,
        "propensity_model": "LightGBM",
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "cltv_rmse": rmse,
        "propensity_auc": auc,
    }
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"cltv_rmse": rmse, "propensity_auc": auc}


def main() -> None:
    metrics = train_models()
    print(f"Saved CLTV model to {CLTV_MODEL_FILE}")
    print(f"Saved propensity model to {PROPENSITY_LGBM_FILE}")
    print(metrics)


if __name__ == "__main__":
    main()
