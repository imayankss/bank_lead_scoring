"""Model training for CLTV and propensity scoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

from lead_scoring.config import PipelineConfig
from lead_scoring.paths import (
    CLTV_MODEL_FILE,
    MODEL_DIR,
    MODEL_METADATA_FILE,
    PREPROCESSED_DATA_DIR,
    PROPENSITY_MODEL_FILE,
    ensure_directories,
)


def _read_target(path: str, column: str) -> pd.Series:
    return pd.read_parquet(PREPROCESSED_DATA_DIR / path)[column]


def train_models(config: PipelineConfig = PipelineConfig()) -> dict[str, float]:
    """Train and persist CLTV and propensity models."""
    ensure_directories()
    X_train = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet")
    y_reg_train = _read_target("y_reg_train.parquet", "future_revenue_12m")
    y_reg_test = _read_target("y_reg_test.parquet", "future_revenue_12m")
    y_clf_train = _read_target("y_clf_train.parquet", "converted_12m").astype(int)
    y_clf_test = _read_target("y_clf_test.parquet", "converted_12m").astype(int)

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
        classifier = DummyClassifier(strategy="prior")
    else:
        classifier = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=config.random_state,
        )
    classifier.fit(X_train, y_clf_train)

    pred_reg = np.clip(regressor.predict(X_test), 0, None)
    pred_prop = classifier.predict_proba(X_test)[:, 1]
    rmse = float(mean_squared_error(y_reg_test, pred_reg) ** 0.5)
    auc = float(roc_auc_score(y_clf_test, pred_prop)) if y_clf_test.nunique() > 1 else float("nan")

    joblib.dump(regressor, CLTV_MODEL_FILE)
    joblib.dump(classifier, PROPENSITY_MODEL_FILE)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "cltv_model": type(regressor.regressor_).__name__,
        "propensity_model": type(classifier).__name__,
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
    print(f"Saved propensity model to {PROPENSITY_MODEL_FILE}")
    print(metrics)


if __name__ == "__main__":
    main()
