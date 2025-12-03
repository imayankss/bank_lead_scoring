from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.common.config import PROJECT_ROOT
from src.ml.unified_features import (
    load_unified_leads_dataset,
    UNIFIED_LEADS_PATH,
)
from src.ml.evaluation import evaluate_binary_classifier, ClassificationMetrics


MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

OUTPUT_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "unified_lead_scores.csv"
FEATURES_JSON_PATH = MODELS_DIR / "unified_lgbm_features.json"
MODEL_PATH = MODELS_DIR / "unified_lgbm_cls.pkl"

# Log file where each run appends its result
MODEL_SELECT_LOG_PATH = MODELS_DIR / "model_select_results.txt"


def _log_model_performance(
    *,
    model_name: str,
    metrics: ClassificationMetrics,
    cm: np.ndarray,
    pr_auc: float,
    log_path: Path,
) -> None:
    """
    Append the performance summary to a text file with timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("============================================================\n")
        f.write(f"Run timestamp      : {timestamp}\n")
        f.write(f"Model              : {model_name}\n")
        f.write(f"Threshold          : {metrics.threshold:.2f}\n")
        f.write(f"Accuracy           : {metrics.accuracy * 100:6.2f}%\n")
        f.write(f"AUC-ROC            : {metrics.auc_roc:6.3f}\n")
        f.write(f"Average Precision  : {pr_auc:6.3f} (PR-AUC)\n")
        f.write(f"Precision (pos=1)  : {metrics.precision * 100:6.2f}%\n")
        f.write(f"Recall   (pos=1)   : {metrics.recall * 100:6.2f}%\n")
        f.write("\nConfusion Matrix (rows=true, cols=pred)\n")
        f.write("             Pred 0    Pred 1\n")
        f.write(f"True 0   : {cm[0, 0]:7d} {cm[0, 1]:8d}\n")
        f.write(f"True 1   : {cm[1, 0]:7d} {cm[1, 1]:8d}\n")
        f.write("============================================================\n")


def train_unified_lgbm(threshold: float = 0.5) -> None:
    """
    Train unified LGBM model with 5-fold Stratified CV, print and log key metrics,
    fit final model on full data, and export lead scores.

    threshold : probability cut-off used for Accuracy / Precision / Recall.
    """
    model_name = "unified_lgbm_cls"

    # -------------------------------------------------------------------------
    # 1. Load features + target
    # -------------------------------------------------------------------------
    X, y = load_unified_leads_dataset()
    X_np = X.to_numpy(float)
    y_np = y.to_numpy(int)

    print("Training unified LGBM on:")
    print("  X:", X_np.shape, "y:", y_np.shape)

    # -------------------------------------------------------------------------
    # 2. 5-fold Stratified CV to get out-of-fold probabilities
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 3. Global CV metrics (AUC, PR-AUC, Accuracy, Precision, Recall, CM)
    # -------------------------------------------------------------------------
    pr_auc = float(average_precision_score(y_np, oof))

    metrics, cm = evaluate_binary_classifier(
        y_true=y_np,
        y_proba=oof,
        threshold=threshold,
        model_name=model_name,
        plot_roc=True,  # Set False if you don't want the ROC plot in CLI runs
        title_prefix=f"{model_name} – 5-fold OOF",
    )

    print("\n=== Cross-validated metrics (out-of-fold) ===")
    print(f"AUC-ROC          : {metrics.auc_roc:6.3f}")
    print(f"Average Precision: {pr_auc:6.3f} (PR-AUC)")
    print(f"Accuracy         : {metrics.accuracy * 100:6.2f}%")
    print(f"Precision        : {metrics.precision * 100:6.2f}%")
    print(f"Recall           : {metrics.recall * 100:6.2f}%")
    print(f"Threshold        : {metrics.threshold:6.2f}")

    # -------------------------------------------------------------------------
    # 4. Log to model_select_results.txt (append, with timestamp)
    # -------------------------------------------------------------------------
    _log_model_performance(
        model_name=model_name,
        metrics=metrics,
        cm=cm,
        pr_auc=pr_auc,
        log_path=MODEL_SELECT_LOG_PATH,
    )
    print(f"\n[OK] Appended metrics to -> {MODEL_SELECT_LOG_PATH}")

    # -------------------------------------------------------------------------
    # 5. Fit final model on full data
    # -------------------------------------------------------------------------
    base.fit(X_np, y_np)
    joblib.dump(base, MODEL_PATH)
    print(f"[OK] Saved unified model -> {MODEL_PATH}")

    # Save feature names
    X.columns.to_series().to_json(FEATURES_JSON_PATH)
    print(f"[OK] Saved feature list -> {FEATURES_JSON_PATH}")

    # -------------------------------------------------------------------------
    # 6. Compute scores for all leads and export a CSV
    #    Include both Customer_ID and cust_id so we can bridge to existing tables
    # -------------------------------------------------------------------------
    df_ids = pd.read_csv(UNIFIED_LEADS_PATH)[["cust_id", "Customer_ID", "lead_id"]]
    proba = base.predict_proba(X_np)[:, 1]

    out = df_ids.copy()
    out["ml_unified_proba"] = proba
    out.to_csv(OUTPUT_SCORES_PATH, index=False)
    print(f"[OK] Saved unified lead scores -> {OUTPUT_SCORES_PATH}")


if __name__ == "__main__":
    train_unified_lgbm()
