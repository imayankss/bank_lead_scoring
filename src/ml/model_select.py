# src/ml/model_select.py
from __future__ import annotations

"""
Unified lead-scoring model selection for Bank of India project.

Run this script to:
  - Train several candidate models (Logistic Regression, LGBM, XGBoost,
    HistGradientBoosting) using K-fold out-of-fold evaluation on the
    unified leads dataset.
  - Print `=== Model Performance Summary ===` for EACH model in the terminal.
  - Select a CHAMPION model based on a chosen metric (AUC-ROC by default).
  - Append all metrics (for all models) to `models/model_select_results.txt`,
    clearly marking which one was the champion for that run.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.common.config import PROJECT_ROOT
from src.ml.unified_features import load_unified_leads_dataset
from src.ml.evaluation import evaluate_binary_classifier, ClassificationMetrics


MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Log file for model selection runs
MODEL_SELECT_LOG_PATH = MODELS_DIR / "model_select_results.txt"


@dataclass
class CandidateResult:
    name: str
    metrics: ClassificationMetrics
    pr_auc: float
    confusion_matrix: np.ndarray


def _get_model_factories(random_state: int = 42) -> Dict[str, Callable[[], object]]:
    """
    Define candidate models for unified lead scoring.

    You can adjust hyperparameters here if needed. All models should support
    predict_proba(X) for binary classification.
    """
    return {
        "logreg_baseline": lambda: LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "unified_lgbm": lambda: LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
        ),
        "xgb_cls": lambda: XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gb": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_depth=None,
            min_samples_leaf=20,
            random_state=random_state,
        ),
    }


def _cv_evaluate_model(
    name: str,
    make_model: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    threshold: float,
    random_state: int,
    plot_roc: bool = False,
) -> Tuple[ClassificationMetrics, np.ndarray, float]:
    """
    Run Stratified K-fold CV for a single candidate model and return:
      - metrics from evaluate_binary_classifier on out-of-fold predictions,
      - confusion matrix,
      - PR-AUC (average precision).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)

    print(f"\n================ Candidate: {name} ================")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        clf = make_model()
        clf.fit(X[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X[va_idx])
        if proba.ndim == 2:
            proba = proba[:, 1]
        oof[va_idx] = proba
        print(f"  [fold {fold}] done")

    pr_auc = float(average_precision_score(y, oof))

    # This prints the "=== Model Performance Summary ===" block
    metrics, cm = evaluate_binary_classifier(
        y_true=y,
        y_proba=oof,
        threshold=threshold,
        model_name=name,
        plot_roc=plot_roc,
        title_prefix=f"{name} – {n_splits}-fold OOF",
    )

    print("\n--- Derived metrics (OOF) ---")
    print(f"AUC-ROC          : {metrics.auc_roc:6.3f}")
    print(f"Average Precision: {pr_auc:6.3f} (PR-AUC)")
    print(f"Accuracy         : {metrics.accuracy * 100:6.2f}%")
    print(f"Precision        : {metrics.precision * 100:6.2f}%")
    print(f"Recall           : {metrics.recall * 100:6.2f}%")
    print(f"Threshold        : {metrics.threshold:6.2f}")

    return metrics, cm, pr_auc


def _select_champion(
    results: List[CandidateResult],
    selection_metric: str = "auc_roc",
) -> CandidateResult:
    """
    Choose the champion model among candidates based on selection_metric.

    selection_metric can be:
      - 'auc_roc'  (default)
      - 'recall'
      - 'pr_auc'
    """
    if selection_metric not in {"auc_roc", "recall", "pr_auc"}:
        raise ValueError(f"Invalid selection_metric={selection_metric!r}")

    records = []
    for r in results:
        records.append(
            {
                "name": r.name,
                "auc_roc": r.metrics.auc_roc,
                "recall": r.metrics.recall,
                "pr_auc": r.pr_auc,
            }
        )
    df = pd.DataFrame.from_records(records)

    df = df.replace({np.nan: -np.inf})
    best_idx = int(df[selection_metric].idxmax())
    best_name = df.loc[best_idx, "name"]

    return next(r for r in results if r.name == best_name)


def _append_log(
    results: List[CandidateResult],
    champion: CandidateResult,
    selection_metric: str,
    threshold: float,
    n_splits: int,
    log_path: Path,
) -> None:
    """
    Append all models' metrics to the model_select_results.txt log file,
    marking the champion model.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("############################################################\n")
        f.write(f"Model selection run at: {timestamp}\n")
        f.write(f"Dataset              : unified_leads (load_unified_leads_dataset)\n")
        f.write(f"CV folds             : {n_splits}\n")
        f.write(f"Selection metric     : {selection_metric}\n")
        f.write(f"Threshold            : {threshold:.2f}\n")
        f.write("############################################################\n")

        for r in results:
            is_champion = (r.name == champion.name)
            tag = "  <-- CHAMPION" if is_champion else ""
            f.write("\n")
            f.write(f"Model              : {r.name}{tag}\n")
            f.write(f"Accuracy           : {r.metrics.accuracy * 100:6.2f}%\n")
            f.write(f"AUC-ROC            : {r.metrics.auc_roc:6.3f}\n")
            f.write(f"Average Precision  : {r.pr_auc:6.3f} (PR-AUC)\n")
            f.write(f"Precision (pos=1)  : {r.metrics.precision * 100:6.2f}%\n")
            f.write(f"Recall   (pos=1)   : {r.metrics.recall * 100:6.2f}%\n")
            f.write("Confusion Matrix (rows=true, cols=pred)\n")
            f.write("             Pred 0    Pred 1\n")
            f.write(f"True 0   : {r.confusion_matrix[0, 0]:7d} {r.confusion_matrix[0, 1]:8d}\n")
            f.write(f"True 1   : {r.confusion_matrix[1, 0]:7d} {r.confusion_matrix[1, 1]:8d}\n")

        f.write("############################################################\n")


def run_model_selection(
    n_splits: int = 5,
    threshold: float = 0.4,
    selection_metric: str = "auc_roc",
    random_state: int = 42,
    save_champion: bool = False,
) -> None:
    """
    Entry point for unified model selection.
    """
    print("Loading unified leads dataset...")
    X, y = load_unified_leads_dataset()
    X_np = X.to_numpy(float)
    y_np = y.to_numpy(int)

    print("  X:", X_np.shape, "y:", y_np.shape)

    factories = _get_model_factories(random_state=random_state)
    results: List[CandidateResult] = []

    for name, make_model in factories.items():
        metrics, cm, pr_auc = _cv_evaluate_model(
            name=name,
            make_model=make_model,
            X=X_np,
            y=y_np,
            n_splits=n_splits,
            threshold=threshold,
            random_state=random_state,
            plot_roc=False,
        )
        results.append(
            CandidateResult(
                name=name,
                metrics=metrics,
                pr_auc=pr_auc,
                confusion_matrix=cm,
            )
        )

    # Summary table in terminal
    print("\n================ All Candidate Metrics (OOF) ================")
    rows = []
    for r in results:
        rows.append(
            {
                "model_name": r.name,
                "accuracy": r.metrics.accuracy,
                "auc_roc": r.metrics.auc_roc,
                "precision": r.metrics.precision,
                "recall": r.metrics.recall,
                "pr_auc": r.pr_auc,
                "threshold": r.metrics.threshold,
            }
        )
    metrics_df = pd.DataFrame.from_records(rows)
    print(metrics_df.to_string(index=False))

    # Champion selection
    champion = _select_champion(results, selection_metric=selection_metric)
    print("\n================ Champion Model Selected ================")
    print(f"Champion model (by {selection_metric}): {champion.name}")
    print(
        f"AUC-ROC={champion.metrics.auc_roc:.4f}, "
        f"Recall={champion.metrics.recall:.4f}, "
        f"PR-AUC={champion.pr_auc:.4f}, "
        f"Accuracy={champion.metrics.accuracy:.4f}"
    )

    # Write log file with all models + CHAMPION tag
    _append_log(
        results=results,
        champion=champion,
        selection_metric=selection_metric,
        threshold=threshold,
        n_splits=n_splits,
        log_path=MODEL_SELECT_LOG_PATH,
    )
    print(f"\n[OK] Appended model selection results to -> {MODEL_SELECT_LOG_PATH}")

    # Optionally: fit champion on full data and save (does NOT touch your existing LGBM model path)
    if save_champion:
        print("\nFitting champion model on full dataset and saving...")
        factories = _get_model_factories(random_state=random_state)
        champ_model = factories[champion.name]()  # fresh instance
        champ_model.fit(X_np, y_np)

        model_path = MODELS_DIR / f"unified_champion_{champion.name}.pkl"
        joblib.dump(champ_model, model_path)
        print(f"[OK] Saved champion model -> {model_path}")


if __name__ == "__main__":
    # Adjust these defaults if you want:
    run_model_selection(
        n_splits=5,
        threshold=0.4,          # lower threshold if you want more recall
        selection_metric="auc_roc",  # or "recall" / "pr_auc"
        random_state=42,
        save_champion=False,    # True if you want to save unified_champion_<name>.pkl
    )




