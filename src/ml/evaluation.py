# src/ml/evaluation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    RocCurveDisplay,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    auc_roc: float
    precision: float
    recall: float
    threshold: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
        }

    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.as_dict()])


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    model_name: Optional[str] = None,
    pos_label: int = 1,
    average: str = "binary",
    plot_roc: bool = True,
    title_prefix: str = "Unified LGBM",
) -> Tuple[ClassificationMetrics, np.ndarray]:
    """
    Compute and print core metrics and optionally plot ROC + confusion matrix.
    """

    y_true = np.asarray(y_true)

    proba = np.asarray(y_proba)
    if proba.ndim > 1:
        if proba.shape[1] < 2:
            raise ValueError("y_proba has shape (n_samples, n_classes) but n_classes < 2.")
        proba_pos = proba[:, 1]
    else:
        proba_pos = proba

    y_pred = (proba_pos >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, proba_pos)
    except ValueError:
        auc = float("nan")

    prec = precision_score(
        y_true, y_pred, pos_label=pos_label, average=average, zero_division=0
    )
    rec = recall_score(
        y_true, y_pred, pos_label=pos_label, average=average, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = ClassificationMetrics(
        accuracy=acc,
        auc_roc=auc,
        precision=prec,
        recall=rec,
        threshold=threshold,
    )

    # --------- Print summary to terminal ----------
    print("\n=== Model Performance Summary ===")
    if model_name:
        print(f"Model              : {model_name}")
    print(f"Threshold          : {threshold:.2f}")
    print(f"Accuracy           : {acc * 100:6.2f}%")
    print(f"AUC-ROC            : {auc:6.3f}")
    print(f"Precision (pos=1)  : {prec * 100:6.2f}%")
    print(f"Recall    (pos=1)  : {rec * 100:6.2f}%  <-- important for not missing good leads")

    print("\nConfusion Matrix (rows=true, cols=pred)")
    print("             Pred 0    Pred 1")
    print(f"True 0   : {cm[0, 0]:7d} {cm[0, 1]:8d}")
    print(f"True 1   : {cm[1, 0]:7d} {cm[1, 1]:8d}")

    # --------- Optional ROC curve ----------
    if plot_roc:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, proba_pos, name=title_prefix, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(f"{title_prefix} – ROC Curve (AUC={auc:0.3f})")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return metrics, cm

