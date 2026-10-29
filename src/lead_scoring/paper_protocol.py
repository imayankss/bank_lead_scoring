"""Next-month, per-product AUC on the test snapshot, next to the MBD paper's numbers.

The paper (arXiv 2409.17587, Table 3) predicts each of 4 products for the next month on full MBD with
client folds and up to three modalities. We use MBD-mini, a time split, and transactions plus dialog
timestamps, so the rows show scale, not a like-for-like tie.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lead_scoring.data import TARGETS
from lead_scoring.paths import PREPROCESSED_DATA_DIR, TABLES_DIR
from lead_scoring.propensity import load_choice

NEXT_MONTH = [f"y1_{t}" for t in TARGETS]
PUBLISHED = [
    {"model": "Published: transaction aggregates (full MBD, client folds)", "source": "paper", "mean_auc": 0.783},
    {"model": "Published best: supervised RNN, transactions + geo + dialogs", "source": "paper", "mean_auc": 0.824},
]


def build_paper_protocol(X_train, labels_train, X_test, labels_test, params: dict, rounds: int) -> pd.DataFrame:
    aucs = {t: float(roc_auc_score(labels_test[t], lgb.train(params, lgb.Dataset(X_train, labels_train[t]),
                                                             num_boost_round=rounds).predict(X_test)))
            for t in NEXT_MONTH}
    ours = {"model": "LightGBM, one model per product", "source": "ours", "mean_auc": float(np.mean(list(aucs.values()))),
            **{f"auc_{t}": v for t, v in aucs.items()}}
    return pd.DataFrame([ours, *PUBLISHED])


def write_paper_protocol() -> pd.DataFrame:
    choice = load_choice()
    table = build_paper_protocol(
        pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet"),
        pd.read_parquet(PREPROCESSED_DATA_DIR / "y_labels_train.parquet"),
        pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet"),
        pd.read_parquet(PREPROCESSED_DATA_DIR / "y_labels_test.parquet"),
        choice["lgbm_params"], choice["lgbm_rounds"],  # the any-product choice, reused for each product
    )
    table.to_csv(TABLES_DIR / "paper_protocol.csv", index=False)
    return table
