"""Next best product: one 3-month LightGBM per product, measured on the test month, scored on the current book.

Each product model reuses the any-product settings chosen on validation (models/propensity_choice.json);
nothing here is tuned on the test month.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lead_scoring.data import TARGETS
from lead_scoring.paths import PREPROCESSED_DATA_DIR, PROCESSED_DATA_DIR, PROPENSITY_LGBM_FILE, TABLES_DIR
from lead_scoring.propensity import load_choice

PRODUCTS = [f"y3_{t}" for t in TARGETS]
NEXT_PRODUCT_SCORES_FILE = PROCESSED_DATA_DIR / "next_product_scores.csv"
DRIVERS_FILE = PROCESSED_DATA_DIR / "lead_drivers.csv"
FAMILIES = ("product_history", "conversations", "transaction_mix", "activity")


def fit_product_models(X: pd.DataFrame, labels: pd.DataFrame, params: dict, rounds: int) -> dict[str, lgb.Booster]:
    return {p: lgb.train(params, lgb.Dataset(X, labels[p]), num_boost_round=rounds) for p in PRODUCTS}


def predict_products(models: dict[str, lgb.Booster], X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({p: models[p].predict(X) for p in PRODUCTS}, index=X.index)


def evaluate_next_product(probs: pd.DataFrame, labels: pd.DataFrame, labels_train: pd.DataFrame) -> pd.DataFrame:
    """Per-product AUC, and how often the top-ranked product is one a buyer actually took."""
    rows = [{"measure": "auc", "product": p, "model": "lightgbm", "value": float(roc_auc_score(labels[p], probs[p])),
             "buyers": int(labels[p].sum())} for p in PRODUCTS]
    buyers = labels[PRODUCTS].sum(axis=1) > 0
    took = labels.loc[buyers, PRODUCTS].to_numpy()
    picks = {
        "lightgbm": probs.loc[buyers, PRODUCTS].to_numpy().argmax(axis=1),
        # Baseline: offer everyone the product most bought in the training months.
        "most_popular_product": np.full(int(buyers.sum()), int(labels_train[PRODUCTS].mean().to_numpy().argmax())),
    }
    rows += [{"measure": "top1_hit_rate_among_buyers", "product": "any", "model": name,
              "value": float(took[np.arange(len(pick)), pick].mean()), "buyers": int(buyers.sum())}
             for name, pick in picks.items()]
    return pd.DataFrame(rows)


def family(feature: str) -> str:
    if feature.startswith("hist_") or feature.endswith("_before"):
        return "product_history"
    if feature.startswith("dlg_"):
        return "conversations"
    if feature.startswith(("cat_", "days_since_evt_")):
        return "transaction_mix"
    return "activity"


def driver_contributions(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM's own SHAP contributions to the propensity score, summed by feature family (log-odds)."""
    contrib = pd.DataFrame(booster.predict(X, pred_contrib=True)[:, :-1], columns=X.columns, index=X.index)
    return contrib.T.groupby(contrib.columns.map(family)).sum().T.reindex(columns=list(FAMILIES), fill_value=0.0)


def write_next_product() -> pd.DataFrame:
    choice = load_choice()
    read = lambda name: pd.read_parquet(PREPROCESSED_DATA_DIR / name)  # noqa: E731
    X_train, labels_train = read("X_train.parquet"), read("y_labels_train.parquet")
    models = fit_product_models(X_train, labels_train, choice["lgbm_params"], choice["lgbm_rounds"])

    table = evaluate_next_product(predict_products(models, read("X_test.parquet")), read("y_labels_test.parquet"), labels_train)
    table.to_csv(TABLES_DIR / "next_product_test.csv", index=False)

    X_scoring = read("X_scoring.parquet")
    ids = pd.read_csv(PREPROCESSED_DATA_DIR / "scoring_ids.csv")["customer_id"]
    scores = predict_products(models, X_scoring).set_axis([f"p_{t}" for t in TARGETS], axis=1)
    scores["next_product"] = scores.to_numpy().argmax(axis=1) + 1
    scores.insert(0, "customer_id", ids.to_numpy())
    scores.to_csv(NEXT_PRODUCT_SCORES_FILE, index=False)

    drivers = driver_contributions(lgb.Booster(model_file=str(PROPENSITY_LGBM_FILE)), X_scoring)
    drivers.insert(0, "customer_id", ids.to_numpy())
    drivers.to_csv(DRIVERS_FILE, index=False)
    return table


def main() -> None:
    print(write_next_product().to_string(index=False))


if __name__ == "__main__":
    main()
