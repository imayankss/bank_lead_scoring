"""Model and baseline comparison on the held-out test months.

Every candidate is scored on the same rows and the same bootstrap resamples.
Rankers are ordered by lift in the top 10% (the share a relationship-manager
team can call), value models by RMSE. No candidate gets a bonus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from lead_scoring.metrics import bootstrap_ci, lift_at_fraction, regression_metrics
from lead_scoring.paths import PREPROCESSED_DATA_DIR, TABLES_DIR, ensure_directories

RANKERS = {
    "prior_baseline": ("Prior rate (no ranking)", "baseline"),
    "recency_rule": ("Recency rule: most recent activity first", "baseline"),
    "activity_rule": ("Activity rule: most transactions in last 90 days", "baseline"),
    "product_history_rule": ("Product-history rule: took a product before, then most active", "baseline"),
    "logistic_regression": ("Logistic regression", "baseline"),
    "propensity_gbm": ("Propensity gradient boosting", "model"),
    "expected_value_ranker": ("Expected value ranker (lead score)", "model"),
}
VALUE_MODELS = {
    "mean_baseline": ("Mean of training clients", "baseline"),
    "last_90_days_baseline": ("Last 90 days repeat", "baseline"),
    "cltv_gbm": ("CLTV gradient boosting", "model"),
}
RANKING_METRICS = {
    "auc": lambda y, s: roc_auc_score(y, s),
    "pr_auc": lambda y, s: average_precision_score(y, s),
    "lift_at_10pct": lambda y, s: lift_at_fraction(y, s, 0.10),
}


def _ranked(rows: list[dict], key: str, ascending: bool) -> pd.DataFrame:
    table = pd.DataFrame(rows).sort_values(key, ascending=ascending, kind="stable").reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table["is_best"] = table["rank"] == 1
    return table


def compare_rankers(y_true: pd.Series, scores: dict[str, pd.Series], *, n_boot: int = 300, seed: int = 42) -> pd.DataFrame:
    """AUC, PR-AUC and top-decile lift with 95% bootstrap intervals, best lift first."""
    rows = []
    for model_id, score in scores.items():
        row = {"model_id": model_id, "task": "propensity"}
        for metric, ci in bootstrap_ci(y_true, score, RANKING_METRICS, n_boot=n_boot, seed=seed).items():
            row.update({metric: ci["point"], f"{metric}_ci_lower": ci["lower"], f"{metric}_ci_upper": ci["upper"]})
        rows.append(row)
    return _ranked(rows, "lift_at_10pct", ascending=False)


def compare_value_models(actual: pd.Series, predictions: dict[str, pd.Series]) -> pd.DataFrame:
    """Regression error for each value model, lowest RMSE first."""
    rows = [{"model_id": model_id, "task": "value", **regression_metrics(actual, pred)} for model_id, pred in predictions.items()]
    return _ranked(rows, "rmse", ascending=True)


def _logistic_scores(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, seed: int) -> np.ndarray:
    model = make_pipeline(
        FunctionTransformer(lambda X: np.log1p(np.clip(X, 0, None))),  # counts and amounts are heavy-tailed
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    return model.fit(X_train, y_train).predict_proba(X_test)[:, 1]


def build_model_comparison(seed: int = 42) -> pd.DataFrame:
    """Compare the trained models with rule and linear baselines on the test snapshot."""
    ensure_directories()
    X_train = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PREPROCESSED_DATA_DIR / "y_clf_train.parquet")["converted"]
    y_reg_train = pd.read_parquet(PREPROCESSED_DATA_DIR / "y_reg_train.parquet")["future_value"]
    predictions = pd.read_csv(TABLES_DIR / "test_predictions.csv")
    y_test = predictions["actual_converted"]

    activity_tiebreak = X_test["txn_count_365"] / (X_test["txn_count_365"].max() + 1)
    rankers = compare_rankers(y_test, {
        "prior_baseline": pd.Series(0.0, index=X_test.index),
        "recency_rule": -X_test["recency_days"],
        "activity_rule": X_test["txn_count_90"],
        "product_history_rule": X_test["products_taken_before"] + activity_tiebreak,
        "logistic_regression": pd.Series(_logistic_scores(X_train, y_train, X_test, seed)),
        "propensity_gbm": predictions["predicted_propensity"],
        "expected_value_ranker": predictions["expected_value"],
    }, seed=seed)
    values = compare_value_models(predictions["actual_future_value"], {
        "mean_baseline": pd.Series(float(y_reg_train.mean()), index=X_test.index),
        # Next 3 months repeat the last 90 days (amounts already capped).
        "last_90_days_baseline": X_test["total_txn_amt_90"],
        "cltv_gbm": predictions["predicted_cltv"],
    })

    output = pd.concat([rankers, values], ignore_index=True)
    names = {**RANKERS, **VALUE_MODELS}
    output.insert(1, "model_name", output["model_id"].map(lambda m: names[m][0]))
    output.insert(2, "role", output["model_id"].map(lambda m: names[m][1]))
    output["production_model"] = output["model_id"].isin(["expected_value_ranker", "cltv_gbm"])
    output.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    return output


def main() -> None:
    output = build_model_comparison()
    print(output[["task", "rank", "model_name", "auc", "lift_at_10pct", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
