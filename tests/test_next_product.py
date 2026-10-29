import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from lead_scoring.next_product import (
    PRODUCTS,
    driver_contributions,
    evaluate_next_product,
    family,
    fit_product_models,
    predict_products,
)

PARAMS = {"objective": "binary", "verbose": -1, "min_child_samples": 5}


def _data(n=600):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.random(n) for i in range(4)})
    # Product k is driven by feature k, so the right model can tell the products apart.
    labels = pd.DataFrame({p: (X[f"f{k}"] + rng.normal(0, 0.1, n) > 0.9).astype(int) for k, p in enumerate(PRODUCTS)})
    return X, labels


def test_each_product_model_learns_its_own_product():
    X, labels = _data()
    probs = predict_products(fit_product_models(X, labels, PARAMS, rounds=20), X)
    table = evaluate_next_product(probs, labels, labels)
    assert (table.query("measure == 'auc'")["value"] > 0.8).all()
    hits = table.query("measure == 'top1_hit_rate_among_buyers'").set_index("model")["value"]
    assert hits["lightgbm"] > hits["most_popular_product"]


def test_hit_rate_counts_any_product_the_buyer_took():
    labels = pd.DataFrame([[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], columns=PRODUCTS)
    probs = pd.DataFrame([[0.1, 0.9, 0, 0], [0.9, 0, 0, 0.1], [1, 0, 0, 0]], columns=PRODUCTS)
    hits = evaluate_next_product(probs, labels, labels).query("model == 'lightgbm' and product == 'any'")
    assert hits["value"].item() == pytest.approx(0.5)  # 2 buyers: first right (took product 2), second wrong
    assert hits["buyers"].item() == 2


def test_features_map_to_readable_families():
    assert family("hist_target_3_m1") == "product_history"
    assert family("target_2_before") == "product_history"
    assert family("dlg_count_90") == "conversations"
    assert family("cat_event_type_40_365") == "transaction_mix"
    assert family("recency_days") == "activity"


def test_driver_contributions_sum_to_the_model_score():
    X = pd.DataFrame({"hist_target_1_m0": [0, 1] * 100, "dlg_count_90": np.arange(200) % 7, "recency_days": np.arange(200)})
    y = (X["hist_target_1_m0"] == 1).astype(int)
    booster = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=10)
    drivers = driver_contributions(booster, X)
    bias = booster.predict(X, pred_contrib=True)[:, -1]
    assert np.allclose(drivers.sum(axis=1) + bias, booster.predict(X, raw_score=True))
    assert drivers["product_history"].abs().mean() > drivers["conversations"].abs().mean()
