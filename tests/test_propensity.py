import json

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from lead_scoring import propensity
from lead_scoring.paper_protocol import NEXT_MONTH, build_paper_protocol


def _setup(tmp_path, monkeypatch, weight):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.random(200)})
    y = (X["f"] + rng.normal(0, 0.2, 200) > 0.6).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=5)
    booster.save_model(str(tmp_path / "lgbm.txt"))
    (tmp_path / "choice.json").write_text(json.dumps({"blend": {"weight": weight}, "calibrator": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}))
    monkeypatch.setattr(propensity, "PROPENSITY_LGBM_FILE", tmp_path / "lgbm.txt")
    monkeypatch.setattr(propensity, "PROPENSITY_CHOICE_FILE", tmp_path / "choice.json")
    return X


def test_lgbm_scores_are_ranked_and_calibrated(tmp_path, monkeypatch):
    out = propensity.score_propensity(_setup(tmp_path, monkeypatch, weight=0.0))
    assert list(out.columns) == ["lgbm_score", "blend_score", "predicted_propensity"]
    assert out["blend_score"].corr(out["lgbm_score"], method="spearman") == pytest.approx(1.0)
    assert out["predicted_propensity"].between(0, 0.5).all()


def test_a_blend_with_the_gru_is_refused_rather_than_scored_without_it(tmp_path, monkeypatch):
    with pytest.raises(NotImplementedError, match="GRU"):
        propensity.score_propensity(_setup(tmp_path, monkeypatch, weight=0.4))


def test_a_missing_choice_file_says_what_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(propensity, "PROPENSITY_CHOICE_FILE", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="validate_propensity.py"):
        propensity.load_choice()


def test_paper_table_has_our_row_then_the_published_ones():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.random(400), "g": rng.random(400)})
    labels = pd.DataFrame({t: (X["f"] + rng.normal(0, 0.3, 400) > 0.7).astype(int) for t in NEXT_MONTH})
    table = build_paper_protocol(X, labels, X, labels, {"objective": "binary", "verbose": -1, "min_child_samples": 5}, rounds=10)
    assert table["source"].tolist() == ["ours", "paper", "paper"]
    ours = table.iloc[0]
    assert ours["mean_auc"] == pytest.approx(np.mean([ours[f"auc_{t}"] for t in NEXT_MONTH]))
    assert ours["mean_auc"] > 0.6
