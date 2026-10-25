import numpy as np

from lead_scoring.blend import blend, calibrate, choose_weight, fit_calibrator


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    return y, y + rng.normal(0, 1, n), y + rng.normal(0, 1, n)  # two independent noisy views of y


def test_two_independent_views_blend_near_half():
    y, a, b = _toy()
    choice = choose_weight(y, a, b)
    assert 0.3 <= choice["weight"] <= 0.7 and choice["auc"] > choice["lgbm_auc"] + 0.005


def test_a_useless_gru_falls_back_to_lgbm_alone():
    y, a, _ = _toy()
    choice = choose_weight(y, a, np.random.default_rng(1).random(len(y)))
    assert choice["weight"] == 0.0 and choice["auc"] == choice["lgbm_auc"]


def test_no_gru_scores_means_lgbm_alone():
    y, a, _ = _toy()
    choice = choose_weight(y, a)
    assert choice["weight"] == 0.0 and choice["auc"] == choice["lgbm_auc"] and choice["gru_auc"] is None


def test_blend_is_rank_based_so_scale_does_not_matter():
    _, a, b = _toy()
    np.testing.assert_allclose(blend(a, b, 0.5), blend(a * 1000, np.exp(b), 0.5))
    np.testing.assert_allclose(blend(a, None, 0.0), blend(a, b, 0.0))


def test_calibrated_probability_is_monotone_in_range_and_keeps_the_base_rate():
    y, a, b = _toy()
    s = blend(a, b, 0.5)
    p = calibrate(s, fit_calibrator(y, s))
    order = np.argsort(s)
    assert np.all(np.diff(p[order]) >= -1e-12) and p.min() >= 0 and p.max() <= 1
    assert abs(p.mean() - y.mean()) < 1e-9
