import numpy as np
import pandas as pd
import pytest

from lead_scoring.config import PipelineConfig
from lead_scoring.data import amount_cap, build_customer_features

SNAP = "2022-01-31"
CFG = PipelineConfig(workers=1)


def _features(mbd_dir, snapshot=SNAP, include_target=True, cap=1_000.0, config=CFG):
    df = build_customer_features(snapshot, include_target=include_target, config=config, mbd_dir=mbd_dir, amount_cap=cap)
    return df.set_index("customer_id")


def test_every_labelled_client_is_scored_even_without_transactions(mbd_dir):
    df = _features(mbd_dir)
    assert sorted(df.index) == ["a", "b", "c", "d"]
    assert df.loc["d", "txn_count_365"] == 0


def test_features_use_moscow_time_and_stop_at_snapshot(mbd_dir):
    a = _features(mbd_dir).loc["a"]
    assert a["txn_count_365"] == 2  # Jun-2021 and 23:00 Moscow on the snapshot day; 01:00 Feb 1 excluded
    assert a["total_txn_amt_365"] == 150
    assert a["recency_days"] == 0


def test_lookback_window_drops_older_transactions(mbd_dir):
    assert _features(mbd_dir).loc["c", "txn_count_365"] == 0


def test_amounts_are_capped(mbd_dir):
    assert _features(mbd_dir, cap=1_000.0).loc["b", "max_txn_amt_365"] == 1_000


def test_target_counts_products_only_inside_horizon(mbd_dir):
    df = _features(mbd_dir)
    assert df["converted"].to_dict() == {"a": 1, "b": 0, "c": 1, "d": 0}
    assert df.loc["a", "future_value"] == 999 + 70  # Feb 1 Moscow and Mar-2022; Jun-2022 is after the window


def test_product_history_uses_only_earlier_label_months(mbd_dir):
    df = _features(mbd_dir, snapshot="2022-04-30")
    assert df.loc["a", "products_taken_before"] == 1  # Mar-2022 label
    assert df.loc["c", "products_taken_before"] == 1  # Feb-2022 label
    assert df.loc["b", "products_taken_before"] == 0  # May-2022 label is in the future


def test_scoring_features_have_no_targets(mbd_dir):
    df = _features(mbd_dir, snapshot="2022-12-31", include_target=False)
    assert not {"converted", "future_value"} & set(df.columns)


def test_unexplained_mbd_columns_never_become_features(mbd_dir):
    assert not {"trans_count", "diff_trans_date"} & set(_features(mbd_dir).columns)


def test_parallel_folds_match_serial(mbd_dir):
    serial = _features(mbd_dir).sort_index()
    parallel = _features(mbd_dir, config=PipelineConfig(workers=2)).sort_index()
    pd.testing.assert_frame_equal(serial, parallel)


def test_amount_cap_uses_only_pre_snapshot_amounts(mbd_dir):
    cap = amount_cap(mbd_dir, before=SNAP)
    # Amounts on or before 2022-01-31 (Moscow time); 999, 70 and 500 come later and must not count.
    expected = np.quantile([5.0, 50.0, 100.0, 10_000_000.0], 0.999)
    assert np.isfinite(cap) and cap < 10_000_000
    assert cap == pytest.approx(expected)
