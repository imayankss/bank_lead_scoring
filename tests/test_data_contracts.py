import numpy as np
import pandas as pd
import pytest

from lead_scoring.config import PipelineConfig
from lead_scoring.data import CATEGORY_COLUMNS, LABELS, amount_cap, build_customer_features, category_vocab

SNAP = "2022-01-31"
CFG = PipelineConfig(workers=1)


EMPTY_VOCAB = {c: [] for c in CATEGORY_COLUMNS}


def _features(mbd_dir, snapshot=SNAP, include_target=True, cap=1_000.0, config=CFG, vocab=None):
    df = build_customer_features(snapshot, include_target=include_target, config=config, mbd_dir=mbd_dir,
                                 amount_cap=cap, vocab=vocab)
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
    assert not {"future_value", *LABELS} & set(df.columns)


def test_product_history_is_a_fixed_two_month_window(mbd_dir):
    df = _features(mbd_dir, snapshot="2022-04-30")
    assert df.loc["a", "hist_target_2_m1"] == 1 and df.loc["a", "hist_any_m1"] == 1  # Mar-2022
    assert df.loc["c", "hist_target_4_m2"] == 1  # Feb-2022
    assert df.loc["b", ["hist_any_m1", "hist_any_m2"]].tolist() == [0, 0]  # May-2022 is in the future


def test_history_before_the_first_label_month_is_missing_not_zero(mbd_dir):
    mar = _features(mbd_dir, snapshot="2022-03-31")
    assert mar["hist_any_m2"].isna().all()  # Jan-2022 has no labels at all
    assert mar.loc["c", "hist_target_4_m1"] == 1 and mar.loc["a", "hist_any_m1"] == 0
    jan = _features(mbd_dir)
    assert jan[[c for c in jan.columns if c.startswith("hist_")]].isna().all().all()


def test_snapshot_month_purchase_is_history_not_label(mbd_dir):
    df = _features(mbd_dir, snapshot="2022-03-31")
    assert df.loc["a", ["hist_target_2_m0", "hist_any_m0"]].tolist() == [1, 1]  # bought during March
    assert df.loc["a", "converted"] == 0  # the label window starts in April
    assert df.loc["c", "hist_any_m0"] == 0 and df.loc["c", "hist_any_m1"] == 1  # bought in February


def test_months_since_last_purchase(mbd_dir):
    df = _features(mbd_dir, snapshot="2022-04-30")
    assert df.loc["a", ["hist_target_2_months_since", "hist_any_months_since"]].tolist() == [1, 1]  # March
    assert df.loc["c", "hist_target_4_months_since"] == 2  # February
    assert df.loc[["b", "d"], "hist_any_months_since"].isna().all()  # no purchase yet: unknown, not zero


def test_per_product_labels_for_three_months_and_next_month(mbd_dir):
    df = _features(mbd_dir)
    assert df.loc["a", ["y3_target_2", "y1_target_2"]].tolist() == [1, 0]  # bought in Mar, not Feb
    assert df.loc["c", ["y3_target_4", "y1_target_4"]].tolist() == [1, 1]  # bought in Feb
    assert df.loc["b", "y3_target_1"] == 0  # May is past the window


def test_category_vocab_counts_only_events_before_the_cutoff(mbd_dir):
    vocab = category_vocab(mbd_dir, before=SNAP, top=10)
    assert set(vocab) == set(CATEGORY_COLUMNS)
    assert vocab["event_type"] == [1, 2]  # type 3 happens at 01:00 Moscow on Feb 1


def test_category_counts_follow_the_vocab_and_window(mbd_dir):
    df = _features(mbd_dir, vocab={**EMPTY_VOCAB, "event_type": [1, 2]})
    a = df.loc["a"]
    assert a["cat_event_type_1_365"] == 1 and a["cat_event_type_2_365"] == 1  # Jun-2021 and the snapshot day
    assert a["cat_event_type_2_90"] == 1 and a["cat_event_type_1_90"] == 0
    assert "cat_event_type_3_365" not in df.columns and df["cat_event_type_1_365"].dtype == "int32"


def test_trend_blocks_and_ratios(mbd_dir):
    a = _features(mbd_dir).loc["a"]
    assert a["txn_count_d30_1"] == 1 and a["txn_amt_d30_1"] == 50  # the snapshot-day transaction
    assert a[[f"txn_count_d30_{i}" for i in range(2, 7)]].sum() == 0  # Jun-2021 is 8 months back
    assert a["ratio_30_90"] == 1.0 and a["ratio_90_365"] == 0.5


def test_days_since_each_event_type(mbd_dir):
    df = _features(mbd_dir, vocab={**EMPTY_VOCAB, "event_type": [1, 2]})
    assert df.loc["a", "days_since_evt_2"] == 0
    assert df.loc["a", "days_since_evt_1"] == (pd.Timestamp(SNAP) - pd.Timestamp("2021-06-01 13:00")).days
    assert df.loc["d", "days_since_evt_1"] == CFG.lookback_days + 1


def test_dialog_counts_and_recency_stop_at_snapshot(mbd_dir):
    df = _features(mbd_dir)
    assert df.loc["a", ["dlg_count_30", "dlg_count_90", "dlg_count_365"]].tolist() == [0, 1, 1]  # Feb-1 Moscow excluded
    assert df.loc["d", "dlg_count_30"] == 1  # no transactions, but a dialog on Jan 20
    assert df.loc["d", "dlg_days_since"] == (pd.Timestamp(SNAP) - pd.Timestamp("2022-01-20 13:00")).days
    assert df.loc["c", "dlg_count_365"] == 0 and df.loc["c", "dlg_days_since"] > CFG.lookback_days
    assert df.loc["b", "dlg_days_since"] == CFG.lookback_days + 1  # never talked: same value as never transacted


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
