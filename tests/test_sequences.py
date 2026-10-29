import numpy as np
import pytest

from lead_scoring.data import CATEGORY_COLUMNS
from lead_scoring.sequences import VOCAB_SIZE, EventStore, build_event_store, collate, cutoff_ms

EMPTY_VOCAB = {c: [] for c in CATEGORY_COLUMNS}


def _store(mbd_dir, tmp_path, vocab=EMPTY_VOCAB):
    build_event_store(tmp_path / "events", mbd_dir=mbd_dir, cap=1_000.0, vocab=vocab)
    return EventStore(tmp_path / "events")


def test_window_stops_strictly_before_the_end_of_the_snapshot_day(mbd_dir, tmp_path):
    codes, cont = _store(mbd_dir, tmp_path).window("a", cutoff_ms("2022-01-31"), max_len=512)
    assert len(codes) == 2  # Jun-2021 and 23:00 Moscow on Jan 31; 01:00 Feb 1 is excluded
    assert cont[-1, 0] == pytest.approx(np.log1p(50))


def test_window_keeps_the_latest_events_and_caps_amounts(mbd_dir, tmp_path):
    store = _store(mbd_dir, tmp_path)
    codes, cont = store.window("a", cutoff_ms("2022-12-31"), max_len=2)
    assert len(codes) == 2 and cont[-1, 0] == pytest.approx(np.log1p(500))  # Jun-2022 is the latest
    _, b = store.window("b", cutoff_ms("2022-01-31"), max_len=512)
    assert b[0, 0] == pytest.approx(np.log1p(1_000))  # 10M capped at 1,000


def test_client_without_events_gets_one_padding_step(mbd_dir, tmp_path):
    codes, cont = _store(mbd_dir, tmp_path).window("d", cutoff_ms("2022-01-31"), max_len=512)
    assert codes.shape == (1, len(CATEGORY_COLUMNS)) and not codes.any() and not cont.any()


def test_vocab_values_map_from_two_and_everything_else_to_one(mbd_dir, tmp_path):
    store = _store(mbd_dir, tmp_path, vocab={**EMPTY_VOCAB, "event_type": [2, 1]})
    codes, _ = store.window("a", cutoff_ms("2022-01-31"), max_len=512)
    assert codes[:, CATEGORY_COLUMNS.index("event_type")].tolist() == [3, 2]  # type 1 is rank 1, type 2 rank 0
    assert (codes[:, CATEGORY_COLUMNS.index("currency")] == 1).all()  # empty vocab: all "other"
    assert codes.max() < VOCAB_SIZE


def test_collate_left_pads_to_a_fixed_length():
    two = (np.ones((2, 11), np.int64), np.ones((2, 3), np.float32))
    one = (np.ones((1, 11), np.int64), np.ones((1, 3), np.float32))
    codes, cont = collate([two, one], max_len=3)
    assert codes.shape == (2, 3, 11) and cont.shape == (2, 3, 3)
    assert codes[0, 0].sum() == 0 and codes[0, 1:].all()
    assert codes[1, :2].sum() == 0 and codes[1, 2].all()
