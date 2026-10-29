import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lead_scoring.data import CATEGORY_COLUMNS
from lead_scoring.gru import HEADS, N_SIDE, SIDE_COLUMNS, EventGRU, load, predict, save, side_features
from lead_scoring.sequences import EventStore, build_event_store

CPU = torch.device("cpu")


def _no_history(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.assign(**{c: np.nan for c in SIDE_COLUMNS})


def test_model_outputs_one_logit_per_head():
    out = EventGRU()(torch.zeros(4, 7, len(CATEGORY_COLUMNS), dtype=torch.long), torch.zeros(4, 7, 3),
                     torch.zeros(4, N_SIDE))
    assert out.shape == (4, len(HEADS)) == (4, 9)


def test_side_features_mark_unknown_months_and_turn_recency_into_a_bounded_score():
    samples = _no_history(pd.DataFrame({"customer_id": ["jan", "jul"]}))
    samples.loc[1, [c for c in SIDE_COLUMNS if not c.endswith("months_since")]] = 0.0
    samples.loc[1, ["hist_any_m0", "hist_target_2_m0", "hist_any_months_since"]] = [1.0, 1.0, 0.0]
    x = side_features(samples)
    assert x.shape == (2, N_SIDE) and x.dtype == np.float32 and np.isfinite(x).all()
    assert not x[0].any()  # Jan-2022 snapshot: no label months yet, all unknown
    assert x[1, 15:18].tolist() == [1, 1, 1]  # all three months known
    assert x[1, -1] == 1.0  # bought this month: 1 / (1 + 0); never bought stays 0


def test_gru_learns_from_purchase_history_when_sequences_are_identical():
    torch.manual_seed(0)
    n = 64
    codes = torch.ones(n, 5, len(CATEGORY_COLUMNS), dtype=torch.long)
    y = (torch.arange(n) % 2).float()
    side = torch.zeros(n, N_SIDE)
    side[:, 0] = y  # positives bought last month; the sequences carry no signal
    labels = y[:, None].repeat(1, len(HEADS))
    model = EventGRU(hidden=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)  # one weight must grow to about 5; 1e-2 x 60 steps can't
    for _ in range(100):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(codes, torch.zeros(n, 5, 3), side), labels)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_gru_learns_a_separable_pattern_on_cpu():
    torch.manual_seed(0)
    n = 64
    codes = torch.ones(n, 5, len(CATEGORY_COLUMNS), dtype=torch.long)
    y = (torch.arange(n) % 2).float()
    codes[y == 1, -1, 0] = 7  # positives end with event-type code 7
    labels = y[:, None].repeat(1, len(HEADS))
    model = EventGRU(hidden=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(codes, torch.zeros(n, 5, 3), torch.zeros(n, N_SIDE)), labels)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_scores_keep_sample_order_survive_save_and_ignore_batch_size(mbd_dir, tmp_path):
    build_event_store(tmp_path / "events", mbd_dir=mbd_dir, cap=1_000.0, vocab={c: [] for c in CATEGORY_COLUMNS})
    store = EventStore(tmp_path / "events")
    samples = _no_history(pd.DataFrame({"customer_id": ["a", "d", "b"], "snapshot_date": "2022-01-31"}))
    model = EventGRU(hidden=8)
    p = predict(model, store, samples, max_len=16, batch_size=2, dev=CPU)
    assert p.shape == (3, 9) and ((p > 0) & (p < 1)).all()
    save(model, tmp_path / "m.pt", max_len=16)
    again, meta = load(tmp_path / "m.pt")
    assert meta["max_len"] == 16 and meta["hidden"] == 8
    np.testing.assert_allclose(predict(again, store, samples, max_len=16, batch_size=3, dev=CPU), p, rtol=1e-5)
