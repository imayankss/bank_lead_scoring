import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lead_scoring.data import CATEGORY_COLUMNS
from lead_scoring.gru import HEADS, EventGRU, load, predict, save
from lead_scoring.sequences import EventStore, build_event_store

CPU = torch.device("cpu")


def test_model_outputs_one_logit_per_head():
    out = EventGRU()(torch.zeros(4, 7, len(CATEGORY_COLUMNS), dtype=torch.long), torch.zeros(4, 7, 3))
    assert out.shape == (4, len(HEADS)) == (4, 9)


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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(codes, torch.zeros(n, 5, 3)), labels)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_scores_keep_sample_order_survive_save_and_ignore_batch_size(mbd_dir, tmp_path):
    build_event_store(tmp_path / "events", mbd_dir=mbd_dir, cap=1_000.0, vocab={c: [] for c in CATEGORY_COLUMNS})
    store = EventStore(tmp_path / "events")
    samples = pd.DataFrame({"customer_id": ["a", "d", "b"], "snapshot_date": "2022-01-31"})
    model = EventGRU(hidden=8)
    p = predict(model, store, samples, max_len=16, batch_size=2, dev=CPU)
    assert p.shape == (3, 9) and ((p > 0) & (p < 1)).all()
    save(model, tmp_path / "m.pt", max_len=16)
    again, meta = load(tmp_path / "m.pt")
    assert meta["max_len"] == 16
    np.testing.assert_allclose(predict(again, store, samples, max_len=16, batch_size=3, dev=CPU), p, rtol=1e-5)
