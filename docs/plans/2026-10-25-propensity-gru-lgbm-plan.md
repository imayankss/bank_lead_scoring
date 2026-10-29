# Propensity Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise propensity ranking on the test months from AUC 0.736 / top-10% lift 3.23 toward AUC ≥ 0.78 / lift ≥ 4.0, with every choice made on a validation month.

**Architecture:** The DuckDB feature build gains windowed product history, category-mix counts and trend blocks, which feed LightGBM. A separate numpy event store feeds a PyTorch GRU over each client's last 512 transactions. A rank blend of the two, with its weight and an isotonic calibrator chosen on the 2022-07-31 validation snapshot, becomes the production propensity. The test snapshot (2022-10-31) is scored by the pipeline only after every choice is frozen in `models/propensity_choice.json`.

**Tech Stack:** Python 3.12 (arm64), DuckDB, pandas, numpy, scikit-learn, LightGBM, PyTorch (`mps` on the M2, CPU in CI and pipeline inference), pytest.

**Spec:** `docs/plans/2026-10-25-propensity-gru-lgbm-design.md`

## Global Constraints

- Test snapshot `2022-10-31` is never read by selection code: fit snapshots `2022-01-31, 2022-02-28, 2022-03-31, 2022-04-30`; validation `2022-07-31`; final train = the 7 month-ends `2022-01-31 … 2022-07-31`.
- Leakage stops: any validation AUC > 0.85 raises; any single feature with validation AUC > 0.80 raises (`LeakageError`).
- Blend weight grid 0 to 1 step 0.05, chosen by validation AUC; GRU kept only if it adds ≥ 0.005 validation AUC over LightGBM alone.
- LightGBM grid: `num_leaves` {31, 63} × `min_child_samples` {100, 400}; `learning_rate` 0.05; `feature_fraction` 0.5; early stopping on validation AUC; final rounds = best iteration × 7/4.
- GRU: 1 layer, hidden 128, dropout 0.2, 11 embeddings of 16 dims, vocab 1,024 per column (0 = padding, 1 = other), last 512 events left-padded to exactly 512, AdamW lr 1e-3, batch 256, up to 12 epochs.
- Category mix: top 10 values per categorical column, counts over 365 and 90 days, int32.
- Product history window: 2 label months before the snapshot month; NaN when that month is before the first label month (2022-02-28).
- Memory budget: 8 GB RAM. Never load all 38.8M events into pandas; the event store is built one fold at a time.
- Commits: no `Co-Authored-By: Claude` line and no "Generated with Claude Code" line (user's global rule). Nothing is pushed without the user's OK.
- Run every command with the project venv: `.venv/bin/python`, `.venv/bin/pytest`.
- Checkpoints after Task 5 and Task 7 are hard stops: report numbers to the user and wait.

## File Map

| File | Change | Responsibility |
|---|---|---|
| `requirements.txt`, `.github/workflows/ci.yml`, `.gitignore` | modify | pin `lightgbm`, `torch`; CPU torch in CI; ignore model binaries |
| `src/lead_scoring/paths.py` | modify | new artifact paths; drop `PROPENSITY_MODEL_FILE` (Task 8) |
| `src/lead_scoring/config.py` | modify | fit / validation / monthly train snapshots |
| `src/lead_scoring/temporal.py` | modify | selection-snapshot guard, single-feature leak report, AUC ceiling |
| `src/lead_scoring/data.py` | modify | labels, windowed history, category vocab and mix, trend blocks |
| `src/lead_scoring/features.py` | modify | dynamic feature list, cumulative history forbidden, float32, label matrices |
| `src/lead_scoring/blend.py` | create | rank blend, weight choice, isotonic calibrator as JSON |
| `scripts/validate_propensity.py` | create | LightGBM selection, leak report, blend choice → `models/propensity_choice.json` |
| `src/lead_scoring/sequences.py` | create | numpy event store, windows, fixed-length collate |
| `src/lead_scoring/gru.py` | create | `EventGRU`, train/predict loops, save/load, `score_snapshot` |
| `scripts/train_gru.py` | create | validation run (epoch choice, val scores) and final refit |
| `src/lead_scoring/propensity.py` | create | production scorer: LightGBM (+ GRU) → blend → calibrated propensity |
| `src/lead_scoring/training.py`, `evaluation.py`, `scoring.py`, `pipeline.py` | modify | LightGBM final fit; use `score_propensity` |
| `src/lead_scoring/model_comparison.py` | modify | new rows; rules read raw columns; previous-model row |
| `src/lead_scoring/paper_protocol.py` | create | next-month per-product AUC vs the paper |
| `src/lead_scoring/model_registry.py`, `provenance.py` | modify | describe the new model; record lightgbm/torch versions |
| `README.md`, `docs/*.md`, `tests/test_readme_numbers.py` | modify | measured results only |
| `tests/conftest.py` + new/extended tests | modify/create | fixture gains all 11 categorical columns |

---

### Task 1: Native arm64 environment with LightGBM and PyTorch

**Files:**
- Modify: `requirements.txt`, `.github/workflows/ci.yml`, `.gitignore`, `src/lead_scoring/paths.py`, `src/lead_scoring/provenance.py`

**Interfaces:**
- Produces (paths.py): `PROPENSITY_LGBM_FILE`, `PROPENSITY_CHOICE_FILE`, `GRU_MODEL_FILE`, `EVENT_STORE_DIR`, `GRU_VAL_SCORES_FILE`.

- [ ] **Step 1: Rebuild the venv on arm64 (the old one is kept, not deleted)**

```bash
mv .venv .venv-x86-backup
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
.venv/bin/python -c "import platform; print(platform.machine())"
```
Expected: `arm64`

- [ ] **Step 2: Install pinned requirements plus the two new packages**

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install lightgbm torch
.venv/bin/pip freeze | grep -iE '^(lightgbm|torch)=='
```
Append the two printed `name==version` lines to `requirements.txt`.

- [ ] **Step 3: Check both libraries load and the M2 GPU is visible**

```bash
.venv/bin/python -c "import lightgbm, torch; print(lightgbm.__version__, torch.__version__, torch.backends.mps.is_available())"
```
Expected: two versions and `True`. If LightGBM raises `OSError` about `libomp.dylib`, stop and ask the user before running `brew install libomp`.

- [ ] **Step 4: CI installs CPU-only torch first** — in `.github/workflows/ci.yml` replace `- run: pip install -r requirements.txt` with:

```yaml
      # CPU-only torch keeps the CI install small; requirements.txt pins the same version.
      - run: pip install "torch==$(grep -oP '(?<=^torch==).*' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements.txt
```

- [ ] **Step 5: New artifact paths** — in `src/lead_scoring/paths.py`, after `MODEL_METADATA_FILE`:

```python
PROPENSITY_LGBM_FILE = MODEL_DIR / "propensity_lgbm.txt"
PROPENSITY_CHOICE_FILE = MODEL_DIR / "propensity_choice.json"  # committed: the choices made on validation
GRU_MODEL_FILE = MODEL_DIR / "gru_final.pt"
EVENT_STORE_DIR = PROCESSED_DATA_DIR / "events"
GRU_VAL_SCORES_FILE = PROCESSED_DATA_DIR / "gru_val_scores.parquet"
```

In `.gitignore`, under `models/*.pkl`, add:

```
models/*.txt
models/*.pt
```

In `src/lead_scoring/provenance.py`: `PACKAGES = ["pandas", "numpy", "scikit-learn", "duckdb", "pyarrow", "joblib", "lightgbm", "torch"]`

- [ ] **Step 6: Full suite passes on the new venv**

Run: `.venv/bin/pytest -q`
Expected: all existing tests pass (63 at the start of this branch).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .github/workflows/ci.yml .gitignore src/lead_scoring/paths.py src/lead_scoring/provenance.py
git commit -m "Pin LightGBM and PyTorch, install CPU torch in CI, add model artifact paths"
```

---

### Task 2: Time protocol and leakage guards

**Files:**
- Modify: `src/lead_scoring/config.py`, `src/lead_scoring/temporal.py`
- Test: `tests/test_temporal.py`

**Interfaces:**
- Produces: `PipelineConfig.fit_snapshots: tuple[str, ...]`, `PipelineConfig.validation_snapshot: str`, 7-month `train_snapshots`;
  `assert_selection_snapshots(snapshots, fit_snapshots, validation_snapshot, horizon_months) -> None`;
  `single_feature_auc(X: pd.DataFrame, y) -> pd.DataFrame[feature, auc]`;
  `assert_no_single_feature_leak(table, threshold=0.80) -> None`; `assert_plausible_auc(auc, ceiling=0.85) -> None`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_temporal.py` (and add `import numpy as np`, `from lead_scoring.config import PipelineConfig`, plus the four new names to the `lead_scoring.temporal` import):

```python
def test_selection_accepts_fit_and_validation_only():
    assert_selection_snapshots(["2022-01-31", "2022-07-31"], ("2022-01-31", "2022-04-30"), "2022-07-31", 3)


def test_selection_raises_on_test_snapshot_rows():
    with pytest.raises(LeakageError, match="2022-10-31"):
        assert_selection_snapshots(["2022-04-30", "2022-10-31"], ("2022-04-30",), "2022-07-31", 3)


def test_selection_raises_when_fit_labels_reach_validation():
    with pytest.raises(LeakageError, match="2022-05-31"):
        assert_selection_snapshots(["2022-05-31"], ("2022-05-31",), "2022-07-31", 3)


def test_default_config_windows_are_leak_free():
    cfg = PipelineConfig()
    assert_selection_snapshots([*cfg.fit_snapshots, cfg.validation_snapshot], cfg.fit_snapshots,
                               cfg.validation_snapshot, cfg.horizon_months)
    assert_target_windows_before(cfg.train_snapshots, cfg.horizon_months, cfg.test_snapshot)
    assert set(cfg.fit_snapshots) < set(cfg.train_snapshots) and cfg.validation_snapshot in cfg.train_snapshots
    assert len(cfg.train_snapshots) == 7


def test_single_feature_leak_raises_on_a_copy_of_the_target():
    y = pd.Series([0, 1] * 50)
    X = pd.DataFrame({"leak": y * 3.0, "noise": np.arange(100) % 7})
    table = single_feature_auc(X, y)
    assert table.iloc[0]["feature"] == "leak" and table.iloc[0]["auc"] == 1.0
    with pytest.raises(LeakageError, match="leak"):
        assert_no_single_feature_leak(table)


def test_a_reversed_feature_counts_as_signal():
    y = pd.Series([0, 1] * 50)
    assert single_feature_auc(pd.DataFrame({"neg": -y}), y).iloc[0]["auc"] == 1.0


def test_auc_above_the_ceiling_raises():
    assert_plausible_auc(0.82)
    with pytest.raises(LeakageError, match="0.910"):
        assert_plausible_auc(0.91)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/pytest tests/test_temporal.py -q`
Expected: ImportError for `assert_selection_snapshots`.

- [ ] **Step 3: Implement** — in `src/lead_scoring/config.py` replace `train_snapshots` with:

```python
    # Model selection: fit on Jan–Apr 2022, validate on Jul 2022 (its labels end on the test snapshot).
    fit_snapshots: tuple[str, ...] = ("2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30")
    validation_snapshot: str = "2022-07-31"
    train_snapshots: tuple[str, ...] = (
        "2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31", "2022-06-30", "2022-07-31",
    )
```

Append to `src/lead_scoring/temporal.py` (add `from sklearn.metrics import roc_auc_score` to the imports):

```python
def assert_selection_snapshots(snapshots: Iterable, fit_snapshots: Iterable, validation_snapshot, horizon_months: int) -> None:
    """Fail if model selection sees anything but fit and validation rows, or fit labels reach validation."""
    fit_snapshots = list(fit_snapshots)
    allowed = {str(pd.Timestamp(s).date()) for s in (*fit_snapshots, validation_snapshot)}
    extra = sorted({str(pd.Timestamp(s).date()) for s in snapshots} - allowed)
    if extra:
        raise LeakageError(f"model selection saw snapshots {extra}; only fit and validation snapshots are allowed")
    assert_target_windows_before(fit_snapshots, horizon_months, validation_snapshot)


def single_feature_auc(X: pd.DataFrame, y) -> pd.DataFrame:
    """AUC of each feature used alone, folded so 0.5 is no signal and 1.0 a perfect ranker either way."""
    rows = []
    for col in X.columns:
        auc = float(roc_auc_score(y, X[col])) if X[col].nunique() > 1 else 0.5
        rows.append({"feature": col, "auc": max(auc, 1.0 - auc)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False, kind="stable").reset_index(drop=True)


def assert_no_single_feature_leak(table: pd.DataFrame, threshold: float = 0.80) -> None:
    """Fail if any one feature ranks validation almost on its own: likely a label in disguise."""
    leaked = table.loc[table["auc"] > threshold, "feature"].tolist()
    if leaked:
        raise LeakageError(f"features {leaked} reach validation AUC above {threshold} alone; audit them first")


def assert_plausible_auc(auc: float, ceiling: float = 0.85) -> None:
    """Fail above the ceiling: the published best on full MBD is 0.824, so higher means audit first."""
    if auc > ceiling:
        raise LeakageError(f"validation AUC {auc:.3f} is above {ceiling}; audit for leakage before reporting it")
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/lead_scoring/config.py src/lead_scoring/temporal.py tests/test_temporal.py
git commit -m "Add monthly snapshots, a validation month and guards that keep selection off the test month"
```

---

### Task 3: Feature build — labels, windowed history, category mix, trends

**Files:**
- Modify: `src/lead_scoring/data.py`, `tests/conftest.py`
- Test: `tests/test_data_contracts.py`

**Interfaces:**
- Produces (data.py): `CATEGORY_COLUMNS: list[str]` (11 names), `TOP_CATEGORIES = 10`, `LABELS: list[str]` = `["converted", "y3_target_1".."y3_target_4", "y1_target_1".."y1_target_4"]`;
  `category_vocab(mbd_dir=MBD_DIR, before=PipelineConfig.fit_snapshots[-1], top=TOP_CATEGORIES) -> dict[str, list[int]]` (most frequent first);
  `build_customer_features(..., vocab: dict[str, list[int]] | None = None)`.
- New columns: `hist_target_{k}_m1/_m2`, `hist_any_m1/_m2`, `cat_{col}_{code}_365/_90`, `txn_count_d30_{1..6}`, `txn_amt_d30_{1..6}`, `ratio_30_90`, `ratio_90_365`, `days_since_evt_{code}`, plus the 8 new labels.

- [ ] **Step 1: Fixture gains all categorical columns** — in `tests/conftest.py` replace `_tx` and the `a` rows:

```python
CATEGORY_DEFAULTS = {"event_subtype": 2, "currency": 11, "src_type11": 3, "src_type12": 4, "dst_type11": 5,
                     "dst_type12": 6, "src_type21": 7, "src_type22": 8, "src_type31": 9, "src_type32": 10}


def _tx(client, fold, utc_time, amount, event_type=1):
    return {"client_id": client, "fold": fold, "event_time": pd.Timestamp(utc_time), "amount": float(amount),
            "event_type": event_type, **CATEGORY_DEFAULTS}
```

```python
        _tx("a", 0, "2022-01-31 20:00:00", 50, event_type=2),   # 23:00 Moscow on snapshot day: counted
        _tx("a", 0, "2022-01-31 22:00:00", 999, event_type=3),  # 01:00 Moscow on Feb 1: after snapshot
```

Run: `.venv/bin/pytest -q` — Expected: all still pass.

- [ ] **Step 2: Write the failing tests** — in `tests/test_data_contracts.py`, change the import to `from lead_scoring.data import CATEGORY_COLUMNS, LABELS, amount_cap, build_customer_features, category_vocab`, give the helper a vocab, and replace `test_scoring_features_have_no_targets`:

```python
EMPTY_VOCAB = {c: [] for c in CATEGORY_COLUMNS}


def _features(mbd_dir, snapshot=SNAP, include_target=True, cap=1_000.0, config=CFG, vocab=None):
    df = build_customer_features(snapshot, include_target=include_target, config=config, mbd_dir=mbd_dir,
                                 amount_cap=cap, vocab=vocab)
    return df.set_index("customer_id")


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
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/pytest tests/test_data_contracts.py -q`
Expected: ImportError for `CATEGORY_COLUMNS`.

- [ ] **Step 4: Implement in `src/lead_scoring/data.py`** — add `import json` and `import numpy as np`; after `TARGETS` add:

```python
CATEGORY_COLUMNS = ["event_type", "event_subtype", "currency", "src_type11", "src_type12", "dst_type11",
                    "dst_type12", "src_type21", "src_type22", "src_type31", "src_type32"]
TOP_CATEGORIES = 10  # top 20 would hold ~3 GB of int64 during the build on an 8 GB machine
# Any product within the horizon (headline), each product within the horizon, each product next month.
LABELS = ["converted", *(f"y3_{t}" for t in TARGETS), *(f"y1_{t}" for t in TARGETS)]
```

After `amount_cap`, add:

```python
def category_vocab(mbd_dir: Path = MBD_DIR, before: str = PipelineConfig.fit_snapshots[-1],
                   top: int = TOP_CATEGORIES) -> dict[str, list[int]]:
    """Most frequent values of each categorical column, counting events up to the end of the `before` day."""
    trx = str(mbd_dir / "detail" / "trx" / "fold=*" / "*.parquet")
    cutoff = pd.Timestamp(before).normalize() + pd.Timedelta(days=1)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    vocab = {}
    for col in CATEGORY_COLUMNS:
        rows = con.execute(f"""SELECT {col} FROM read_parquet($trx)
                               WHERE {LOCAL_TS} < $cutoff AND {col} IS NOT NULL
                               GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {int(top)}""",
                           {"trx": trx, "cutoff": cutoff}).fetchall()
        vocab[col] = [int(r[0]) for r in rows]
    con.close()
    return vocab


def _block(i: int) -> str:
    """Count and amount in the i-th 30-day block before the snapshot (1 = the latest)."""
    upper = "$cutoff" if i == 1 else f"$d30_{i - 1}"
    window = f"ts >= $d30_{i} AND ts < {upper}"
    return f"count(*) FILTER (WHERE {window}) AS txn_count_d30_{i}, sum(amount) FILTER (WHERE {window}) AS txn_amt_d30_{i}"
```

Replace `_fold_features` with:

```python
def _fold_features(fold: int, *, mbd_dir: Path, snapshot: str, include_target: bool,
                   config: PipelineConfig, cap: float, vocab: dict[str, list[int]]) -> pd.DataFrame:
    snap = pd.Timestamp(snapshot)
    day_after = snap + pd.Timedelta(days=1)
    horizon_end = month_end(snap, config.horizon_months)
    trx = str(mbd_dir / "detail" / "trx" / f"fold={fold}" / "*.parquet")
    targets = str(mbd_dir / "targets" / f"fold={fold}" / "*.parquet")
    params = {
        "trx": trx, "targets": targets, "cap": cap,
        "start": day_after - pd.Timedelta(days=config.lookback_days), "start90": day_after - pd.Timedelta(days=90),
        "cutoff": day_after, "horizon_cutoff": horizon_end + pd.Timedelta(days=1),
        "snap_mon": snap.date().isoformat(), "horizon_mon": horizon_end.date().isoformat(),
        "next_mon": month_end(snap, 1).date().isoformat(),
        # The two label months before the snapshot month; the snapshot month itself is never used.
        "m1": month_end(snap, -1).date().isoformat(), "m2": month_end(snap, -2).date().isoformat(),
        **{f"d30_{i}": day_after - pd.Timedelta(days=30 * i) for i in range(1, 7)},
    }
    any_product = "greatest(" + ", ".join(TARGETS) + ")"
    history = [*((t, t) for t in TARGETS), ("any", any_product)]
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    first_label = str(con.execute("SELECT min(mon) FROM read_parquet($targets)", {"targets": targets}).fetchone()[0])
    clients = con.execute(f"""
        SELECT client_id,
               sum({any_product}) FILTER (WHERE mon < $snap_mon) AS products_taken_before,
               {", ".join(f"sum({t}) FILTER (WHERE mon < $snap_mon) AS {t}_before" for t in TARGETS)},
               {", ".join(f"max({col}) FILTER (WHERE mon = ${m}) AS hist_{name}_{m}" for name, col in history for m in ("m1", "m2"))},
               max({any_product}) FILTER (WHERE mon > $snap_mon AND mon <= $horizon_mon) AS converted,
               {", ".join(f"max({t}) FILTER (WHERE mon > $snap_mon AND mon <= $horizon_mon) AS y3_{t}" for t in TARGETS)},
               {", ".join(f"max({t}) FILTER (WHERE mon = $next_mon) AS y1_{t}" for t in TARGETS)}
        FROM read_parquet($targets) GROUP BY client_id""",
        {k: params[k] for k in ("targets", "snap_mon", "horizon_mon", "next_mon", "m1", "m2")}).df()
    aggs = [
        "sum(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS total_txn_amt_365",
        "count(*) FILTER (WHERE ts >= $start AND ts < $cutoff) AS txn_count_365",
        "avg(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS avg_txn_amt_365",
        "max(amount) FILTER (WHERE ts >= $start AND ts < $cutoff) AS max_txn_amt_365",
        "count(DISTINCT CAST(ts AS DATE)) FILTER (WHERE ts >= $start AND ts < $cutoff) AS active_days_365",
        "count(*) FILTER (WHERE ts >= $start90 AND ts < $cutoff) AS txn_count_90",
        "sum(amount) FILTER (WHERE ts >= $start90 AND ts < $cutoff) AS total_txn_amt_90",
        "min(ts) FILTER (WHERE ts < $cutoff) AS first_txn",
        "max(ts) FILTER (WHERE ts < $cutoff) AS last_txn",
        "sum(amount) FILTER (WHERE ts >= $cutoff AND ts < $horizon_cutoff) AS future_value",
        *(_block(i) for i in range(1, 7)),
        *(f"count(*) FILTER (WHERE ts >= ${w} AND ts < $cutoff AND {col} = {int(code)}) AS cat_{col}_{int(code)}_{days}"
          for col, codes in vocab.items() for code in codes for w, days in (("start", 365), ("start90", 90))),
        *(f"max(ts) FILTER (WHERE ts < $cutoff AND event_type = {int(code)}) AS last_evt_{int(code)}"
          for code in vocab["event_type"]),
    ]
    tx = con.execute(f"""
        WITH tx AS (SELECT client_id, {LOCAL_TS} AS ts, least(amount, $cap) AS amount, {", ".join(CATEGORY_COLUMNS)}
                    FROM read_parquet($trx))
        SELECT client_id, {", ".join(aggs)} FROM tx GROUP BY client_id""",
        {k: params[k] for k in ("trx", "cap", "start", "start90", "cutoff", "horizon_cutoff", *(f"d30_{i}" for i in range(1, 7)))}).df()
    con.close()

    df = clients.merge(tx, on="client_id", how="left").rename(columns={"client_id": "customer_id"})
    assert_events_before(df["last_txn"], snap)
    filled = [c for c in df.columns if c.endswith(("_365", "_90", "_before"))
              or c.startswith(("cat_", "txn_count_d30_", "txn_amt_d30_")) or c in (*LABELS, "future_value")]
    df[filled] = df[filled].fillna(0)
    for m in ("m1", "m2"):
        cols = [c for c in df.columns if c.startswith("hist_") and c.endswith(f"_{m}")]
        # A month before the first label month has no labels at all: unknown, not "bought nothing".
        df[cols] = np.nan if params[m] < first_label else df[cols].fillna(0)
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    df[cat_cols] = df[cat_cols].astype("int32")
    df["recency_days"] = (snap - df["last_txn"]).dt.days.fillna(config.lookback_days + 1).clip(lower=0)
    df["tenure_days"] = (snap - df["first_txn"]).dt.days.fillna(0).clip(lower=0)
    df["ratio_30_90"] = df["txn_count_d30_1"] / df["txn_count_90"].clip(lower=1)
    df["ratio_90_365"] = df["txn_count_90"] / df["txn_count_365"].clip(lower=1)
    for code in vocab["event_type"]:
        last = pd.to_datetime(df.pop(f"last_evt_{int(code)}"))
        df[f"days_since_evt_{int(code)}"] = (snap - last).dt.days.fillna(config.lookback_days + 1).clip(lower=0)
    df["fold"] = fold
    if include_target:
        df[LABELS] = df[LABELS].astype(int)
    else:
        df = df.drop(columns=["future_value", *LABELS])
    return df
```

In `build_customer_features`, add the keyword `vocab: dict[str, list[int]] | None = None`, and before `work = partial(...)`:

```python
    vocab = vocab if vocab is not None else category_vocab(mbd_dir, before=config.fit_snapshots[-1])
```

Pass `vocab=vocab` into `partial(_fold_features, ...)`. In `save_customer_features`, compute `vocab = category_vocab(mbd_dir, before=config.fit_snapshots[-1])` once after `cap`, pass `vocab=vocab` to both `build_customer_features` calls, and add `"category_vocab": json.dumps(vocab)` to `metadata`.

- [ ] **Step 5: Run the suite**

Run: `.venv/bin/pytest -q`
Expected: all pass, including the unchanged `test_parallel_folds_match_serial` and `test_product_history_uses_only_earlier_label_months`.

- [ ] **Step 6: Commit**

```bash
git add src/lead_scoring/data.py tests/conftest.py tests/test_data_contracts.py
git commit -m "Build windowed product history, per-product labels, category-mix and trend features"
```

---

### Task 4: Model matrices from the new features

**Files:**
- Modify: `src/lead_scoring/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `LABELS` from `lead_scoring.data`.
- Produces: `X_*.parquet` as float32; `y_labels_train.parquet`, `y_labels_test.parquet` (columns `LABELS`, int8); `train_ids.csv` keeps `snapshot_date` (selection uses it). `CUMULATIVE_HISTORY` set is forbidden as features.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_features.py` (import `transform_features` too):

```python
def test_cumulative_history_is_forbidden_but_windowed_history_is_a_feature():
    df = _stacked(["2022-01-31", "2022-04-30"]).assign(
        products_taken_before=1, target_1_before=1, hist_any_m1=[None, 1.0], cat_event_type_1_365=3,
        txn_count_d30_1=1, ratio_30_90=0.5, days_since_evt_1=4, y3_target_1=1, y1_target_1=0)
    spec = fit_preprocessor(df)
    assert not {"products_taken_before", "target_1_before", "y3_target_1", "y1_target_1"} & set(spec["feature_cols"])
    assert {"hist_any_m1", "hist_any_m1_missing", "cat_event_type_1_365", "txn_count_d30_1",
            "ratio_30_90", "days_since_evt_1"} <= set(spec["feature_cols"])


def test_model_matrix_is_float32():
    df = _stacked(["2022-01-31", "2022-04-30"])
    assert (transform_features(df, fit_preprocessor(df)).dtypes == "float32").all()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/pytest tests/test_features.py -q`
Expected: FAIL — `products_taken_before` is in `feature_cols`.

- [ ] **Step 3: Implement in `src/lead_scoring/features.py`** — add `from lead_scoring.data import LABELS`; replace `TARGET_COLUMNS`, the candidate list, and `FORBIDDEN_COLUMNS`:

```python
TARGET_COLUMNS = {"future_value", *LABELS}
# Cumulative history spans 0 months at the Jan-2022 snapshot and 8 at the test one (labels start 2022-02),
# so the same value means different things in training and test. Kept for display and the rule baseline only.
CUMULATIVE_HISTORY = {"products_taken_before", "target_1_before", "target_2_before", "target_3_before", "target_4_before"}
FORBIDDEN_COLUMNS = TARGET_COLUMNS | IDENTIFIER_COLUMNS | PII_COLUMNS | DATE_COLUMNS | UNEXPLAINED_COLUMNS | CUMULATIVE_HISTORY

NUMERIC_FEATURE_CANDIDATES = [
    "total_txn_amt_365", "txn_count_365", "avg_txn_amt_365", "max_txn_amt_365", "active_days_365",
    "txn_count_90", "total_txn_amt_90", "recency_days", "tenure_days", "ratio_30_90", "ratio_90_365",
]
# Built per vocabulary and window by lead_scoring.data, so they are matched by prefix.
GENERATED_FEATURE_PREFIXES = ("hist_", "cat_", "txn_count_d30_", "txn_amt_d30_", "days_since_evt_")
```

In `fit_preprocessor`, replace the `numeric_cols = ...` line with:

```python
    numeric_cols = _available(train_df.columns, NUMERIC_FEATURE_CANDIDATES)
    numeric_cols += [c for c in train_df.columns if c.startswith(GENERATED_FEATURE_PREFIXES)]
```

In `transform_features`, make the return `return out.reindex(columns=spec["feature_cols"], fill_value=0.0).astype("float32")`.

In `prepare_feature_sets`, add `"y_labels_train"` and `"y_labels_test"` to `paths` (`PREPROCESSED_DATA_DIR / "y_labels_train.parquet"` and `PREPROCESSED_DATA_DIR / "y_labels_test.parquet"`), and after the existing target writes:

```python
    train_df[LABELS].astype("int8").to_parquet(paths["y_labels_train"], index=False)
    test_df[LABELS].astype("int8").to_parquet(paths["y_labels_test"], index=False)
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Build the real features and matrices (about 10 min; watch memory)**

```bash
.venv/bin/python -m lead_scoring.data && .venv/bin/python -m lead_scoring.features
.venv/bin/python -c "import pandas as pd; X = pd.read_parquet('data/processed/preprocessed/X_train.parquet'); print(X.shape, X.dtypes.unique(), round(X.memory_usage().sum() / 1e9, 2), 'GB')"
```
Expected: about 700,000 rows × about 270 columns, `float32`, under 1 GB. If the build is killed for memory, lower `TOP_CATEGORIES` to 5 and report it.

- [ ] **Step 6: Commit**

```bash
git add src/lead_scoring/features.py tests/test_features.py
git commit -m "Feed windowed history and transaction-mix features to the models as float32"
```

---

### Task 5: Blend math and LightGBM selection on validation — CHECKPOINT 1

**Files:**
- Create: `src/lead_scoring/blend.py`, `scripts/validate_propensity.py`
- Test: `tests/test_blend.py`

**Interfaces:**
- Produces (blend.py): `WEIGHT_GRID`, `MIN_GAIN = 0.005`, `rank_pct(scores) -> np.ndarray`,
  `blend(lgbm, gru, weight: float) -> np.ndarray` (returns `rank_pct(lgbm)` when `weight == 0`, so `gru` may be `None`),
  `choose_weight(y, lgbm, gru=None) -> {"weight", "auc", "lgbm_auc", "gru_auc"}`,
  `fit_calibrator(y, blended) -> {"x": list, "y": list}`, `calibrate(blended, calibrator) -> np.ndarray`.
- Produces (script): `models/propensity_choice.json` = `{"lgbm_params": dict, "lgbm_rounds": int, "blend": choose_weight(...), "calibrator": {...}, "validation_snapshot": str}`; `reports/tables/propensity_validation.csv`; `reports/tables/single_feature_auc.csv`.

> Learn by Doing: when executing, offer the user the simplicity-rule lines in `choose_weight` (the `if aucs[best] - lgbm_auc < min_gain` branch) before writing them.

- [ ] **Step 1: Write the failing tests** — create `tests/test_blend.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/pytest tests/test_blend.py -q`
Expected: ModuleNotFoundError `lead_scoring.blend`.

- [ ] **Step 3: Implement `src/lead_scoring/blend.py`**

```python
"""Blend two propensity rankers and turn the blend into a calibrated probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

WEIGHT_GRID = tuple(float(w) for w in np.round(np.arange(0, 1.0001, 0.05), 2))
MIN_GAIN = 0.005  # validation AUC the GRU must add to earn a place in production


def rank_pct(scores) -> np.ndarray:
    """Percentile ranks, so models on different scales blend fairly."""
    return pd.Series(np.asarray(scores, dtype=float)).rank(method="average", pct=True).to_numpy()


def blend(lgbm, gru, weight: float) -> np.ndarray:
    if weight == 0:
        return rank_pct(lgbm)
    return weight * rank_pct(gru) + (1 - weight) * rank_pct(lgbm)


def choose_weight(y, lgbm, gru=None, grid=WEIGHT_GRID, min_gain: float = MIN_GAIN) -> dict:
    """GRU weight with the best validation AUC; 0 unless the GRU adds at least `min_gain`."""
    lgbm_auc = float(roc_auc_score(y, lgbm))
    if gru is None:
        return {"weight": 0.0, "auc": lgbm_auc, "lgbm_auc": lgbm_auc, "gru_auc": None}
    aucs = {w: float(roc_auc_score(y, blend(lgbm, gru, w))) for w in grid}
    best = max(aucs, key=lambda w: (aucs[w], -w))  # ties go to the smaller GRU weight
    if aucs[best] - lgbm_auc < min_gain:
        return {"weight": 0.0, "auc": lgbm_auc, "lgbm_auc": lgbm_auc, "gru_auc": aucs[1.0]}
    return {"weight": best, "auc": aucs[best], "lgbm_auc": lgbm_auc, "gru_auc": aucs[1.0]}


def fit_calibrator(y, blended) -> dict[str, list[float]]:
    """Isotonic map from blend rank to conversion rate, stored as JSON-friendly breakpoints."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(blended, y)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def calibrate(blended, calibrator: dict[str, list[float]]) -> np.ndarray:
    return np.interp(np.asarray(blended, dtype=float), calibrator["x"], calibrator["y"])
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_blend.py -q`
Expected: 5 pass.

- [ ] **Step 5: Create `scripts/validate_propensity.py`**

```python
"""Choose the propensity model on the validation snapshot, never on the test months.

Fits LightGBM on the fit snapshots (Jan–Apr 2022) and validates on Jul 2022. When
`scripts/train_gru.py --stage validate` has run, it also picks the GRU blend weight there.
Writes reports/tables/propensity_validation.csv, reports/tables/single_feature_auc.csv and
models/propensity_choice.json. Run after: python -m lead_scoring.data && python -m lead_scoring.features
"""

from pathlib import Path
import itertools
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lead_scoring.blend import blend, choose_weight, fit_calibrator
from lead_scoring.config import PipelineConfig
from lead_scoring.metrics import lift_at_fraction
from lead_scoring.paths import GRU_VAL_SCORES_FILE, PREPROCESSED_DATA_DIR, PROPENSITY_CHOICE_FILE, TABLES_DIR
from lead_scoring.temporal import (
    assert_no_single_feature_leak,
    assert_plausible_auc,
    assert_selection_snapshots,
    single_feature_auc,
)

BASE = dict(objective="binary", metric="auc", learning_rate=0.05, feature_fraction=0.5,
            deterministic=True, force_row_wise=True, verbose=-1, seed=42)
GRID = [dict(num_leaves=leaves, min_child_samples=leaf_rows) for leaves, leaf_rows in itertools.product((31, 63), (100, 400))]


def _row(model: str, y, score, **extra) -> dict:
    return {"model": model, **extra, "val_auc": float(roc_auc_score(y, score)),
            "val_lift_at_10pct": float(lift_at_fraction(y, score, 0.10))}


def main() -> None:
    cfg = PipelineConfig()
    X = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(PREPROCESSED_DATA_DIR / "y_clf_train.parquet")["converted"].to_numpy()
    ids = pd.read_csv(PREPROCESSED_DATA_DIR / "train_ids.csv")
    fit = ids["snapshot_date"].isin(cfg.fit_snapshots).to_numpy()
    val = (ids["snapshot_date"] == cfg.validation_snapshot).to_numpy()
    assert_selection_snapshots(ids.loc[fit | val, "snapshot_date"].unique(), cfg.fit_snapshots,
                               cfg.validation_snapshot, cfg.horizon_months)

    leaks = single_feature_auc(X[val], y[val])
    leaks.to_csv(TABLES_DIR / "single_feature_auc.csv", index=False)
    assert_no_single_feature_leak(leaks)

    train_set = lgb.Dataset(X[fit], y[fit], free_raw_data=False)
    val_set = lgb.Dataset(X[val], y[val], reference=train_set)
    rows, best = [], None
    for params in GRID:
        booster = lgb.train({**BASE, **params}, train_set, num_boost_round=3000, valid_sets=[val_set],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        score = booster.predict(X[val], num_iteration=booster.best_iteration)
        row = _row("lightgbm", y[val], score, **params, best_iteration=int(booster.best_iteration))
        rows.append(row)
        print(row, flush=True)
        if best is None or row["val_auc"] > best[0]["val_auc"]:
            best = (row, score)
    chosen, lgbm_val = best
    assert_plausible_auc(chosen["val_auc"])

    gru_val = None
    if GRU_VAL_SCORES_FILE.exists():
        scores = pd.read_parquet(GRU_VAL_SCORES_FILE)
        gru_val = ids.loc[val, ["customer_id", "snapshot_date"]].merge(
            scores, on=["customer_id", "snapshot_date"], how="left")["converted"].to_numpy()
        if np.isnan(gru_val).any():
            raise ValueError("GRU validation scores miss some validation clients; rerun train_gru.py --stage validate")
        rows.append(_row("gru", y[val], gru_val))
        assert_plausible_auc(rows[-1]["val_auc"])
    choice = choose_weight(y[val], lgbm_val, gru_val)
    blended = blend(lgbm_val, gru_val, choice["weight"])
    rows.append(_row("production_blend", y[val], blended, gru_weight=choice["weight"]))
    assert_plausible_auc(choice["auc"])

    table = pd.DataFrame(rows).assign(validation_snapshot=cfg.validation_snapshot)
    table.to_csv(TABLES_DIR / "propensity_validation.csv", index=False)
    PROPENSITY_CHOICE_FILE.write_text(json.dumps({
        "lgbm_params": {**BASE, "num_leaves": chosen["num_leaves"], "min_child_samples": chosen["min_child_samples"]},
        # More rows in the final refit (7 snapshots vs 4), so proportionally more rounds.
        "lgbm_rounds": int(round(chosen["best_iteration"] * len(cfg.train_snapshots) / len(cfg.fit_snapshots))),
        "blend": choice,
        "calibrator": fit_calibrator(y[val], blended),
        "validation_snapshot": cfg.validation_snapshot,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run it on the real data (LightGBM only — no GRU scores exist yet)**

Run: `.venv/bin/python scripts/validate_propensity.py`
Expected: four LightGBM rows plus `production_blend` with `gru_weight` 0.0; no `LeakageError`; `models/propensity_choice.json` written. Also print the top of `reports/tables/single_feature_auc.csv`.

- [ ] **Step 7: Commit**

```bash
git add src/lead_scoring/blend.py scripts/validate_propensity.py tests/test_blend.py models/propensity_choice.json reports/tables/propensity_validation.csv reports/tables/single_feature_auc.csv
git commit -m "Choose LightGBM settings and the blend on the validation month"
```

- [ ] **Step 8: CHECKPOINT 1 — stop and report to the user:** best validation AUC and lift at 10% with the chosen params, the top 5 single-feature AUCs, and peak memory during Task 4. Do not start Task 6 until the user says to continue.

---

### Task 6: Event store for transaction sequences

**Files:**
- Create: `src/lead_scoring/sequences.py`
- Test: `tests/test_sequences.py`

**Interfaces:**
- Consumes: `CATEGORY_COLUMNS`, `LOCAL_TS`, `list_folds` from `lead_scoring.data`.
- Produces: `VOCAB_SIZE = 1024`, `N_CONTINUOUS = 3`, `cutoff_ms(snapshot: str) -> int`,
  `build_event_store(out_dir: Path, *, mbd_dir: Path = MBD_DIR, cap: float, vocab: dict[str, list[int]]) -> Path`,
  `EventStore(store_dir)` with `.window(client: str, cutoff: int, max_len: int) -> (codes int64 [n, 11], cont float32 [n, 3])`, n ≥ 1,
  `collate(windows, max_len: int) -> (codes int64 [B, max_len, 11], cont float32 [B, max_len, 3])`, left-padded.

- [ ] **Step 1: Write the failing tests** — create `tests/test_sequences.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/pytest tests/test_sequences.py -q`
Expected: ModuleNotFoundError `lead_scoring.sequences`.

- [ ] **Step 3: Implement `src/lead_scoring/sequences.py`**

```python
"""Per-client transaction sequences for the GRU, cached as numpy arrays.

Events are sorted by client and time once, one fold at a time to stay inside 8 GB
of RAM. A sample is the last `max_len` events strictly before the end of the
snapshot day (Moscow time), found by binary search on time.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from lead_scoring.data import CATEGORY_COLUMNS, LOCAL_TS, list_folds
from lead_scoring.paths import MBD_DIR

VOCAB_SIZE = 1024  # 0 = padding, 1 = rare or unseen value, 2.. = the 1,022 most frequent values
N_CONTINUOUS = 3  # log amount, log hours since the previous event, log days before the snapshot


def cutoff_ms(snapshot: str) -> int:
    """End of the snapshot day in local-time epoch milliseconds; events must be strictly earlier."""
    return int((pd.Timestamp(snapshot).normalize() + pd.Timedelta(days=1)).value // 1_000_000)


def _encode(values: np.ndarray, vocab: list[int]) -> np.ndarray:
    if not vocab:
        return np.ones(len(values), np.int16)
    known = np.asarray(vocab, dtype=np.int64)
    order = np.argsort(known)
    pos = np.searchsorted(known[order], values).clip(max=len(known) - 1)
    hit = known[order][pos] == values
    return np.where(hit, 2 + order[pos], 1).astype(np.int16)


def build_event_store(out_dir: Path, *, mbd_dir: Path = MBD_DIR, cap: float, vocab: dict[str, list[int]]) -> Path:
    """Write one .npz per client fold: codes, time, log amount, clients and their offsets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in list_folds(mbd_dir):
        trx = str(mbd_dir / "detail" / "trx" / f"fold={fold}" / "*.parquet")
        con = duckdb.connect()
        con.execute("SET enable_progress_bar = false")
        counts = con.execute("SELECT client_id, count(*) AS n FROM read_parquet($trx) GROUP BY 1 ORDER BY 1",
                             {"trx": trx}).df()
        events = con.execute(f"""
            SELECT epoch_ms({LOCAL_TS}) AS t, ln(1 + least(amount, $cap)) AS log_amount,
                   {", ".join(f"coalesce({c}, -1) AS {c}" for c in CATEGORY_COLUMNS)}
            FROM read_parquet($trx) ORDER BY client_id, t""", {"trx": trx, "cap": cap}).fetchnumpy()
        con.close()
        np.savez(
            out_dir / f"fold={fold}.npz",
            codes=np.stack([_encode(events[c], vocab[c]) for c in CATEGORY_COLUMNS], axis=1),
            time=events["t"].astype(np.int64),
            log_amount=events["log_amount"].astype(np.float32),
            clients=counts["client_id"].astype(str).to_numpy(),
            offsets=np.concatenate([[0], np.cumsum(counts["n"].to_numpy())]),
        )
    return out_dir


class EventStore:
    """All folds' events in memory (about 1.2 GB for MBD-mini) with a client → row-range index."""

    def __init__(self, store_dir: Path):
        parts = [np.load(p) for p in sorted(Path(store_dir).glob("fold=*.npz"))]
        if not parts:
            raise FileNotFoundError(f"no event store in {store_dir}; run scripts/train_gru.py to build it")
        self.codes = np.concatenate([p["codes"] for p in parts])
        self.time = np.concatenate([p["time"] for p in parts])
        self.log_amount = np.concatenate([p["log_amount"] for p in parts])
        self.index: dict[str, tuple[int, int]] = {}
        base = 0
        for p in parts:
            offsets = p["offsets"]
            for client, start, end in zip(p["clients"], offsets[:-1], offsets[1:]):
                self.index[str(client)] = (base + int(start), base + int(end))
            base += len(p["time"])

    def window(self, client: str, cutoff: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = self.index.get(client, (0, 0))
        stop = start + int(np.searchsorted(self.time[start:end], cutoff, side="left"))
        begin = max(start, stop - max_len)
        if stop == begin:
            return np.zeros((1, len(CATEGORY_COLUMNS)), np.int64), np.zeros((1, N_CONTINUOUS), np.float32)
        t = self.time[begin:stop]
        gap_hours = np.diff(t, prepend=t[0]) / 3.6e6
        age_days = (cutoff - t) / 8.64e7
        cont = np.stack([self.log_amount[begin:stop], np.log1p(gap_hours), np.log1p(age_days)], axis=1)
        return self.codes[begin:stop].astype(np.int64), cont.astype(np.float32)


def collate(windows: list[tuple[np.ndarray, np.ndarray]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad every window to exactly max_len, so a score never depends on its batch-mates."""
    codes = np.zeros((len(windows), max_len, len(CATEGORY_COLUMNS)), np.int64)
    cont = np.zeros((len(windows), max_len, N_CONTINUOUS), np.float32)
    for i, (c, x) in enumerate(windows):
        codes[i, max_len - len(c):] = c
        cont[i, max_len - len(x):] = x
    return codes, cont
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_sequences.py -q`
Expected: 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/lead_scoring/sequences.py tests/test_sequences.py
git commit -m "Cache per-client transaction sequences for the GRU, one fold at a time"
```

---

### Task 7: GRU model and its training script — CHECKPOINT 2

**Files:**
- Create: `src/lead_scoring/gru.py`, `scripts/train_gru.py`
- Test: `tests/test_gru.py`

**Interfaces:**
- Consumes: `EventStore`, `collate`, `cutoff_ms`, `VOCAB_SIZE`, `N_CONTINUOUS` (Task 6); `LABELS`, `CATEGORY_COLUMNS`, `amount_cap`, `category_vocab` (Task 3).
- Produces (gru.py): `HEADS = LABELS`; `EventGRU(hidden=128, ...)` with `forward(codes, cont) -> logits [B, 9]`; `device()`;
  `train_epoch(model, opt, store, samples, labels, *, max_len, batch_size, seed, dev) -> float`;
  `predict(model, store, samples, *, max_len, batch_size, dev) -> np.ndarray [n, 9]` (probabilities, in `samples` order);
  `save(model, path, **meta)`; `load(path) -> (EventGRU, meta)`;
  `score_snapshot(client_ids: pd.Series, snapshot: str, *, model_path=GRU_MODEL_FILE, store_dir=EVENT_STORE_DIR) -> np.ndarray [n, 9]`.
- Produces (script): `reports/tables/gru_validation.csv` (epoch, train_loss, val_auc, val_lift_at_10pct, epoch_seconds); `data/processed/gru_val_scores.parquet` (customer_id, snapshot_date, converted); `models/gru_final.pt`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_gru.py`:

```python
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
```

Note: `load` must rebuild with `hidden` from `meta`, so `save` stores `hidden=model.gru.hidden_size`.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/pytest tests/test_gru.py -q`
Expected: ModuleNotFoundError `lead_scoring.gru`.

- [ ] **Step 3: Implement `src/lead_scoring/gru.py`**

```python
"""GRU over a client's transaction sequence, with one logit per prediction target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from lead_scoring.data import CATEGORY_COLUMNS, LABELS
from lead_scoring.paths import EVENT_STORE_DIR, GRU_MODEL_FILE
from lead_scoring.sequences import N_CONTINUOUS, VOCAB_SIZE, EventStore, collate, cutoff_ms

HEADS = LABELS  # any product in 3 months (blended), each product in 3 months, each product next month


class EventGRU(nn.Module):
    def __init__(self, hidden: int = 128, emb_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=0) for _ in CATEGORY_COLUMNS)
        self.gru = nn.GRU(len(CATEGORY_COLUMNS) * emb_dim + N_CONTINUOUS, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, len(HEADS)))

    def forward(self, codes: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb(codes[..., i]) for i, emb in enumerate(self.embeddings)] + [cont], dim=-1)
        return self.head(self.gru(x)[1][-1])


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _batches(store: EventStore, samples: pd.DataFrame, max_len: int, batch_size: int, shuffle: bool, seed: int):
    clients = samples["customer_id"].astype(str).to_numpy()
    cutoffs = samples["snapshot_date"].map({s: cutoff_ms(s) for s in samples["snapshot_date"].unique()}).to_numpy()
    order = np.random.default_rng(seed).permutation(len(samples)) if shuffle else np.arange(len(samples))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        codes, cont = collate([store.window(clients[j], int(cutoffs[j]), max_len) for j in idx], max_len)
        yield torch.from_numpy(codes), torch.from_numpy(cont), idx


def train_epoch(model, opt, store, samples, labels: np.ndarray, *, max_len: int, batch_size: int,
                seed: int, dev: torch.device) -> float:
    model.train()
    loss_fn, total = nn.BCEWithLogitsLoss(), 0.0
    for codes, cont, idx in _batches(store, samples, max_len, batch_size, True, seed):
        opt.zero_grad()
        loss = loss_fn(model(codes.to(dev), cont.to(dev)), torch.from_numpy(labels[idx]).to(dev))
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / len(samples)


@torch.no_grad()
def predict(model, store, samples, *, max_len: int, batch_size: int, dev: torch.device) -> np.ndarray:
    model.eval()
    out = np.zeros((len(samples), len(HEADS)), np.float32)
    for codes, cont, idx in _batches(store, samples, max_len, batch_size, False, 0):
        out[idx] = torch.sigmoid(model(codes.to(dev), cont.to(dev))).cpu().numpy()
    return out


def save(model: EventGRU, path: Path, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": {"hidden": model.gru.hidden_size, **meta}}, path)


def load(path: Path) -> tuple[EventGRU, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model = EventGRU(hidden=ckpt["meta"]["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["meta"]


def score_snapshot(client_ids: pd.Series, snapshot: str, *, model_path: Path = GRU_MODEL_FILE,
                   store_dir: Path = EVENT_STORE_DIR) -> np.ndarray:
    """All heads' probabilities for each client at a snapshot, on CPU so reruns are identical."""
    model, meta = load(model_path)
    samples = pd.DataFrame({"customer_id": client_ids.astype(str).to_numpy(), "snapshot_date": snapshot})
    return predict(model, EventStore(store_dir), samples, max_len=meta["max_len"], batch_size=512,
                   dev=torch.device("cpu"))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_gru.py -q`
Expected: 3 pass.

- [ ] **Step 5: Create `scripts/train_gru.py`**

```python
"""Train the transaction GRU: choose the epoch count on validation, then refit for production.

--stage validate  fit on the fit snapshots (Jan–Apr 2022), score Jul 2022 after every epoch and keep
                  the best epoch's validation scores for scripts/validate_propensity.py.
--stage final     refit on all 7 training snapshots for the chosen number of epochs.
Run after `python -m lead_scoring.data`. Builds data/processed/events on first use; delete that folder
to rebuild it after changing the amount cap or vocabulary.
"""

from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from lead_scoring.config import PipelineConfig
from lead_scoring.data import LABELS, amount_cap, category_vocab
from lead_scoring.gru import EventGRU, device, predict, save, train_epoch
from lead_scoring.metrics import lift_at_fraction
from lead_scoring.paths import CUSTOMER_FEATURES_FILE, EVENT_STORE_DIR, GRU_MODEL_FILE, GRU_VAL_SCORES_FILE, TABLES_DIR
from lead_scoring.sequences import VOCAB_SIZE, EventStore, build_event_store
from lead_scoring.temporal import assert_plausible_auc, assert_selection_snapshots


def _store(cfg: PipelineConfig) -> EventStore:
    if not any(EVENT_STORE_DIR.glob("fold=*.npz")):
        print("Building the event store (one fold at a time)...", flush=True)
        build_event_store(EVENT_STORE_DIR, cap=amount_cap(before=cfg.test_snapshot),
                          vocab=category_vocab(before=cfg.fit_snapshots[-1], top=VOCAB_SIZE - 2))
    return EventStore(EVENT_STORE_DIR)


def _fit(cfg, store, samples, epochs, args, on_epoch=None) -> EventGRU:
    torch.manual_seed(cfg.random_state)
    dev = device()
    model = EventGRU().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    labels = samples[LABELS].to_numpy(np.float32)
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train_epoch(model, opt, store, samples, labels, max_len=args.max_len, batch_size=args.batch_size,
                           seed=cfg.random_state + epoch, dev=dev)
        if on_epoch:
            on_epoch(epoch, model, loss, time.time() - start)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["validate", "final"], required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="train on a random subset of this many rows (smoke runs)")
    args = parser.parse_args()
    cfg = PipelineConfig()
    table = pd.read_parquet(CUSTOMER_FEATURES_FILE, columns=["customer_id", "snapshot_date", *LABELS])
    store = _store(cfg)

    if args.stage == "validate":
        assert_selection_snapshots([*cfg.fit_snapshots, cfg.validation_snapshot], cfg.fit_snapshots,
                                   cfg.validation_snapshot, cfg.horizon_months)
        fit = table[table["snapshot_date"].isin(cfg.fit_snapshots)]
        if args.limit:
            fit = fit.sample(n=min(args.limit, len(fit)), random_state=cfg.random_state)
        fit = fit.reset_index(drop=True)
        val = table[table["snapshot_date"] == cfg.validation_snapshot].reset_index(drop=True)
        rows, best = [], {"val_auc": -1.0}

        def on_epoch(epoch, model, loss, seconds):
            nonlocal best
            scores = predict(model, store, val, max_len=args.max_len, batch_size=512, dev=device())[:, 0]
            row = {"epoch": epoch, "train_loss": loss, "val_auc": float(roc_auc_score(val["converted"], scores)),
                   "val_lift_at_10pct": float(lift_at_fraction(val["converted"], scores, 0.10)), "epoch_seconds": seconds}
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(TABLES_DIR / "gru_validation.csv", index=False)
            if row["val_auc"] > best["val_auc"]:
                best = row
                val[["customer_id", "snapshot_date"]].assign(converted=scores).to_parquet(GRU_VAL_SCORES_FILE, index=False)

        _fit(cfg, store, fit, args.epochs, args, on_epoch)
        assert_plausible_auc(best["val_auc"])
    else:
        history = pd.read_csv(TABLES_DIR / "gru_validation.csv")
        epochs = int(history.loc[history["val_auc"].idxmax(), "epoch"])
        train = table[table["snapshot_date"].isin(cfg.train_snapshots)].reset_index(drop=True)
        model = _fit(cfg, store, train, epochs, args)
        save(model.cpu(), GRU_MODEL_FILE, max_len=args.max_len, epochs=epochs, snapshots=list(cfg.train_snapshots))
        print(f"Saved {GRU_MODEL_FILE} after {epochs} epochs on {len(train):,} sequences")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke run (builds the event store, one short epoch)**

Run: `.venv/bin/python scripts/train_gru.py --stage validate --epochs 1 --limit 20000`
Expected: the event store builds without running out of memory; one row printed with `epoch_seconds`. Multiply `epoch_seconds` by (number of fit rows ÷ 20,000) to estimate a full epoch.

- [ ] **Step 7: Full validation run in the background (about 1.5 hours if the estimate holds)**

Only if the estimated full epoch is under 30 minutes. Otherwise stop and go to the checkpoint.

Run (in the background): `.venv/bin/python scripts/train_gru.py --stage validate --epochs 12`
Expected: 12 rows in `reports/tables/gru_validation.csv`; `data/processed/gru_val_scores.parquet` from the best epoch.

- [ ] **Step 8: Commit**

```bash
git add src/lead_scoring/gru.py scripts/train_gru.py tests/test_gru.py reports/tables/gru_validation.csv
git commit -m "Add the transaction GRU and pick its epoch count on the validation month"
```

- [ ] **Step 9: CHECKPOINT 2 — stop and report to the user:** real seconds per epoch, best epoch, GRU validation AUC and lift, compared with LightGBM's from Checkpoint 1. If an epoch took over 30 minutes, present the Colab and AWS options with costs; spend nothing without a yes. Wait for the user before Task 8.

---

### Task 8: Production scorer and pipeline wiring

**Files:**
- Create: `src/lead_scoring/propensity.py`
- Modify: `src/lead_scoring/training.py`, `src/lead_scoring/evaluation.py`, `src/lead_scoring/scoring.py`, `src/lead_scoring/pipeline.py`, `src/lead_scoring/paths.py`, `scripts/repo_health_check.py` (only if it names `propensity_model.pkl`)
- Test: `tests/test_propensity.py`

**Interfaces:**
- Consumes: `blend`, `calibrate` (Task 5); `HEADS`, `score_snapshot` (Task 7); `models/propensity_choice.json`.
- Produces: `load_choice() -> dict`; `score_propensity(X, client_ids, snapshot) -> pd.DataFrame` with columns `lgbm_score`, `gru_<head>` for each of the 9 heads (only when `models/gru_final.pt` exists), `blend_score`, `predicted_propensity`. `test_predictions.csv` gains those score columns. `evaluate_models(config)` and `score_batch(config)` take the config.

- [ ] **Step 1: Write the failing tests** — create `tests/test_propensity.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from lead_scoring import propensity


def _setup(tmp_path, monkeypatch, weight):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.random(200)})
    y = (X["f"] + rng.normal(0, 0.2, 200) > 0.6).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=5)
    booster.save_model(str(tmp_path / "lgbm.txt"))
    (tmp_path / "choice.json").write_text(json.dumps({"blend": {"weight": weight}, "calibrator": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}))
    monkeypatch.setattr(propensity, "PROPENSITY_LGBM_FILE", tmp_path / "lgbm.txt")
    monkeypatch.setattr(propensity, "PROPENSITY_CHOICE_FILE", tmp_path / "choice.json")
    monkeypatch.setattr(propensity, "GRU_MODEL_FILE", tmp_path / "missing.pt")
    return X


def test_lgbm_only_scores_are_ranked_and_calibrated(tmp_path, monkeypatch):
    X = _setup(tmp_path, monkeypatch, weight=0.0)
    out = propensity.score_propensity(X, pd.Series([str(i) for i in range(len(X))]), "2022-10-31")
    assert list(out.columns) == ["lgbm_score", "blend_score", "predicted_propensity"]
    assert out["blend_score"].corr(out["lgbm_score"], method="spearman") == pytest.approx(1.0)
    assert out["predicted_propensity"].between(0, 0.5).all()


def test_a_blend_without_a_trained_gru_says_what_to_run(tmp_path, monkeypatch):
    X = _setup(tmp_path, monkeypatch, weight=0.4)
    with pytest.raises(FileNotFoundError, match="train_gru.py --stage final"):
        propensity.score_propensity(X, pd.Series(["a"] * len(X)), "2022-10-31")


def test_a_missing_choice_file_says_what_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(propensity, "PROPENSITY_CHOICE_FILE", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="validate_propensity.py"):
        propensity.load_choice()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/pytest tests/test_propensity.py -q`
Expected: ImportError `cannot import name 'propensity'`.

- [ ] **Step 3: Implement `src/lead_scoring/propensity.py`**

```python
"""Production propensity: LightGBM, blended with the transaction GRU when validation chose it, then calibrated."""

from __future__ import annotations

import json

import lightgbm as lgb
import pandas as pd

from lead_scoring.blend import blend, calibrate
from lead_scoring.paths import GRU_MODEL_FILE, PROPENSITY_CHOICE_FILE, PROPENSITY_LGBM_FILE


def load_choice() -> dict:
    if not PROPENSITY_CHOICE_FILE.exists():
        raise FileNotFoundError(f"{PROPENSITY_CHOICE_FILE} is missing; run scripts/validate_propensity.py first")
    return json.loads(PROPENSITY_CHOICE_FILE.read_text(encoding="utf-8"))


def score_propensity(X: pd.DataFrame, client_ids: pd.Series, snapshot: str) -> pd.DataFrame:
    """LightGBM score, GRU head probabilities (when trained), blend rank and calibrated propensity."""
    choice = load_choice()
    weight = choice["blend"]["weight"]
    out = pd.DataFrame({"lgbm_score": lgb.Booster(model_file=str(PROPENSITY_LGBM_FILE)).predict(X)})
    gru = None
    if GRU_MODEL_FILE.exists():
        from lead_scoring.gru import HEADS, score_snapshot  # torch loads only when a GRU exists

        probs = score_snapshot(client_ids, snapshot, model_path=GRU_MODEL_FILE)
        out[[f"gru_{h}" for h in HEADS]] = probs
        gru = probs[:, 0]
    elif weight > 0:
        raise FileNotFoundError(f"the blend uses the GRU but {GRU_MODEL_FILE} is missing; run scripts/train_gru.py --stage final")
    out["blend_score"] = blend(out["lgbm_score"], gru, weight)
    out["predicted_propensity"] = calibrate(out["blend_score"], choice["calibrator"])
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_propensity.py -q`
Expected: 3 pass.

- [ ] **Step 5: LightGBM final fit in `src/lead_scoring/training.py`** — remove the `DummyClassifier` and `HistGradientBoostingClassifier` imports, import `lightgbm as lgb`, `from lead_scoring.propensity import load_choice`, and `PROPENSITY_LGBM_FILE` instead of `PROPENSITY_MODEL_FILE`. Replace the classifier block, the `pred_prop` line and the classifier dump with:

```python
    if y_clf_train.nunique() < 2:
        raise ValueError("propensity training labels have a single class")
    choice = load_choice()
    booster = lgb.train(choice["lgbm_params"], lgb.Dataset(X_train, y_clf_train), num_boost_round=choice["lgbm_rounds"])
    booster.save_model(str(PROPENSITY_LGBM_FILE))
    pred_prop = booster.predict(X_test)
```

In `metadata` set `"propensity_model": "LightGBM"`, and change the `main()` print to `PROPENSITY_LGBM_FILE`.

- [ ] **Step 6: Evaluation uses the production scorer** — in `src/lead_scoring/evaluation.py` import `PipelineConfig` and `score_propensity`; drop `PROPENSITY_MODEL_FILE`. Change the signature to `def evaluate_models(config: PipelineConfig = PipelineConfig()) -> dict[str, float]:`, move `test_ids = pd.read_csv(PREPROCESSED_DATA_DIR / "test_ids.csv")` to the top of the function (delete the later read), and replace the propensity lines with:

```python
    scores = score_propensity(X_test, test_ids["customer_id"], config.test_snapshot)
    pred_propensity = scores["predicted_propensity"].to_numpy()
    rank_score = scores["blend_score"].to_numpy()  # finer than the calibrated steps, so ties don't blur the ranking
    expected_value = pred_cltv * pred_propensity
```

Use `rank_score` in `"propensity_auc": _safe_auc(y_clf_test, rank_score)` and in `bootstrap_ci(y_clf_test, rank_score, RANKING_METRICS)`; Brier and the calibration plot keep `pred_propensity`. In the `predictions` DataFrame, after `"expected_value": expected_value,` add the score columns:

```python
            **{c: scores[c].to_numpy() for c in scores.columns if c != "predicted_propensity"},
```

- [ ] **Step 7: Scoring uses the production scorer** — in `src/lead_scoring/scoring.py` import `PipelineConfig` and `score_propensity`; drop `PROPENSITY_MODEL_FILE`; change to `def score_batch(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:`, delete `propensity_model = joblib.load(...)`, and set:

```python
    predicted_propensity = score_propensity(X_scoring, scoring_ids["customer_id"], config.score_snapshot)["predicted_propensity"].to_numpy()
```

In `src/lead_scoring/pipeline.py` call `evaluate_models(config)` and `score_batch(config)`. Delete `PROPENSITY_MODEL_FILE` from `paths.py`, then:

Run: `grep -rn "PROPENSITY_MODEL_FILE\|propensity_model.pkl" src scripts app tests`
Expected: no output. Replace any hit in `scripts/repo_health_check.py` with `PROPENSITY_LGBM_FILE`.

- [ ] **Step 8: Run the suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 9: Real runs, in order**

```bash
.venv/bin/python scripts/validate_propensity.py
.venv/bin/python scripts/train_gru.py --stage final
.venv/bin/python scripts/run_pipeline.py
```
The first picks up the GRU validation scores and writes the blend weight and calibrator. Run the second in the background (about 7/4 × the validation epoch time × the best epoch). The third scores the test month with everything frozen.
Expected: `reports/tables/metrics_summary.csv` holds the new headline. If any step raises `LeakageError`, stop and report.

- [ ] **Step 10: Commit**

```bash
git add src/lead_scoring/propensity.py src/lead_scoring/training.py src/lead_scoring/evaluation.py src/lead_scoring/scoring.py src/lead_scoring/pipeline.py src/lead_scoring/paths.py scripts/repo_health_check.py tests/test_propensity.py models/propensity_choice.json models/model_metadata.json reports/tables/propensity_validation.csv reports/tables/metrics_summary.csv reports/tables/decile_lift.csv reports/figures/calibration.png
git commit -m "Score propensity with LightGBM and the validated GRU blend, calibrated on validation"
```

---

### Task 9: Comparison rows and the paper-protocol table

**Files:**
- Modify: `src/lead_scoring/model_comparison.py`, `src/lead_scoring/model_registry.py`, `src/lead_scoring/pipeline.py`
- Create: `src/lead_scoring/paper_protocol.py`
- Test: `tests/test_paper_protocol.py`

**Interfaces:**
- Consumes: `test_predictions.csv` score columns (Task 8); `y_labels_*.parquet` (Task 4); `load_choice` (Task 8); `blend` (Task 5).
- Produces: `model_comparison.csv` rows `previous_gbm`, `lightgbm`, `gru` (when present), `propensity_blend` (when weight > 0); `reports/tables/paper_protocol.csv` with columns `model, source, mean_auc, auc_y1_target_1..4`;
  `build_paper_protocol(X_train, labels_train, X_test, labels_test, params, rounds, gru_test, weight) -> pd.DataFrame`; `write_paper_protocol() -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test** — create `tests/test_paper_protocol.py`:

```python
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from lead_scoring.paper_protocol import NEXT_MONTH, build_paper_protocol


def test_paper_table_has_our_row_then_the_published_ones():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.random(400), "g": rng.random(400)})
    labels = pd.DataFrame({t: (X["f"] + rng.normal(0, 0.3, 400) > 0.7).astype(int) for t in NEXT_MONTH})
    params = {"objective": "binary", "verbose": -1, "min_child_samples": 5}
    table = build_paper_protocol(X, labels, X, labels, params, rounds=10, gru_test=None, weight=0.0)
    assert table["source"].tolist() == ["ours", "paper", "paper"]
    ours = table.iloc[0]
    assert ours["mean_auc"] == pytest.approx(np.mean([ours[f"auc_{t}"] for t in NEXT_MONTH]))
    assert ours["mean_auc"] > 0.6
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/pytest tests/test_paper_protocol.py -q`
Expected: ModuleNotFoundError `lead_scoring.paper_protocol`.

- [ ] **Step 3: Implement `src/lead_scoring/paper_protocol.py`**

```python
"""Next-month, per-product AUC on the test snapshot, next to the MBD paper's numbers.

The paper (arXiv 2409.17587, Table 3) predicts each of 4 products for the next month on full MBD with
client folds and up to three modalities. We use MBD-mini, a time split and transactions only, so the
rows show scale, not a like-for-like tie.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lead_scoring.blend import blend
from lead_scoring.data import TARGETS
from lead_scoring.paths import PREPROCESSED_DATA_DIR, TABLES_DIR
from lead_scoring.propensity import load_choice

NEXT_MONTH = [f"y1_{t}" for t in TARGETS]
PUBLISHED = [
    {"model": "Published: transaction aggregates (full MBD, client folds)", "source": "paper", "mean_auc": 0.783},
    {"model": "Published best: supervised RNN, transactions + geo + dialogs", "source": "paper", "mean_auc": 0.824},
]


def _row(model: str, labels: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict:
    aucs = {t: float(roc_auc_score(labels[t], scores[t])) for t in NEXT_MONTH}
    return {"model": model, "source": "ours", "mean_auc": float(np.mean(list(aucs.values()))),
            **{f"auc_{t}": v for t, v in aucs.items()}}


def build_paper_protocol(X_train, labels_train, X_test, labels_test, params: dict, rounds: int,
                         gru_test: pd.DataFrame | None, weight: float) -> pd.DataFrame:
    lgbm = {t: lgb.train(params, lgb.Dataset(X_train, labels_train[t]), num_boost_round=rounds).predict(X_test)
            for t in NEXT_MONTH}
    rows = [_row("LightGBM, one model per product", labels_test, lgbm)]
    if gru_test is not None:
        gru = {t: gru_test[f"gru_{t}"].to_numpy() for t in NEXT_MONTH}
        rows.append(_row("GRU next-month heads", labels_test, gru))
        if weight > 0:
            rows.append(_row("LightGBM + GRU blend", labels_test, {t: blend(lgbm[t], gru[t], weight) for t in NEXT_MONTH}))
    return pd.DataFrame(rows + PUBLISHED)


def write_paper_protocol() -> pd.DataFrame:
    choice = load_choice()
    predictions = pd.read_csv(TABLES_DIR / "test_predictions.csv")
    table = build_paper_protocol(
        pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet"),
        pd.read_parquet(PREPROCESSED_DATA_DIR / "y_labels_train.parquet"),
        pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet"),
        pd.read_parquet(PREPROCESSED_DATA_DIR / "y_labels_test.parquet"),
        choice["lgbm_params"], choice["lgbm_rounds"],
        predictions if f"gru_{NEXT_MONTH[0]}" in predictions else None,
        choice["blend"]["weight"],
    )
    table.to_csv(TABLES_DIR / "paper_protocol.csv", index=False)
    return table
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_paper_protocol.py -q`
Expected: 1 pass.

- [ ] **Step 5: New comparison rows in `src/lead_scoring/model_comparison.py`** — add imports `from sklearn.ensemble import HistGradientBoostingClassifier`, `from lead_scoring.config import PipelineConfig`, `from lead_scoring.features import split_training_data`, `from lead_scoring.propensity import load_choice`, and `CUSTOMER_FEATURES_FILE` from paths. Replace `RANKERS` and add the previous model's features:

```python
RANKERS = {
    "prior_baseline": ("Prior rate (no ranking)", "baseline"),
    "recency_rule": ("Recency rule: most recent activity first", "baseline"),
    "activity_rule": ("Activity rule: most transactions in last 90 days", "baseline"),
    "product_history_rule": ("Product-history rule: took a product before, then most active", "baseline"),
    "logistic_regression": ("Logistic regression", "baseline"),
    "previous_gbm": ("Previous model: gradient boosting on 14 features", "baseline"),
    "lightgbm": ("LightGBM on transaction-mix features", "model"),
    "gru": ("GRU on raw transactions", "model"),
    "propensity_blend": ("LightGBM + GRU blend", "model"),
    "expected_value_ranker": ("Expected value ranker (lead score)", "model"),
}
# The 14 columns the previous propensity model used, kept so its row shows the gain on the same test rows.
V1_FEATURES = ["total_txn_amt_365", "txn_count_365", "avg_txn_amt_365", "max_txn_amt_365", "active_days_365",
               "txn_count_90", "total_txn_amt_90", "recency_days", "tenure_days", "products_taken_before",
               "target_1_before", "target_2_before", "target_3_before", "target_4_before"]
```

Add:

```python
def _previous_gbm_scores(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=0.05, random_state=seed)
    return model.fit(train[V1_FEATURES], train["converted"]).predict_proba(test[V1_FEATURES])[:, 1]
```

Change the signature to `def build_model_comparison(config: PipelineConfig = PipelineConfig(), seed: int = 42) -> pd.DataFrame:`. After `y_test = ...`, replace everything up to `rankers = compare_rankers(...)` (inclusive) with:

```python
    raw = pd.read_parquet(CUSTOMER_FEATURES_FILE, columns=["customer_id", "snapshot_date", "converted", *V1_FEATURES])
    raw_train, raw_test = split_training_data(raw, config)
    if not (raw_test["customer_id"].astype(str).to_numpy() == predictions["customer_id"].astype(str).to_numpy()).all():
        raise ValueError("raw test rows are not in the order of test_predictions.csv")
    weight = load_choice()["blend"]["weight"]
    activity_tiebreak = raw_test["txn_count_365"] / (raw_test["txn_count_365"].max() + 1)
    scores = {
        "prior_baseline": pd.Series(0.0, index=raw_test.index),
        "recency_rule": -raw_test["recency_days"],
        "activity_rule": raw_test["txn_count_90"],
        "product_history_rule": raw_test["products_taken_before"] + activity_tiebreak,
        "logistic_regression": pd.Series(_logistic_scores(X_train, y_train, X_test, seed)),
        "previous_gbm": pd.Series(_previous_gbm_scores(raw_train, raw_test, seed)),
        "lightgbm": predictions["lgbm_score"],
    }
    if "gru_converted" in predictions:
        scores["gru"] = predictions["gru_converted"]
    if weight > 0:
        scores["propensity_blend"] = predictions["blend_score"]
    scores["expected_value_ranker"] = predictions["expected_value"]
    rankers = compare_rankers(y_test, scores, seed=seed)
```

Leave the `values = compare_value_models(...)` call as it is (`total_txn_amt_90` is still a model feature). Replace the `production_model` line with:

```python
    production = {"expected_value_ranker", "cltv_gbm", "propensity_blend" if weight > 0 else "lightgbm"}
    output["production_model"] = output["model_id"].isin(production)
```

- [ ] **Step 6: Registry and pipeline** — in `src/lead_scoring/model_registry.py`, replace the `propensity_hist_gradient_boosting` entry with:

```python
    "propensity_lightgbm_gru": ModelRegistryEntry(
        model_id="propensity_lightgbm_gru",
        display_name="Propensity: LightGBM + GRU",
        model_family="LightGBM, GRU (PyTorch), isotonic calibration",
        target="converted",
        role="Conversion propensity",
        description="LightGBM on transaction-mix and windowed product-history features, rank-blended with a GRU over the "
                    "last 512 transactions; weight and calibration chosen on the July 2022 validation month.",
    ),
```

In `src/lead_scoring/pipeline.py`: import `write_paper_protocol`; call `build_model_comparison(config)`; add `paper = write_paper_protocol()` after it and `"paper_protocol_rows": int(len(paper)),` to the returned dict.

- [ ] **Step 7: Run the suite, then the pipeline**

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/run_pipeline.py
```
Expected: tests pass (`tests/test_readme_numbers.py` will fail on the new rows until Task 10; note which, don't fix yet). `reports/tables/paper_protocol.csv` exists with our rows above the two published ones.

- [ ] **Step 8: Commit**

```bash
git add src/lead_scoring/model_comparison.py src/lead_scoring/paper_protocol.py src/lead_scoring/model_registry.py src/lead_scoring/pipeline.py tests/test_paper_protocol.py reports/tables/model_comparison.csv reports/tables/paper_protocol.csv
git commit -m "Compare LightGBM, GRU and blend with the previous model, and report the paper's protocol"
```

---

### Task 10: README, docs and dashboard from the measured results

**Files:**
- Modify: `README.md`, `docs/methodology.md`, `docs/results_summary.md`, `docs/architecture.md`, `tests/test_readme_numbers.py`, `web/public/data/*.json` (regenerated)

**Interfaces:**
- Consumes: `reports/tables/model_comparison.csv`, `metrics_summary.csv`, `paper_protocol.csv`, `propensity_validation.csv`, `models/propensity_choice.json`.

- [ ] **Step 1: Write the failing README tests** — append to `tests/test_readme_numbers.py`:

```python
CHOICE = json.loads((ROOT / "models" / "propensity_choice.json").read_text())


def test_paper_protocol_rows_are_quoted():
    table = pd.read_csv(TABLES / "paper_protocol.csv")
    for row in table[table["source"] == "ours"].to_dict("records"):
        assert f"{row['mean_auc']:.3f}" in README


def test_blend_weight_and_validation_auc_are_quoted():
    assert f"GRU weight {CHOICE['blend']['weight']:.2f}" in README
    assert f"validation AUC {CHOICE['blend']['auc']:.3f}" in README
```

Run: `.venv/bin/pytest tests/test_readme_numbers.py -q`
Expected: FAIL on the new rows and the two new tests.

- [ ] **Step 2: Rewrite the README results** — every number comes from the files above:
  - The results table gets the `previous_gbm`, `lightgbm`, `gru` and `propensity_blend` rows, as `lift (lower–upper)` and AUC to 3 decimals.
  - A "Paper protocol" table quotes `paper_protocol.csv` with its three stated differences: MBD-mini, a time split, transactions only.
  - The model section states `GRU weight X.XX` and `validation AUC X.XXX` exactly as the test expects.
  - The "Run it" section lists, in order: `python -m lead_scoring.data`, `python -m lead_scoring.features`, `python scripts/train_gru.py --stage validate`, `python scripts/validate_propensity.py`, `python scripts/train_gru.py --stage final`, `python scripts/run_pipeline.py`.
  - If the headline is under the 0.78 target, say so plainly.

Update `docs/methodology.md` (time protocol, validation month, leakage stops), `docs/architecture.md` (event store, GRU, blend) and `docs/results_summary.md` to match.

- [ ] **Step 3: Dashboard data and build**

```bash
.venv/bin/python scripts/run_export_web_data.py
cd web && npm run lint && npm run build && cd ..
```
Expected: both pass. Open the models page in the browser and check the new leaderboard rows render with their intervals.

- [ ] **Step 4: Full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add README.md docs tests/test_readme_numbers.py web/public/data
git commit -m "Report the LightGBM + GRU results and the paper-protocol comparison"
```

- [ ] **Step 6: Clean reproducibility run** — with a clean tracked tree (`git status --porcelain --untracked-files=no` prints nothing):

```bash
.venv/bin/python scripts/run_pipeline.py
git status --porcelain --untracked-files=no
```
Expected: no result file changes except `reports/run_manifest.json` (timestamp and SHA). LightGBM is deterministic and GRU inference runs on CPU, so `metrics_summary.csv` and `model_comparison.csv` must not change. Commit the manifest:

```bash
git add reports/run_manifest.json
git commit -m "Record the run manifest for the LightGBM + GRU results"
```

- [ ] **Step 7: Report to the user** — headline AUC and lift with intervals versus 0.736 / 3.23, the paper-protocol mean against 0.783 / 0.824, the blend weight, and what was not verified. Do not push.
