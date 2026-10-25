# Propensity upgrade: richer features, GRU on transactions, and a validated blend

Date: 2026-10-25. Branch: `feat/propensity-gru-blend`, built on `feat/mbd-honest-eval` (16 local commits, unpushed).

## 1. Goal and success bar

Raise the propensity ranking on the test months (snapshot 2022-10-31, conversions Nov 2022–Jan 2023, 100,224 clients) as far as it honestly goes.

| Measure | Now (propensity GBM) | Target | Stretch |
|---|---:|---:|---:|
| AUC | 0.736 (0.726–0.747) | ≥ 0.78 | 0.82 |
| Lift in top 10% | 3.23 (3.05–3.41) | ≥ 4.0 | 5.0 |

Reference ceiling: on full MBD, the paper ([arXiv 2409.17587](https://arxiv.org/abs/2409.17587), Table 3) reports a mean per-product next-month ROC-AUC of 0.783 for aggregates and 0.824 for its best model (supervised RNN on transactions, geo and dialogs). We have MBD-mini and transactions only.

**Leakage rule:** if any validation or test AUC goes above 0.85, or any single feature alone scores above 0.80 AUC on validation, we stop and audit before reporting it.

Out of scope: geo and dialog data (worth 0.5–1 point in the paper), the CLTV model choice (flagged separately on the previous branch), and deployment.

## 2. Evidence that shaped the design

- Logistic regression (0.738) ties the GBM (0.736) on the same 14 features, so the model is limited by its features, not by the learner.
- 11 categorical transaction columns are unused: `event_type` (55 values), `event_subtype` (60), `currency` (13), `src_type11` (60), `src_type12` (262), `dst_type11` (66), `dst_type12` (322), `src_type21` (10,081), `src_type22` (85), `src_type31` (1,722), `src_type32` (86).
- Product history is the strongest signal: the best rule, "took a product before", reaches a lift of 2.86. But the targets table starts in 2022-02, so `target_*_before` covers 0 months at the 2022-01-31 snapshot and 8 months at the test snapshot. Its meaning shifts between training and test.
- Clients have a median of 146 transactions before 2022-07-31, 810 at p90 and 1,840 at p99.
- Hardware: Apple M2 with 8 GB of RAM. A GRU training step at batch 256 × 512 events takes 0.26 s on the M2 GPU (`mps`) and 0.55 s on the CPU, about 7 minutes per epoch for 400k sequences. The project `.venv` is an x86_64 Python under Rosetta, which can't use `mps`, and PyTorch publishes no Intel-Mac wheels after 2.2. The system Python 3.12 is arm64.

## 3. Time protocol

The test snapshot is used once, for the final numbers. Every choice is made on validation.

| Role | Snapshots (month-ends) | Target window ends |
|---|---|---|
| Fit (for model selection) | 2022-01-31, 02-28, 03-31, 04-30 | by 2022-07-31 |
| Validation | 2022-07-31 | 2022-10-31 |
| Final refit | 2022-01-31 … 2022-07-31 (7 monthly) | by 2022-10-31 |
| Test | 2022-10-31 | 2023-01-31 |
| Scoring | 2022-12-31 | n/a |

- `PipelineConfig` gains `fit_snapshots` and `validation_snapshot`, and `train_snapshots` becomes the 7 monthly snapshots.
- `assert_target_windows_before` is checked twice: fit against validation, and train against test.
- Model selection code receives only fit and validation frames. A guard raises if a test-snapshot row reaches it.

## 4. Features for LightGBM (`data.py`, `features.py`)

All features are computed in the existing DuckDB per-fold query, from events strictly before the end of the snapshot day.

1. **Category mix:** for each of the 11 categorical columns, take the top 20 values by frequency in the fit snapshots and record transaction counts in the last 365 and 90 days for each. Other values are pooled as "other". This adds roughly 400 columns, mostly sparse.
2. **Trend:** transaction count and amount for each of the last 6 months; ratios of the last 30 days to 90 and of 90 to 365; days since the first and last transaction of each `event_type` among the top 10.
3. **Fixed-window product history,** replacing the cumulative `target_*_before` and `products_taken_before`:
   - For each product, a purchase flag in the last 2 months and months since the last purchase, capped at 2.
   - When the window starts before 2022-02 (the first target month), the value is **missing (NaN)**, not 0.
   - LightGBM handles NaN natively, and the linear baselines get the existing missing-value indicators.
   - With a 2-month window, history is complete from the 2022-04-30 snapshot onward. That covers 1 of 4 fit snapshots and 4 of 7 final snapshots.
   - Why 2 months: it is the longest window that is complete at any fit snapshot.
4. The remaining 365-day and 90-day aggregates stay as they are.
5. **Labels:** alongside `converted` (any product within 3 months), add each product within 3 months and each product in the next month. They feed the GRU heads and the paper-protocol table, and are added to `TARGET_COLUMNS` so the forbidden-column guard keeps them out of the features.

Model: LightGBM binary, early-stopped on validation AUC, with a small grid on validation: `num_leaves` {31, 63}, `learning_rate` 0.05, `min_child_samples` {100, 400}, `feature_fraction` 0.5. The final refit uses the chosen settings and the best iteration × 7/4, scaled for the extra data.

## 5. GRU on raw transactions (`sequences.py`, `gru.py`)

- **Event store:** all 38.8M events are sorted by (client, time) once and cached as `.npy` arrays: 11 category codes as int16 (`src_type21` and `src_type31` hashed to 1,024 buckets), log1p of the capped amount as float32, and the timestamp as int64. A per-client offset index sits beside them. That is about 1.2 GB in RAM, which fits in 8 GB. The cache lives under `data/processed/` (gitignored).
- **Sample:** for a given (client, snapshot), the last 512 events before the snapshot cutoff, found with a binary search on time. Each event gets 11 embeddings (16 dims each), log amount, log hours since the previous event and log days before the snapshot. Clients with no events get one padding token.
- **Model:** one GRU layer with 128 hidden units and dropout 0.2. Its last state feeds 9 heads, trained jointly with summed binary cross-entropy: any product within 3 months (the headline), each of the 4 products within 3 months, and each of the 4 products in the next month (for the paper row). Only the "any" head is blended.
- **Training:** AdamW at 1e-3 and batch 256 on `mps` (falling back to CPU), up to 12 epochs, keeping the best epoch by validation AUC. The final refit trains for that many epochs on the 7 snapshots. Seeds are fixed. MPS training is not bit-reproducible, so the final weights are saved and inference runs on the CPU.
- **Where it runs:** `scripts/train_gru.py` runs the validation fit and then the final fit, and writes `models/gru_final.pt` plus `reports/tables/gru_validation.csv`. `run_pipeline` loads `gru_final.pt` and only runs inference (about 3 min on CPU), so the pipeline stays fast and reproducible. Weights are about 2 MB and stay out of git, like `models/*.pkl`.

## 6. Blend and model selection (`scripts/validate_propensity.py`)

This follows the existing `scripts/cltv_validation.py` pattern: fit on the fit snapshots, score validation, and write `reports/tables/propensity_validation.csv` plus `models/propensity_choice.json`.

- **Blend:** `w · rank(GRU) + (1 − w) · rank(LightGBM)`, with w from 0 to 1 in steps of 0.05, chosen by validation AUC. Validation lift at 10% is reported alongside but not used to choose: with about 700 positives in the top 10%, its 95% interval is roughly ±0.18, too wide to separate close candidates.
- **Calibration:** the blend is a rank, so the production probability comes from isotonic regression fitted on validation (blend rank percentile → converted), keeping the Brier score and the calibration plot meaningful. Shortcut: the mapping is fitted on the fit-snapshot models and reused for the final refit. It is percentile-based, so it transfers, but any base-rate drift between July and October shows up as calibration error on test, and that error is reported, not hidden.
- **Simplicity rule:** if the blend's validation AUC does not beat LightGBM alone by at least 0.005, ship LightGBM alone and report the GRU as a row only.

## 7. Evaluation and reporting

- **`model_comparison.csv`:** new rows for LightGBM (rich features), GRU, and the blend. The old 14-feature GBM stays as a row, so the gain is visible against the same baselines and bootstrap resamples.
- **Paper-protocol table `reports/tables/paper_protocol.csv`:**
  - Mean and per-product ROC-AUC for the next month (November 2022) from the 2022-10-31 snapshot, for 4 per-product LightGBMs, the GRU heads and their blend.
  - It sits next to the paper's 0.783 and 0.824, with the differences stated: MBD-mini, a time split instead of client folds, transactions only.
- **Leakage report `reports/tables/single_feature_auc.csv`:** the validation AUC of every LightGBM feature on its own. The run fails if any is above 0.80.
- **Downstream outputs:** the README, `test_readme_numbers.py`, the dashboard leaderboard and the headline cards pick up the new production model. Numbers come only from the result files.

## 8. Environment and CI

- Rebuild `.venv` on the arm64 `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.
- Pin `lightgbm` and `torch` in `requirements.txt`. CI installs torch from the PyTorch CPU index (`--extra-index-url https://download.pytorch.org/whl/cpu`), which keeps the install small on Linux.
- A clean install from the pinned file must still pass all tests, with the full suite under about 30 s apart from the first-run compile.

## 9. Tests (written first)

| Test | Checks |
|---|---|
| `test_sequences.py` | No event at or after the snapshot cutoff is returned; truncation keeps the *last* 512; a client with no events gets a padding token; hashed codes stay in range |
| `test_features.py` (extended) | Product-history windows are NaN when they start before the first target month and complete otherwise; top-20 category vocabularies come from fit snapshots only |
| `test_temporal.py` (extended) | Fit vs validation and train vs test windows; selection code raises on a test-snapshot row |
| `test_gru.py` | A tiny GRU on CPU learns a separable toy pattern within a few steps; the output shape is 9 heads |
| `test_blend.py` | Weight search picks the known-best w on toy scores; the simplicity rule falls back to LightGBM; isotonic output is monotone in rank |
| `test_readme_numbers.py` | Extended to the new headline and paper-protocol numbers |

## 10. Order of work

1. Rebuild the arm64 environment; pin `lightgbm` and `torch`; set CI to the torch CPU index.
2. Add the monthly snapshots and validation config, with guards (test-first).
3. Fixed-window product history plus category-mix and trend features (test-first). Run LightGBM selection on validation, then commit. **Checkpoint:** report validation AUC and lift to you before starting the GRU.
4. Event store and sequence sampler (test-first).
5. GRU model and training script; validation run on the M2. **Checkpoint:** report validation numbers and the real epoch time. If an epoch takes over 30 minutes, stop and discuss compute (Colab, then AWS) before spending anything.
6. Blend selection, isotonic calibration, final refit, and a single test scoring.
7. Paper-protocol table, single-feature leakage report, model comparison rows.
8. README, docs, dashboard, run manifest, and a clean rerun that reproduces the committed results.

## 11. Risks

- **The gain may be smaller than hoped.** Our time split is stricter than the paper's client folds, and MBD-mini is about 1/15 of MBD. Reaching 0.78 would match the paper's aggregate baseline. Numbers are reported as measured, either way.
- **8 GB of RAM.** The event store plus a training batch sits near 3 GB. If memory runs short, cut to the last 256 events and store timestamps as int32 seconds since 2020.
- **Product history is NaN in 3 of 4 fit snapshots,** which weakens what the selection model learns from it. The final refit has 4 of 7 complete. If the validation gain from history is weak, try using 2022-03-31 and 2022-04-30 as the only fit snapshots.
- **MPS nondeterminism.** Covered by saving the final weights and running CPU inference in the pipeline.
- **CI install size** grows with torch, even from the CPU index.
