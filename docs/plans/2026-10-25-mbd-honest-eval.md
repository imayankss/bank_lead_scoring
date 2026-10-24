# Real data and honest evaluation: plan

**Goal:** Keep the lead-scoring platform's design (CLTV model, propensity model, expected-value lead score, Hot/Medium/Cold, risk, scenario, seasonality, 8-page dashboard). Replace the 500 synthetic customers with public bank data, and fix the evaluation so every reported number holds up.

**Framing:** The platform is built for a Bank of India-style lead workflow. It is demonstrated on the public Multimodal Banking Dataset (MBD) from Sber AI Lab (business clients, Feb 2021 to Jan 2023, CC BY 4.0). MBD clients are never described as Bank of India customers. MBD's four products stay "Product 1–4", because the dataset doesn't name them.

## Why (audit of `main` at 17a5774)

- AUC 0.913 came from current-state account features (66 of 700 accounts opened after the training snapshot) and a random split. `account_count` alone scored 0.912.
- Without those features, the model (0.833) lost to a one-line recency rule (0.852).
- The "baselines" reused the model's own predictions, and `business_score` gave the production model a +0.05 bonus.
- CLTV R² was −0.37, worse than predicting the mean.
- Re-running gave different numbers: package versions were unpinned and no run recorded its inputs.

## Data

- **Source:** `ai-lab/MBD-mini` on Hugging Face: 100,224 clients in 5 client folds, 38.8M transactions.
  - Monthly product labels `target_1..4` for month-ends Feb 2022 to Jan 2023.
- **Download:** `scripts/fetch_mbd.py` writes to `data/raw/mbd/`, which git ignores, and checks SHA-256.
- **What's excluded and why:**
  - `trans_count` and `diff_trans_date`: no transaction window reproduces them (best match 11% of rows), so their source is unknown.
  - The geo and dialog tables: out of scope.
- **Timestamps:** stored in UTC. Months are taken in Moscow time (UTC+3).

## Target and split

- **Snapshot:** a month-end S. Features use transactions in (S − 365 d, S] and product labels with `mon` < S.
- **Targets:**
  - `converted`: any of the 4 products taken in a label month within (S, S + 3 months].
  - `future_value`: transaction amount in (S, S + 3 months], capped.
- **Snapshots:**

| Role | Month-end snapshots |
|---|---|
| Train | 2022-01-31, 2022-04-30, 2022-07-31 |
| Test | 2022-10-31 (labels Nov 2022 – Jan 2023) |
| Current scoring | 2022-12-31 |

- **Guards** (`src/lead_scoring/temporal.py`): each raises `LeakageError` if
  - a training target window ends after the test snapshot;
  - a feature row reads past its snapshot;
  - a forbidden column reaches the model.

## Scaling

Transactions are aggregated with DuckDB straight from the parquet files, one MBD client fold per worker process (`ProcessPoolExecutor`). The folds share no clients, so each one is an independent partition. The same code runs MBD-mini on a laptop or full MBD (2.2M clients, 948M transactions) fold by fold.

## Tasks (one commit each)

1. **CI:** `.github/workflows/ci.yml` runs pytest, then web lint and build.
2. **Leakage guards, test-first:** `temporal.py`, `tests/test_temporal.py`.
3. **Data layer:**
   - `fetch_mbd.py`, and `data.py` rewritten for MBD + DuckDB by fold.
   - Tests run on tiny MBD-shaped fixtures in `tests/conftest.py`.
   - Synthetic CSVs removed; `data/processed/` and `models/` no longer committed.
4. **Features and split:** stacked snapshots, time split, guards wired in. Columns renamed `converted_12m` → `converted` and `future_revenue_12m` → `future_value`.
5. **Metrics, test-first:**
   - Bootstrap CI for AUC, PR-AUC and lift@10%.
   - A decile table with CIs.
   - Calibration figure.
6. **Model comparison:**
   - Real baselines: prior rate, recency rule, "took a product before" rule, logistic regression.
   - Against them: the gradient-boosting propensity model and the expected-value ranker.
   - No bonus term in the ranking.
7. **Downstream modules:**
   - Risk, fallback and diagnostics use only columns MBD has.
   - Seasonality is computed by DuckDB.
   - Web export caps row counts.
   - Dashboard: leaderboard shows lift with CI, and there is no currency label.
8. **Provenance and docs:**
   - `reports/run_manifest.json` records config, input hashes, git SHA and package versions; `requirements.txt` is pinned.
   - README and docs rewritten around measured results.
   - `tests/test_readme_numbers.py` checks the README against the results files.

## Verify

- `pytest` passes in under 60 s, on fixtures only.
- The full pipeline on MBD-mini runs end to end.
- `cd web && npm run lint && npm run build` passes.
- Every README number matches a committed results file.

## Out of scope

Geo and dialog features, deep sequence models, deployment (Vercel needs your OK first).
