# Methodology

## Question

At a month-end snapshot, which clients will take any of the bank's 4 products in the next 3 months, and what will they transact over that period? The lead score combines both: `expected_value = predicted_cltv × predicted_propensity`, ranked to 0–100, Hot ≥ 70, Cold < 30.

## Data

Sber AI Lab's MBD-mini (CC BY 4.0): 100,224 business clients, 38.8M transactions (Jan 2021 to Dec 2022), monthly purchase flags for 4 unnamed products (Feb 2022 to Jan 2023). See [data/README.md](../data/README.md).

## Snapshots and targets

For a month-end snapshot S (Moscow time; MBD stores UTC):

- **Features:** transactions in (S − 365 days, S], dialog timestamps before the end of S, and purchase flags for S and the 2 months before it (a purchase row stamped month M is read as "bought during M"; see [data/README.md](../data/README.md#label-timing)). Windows that start before the first label month (2022-02) are missing, not 0.
- **Targets:** `converted` = any product flag in a label month within (S, S + 3 months]; `future_value` = transaction amount in the same window.
- **Amount cap:** every amount is capped at the 0.999 quantile of transactions up to the test snapshot, because MBD adds noise to amounts.

| Role | Snapshots |
|---|---|
| Fit, for choosing settings | 2022-01-31, 02-28, 03-31, 04-30 (labels end by 2022-07-31) |
| Validation | 2022-07-31 (labels Aug to Oct 2022) |
| Final refit | the 7 month-ends 2022-01-31 to 2022-07-31 |
| Test, scored once | 2022-10-31 (labels Nov 2022 to Jan 2023) |
| Current scoring | 2022-12-31 |

Every choice (LightGBM settings and round count, the GRU blend weight, the calibration map, each feature idea) was made on the validation month. `scripts/test_propensity.py` scored the test month once and refuses to run again while its result file exists.

## Leakage guards (`src/lead_scoring/temporal.py`)

Each raises `LeakageError` instead of warning:

- a training snapshot whose target window ends after the test snapshot;
- a feature built from a transaction after its snapshot;
- a target, identifier, date or unexplained column (`trans_count`, `diff_trans_date`) in the model matrix;
- model selection that sees any snapshot other than the fit and validation months.

Two more checks stop a run: any single feature above 0.80 AUC on its own (`reports/tables/single_feature_auc.csv`), and any model above 0.85 AUC. The project's own audit line for anything quoted is the same 0.85.

## Evaluation

- All models and baselines are scored on the same test rows and the same seeded bootstrap resamples (300 draws, percentile 95% intervals).
- **Rankers** are ordered by lift in the top 10%. Ties at the cutoff count at their mean rate, so rules with many tied scores aren't flattered by row order.
- **Baselines:** prior rate, recency rule, 90-day activity rule, product-history rule, logistic regression on the same features, and the previous gradient-boosting model on its original 14 features.
- **Paper protocol:** next-month AUC per product, one LightGBM per product, next to the MBD paper's published means (`reports/tables/paper_protocol.csv`). Different data size, split and modalities, so it shows scale, not a tie.
- **Value models** are ordered by RMSE against the training mean and "last 90 days repeat".
- The CLTV estimator was compared on the Jul 2022 validation snapshot (`scripts/cltv_validation.py`), never on test months.

## Heuristics, labelled as such

Risk scores, follow-up flags and scenarios are rules, not fitted models. They aren't evaluated and aren't presented as predictions.
