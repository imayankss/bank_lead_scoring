# Methodology

## Question

At a month-end snapshot, which clients will take any of the bank's 4 products in the next 3 months, and what will they transact over that period? The lead score combines both: `expected_value = predicted_cltv × predicted_propensity`, ranked to 0–100, Hot ≥ 70, Cold < 30.

## Data

Sber AI Lab's MBD-mini (CC BY 4.0): 100,224 business clients, 38.8M transactions (Jan 2021 to Dec 2022), monthly purchase flags for 4 unnamed products (Feb 2022 to Jan 2023). See [data/README.md](../data/README.md).

## Snapshots and targets

For a month-end snapshot S (Moscow time; MBD stores UTC):

- **Features:** transactions in (S − 365 days, S] and product labels for months before S.
- **Targets:** `converted` = any product flag in a label month within (S, S + 3 months]; `future_value` = transaction amount in the same window.
- **Amount cap:** every amount is capped at the 0.999 quantile of transactions up to the test snapshot, because MBD adds noise to amounts.

| Role | Snapshots |
|---|---|
| Train | 2022-01-31, 2022-04-30, 2022-07-31 |
| Test | 2022-10-31 (labels Nov 2022 to Jan 2023) |
| Current scoring | 2022-12-31 |

## Leakage guards (`src/lead_scoring/temporal.py`)

Each raises `LeakageError` instead of warning:

- a training snapshot whose target window ends after the test snapshot;
- a feature built from a transaction after its snapshot;
- a target, identifier, date or unexplained column (`trans_count`, `diff_trans_date`) in the model matrix.

## Evaluation

- All models and baselines are scored on the same test rows and the same seeded bootstrap resamples (300 draws, percentile 95% intervals).
- **Rankers** are ordered by lift in the top 10%. Ties at the cutoff count at their mean rate, so rules with many tied scores aren't flattered by row order.
- **Baselines:** prior rate, recency rule, 90-day activity rule, product-history rule, logistic regression.
- **Value models** are ordered by RMSE against the training mean and "last 90 days repeat".
- The CLTV estimator was compared on the Jul 2022 validation snapshot (`scripts/cltv_validation.py`), never on test months.

## Heuristics, labelled as such

Risk scores, follow-up flags and scenarios are rules, not fitted models. They aren't evaluated and aren't presented as predictions.
