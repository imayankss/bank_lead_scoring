# Methodology

## Data Sources

The project uses five source files:

- `data/raw/customers.csv`
- `data/raw/account_master.csv`
- `data/raw/transactions_history.csv`
- `data/raw/transactions_recent.csv`
- `data/raw/transaction_change_log.csv`

The current pipeline uses the first four files. The change log is retained as a raw source for future audit or anomaly features.

## Feature Window

The training table uses a 365 day lookback window ending one year before the latest transaction date. This creates historical customer features before the outcome period starts.

The current scoring table uses the latest transaction date as the snapshot. It has no future target columns.

## Targets

- `future_revenue_12m`: total customer transaction amount in the 365 days after the training snapshot.
- `converted_12m`: binary label indicating whether `future_revenue_12m` is greater than zero.

This replaces the earlier logic that used the max transaction date as the snapshot and produced only 3 positive customers.

## Modeling

The pipeline trains:

- A CLTV model using `HistGradientBoostingRegressor` wrapped in a log-transformed target regressor.
- A propensity model using `HistGradientBoostingClassifier`.

The project no longer depends on LightGBM, which makes the repository easier to run in a fresh Python environment.

## Scoring

Expected value is calculated as:

```text
predicted_cltv * predicted_propensity
```

Lead score is assigned from expected-value rank percentiles on a 0-100 scale. This avoids the previous two-value 0/100 score output.

## Explainability

The pipeline saves permutation feature importance for the CLTV model and lightweight per-lead driver summaries. These are designed for portfolio inspection, not regulatory model explanations.
