# Results Summary

## Latest Pipeline Run

The latest run completed successfully with:

- 500 training customer rows.
- 500 current scoring customer rows.
- 328 positive training customers.
- 29 model feature columns.
- 500 scored leads.

## Evaluation Metrics

| Metric | Value |
| --- | ---: |
| Test rows | 100 |
| Positive rate | 66.0% |
| Propensity AUC | 0.913 |
| Propensity Brier score | 0.126 |
| CLTV RMSE | INR 145,270 |
| CLTV MAE | INR 82,076 |
| Precision at top 10% | 1.000 |
| Precision at top 20% | 1.000 |

## Lead Distribution

| Category | Customers |
| --- | ---: |
| Hot | 153 |
| Medium | 200 |
| Cold | 147 |

## Interpretation

The corrected pipeline is structurally sound enough for a portfolio presentation: it has no target leakage in the model matrix, scores all customers, and generates a useful score distribution.

The dataset is still synthetic and small. The high propensity AUC and top-decile precision should be presented as prototype validation, not production model proof.
