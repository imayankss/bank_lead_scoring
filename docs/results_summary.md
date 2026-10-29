# Results Summary

The measured results, with 95% intervals and baselines, are in the [README](../README.md#results). The README is checked against the result files by `tests/test_readme_numbers.py`, so this page doesn't repeat the numbers.

Result files:

- `reports/tables/model_comparison.csv`: every model and baseline on the test months.
- `reports/tables/metrics_summary.csv`: headline metrics with intervals.
- `reports/tables/decile_lift.csv`: conversion by lead-score decile, with Wilson intervals and cumulative reach.
- `reports/tables/cltv_validation.csv`: CLTV estimator choice on the validation snapshot.
- `reports/tables/propensity_validation.csv`, `gru_validation.csv`, `single_feature_auc.csv`: propensity choices and leakage checks on the validation month.
- `reports/tables/propensity_test.csv`: the one-time test-month score.
- `reports/tables/paper_protocol.csv`: next-month per-product AUC next to the MBD paper.
- `docs/plans/2026-10-25-auc-lift-strategy.md`: every feature and model idea tried, with its validation result.
- `reports/run_manifest.json`: git SHA, config, data revision and hashes, package versions.
