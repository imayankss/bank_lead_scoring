# Results Summary

The measured results, with 95% intervals and baselines, are in the [README](../README.md#results). The README is checked against the result files by `tests/test_readme_numbers.py`, so this page doesn't repeat the numbers.

Result files:

- `reports/tables/model_comparison.csv`: every model and baseline on the test months.
- `reports/tables/metrics_summary.csv`: headline metrics with intervals.
- `reports/tables/decile_lift.csv`: conversion by lead-score decile, with Wilson intervals and cumulative reach.
- `reports/tables/cltv_validation.csv`: CLTV estimator choice on the validation snapshot.
- `reports/run_manifest.json`: git SHA, config, data revision and hashes, package versions.
