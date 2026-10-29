"""Choose the CLTV estimator on a validation snapshot, never on the test months.

Trains on the Jan and Apr 2022 snapshots and validates on Jul 2022, whose target
window ends on the test snapshot. Writes reports/tables/cltv_validation.csv.
Run after scripts/run_pipeline.py (it reads data/processed/customer_features.parquet).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from lead_scoring.features import fit_preprocessor, transform_features
from lead_scoring.metrics import regression_metrics
from lead_scoring.paths import CUSTOMER_FEATURES_FILE, TABLES_DIR

FIT_SNAPSHOTS = ["2022-01-31", "2022-04-30"]
VALIDATION_SNAPSHOT = "2022-07-31"
GBM = dict(max_iter=250, learning_rate=0.05, l2_regularization=0.05, random_state=42)


def main() -> None:
    df = pd.read_parquet(CUSTOMER_FEATURES_FILE)
    fit, val = df[df["snapshot_date"].isin(FIT_SNAPSHOTS)], df[df["snapshot_date"] == VALIDATION_SNAPSHOT]
    spec = fit_preprocessor(fit)
    X_fit, X_val = transform_features(fit, spec), transform_features(val, spec)
    y_fit, y_val = fit["future_value"], val["future_value"]

    candidates = {
        "GBM, log target (pipeline model)": TransformedTargetRegressor(
            HistGradientBoostingRegressor(**GBM), func=np.log1p, inverse_func=np.expm1),
        "GBM, squared error": HistGradientBoostingRegressor(**GBM),
        "GBM, Poisson loss": HistGradientBoostingRegressor(loss="poisson", **GBM),
    }
    predictions = {"Last 90 days repeat (baseline)": val["total_txn_amt_90"]}
    for name, model in candidates.items():
        predictions[name] = np.clip(model.fit(X_fit, y_fit).predict(X_val), 0, None)
    residual = HistGradientBoostingRegressor(**GBM).fit(X_fit, y_fit - fit["total_txn_amt_90"])
    predictions["GBM on change from last 90 days"] = np.clip(val["total_txn_amt_90"].to_numpy() + residual.predict(X_val), 0, None)

    rows = [{"model": name, **regression_metrics(y_val, pred)} for name, pred in predictions.items()]
    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    table.insert(1, "validation_snapshot", VALIDATION_SNAPSHOT)
    table.to_csv(TABLES_DIR / "cltv_validation.csv", index=False)
    print(table[["model", "rmse", "mae", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
