"""Score the test snapshot (Oct 2022) once with the propensity model chosen on validation.

Refits LightGBM with the parameters and round count in models/propensity_choice.json on all
training snapshots (Jan–Jul 2022), then writes AUC and top-10% lift on the test snapshot to
reports/tables/propensity_test.csv. Refuses to run again once that file exists, so the test
month can't be used to make choices. Run after scripts/validate_propensity.py.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from lead_scoring.config import PipelineConfig
from lead_scoring.metrics import lift_at_fraction
from lead_scoring.paths import PREPROCESSED_DATA_DIR, PROPENSITY_CHOICE_FILE, TABLES_DIR
from lead_scoring.temporal import assert_plausible_auc, assert_target_windows_before

RESULT_FILE = TABLES_DIR / "propensity_test.csv"


def main() -> None:
    if RESULT_FILE.exists():
        raise SystemExit(f"{RESULT_FILE} exists: the test month has been scored. Delete it only to reproduce that run.")
    cfg = PipelineConfig()
    choice = json.loads(PROPENSITY_CHOICE_FILE.read_text())
    if choice["blend"]["weight"] != 0:
        raise SystemExit("The chosen blend uses the GRU; this script scores LightGBM alone.")
    train_ids = pd.read_csv(PREPROCESSED_DATA_DIR / "train_ids.csv")
    test_ids = pd.read_csv(PREPROCESSED_DATA_DIR / "test_ids.csv")
    assert set(train_ids["snapshot_date"]) == set(cfg.train_snapshots)
    assert set(test_ids["snapshot_date"]) == {cfg.test_snapshot}
    assert_target_windows_before(cfg.train_snapshots, cfg.horizon_months, cfg.test_snapshot)

    X = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(PREPROCESSED_DATA_DIR / "y_clf_train.parquet")["converted"].to_numpy()
    X_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PREPROCESSED_DATA_DIR / "y_clf_test.parquet")["converted"].to_numpy()
    booster = lgb.train(choice["lgbm_params"], lgb.Dataset(X, y), num_boost_round=choice["lgbm_rounds"])
    score = booster.predict(X_test[X.columns])

    row = {"model": "lightgbm", "test_snapshot": cfg.test_snapshot, "rows": len(y_test),
           "positives": int(y_test.sum()), "base_rate": float(y_test.mean()),
           "test_auc": float(roc_auc_score(y_test, score)),
           "test_lift_at_10pct": float(lift_at_fraction(y_test, score, 0.10)),
           "validation_auc": choice["blend"]["lgbm_auc"], "rounds": choice["lgbm_rounds"]}
    assert_plausible_auc(row["test_auc"])
    pd.DataFrame([row]).to_csv(RESULT_FILE, index=False)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
