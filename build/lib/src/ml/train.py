from src.common.config import settings
import os, json, numpy as np, duckdb, joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

DB_PATH = str(settings.project.db_path)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM ans.modeling_dataset").fetchdf()
    con.close()
    return df

def train_baseline(df):
    FEATURES = ["acct_cnt","sa_cnt","ca_cnt","txn_cnt_12m","txn_months_12m","txn_amt_12m","avg_txn_amt_12m"]
    X = df[FEATURES].fillna(0).to_numpy(float)
    y = df["y_high_cltv"].to_numpy(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
    for tr, va in skf.split(X, y):
        m = clf
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:,1]
    print(f"[CV] ROC_AUC={roc_auc_score(y,oof):.3f}  PR_AUC={average_precision_score(y,oof):.3f}")

    clf.fit(X, y)
    joblib.dump(clf, MODELS_DIR / "high_cltv_rf.pkl")
    with open(MODELS_DIR / "high_cltv_features.json","w") as f: json.dump(FEATURES, f)

    proba = clf.predict_proba(X)[:,1]
    thr = float(np.quantile(proba, 0.80))
    flag = (proba >= thr).astype(int)

    out = df[["cust_id"]].copy()
    out["ml_high_cltv_proba"] = proba
    out["ml_high_cltv_flag"]  = flag
    out["ml_cutoff"]          = thr

    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS ans")
    con.execute("DROP TABLE IF EXISTS ans.customer_ml_scores")
    con.register("__scores", out)
    con.execute("CREATE TABLE ans.customer_ml_scores AS SELECT * FROM __scores")
    con.unregister("__scores")
    os.makedirs("data/processed", exist_ok=True)
    con.execute("COPY (SELECT * FROM ans.customer_ml_scores) TO 'data/processed/customer_ml_scores.csv' (HEADER, DELIMITER ',')")
    con.close()

    print("[OK] model → models/high_cltv_rf.pkl")
    print("[OK] scores → data/processed/customer_ml_scores.csv")
    print(f"[CUT] top-20% threshold={thr:.4f}; flagged={flag.sum()}/{len(flag)}")

if __name__ == "__main__":
    df = load_data()
    print(f"[DATA] rows={len(df)}  positives={int(df['y_high_cltv'].sum())}")
    train_baseline(df)