import json, numpy as np, pandas as pd, joblib, duckdb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor
from src.ml.features import load_classification_dataset, load_regression_dataset, write_scores, FEATURES_V2

MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

def _ids():
    con=duckdb.connect("data/warehouse/cltv.duckdb")
    v=con.execute("SELECT cust_id FROM ans.modeling_dataset_v2").fetchdf()["cust_id"].tolist()
    con.close(); return v

def train_cls():
    X_df, y_ser = load_classification_dataset()
    X = X_df.to_numpy(float)
    y = y_ser.to_numpy(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    base = XGBClassifier(
        n_estimators=1200, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=-1
    )
    for tr, va in skf.split(X, y):
        m = XGBClassifier(**base.get_params())
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:,1]
    print({"cv_roc": float(roc_auc_score(y,oof)), "cv_pr": float(average_precision_score(y,oof))})

    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    cal.fit(X, y)
    proba = cal.predict_proba(X)[:,1]
    joblib.dump(cal, MODELS / "xgb_cls_cal.pkl")
    with open(MODELS / "xgb_features.json","w") as f: json.dump(FEATURES_V2, f)
    out = pd.DataFrame({"cust_id": _ids(), "ml_xgb_proba_cal": proba})
    write_scores("ans.customer_ml_scores_xgb", out)
    print("[OK] ans.customer_ml_scores_xgb")

def train_reg():
    X_df, y_ser = load_regression_dataset()
    X = X_df.to_numpy(float)
    y = y_ser.to_numpy(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    base = XGBRegressor(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=-1
    )
    for tr, va in kf.split(X):
        m = XGBRegressor(**base.get_params())
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    rmse = float(np.sqrt(mean_squared_error(y,oof)))
    mae  = float(mean_absolute_error(y,oof))
    print({"cv_rmse": rmse, "cv_mae": mae})

    base.fit(X, y)
    joblib.dump(base, MODELS / "xgb_reg.pkl")
    preds = base.predict(X)
    out = pd.DataFrame({"cust_id": _ids(), "ml_xgb_pred_cltv": preds})
    write_scores("ans.customer_cltv_reg_xgb", out)
    print("[OK] ans.customer_cltv_reg_xgb")

if __name__ == "__main__":
    train_cls()
    train_reg()
