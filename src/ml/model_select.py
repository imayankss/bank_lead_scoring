import os, json, numpy as np, duckdb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from src.common.config import settings

DB = str(settings.project.db_path)
rw = settings.scoring.rules_pct
mw = settings.scoring.ml_pct
w_hgb, w_xgb, w_lgbm = 0.4, 0.35, 0.25
DELTA = 0.01

EXPORT_CRM_HYBRID = str(settings.exports.crm_hybrid)


con = duckdb.connect(DB)
df = con.execute("SELECT * FROM ans.modeling_dataset_v2").fetchdf()
FEATURES = ["acct_cnt","sa_cnt","ca_cnt","txn_cnt_12m","txn_months_12m","txn_amt_12m",
            "avg_txn_amt_12m","std_txn_amt_12m","med_txn_amt_12m","debit_cnt_12m","credit_cnt_12m",
            "recency_days","chrg_amt_12m","chrg_months_12m","avg_bal_12m_savings",
            "avg_bal_12m_current_ac","avg_bal_12m_td"]
X = df[FEATURES].fillna(0).to_numpy(float)
y = df["y_high_cltv"].to_numpy(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def oof_probs(clf):
    p=np.zeros(len(y))
    for tr,va in skf.split(X,y):
        m=clf()
        m.fit(X[tr], y[tr])
        p[va]=m.predict_proba(X[va])[:,1]
    return p

oof = {
  "hgb":  oof_probs(lambda: HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=400, l2_regularization=1e-3, random_state=42)),
  "lgbm": oof_probs(lambda: LGBMClassifier(n_estimators=1200, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42)),
  "xgb":  oof_probs(lambda: XGBClassifier(n_estimators=1200, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=-1, random_state=42)),
}
oof["ensemble"] = w_hgb*oof["hgb"] + w_xgb*oof["xgb"] + w_lgbm*oof["lgbm"]

def capture(p,k):
    order=np.argsort(-p); y_ord=y[order]; pos=y.sum(); n=len(y)
    return float(y_ord[:int(np.ceil(k*n))].sum()/pos)

metrics={}
for name,p in oof.items():
    metrics[name]={
      "pr_auc":float(average_precision_score(y,p)),
      "roc_auc":float(roc_auc_score(y,p)),
      "brier":float(brier_score_loss(y,p)),
      "cap10":round(capture(p,0.10),3),
      "cap20":round(capture(p,0.20),3),
      "cap30":round(capture(p,0.30),3),
    }

best_single=max(("hgb","xgb","lgbm"), key=lambda n: metrics[n]["pr_auc"])
champ = "ensemble" if metrics["ensemble"]["pr_auc"] >= metrics[best_single]["pr_auc"] + DELTA else best_single

if champ=="ensemble":
    con.execute(f"""
    CREATE OR REPLACE TABLE ans.crm_export_final AS
    WITH probs AS (
      SELECT l.cust_id, l.lead_score_0_100, l.cltv_decile, l.cltv_category_q, l.recommended_action, l.priority_flag,
             c.ml_high_cltv_proba_cal AS p_hgb, x.ml_xgb_proba_cal AS p_xgb, b.ml_lgbm_proba_cal AS p_lgbm
      FROM ans.customer_lead_scores l
      LEFT JOIN ans.customer_ml_scores_cal  c USING(cust_id)
      LEFT JOIN ans.customer_ml_scores_xgb  x USING(cust_id)
      LEFT JOIN ans.customer_ml_scores_lgbm b USING(cust_id)
    )
    SELECT
      cust_id, lead_score_0_100, cltv_decile, cltv_category_q, recommended_action, priority_flag,
      ({w_hgb})*COALESCE(p_hgb,0)+({w_xgb})*COALESCE(p_xgb,0)+({w_lgbm})*COALESCE(p_lgbm,0) AS ml_proba,
      ROUND({rw}*lead_score_0_100 + {mw}*(100.0*(({w_hgb})*COALESCE(p_hgb,0)+({w_xgb})*COALESCE(p_xgb,0)+({w_lgbm})*COALESCE(p_lgbm,0)))) AS hybrid_score_0_100,
      'ensemble'::VARCHAR AS champion_model
    FROM probs
    """)
else:
    mapping={"hgb":("ans.customer_ml_scores_cal","ml_high_cltv_proba_cal"),
             "xgb":("ans.customer_ml_scores_xgb","ml_xgb_proba_cal"),
             "lgbm":("ans.customer_ml_scores_lgbm","ml_lgbm_proba_cal")}
    tbl,col=mapping[champ]
    con.execute(f"""
    CREATE OR REPLACE TABLE ans.crm_export_final AS
    SELECT l.cust_id, l.lead_score_0_100, l.cltv_decile, l.cltv_category_q, l.recommended_action, l.priority_flag,
           s.{col} AS ml_proba,
           ROUND({rw}*l.lead_score_0_100 + {mw}*(100.0*COALESCE(s.{col},0))) AS hybrid_score_0_100,
           '{champ}'::VARCHAR AS champion_model
    FROM ans.customer_lead_scores l
    LEFT JOIN {tbl} s USING(cust_id)
    """)

os.makedirs(os.path.dirname(EXPORT_CRM_HYBRID), exist_ok=True)
con.execute(f"""
COPY (SELECT * FROM ans.crm_export_final
      ORDER BY hybrid_score_0_100 DESC, lead_score_0_100 DESC)
TO '{EXPORT_CRM_HYBRID}' (HEADER, DELIMITER ',')
""")
os.makedirs("reports", exist_ok=True)
json.dump(
    {
        "gate_delta": DELTA,
        "weights": {"rules_pct": rw, "ml_pct": mw, "hgb": w_hgb, "xgb": w_xgb, "lgbm": w_lgbm},
        "metrics": metrics,
        "champion": champ,
    },
    open("reports/model_select.json", "w"),
    indent=2,
)
print("Champion:", champ, "->", EXPORT_CRM_HYBRID)
con.close()

