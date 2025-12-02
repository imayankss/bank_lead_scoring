import pandas as pd

def compute_rfm(tx: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    tx = tx.copy()
    tx["tx_date"] = pd.to_datetime(tx["tx_date"])
    g = tx.groupby("cust_id", as_index=False)
    last_tx = g["tx_date"].max().rename(columns={"tx_date":"last_tx"})
    freq = g.size().rename(columns={"size":"frequency"})
    monetary = g["amount"].sum().rename(columns={"amount":"monetary"})
    out = last_tx.merge(freq, on="cust_id").merge(monetary, on="cust_id")
    out["recency"] = (pd.to_datetime(asof) - out["last_tx"]).dt.days
    out["monetary"] = out["monetary"].clip(lower=0)
    return out[["cust_id","recency","frequency","monetary"]]
