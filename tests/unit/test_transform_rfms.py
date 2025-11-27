import pandas as pd
from etl.transforms.rfm import compute_rfm

def test_compute_rfm_basic():
    tx = pd.DataFrame({
        "cust_id": [1,1,2],
        "amount": [100,50,200],
        "tx_date": pd.to_datetime(["2025-10-01","2025-10-15","2025-09-20"])
    })
    asof = pd.Timestamp("2025-11-01")
    out = compute_rfm(tx, asof)
    assert set(["cust_id","recency","frequency","monetary"]).issubset(out.columns)
    assert out.loc[out["cust_id"]==1, "frequency"].item() == 2
    assert out["monetary"].min() >= 0
