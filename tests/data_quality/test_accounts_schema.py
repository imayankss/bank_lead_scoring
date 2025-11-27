import pandera.pandas as pa
from pandera import Column, Check
import pandas as pd

schema = pa.DataFrameSchema(
    {
        "account_id": Column(pa.Int, Check.gt(0), nullable=False, unique=True, coerce=True),
        "avg_bal_12m_td": Column(pa.Float, Check.ge(0), coerce=True),
        "avg_bal_12m_savings": Column(pa.Float, Check.ge(0), coerce=True),
        "avg_bal_12m_current_ac": Column(pa.Float, Check.ge(0), coerce=True),
        "as_of_date": Column(
            pa.DateTime,
            Check(lambda s: s.dt.tz_localize(None) <= pd.Timestamp.now(tz=None)),
            coerce=True,
        ),
    }
)

def test_accounts_contract(load_accounts_df):
    df = load_accounts_df()
    schema.validate(df, lazy=True)
