"""
Dataframe schemas for key modeling datasets using Pandera.

These schemas can be used in ETL steps or tests to validate that the
dataframes used for model training and scoring have the expected shape
and basic constraints.

Example usage:

    import pandas as pd
    from src.etl.schemas import validate_modeling_dataset_v2

    df = pd.read_csv("data/processed/modeling_dataset_v2.csv")
    validate_modeling_dataset_v2(df)

"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


# ---------------------------------------------------------------------------
# Schema for data/processed/modeling_dataset_v2.csv
# ---------------------------------------------------------------------------

modeling_dataset_v2_schema = pa.DataFrameSchema(
    {
        "cust_id": Column(pa.Int, nullable=False, coerce=True),

        # target / label
        "y_high_cltv": Column(
            pa.Int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        ),

        # regression target
        "cltv_profit": Column(pa.Float, nullable=False, coerce=True),

        # scaled / derived scores
        "lead_score_0_100": Column(
            pa.Float,
            Check.in_range(0, 100),
            nullable=False,
            coerce=True,
        ),
        "cltv_decile": Column(
            pa.Int,
            Check.in_range(1, 10),
            nullable=False,
            coerce=True,
        ),
        "cltv_category_q": Column(pa.String, nullable=False, coerce=True),

        # profitability decomposition
        "non_interest_income": Column(pa.Float, nullable=True, coerce=True),
        "interest_income": Column(pa.Float, nullable=True, coerce=True),

        # relationship / activity features
        "acct_cnt": Column(pa.Int, Check.ge(0), nullable=False, coerce=True),
        "sa_cnt": Column(pa.Int, Check.ge(0), nullable=False, coerce=True),
        "ca_cnt": Column(pa.Int, Check.ge(0), nullable=False, coerce=True),
        "txn_cnt_12m": Column(pa.Int, Check.ge(0), nullable=False, coerce=True),
        "txn_months_12m": Column(pa.Int, Check.ge(0), nullable=False, coerce=True),
        "txn_amt_12m": Column(pa.Float, Check.ge(0), nullable=False, coerce=True),
        "avg_txn_amt_12m": Column(pa.Float, Check.ge(0), nullable=False, coerce=True),
        "std_txn_amt_12m": Column(pa.Float, Check.ge(0), nullable=True, coerce=True),
        "med_txn_amt_12m": Column(pa.Float, Check.ge(0), nullable=True, coerce=True),
        "debit_cnt_12m": Column(pa.Int, Check.ge(0), nullable=False, coerce=True),
        "credit_cnt_12m": Column(pa.Int, Check.ge(0), nullable=False, coerce=True),

        # recency & charges
        "recency_days": Column(pa.Int, Check.ge(0), nullable=False, coerce=True),
        "chrg_amt_12m": Column(pa.Float, Check.ge(0), nullable=True, coerce=True),
        "chrg_months_12m": Column(pa.Int, Check.ge(0), nullable=True, coerce=True),

        # balances
        "avg_bal_12m_savings": Column(pa.Float, Check.ge(0), nullable=True, coerce=True),
        "avg_bal_12m_current_ac": Column(pa.Float, Check.ge(0), nullable=True, coerce=True),
        "avg_bal_12m_td": Column(pa.Float, Check.ge(0), nullable=True, coerce=True),
    },
    strict=False,  # allow extra columns if you add more features later
    coerce=True,
)


def validate_modeling_dataset_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the modeling_dataset_v2 dataframe.

    Raises pandera.errors.SchemaError on failure.
    Returns the validated dataframe on success (may be type-coerced).
    """
    return modeling_dataset_v2_schema.validate(df, lazy=True)


# ---------------------------------------------------------------------------
# Schema for data/processed/modeling_dataset_leads_unified.csv
# ---------------------------------------------------------------------------
# This dataset has many columns. For now, we strongly validate the *core*
# identifiers and label columns used by the unified lead scoring model, and
# allow other columns to pass through without detailed checks.
# ---------------------------------------------------------------------------

modeling_dataset_leads_unified_schema = pa.DataFrameSchema(
    {
        # identifiers
        "Customer_ID": Column(pa.String, nullable=False, coerce=True),
        "lead_id": Column(pa.String, nullable=False, coerce=True),

        # core lead metadata
        "lead_product_name_norm": Column(pa.String, nullable=False, coerce=True),
        "Date_Of_Lead": Column(pa.DateTime, nullable=False, coerce=True),
        "lead_date": Column(pa.DateTime, nullable=False, coerce=True),
        "Lead_Source": Column(pa.String, nullable=False, coerce=True),

        # labels (binary classification targets)
        "label_conv": Column(
            pa.Int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        ),
        "label_conv_90d": Column(
            pa.Int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        ),
    },
    strict=False,  # other feature columns (AA, CBS, balances, etc.) are allowed
    coerce=True,
)


def validate_modeling_dataset_leads_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the modeling_dataset_leads_unified dataframe.

    Only enforces the presence and basic constraints of the core ID/label
    columns; additional feature columns are allowed.
    """
    return modeling_dataset_leads_unified_schema.validate(df, lazy=True)


__all__ = [
    "modeling_dataset_v2_schema",
    "validate_modeling_dataset_v2",
    "modeling_dataset_leads_unified_schema",
    "validate_modeling_dataset_leads_unified",
]
