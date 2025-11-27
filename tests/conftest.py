import pytest, pandas as pd
from pathlib import Path
FIX = Path(__file__).parent / "fixtures"

@pytest.fixture
def load_accounts_df():
    def _load():
        return pd.read_csv(FIX / "sample_accounts.csv", parse_dates=["as_of_date"])
    return _load
