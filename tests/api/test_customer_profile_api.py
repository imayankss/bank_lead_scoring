from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# -------------------------------------------------------------------
# Ensure the project root is on sys.path so `src.*` imports work
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app  # noqa: E402  (import after sys.path tweak)


client = TestClient(app)


def test_get_existing_customer_profile():
    """
    Basic smoke test: calling the profile endpoint for some cust_id
    should return either 200 with a 'profile' key or 404 if not found.

    We don't assert a specific ID exists here because the dataset may
    change; this is just to verify that the API and DuckDB path work.
    """
    resp = client.get("/api/customers/1/profile")
    assert resp.status_code in (200, 404)

    if resp.status_code == 200:
        data = resp.json()
        assert "profile" in data
        assert "last_transactions" in data
        assert "risk_flags" in data


def test_get_missing_customer_profile():
    """
    Very large cust_id should not exist and must return 404.
    """
    resp = client.get("/api/customers/99999999/profile")
    assert resp.status_code == 404


