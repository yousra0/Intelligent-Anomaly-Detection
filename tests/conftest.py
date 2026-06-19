"""
tests/conftest.py
Fixtures partagées pour les tests FastAPI.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def client(project_root):
    from app.main import app
    with TestClient(app) as c:
        yield c


def _make_transaction_df(n: int = 20, include_fraud_like: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    types = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]

    rows = []
    for i in range(n):
        tx_type = types[i % len(types)]
        amount = float(rng.uniform(100, 500_000))
        old_orig = float(rng.uniform(amount, amount + 1_000_000))
        new_orig = old_orig - amount if include_fraud_like and i % 5 == 0 else float(rng.uniform(0, old_orig))
        rows.append({
            "step": i + 1,
            "type": tx_type,
            "amount": amount,
            "nameOrig": f"C{i:010d}",
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": f"M{i:010d}",
            "oldbalanceDest": float(rng.uniform(0, 1_000_000)),
            "newbalanceDest": float(rng.uniform(0, 1_000_000)),
            "isFraud": 0,
            "isFlaggedFraud": 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def test_csv() -> bytes:
    df = _make_transaction_df(n=20)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def test_csv_invalid() -> bytes:
    """CSV sans la colonne 'amount'."""
    df = _make_transaction_df(n=10).drop(columns=["amount"])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
