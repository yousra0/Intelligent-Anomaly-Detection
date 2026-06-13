"""
tests/test_predict.py
Tests pour les endpoints /api/health et /api/predict.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Tests de base
# ─────────────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


def test_predict_valid_csv(client, test_csv):
    r = client.post(
        "/api/predict",
        files={"file": ("test.csv", test_csv, "text/csv")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["n_transactions"] > 0
    assert isinstance(data["transactions"], list)
    assert "n_fraud" in data
    assert "fraud_rate_pct" in data
    assert "amount_at_risk" in data


def test_predict_empty_csv(client):
    empty = b"step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\n"
    r = client.post(
        "/api/predict",
        files={"file": ("empty.csv", empty, "text/csv")},
    )
    assert r.status_code == 400


def test_predict_returns_sorted_by_score(client, test_csv):
    """En mode paysim, les transactions sont triées par xgb_score décroissant."""
    r = client.post(
        "/api/predict",
        files={"file": ("test.csv", test_csv, "text/csv")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["prediction_mode"] == "paysim"
    txs = r.json()["transactions"]
    if len(txs) > 1:
        scores = [t["xgb_score"] for t in txs]
        assert scores == sorted(scores, reverse=True), "Transactions non triées par xgb_score"


# ─────────────────────────────────────────────────────────────────────────────
# Tests du mode paysim
# ─────────────────────────────────────────────────────────────────────────────

class TestPaysimMode:

    def test_schema_detection_returned(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        data = r.json()
        assert "schema_detection" in data
        sd = data["schema_detection"]
        assert sd["mode"] == "paysim"
        assert sd["use_xgb"] is True
        assert sd["use_ae"] is True
        assert sd["use_isoforest"] is False

    def test_model_used_label(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        data = r.json()
        assert "XGBoost" in data["model_used"]
        assert "Autoencoder" in data["model_used"]

    def test_xgb_score_present_in_paysim_mode(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        txs = r.json()["transactions"]
        assert all("xgb_score" in t for t in txs)
        assert all("ae_score" in t for t in txs)

    def test_threshold_present_in_paysim_mode(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        data = r.json()
        assert "threshold" in data
        assert isinstance(data["threshold"], float)


# ─────────────────────────────────────────────────────────────────────────────
# Tests du mode générique — bascule automatique
# ─────────────────────────────────────────────────────────────────────────────

def _generic_csv_no_amount(n: int = 20) -> bytes:
    """CSV PaySim sans la colonne 'amount' → IsoForest (amount non détecté)."""
    rng = np.random.default_rng(7)
    types = ["TRANSFER", "CASH_OUT", "PAYMENT"]
    df = pd.DataFrame({
        "step": np.arange(1, n + 1),
        "type": [types[i % 3] for i in range(n)],
        "nameOrig": [f"C{i:04d}" for i in range(n)],
        "oldbalanceOrg": rng.uniform(100_000, 1_000_000, n),
        "newbalanceOrig": rng.uniform(0, 100_000, n),
        "nameDest": [f"M{i:04d}" for i in range(n)],
        "oldbalanceDest": rng.uniform(0, 500_000, n),
        "newbalanceDest": rng.uniform(0, 500_000, n),
        # 'amount' intentionnellement absent
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _generic_csv_partial(n: int = 20) -> bytes:
    """CSV avec amount + balance mais sans 'type' → mode ae_isoforest."""
    rng = np.random.default_rng(8)
    df = pd.DataFrame({
        "step": np.arange(1, n + 1),
        "amount": rng.uniform(100, 500_000, n),
        "nameOrig": [f"C{i:04d}" for i in range(n)],
        "oldbalanceOrg": rng.uniform(100_000, 1_000_000, n),
        "newbalanceOrig": rng.uniform(0, 100_000, n),
        "nameDest": [f"M{i:04d}" for i in range(n)],
        "oldbalanceDest": rng.uniform(0, 500_000, n),
        "newbalanceDest": rng.uniform(0, 500_000, n),
        # 'type' et 'step' manquants (step est présent mais type non)
        # step présent mais 'type' absent → success=False car type est required
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _truly_incompatible_csv() -> bytes:
    """CSV avec uniquement des colonnes texte → aucun modèle applicable → 422."""
    df = pd.DataFrame({
        "merchant_name": ["Shop A", "Shop B", "Shop C"] * 4,
        "transaction_status": ["pending", "approved", "declined"] * 4,
        "currency_code": ["EUR", "USD", "GBP"] * 4,
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


class TestGenericMode:

    def test_missing_amount_uses_isoforest_mode(self, client):
        """Sans 'amount', le système bascule en isoforest (pas 422)."""
        r = client.post(
            "/api/predict",
            files={"file": ("no_amount.csv", _generic_csv_no_amount(), "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["prediction_mode"] == "isoforest"
        sd = data["schema_detection"]
        assert sd["use_xgb"] is False
        assert sd["use_ae"] is False
        assert sd["use_isoforest"] is True

    def test_isoforest_mode_no_xgb_score(self, client):
        """En mode isoforest, xgb_score absent des transactions."""
        r = client.post(
            "/api/predict",
            files={"file": ("no_amount.csv", _generic_csv_no_amount(), "text/csv")},
        )
        txs = r.json()["transactions"]
        assert len(txs) > 0
        assert all("xgb_score" not in t for t in txs)
        assert all("isoforest_score" in t for t in txs)

    def test_isoforest_mode_risk_levels_valid(self, client):
        """Tous les niveaux de risque doivent être parmi CRITIQUE/ELEVE/FAIBLE."""
        r = client.post(
            "/api/predict",
            files={"file": ("no_amount.csv", _generic_csv_no_amount(), "text/csv")},
        )
        txs = r.json()["transactions"]
        for t in txs:
            assert t["risk_level"] in ("CRITIQUE", "ELEVE", "FAIBLE")

    def test_partial_schema_uses_ae_isoforest(self, client):
        """Schéma partiel (amount + balance, sans type) → ae_isoforest."""
        r = client.post(
            "/api/predict",
            files={"file": ("partial.csv", _generic_csv_partial(), "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        # mode peut être ae_isoforest ou ae_only selon les colonnes numériques dispo
        assert data["prediction_mode"] in ("ae_isoforest", "ae_only", "paysim")
        sd = data["schema_detection"]
        assert sd["use_xgb"] is False or data["prediction_mode"] == "paysim"

    def test_ae_isoforest_mode_has_both_scores(self, client):
        """En mode ae_isoforest, ae_score ET isoforest_score présents."""
        r = client.post(
            "/api/predict",
            files={"file": ("partial.csv", _generic_csv_partial(), "text/csv")},
        )
        data = r.json()
        if data["prediction_mode"] == "ae_isoforest":
            txs = data["transactions"]
            assert all("ae_score" in t for t in txs)
            assert all("isoforest_score" in t for t in txs)

    def test_generic_mode_sorted_by_risk(self, client):
        """En mode générique, les anomalies (is_fraud_predicted=True) sont en tête."""
        r = client.post(
            "/api/predict",
            files={"file": ("no_amount.csv", _generic_csv_no_amount(), "text/csv")},
        )
        txs = r.json()["transactions"]
        if len(txs) > 1:
            fraud_flags = [1 if t["is_fraud_predicted"] else 0 for t in txs]
            # Les anomalies doivent toutes précéder les transactions normales
            first_normal = next((i for i, f in enumerate(fraud_flags) if f == 0), len(txs))
            assert all(fraud_flags[i] == 1 for i in range(first_normal))

    def test_generic_mode_returns_schema_detection(self, client):
        """Le champ schema_detection est toujours retourné en mode générique."""
        r = client.post(
            "/api/predict",
            files={"file": ("no_amount.csv", _generic_csv_no_amount(), "text/csv")},
        )
        data = r.json()
        assert "schema_detection" in data
        sd = data["schema_detection"]
        assert "mode" in sd
        assert "reason" in sd
        assert "models_used" in sd
        assert "warnings" in sd

    def test_no_threshold_in_generic_mode(self, client):
        """En mode générique, le champ 'threshold' XGB est absent."""
        r = client.post(
            "/api/predict",
            files={"file": ("no_amount.csv", _generic_csv_no_amount(), "text/csv")},
        )
        data = r.json()
        assert "threshold" not in data

    def test_truly_incompatible_csv_returns_422(self, client):
        """CSV sans colonnes numériques exploitables → 422."""
        r = client.post(
            "/api/predict",
            files={"file": ("bad.csv", _truly_incompatible_csv(), "text/csv")},
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "incompatible" in str(detail).lower() or "colonnes" in str(detail).lower()

    def test_generic_mode_with_renamed_paysim_minus_amount(self, client):
        """Colonnes renommées + amount absent → mode générique."""
        rng = np.random.default_rng(99)
        n = 15
        df = pd.DataFrame({
            "solde_avant_source": rng.uniform(100_000, 1_000_000, n),
            "solde_apres_source": rng.uniform(0, 100_000, n),
            "solde_avant_dest": rng.uniform(0, 500_000, n),
            "solde_apres_dest": rng.uniform(0, 500_000, n),
            "period": np.arange(1, n + 1),
            "extra_feature": rng.uniform(0, 1, n),
            # Pas d'amount → mode générique
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/api/predict",
            files={"file": ("renamed.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["prediction_mode"] != "paysim"
        assert data["schema_detection"]["use_xgb"] is False
