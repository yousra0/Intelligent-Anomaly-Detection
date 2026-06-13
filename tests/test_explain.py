"""
tests/test_explain.py
Tests pour l'endpoint GET /api/explain/{tx_id}.
"""

from __future__ import annotations


def test_explain_without_predict_first(client):
    # Réinitialiser le cache
    client.app.state.results_cache = {}
    r = client.get("/api/explain/0")
    assert r.status_code == 400
    assert "predict" in r.json()["detail"].lower()


def test_explain_after_predict(client, test_csv):
    # 1. Lancer predict pour peupler le cache
    r_pred = client.post(
        "/api/predict",
        files={"file": ("test.csv", test_csv, "text/csv")},
    )
    assert r_pred.status_code == 200
    transactions = r_pred.json()["transactions"]
    assert len(transactions) > 0

    first_tx_id = transactions[0]["tx_id"]

    # 2. Explain sur la première transaction
    r = client.get(f"/api/explain/{first_tx_id}")
    assert r.status_code == 200

    data = r.json()
    assert data["tx_id"] == first_tx_id
    assert "xgb_score" in data
    assert "ae_score" in data
    assert "risk_level" in data
    assert isinstance(data["shap_values"], dict)
    assert len(data["shap_values"]) == 14 or "error" in data["shap_values"]
    assert isinstance(data["lime_rules"], list)
    assert isinstance(data["llm"], dict)
    assert "risk_level" in data["llm"]


def test_explain_invalid_tx_id(client, test_csv):
    # Lancer predict
    r_pred = client.post(
        "/api/predict",
        files={"file": ("test.csv", test_csv, "text/csv")},
    )
    assert r_pred.status_code == 200

    # ID inexistant
    r = client.get("/api/explain/999999")
    assert r.status_code == 404
