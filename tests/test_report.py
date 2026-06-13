"""
tests/test_report.py
Tests pour l'endpoint POST /api/report.
"""

from __future__ import annotations


def test_report_generation(client, test_csv):
    # 1. Lancer predict
    r_pred = client.post(
        "/api/predict",
        files={"file": ("test.csv", test_csv, "text/csv")},
    )
    assert r_pred.status_code == 200

    # 2. Générer le rapport (corps vide → utilise le cache)
    r = client.post("/api/report", json={})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert len(r.content) > 0, "Le PDF est vide"
    # Vérifier l'en-tête PDF
    assert r.content[:4] == b"%PDF", "Le contenu n'est pas un PDF valide"


def test_report_without_predict(client):
    # Réinitialiser le cache
    client.app.state.results_cache = {}
    r = client.post("/api/report", json={})
    assert r.status_code == 400
