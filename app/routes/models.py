"""
app/routes/models.py
GET /api/models — Métriques comparatives des 7 modèles
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()

# Modèles dans l'ordre de présentation
MODEL_NAMES = [
    "LR_balanced", "LR_smote", "RF_balanced", "RF_smote",
    "XGB_smote", "IsoForest", "AutoEncoder",
]

BEST_MODEL = "XGB_smote"


@router.get("/models")
def get_models(request: Request):
    baseline = request.app.state.models["baseline_report"]
    thresholds = request.app.state.models["thresholds"]

    # baseline_report.models est une liste de dicts
    raw_models = baseline.get("models", [])
    # Indexer par nom pour lookup rapide
    by_name: dict = {}
    for m in raw_models:
        name = m.get("name", "")
        by_name[name] = m

    result = []
    for name in MODEL_NAMES:
        entry = by_name.get(name, {})
        metrics = entry.get("test_metrics", {})
        result.append({
            "name": name,
            "recall": round(metrics.get("recall", 0.0), 4),
            "precision": round(metrics.get("precision", 0.0), 4),
            "f1": round(metrics.get("f1", 0.0), 4),
            "pr_auc": round(metrics.get("pr_auc", 0.0), 4),
            "roc_auc": round(metrics.get("roc_auc", 0.0), 4),
            "train_time_s": round(entry.get("train_time_s", 0.0), 2),
            "optimal_threshold": round(thresholds.get(name, metrics.get("threshold", 0.5)), 4),
            "is_best": name == BEST_MODEL,
        })

    return {"models": result}
