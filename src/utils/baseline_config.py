"""
baseline_config.py
==================
Chargement des métriques de référence (baseline) depuis baseline_report.json.
"""

from __future__ import annotations
import json
from pathlib import Path

_HARDCODED_BASELINE = {
    "recall":    0.0039,
    "precision": 1.0000,
    "f1":        0.0077,
    "model":     "isFlaggedFraud",
}


def load_baseline_metrics(report_path=None) -> dict:
    """
    Retourne les métriques baseline isFlaggedFraud.
    report_path est optionnel — si None ou fichier absent, retourne
    les valeurs hardcodées (recall=0.0039, precision=1.0, f1=0.0077).
    """
    if report_path is None:
        return _HARDCODED_BASELINE.copy()
    try:
        path = Path(report_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
            bm = report.get("baseline_metier", _HARDCODED_BASELINE)
            return {
                "recall":    bm.get("recall",    _HARDCODED_BASELINE["recall"]),
                "precision": bm.get("precision", _HARDCODED_BASELINE["precision"]),
                "f1":        bm.get("f1",        _HARDCODED_BASELINE["f1"]),
                "model":     bm.get("model",     _HARDCODED_BASELINE["model"]),
            }
    except Exception:
        pass
    return _HARDCODED_BASELINE.copy()


def load_best_baseline_ml(report_path=None) -> dict:
    """
    Retourne les métriques du meilleur modèle ML (RF_smote, NB03).
    report_path est optionnel — si None, retourne les valeurs hardcodées.
    """
    _default = {
        "recall":    0.7949,
        "precision": 0.8158,
        "f1":        0.8052,
        "pr_auc":    0.8405,
        "roc_auc":   0.9940,
        "model":     "RF_smote",
        "threshold": 0.6291,
    }
    if report_path is None:
        return _default.copy()
    try:
        path = Path(report_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
            models = report.get("models", [])
            if models:
                best = max(models, key=lambda m: m.get("test_metrics", {}).get("f1", 0))
                tm = best.get("test_metrics", {})
                return {
                    "recall":    tm.get("recall",    _default["recall"]),
                    "precision": tm.get("precision", _default["precision"]),
                    "f1":        tm.get("f1",        _default["f1"]),
                    "pr_auc":    tm.get("pr_auc",    _default["pr_auc"]),
                    "roc_auc":   tm.get("roc_auc",   _default["roc_auc"]),
                    "model":     best.get("name",    _default["model"]),
                    "threshold": best.get("optimal_threshold", _default["threshold"]),
                }
    except Exception:
        pass
    return _default.copy()