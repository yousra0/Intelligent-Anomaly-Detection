"""
baseline_config.py
==================
Chargement des métriques de référence (baseline) depuis baseline_report.json.
"""

from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

_HARDCODED_BASELINE = {
    "recall":    0.0039,
    "precision": 1.0000,
    "f1":        0.0077,
    "model":     "isFlaggedFraud",
}


def diagnose_baseline_report(report_path) -> tuple[bool, str]:
    """
    Diagnostique le fichier baseline_report.json.
    
    Returns:
        (is_ok, message) tuple.
        - is_ok: True si le fichier existe et est valide
        - message: Description du statut
    """
    path = Path(report_path)
    
    if not path.exists():
        return False, f"WARN: Fichier absent — {path.absolute()}"
    
    if not path.is_file():
        return False, f"WARN: Pas un fichier — {path.absolute()}"
    
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        if "baseline_metier" not in data:
            return False, f"WARN: Clé 'baseline_metier' absente dans {path.name}"
        
        return True, f"OK: {path.name} chargé (recall={data['baseline_metier'].get('recall')})"
    
    except json.JSONDecodeError as e:
        return False, f"WARN: JSON malformé dans {path.name} — {e}"
    except Exception as e:
        return False, f"WARN: Erreur lecture {path.name} — {e}"


def load_baseline_metrics(report_path=None) -> dict:
    """
    Retourne les métriques baseline isFlaggedFraud.
    
    Args:
        report_path: Chemin optionnel vers baseline_report.json.
                    Si None ou fichier absent, retourne les valeurs hardcodées.
    
    Returns:
        dict avec recall, precision, f1, model.
        
    Note:
        Si report_path est fourni mais le fichier n'existe pas ou est malformé,
        une gestion silencieuse retourne les hardcoded defaults.
    """
    if report_path is None:
        warnings.warn(
            "baseline_report.json absent: using hardcoded baseline metrics.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _HARDCODED_BASELINE.copy()
    
    path = Path(report_path)
    if not path.exists():
        warnings.warn(
            f"baseline_report.json absent: using hardcoded baseline metrics for {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _HARDCODED_BASELINE.copy()
    
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        bm = report.get("baseline_metier", _HARDCODED_BASELINE)
        return {
            "recall":    bm.get("recall",    _HARDCODED_BASELINE["recall"]),
            "precision": bm.get("precision", _HARDCODED_BASELINE["precision"]),
            "f1":        bm.get("f1",        _HARDCODED_BASELINE["f1"]),
            "model":     bm.get("model",     _HARDCODED_BASELINE["model"]),
        }
    except (json.JSONDecodeError, KeyError, IOError) as e:
        warnings.warn(
            f"baseline_report.json unreadable ({e}): using hardcoded baseline metrics for {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _HARDCODED_BASELINE.copy()


def load_best_baseline_ml(report_path=None) -> dict:
    """
    Retourne les métriques du meilleur modèle ML (RF_smote, NB03).
    
    Args:
        report_path: Chemin optionnel vers baseline_report.json.
                    Si None ou fichier absent, retourne les valeurs hardcodées.
    
    Returns:
        dict avec recall, precision, f1, pr_auc, roc_auc, model, threshold.
        
    Note:
        Si report_path est fourni mais le fichier n'existe pas ou est malformé,
        une gestion silencieuse retourne les hardcoded defaults.
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
        warnings.warn(
            "baseline_report.json absent: using hardcoded best-ML baseline metrics.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _default.copy()
    
    path = Path(report_path)
    if not path.exists():
        warnings.warn(
            f"baseline_report.json absent: using hardcoded best-ML baseline metrics for {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _default.copy()
    
    try:
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
    except (json.JSONDecodeError, KeyError, IOError, ValueError):
        warnings.warn(
            f"baseline_report.json unreadable: using hardcoded best-ML baseline metrics for {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
    
    return _default.copy()