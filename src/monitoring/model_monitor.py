# -*- coding: utf-8 -*-
"""
src/monitoring/model_monitor.py
=================================
Surveillance des performances du modèle en production.

Enregistre les prédictions et les labels réels (quand disponibles),
calcule des métriques glissantes, et alerte si le recall chute.

Usage :
    monitor = ModelMonitor(threshold_recall_alert=0.70)
    monitor.log_predictions(y_true, y_pred, y_score, batch_name="semaine_23")
    rapport = monitor.compute_performance_report()
    monitor.save_report(rapport)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    confusion_matrix, average_precision_score,
)


class ModelMonitor:
    """
    Surveille les performances du modèle sur des batches successifs.

    Parameters
    ----------
    model_name : str
        Identifiant du modèle surveillé.
    model_version : str
        Version du modèle (ex : "autoencoder_v1", "xgb_smote_v2").
    threshold_recall_alert : float
        Seuil de recall en-dessous duquel une alerte est déclenchée (défaut 0.70).
    log_path : str
        Chemin vers le fichier JSONL de logs de performance.
    """

    def __init__(
        self,
        model_name            : str   = "autoencoder",
        model_version         : str   = "1.0",
        threshold_recall_alert: float = 0.70,
        log_path              : str   = "outputs/audit_trail/performance_log.jsonl",
    ):
        self.model_name             = model_name
        self.model_version          = model_version
        self.threshold_recall_alert = threshold_recall_alert
        self.log_path               = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Buffer en mémoire pour les batches courants
        self._batches: list[dict] = []

    def log_predictions(
        self,
        y_true    : np.ndarray | pd.Series,
        y_pred    : np.ndarray | pd.Series,
        y_score   : Optional[np.ndarray | pd.Series] = None,
        batch_name: str = "batch",
        decision_threshold: float = 0.5,
    ) -> dict:
        """
        Enregistre un batch de prédictions et calcule ses métriques.

        Returns
        -------
        dict : métriques du batch (recall, precision, f1, alert, ...)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        recall    = float(recall_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        f1        = float(f1_score(y_true, y_pred, zero_division=0))
        cm        = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        pr_auc = None
        if y_score is not None:
            try:
                pr_auc = float(average_precision_score(y_true, np.asarray(y_score)))
            except Exception:
                pass

        alert = recall < self.threshold_recall_alert

        record = {
            "timestamp"          : datetime.now(timezone.utc).isoformat(),
            "model_name"         : self.model_name,
            "model_version"      : self.model_version,
            "batch_name"         : batch_name,
            "n_transactions"     : int(len(y_true)),
            "n_fraud_true"       : int(y_true.sum()),
            "n_fraud_pred"       : int(y_pred.sum()),
            "decision_threshold" : decision_threshold,
            "recall"             : round(recall, 4),
            "precision"          : round(precision, 4),
            "f1"                 : round(f1, 4),
            "pr_auc"             : round(pr_auc, 4) if pr_auc is not None else None,
            "tp"                 : int(tp),
            "fp"                 : int(fp),
            "fn"                 : int(fn),
            "tn"                 : int(tn),
            "alert_recall"       : alert,
            "threshold_recall_alert": self.threshold_recall_alert,
        }

        self._batches.append(record)
        # Persistance immédiate (append)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def compute_performance_report(self, last_n_batches: int = 10) -> dict:
        """
        Calcule un rapport de performance agrégé sur les N derniers batches.

        Returns
        -------
        dict : recall_mean, recall_trend, alert_count, recommandation, ...
        """
        if not self._batches:
            return {"status": "no_data", "message": "Aucun batch enregistré."}

        recent = self._batches[-last_n_batches:]
        recalls    = [b["recall"]    for b in recent]
        precisions = [b["precision"] for b in recent]
        f1s        = [b["f1"]        for b in recent]
        alerts     = [b["alert_recall"] for b in recent]

        recall_trend = float(np.polyfit(range(len(recalls)), recalls, 1)[0]) if len(recalls) > 1 else 0.0

        if recall_trend < -0.02:
            trend_status = "degrading"
        elif recall_trend > 0.01:
            trend_status = "improving"
        else:
            trend_status = "stable"

        alert_count = sum(alerts)
        if alert_count >= 3 or (len(recent) > 0 and recalls[-1] < self.threshold_recall_alert):
            recommandation = "RÉENTRAÎNEMENT URGENT — recall en dessous du seuil critique."
        elif trend_status == "degrading":
            recommandation = "Surveillance accrue — tendance à la baisse du recall."
        else:
            recommandation = "Performances stables."

        return {
            "timestamp"        : datetime.now(timezone.utc).isoformat(),
            "model_name"       : self.model_name,
            "model_version"    : self.model_version,
            "n_batches_analysed": len(recent),
            "recall_mean"      : round(float(np.mean(recalls)), 4),
            "recall_min"       : round(float(np.min(recalls)), 4),
            "recall_last"      : round(recalls[-1], 4),
            "recall_trend"     : round(recall_trend, 6),
            "trend_status"     : trend_status,
            "precision_mean"   : round(float(np.mean(precisions)), 4),
            "f1_mean"          : round(float(np.mean(f1s)), 4),
            "alert_count"      : alert_count,
            "recommandation"   : recommandation,
        }

    def print_report(self, report: dict) -> None:
        sep = "=" * 60
        icons = {"degrading": "🔴", "stable": "✅", "improving": "🟢"}
        icon = icons.get(report.get("trend_status", ""), "?")
        print(sep)
        print(f"  SURVEILLANCE MODÈLE  {icon} {report.get('trend_status', '').upper()}")
        print(f"  {report['model_name']} v{report['model_version']}")
        print(sep)
        print(f"  Batches analysés : {report.get('n_batches_analysed', 0)}")
        print(f"  Recall moyen     : {report.get('recall_mean', 0):.4f}")
        print(f"  Recall dernier   : {report.get('recall_last', 0):.4f}")
        print(f"  Recall min       : {report.get('recall_min', 0):.4f}")
        print(f"  Tendance         : {report.get('recall_trend', 0):+.4f} / batch")
        print(f"  Alertes recall   : {report.get('alert_count', 0)}")
        print("-" * 60)
        print(f"  → {report.get('recommandation', '')}")
        print(sep)

    @classmethod
    def load_history(cls, log_path: str = "outputs/audit_trail/performance_log.jsonl") -> list[dict]:
        """Charge l'historique complet des batches depuis le fichier JSONL."""
        p = Path(log_path)
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
