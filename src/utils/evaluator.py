"""
evaluation.py
=============
Fonctions d'évaluation pour la détection de fraude.

Dans un contexte de fraude financière avec ratio 1:774, les métriques
pertinentes sont **différentes** de l'accuracy :

  Priorité 1 → Recall    : fraudes effectivement détectées
  Priorité 2 → PR-AUC    : synthétise le tradeoff Precision/Recall
  Priorité 3 → F1-Score  : équilibre entre Recall et Precision
  Info       → ROC-AUC   : moins informatif sous fort déséquilibre

Référence EDA :
  Baseline isFlaggedFraud → Recall=0.0039, Precision=1.0, F1=0.0077
  Objectif DL             → Recall > 0.0039, F1 > 0.0077
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------------------------
# Métriques principales
# ---------------------------------------------------------------------------

def compute_fraud_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series | None = None,
    threshold: float = 0.5,
    model_name: str = "model",
) -> dict:
    """
    Calcule les métriques de détection de fraude.

    Args:
        y_true:     Labels réels (0/1).
        y_pred:     Prédictions binaires (0/1). Si y_score fourni et
                    y_pred=None, les prédictions sont calculées depuis y_score.
        y_score:    Scores/probabilités de la classe positive (optionnel).
        threshold:  Seuil de décision (défaut 0.5).
        model_name: Nom du modèle pour les logs.

    Returns:
        dict avec : recall, precision, f1, accuracy, roc_auc (si y_score),
                    pr_auc (si y_score), confusion_matrix, threshold.
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)

    # Métriques de base
    recall    = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = float((y_true == y_pred).mean())
    cm        = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "model":      model_name,
        "threshold":  threshold,
        "recall":     round(recall,    4),
        "precision":  round(precision, 4),
        "f1":         round(f1,        4),
        "accuracy":   round(accuracy,  4),
        "confusion_matrix": cm,
        # tp/fp/fn/tn pour lisibilité
        "tp": int(cm[1][1]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tn": int(cm[0][0]),
    }

    if y_score is not None:
        y_score = np.asarray(y_score)
        result["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        result["pr_auc"]  = round(float(average_precision_score(y_true, y_score)), 4)

    return result


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    metric: str = "f1",
    beta: float = 1.0,
) -> tuple[float, float]:
    """
    Trouve le seuil de décision optimal sur la courbe Precision-Recall.

    Dans un contexte fraude, on peut vouloir favoriser le Recall
    (beta > 1) pour ne manquer aucune fraude.

    Args:
        y_true:  Labels réels.
        y_score: Scores de probabilité de la classe positive.
        metric:  "f1" ou "fbeta" (utilise beta).
        beta:    Poids du Recall dans Fbeta (beta=2 → 2× plus important).

    Returns:
        (optimal_threshold, optimal_score)
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    if metric == "f1":
        # F1 = 2 * P * R / (P + R)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(
                (precisions[:-1] + recalls[:-1]) > 0,
                2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
                0.0,
            )
    else:
        # Fbeta
        beta2 = beta ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(
                (beta2 * precisions[:-1] + recalls[:-1]) > 0,
                (1 + beta2) * precisions[:-1] * recalls[:-1]
                / (beta2 * precisions[:-1] + recalls[:-1]),
                0.0,
            )

    idx   = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


# ---------------------------------------------------------------------------
# Rapport formaté
# ---------------------------------------------------------------------------

def print_metrics_report(
    metrics: dict,
    baseline_recall: float = 0.0039,
    baseline_f1: float = 0.0077,
) -> None:
    """
    Affiche un rapport de métriques formaté avec comparaison à la baseline
    isFlaggedFraud (EDA Cell 39).

    Args:
        metrics:          Sortie de compute_fraud_metrics().
        baseline_recall:  Recall de la règle métier (EDA : 0.0039).
        baseline_f1:      F1 de la règle métier (EDA : 0.0077).
    """
    sep = "=" * 58
    recall_ok    = "✅" if metrics["recall"]    > baseline_recall    else "❌"
    f1_ok        = "✅" if metrics["f1"]        > baseline_f1        else "❌"
    pr_auc_str   = f'{metrics["pr_auc"]:.4f}' if "pr_auc"   in metrics else "N/A"
    roc_auc_str  = f'{metrics["roc_auc"]:.4f}' if "roc_auc" in metrics else "N/A"

    print(sep)
    print(f"  {metrics.get('model', 'model'):<30}  seuil={metrics['threshold']:.2f}")
    print(sep)
    print(f"  Recall    : {metrics['recall']:.4f}  {recall_ok}  (baseline {baseline_recall:.4f})")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}  {f1_ok}  (baseline {baseline_f1:.4f})")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  PR-AUC    : {pr_auc_str}")
    print(f"  ROC-AUC   : {roc_auc_str}")
    print("-" * 58)
    cm = metrics["confusion_matrix"]
    print(f"  Matrice de confusion :")
    print(f"    TP={metrics['tp']:>6,}   FN={metrics['fn']:>6,}")
    print(f"    FP={metrics['fp']:>6,}   TN={metrics['tn']:>6,}")
    print(sep)


def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Compare plusieurs modèles dans un DataFrame trié par Recall décroissant.

    Args:
        metrics_list: Liste de dicts retournés par compute_fraud_metrics().

    Returns:
        DataFrame avec colonnes : model, recall, precision, f1, pr_auc, roc_auc,
                                  tp, fp, fn, threshold.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "model":     m.get("model", "?"),
            "recall":    m.get("recall", 0),
            "precision": m.get("precision", 0),
            "f1":        m.get("f1", 0),
            "pr_auc":    m.get("pr_auc", None),
            "roc_auc":   m.get("roc_auc", None),
            "tp":        m.get("tp", 0),
            "fp":        m.get("fp", 0),
            "fn":        m.get("fn", 0),
            "threshold": m.get("threshold", 0.5),
        })
    df = pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)
    return df
