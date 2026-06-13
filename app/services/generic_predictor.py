"""
app/services/generic_predictor.py

Moteur de prédiction en mode générique (schéma inconnu).

Modes supportés
───────────────
 "ae_isoforest"  AE pré-entraîné (features PaySim avec fallbacks)
                 + IsolationForest fitté on-the-fly sur les colonnes numériques brutes
                 Risque : les deux flaggent → CRITIQUE ; un seul → ELEVE

 "ae_only"       AE seul
                 Risque : ae_score ≥ 2×seuil → CRITIQUE ; ≥ seuil → ELEVE

 "isoforest"     IsolationForest seul
                 Risque : decision < -0.15 → CRITIQUE ; prediction=-1 → ELEVE

Notes
─────
 • L'IsoForest est toujours fitté sur le batch courant (transductif).
   Le modèle iso_forest.pkl entraîné sur PaySim n'est PAS utilisé ici.
 • La contamination présumée est fixée à 5 % (ajustable via CONTAMINATION).
 • Le tri de la liste résultante : anomalies en tête, puis par score composé.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.services.schema_detector import SchemaDetectionResult

# Fraction d'anomalies présumée dans le batch (paramètre IsoForest)
CONTAMINATION = 0.05

# Seuils IsoForest (decision_function)
_ISO_CRITICAL = -0.15    # en-dessous → CRITIQUE
# prediction=-1 ET score > _ISO_CRITICAL → ELEVE


def _fit_isoforest(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fitte un IsoForest on-the-fly et retourne (scores, predictions).

    scores      : decision_function — valeurs en [-0.5, 0.5]
                  positif = inlier, négatif = anomalie
    predictions : 1 = normal, -1 = anomalie (sklearn convention)
    """
    X = df[numeric_cols].copy()
    # Imputation par médiane (plus robuste que la moyenne sur données frauduleuses)
    X = X.fillna(X.median(numeric_only=True))
    # Clamp les valeurs extrêmes (NaN résiduels → 0)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    scores = iso.decision_function(X_scaled).astype(np.float32)
    predictions = iso.predict(X_scaled)   # 1 ou -1
    return scores, predictions


def _risk_ae_only(ae_score: float, ae_threshold: float) -> tuple[str, bool]:
    if ae_score >= ae_threshold * 2.0:
        return "CRITIQUE", True
    if ae_score >= ae_threshold:
        return "ELEVE", True
    return "FAIBLE", False


def _risk_iso_only(iso_score: float, iso_pred: int) -> tuple[str, bool]:
    if iso_score < _ISO_CRITICAL:
        return "CRITIQUE", True
    if iso_pred == -1:
        return "ELEVE", True
    return "FAIBLE", False


def _risk_combined(
    ae_score: float,
    ae_threshold: float,
    iso_score: float,
    iso_pred: int,
) -> tuple[str, bool]:
    ae_flag = ae_score >= ae_threshold
    iso_flag = iso_pred == -1
    if ae_flag and iso_flag:
        return "CRITIQUE", True
    if ae_flag or iso_flag:
        return "ELEVE", True
    return "FAIBLE", False


def predict_generic_batch(
    df_original: pd.DataFrame,
    X_arr: Optional[np.ndarray],
    models: dict,
    schema_result: SchemaDetectionResult,
    df_mapped: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """
    Prédiction en mode générique.

    Paramètres
    ----------
    df_original   : DataFrame brut (colonnes non renommées) — pour IsoForest
    X_arr         : array (N, 14) features AE avec fallbacks (None si non construit)
    models        : dict des modèles chargés (ae, ae_threshold)
    schema_result : résultat de detect_schema_mode()
    df_mapped     : DataFrame avec colonnes canoniques (pour type/amount dans la réponse)

    Retourne
    --------
    Liste de dicts triée par risque décroissant — même format que predict_batch()
    mais sans champ xgb_score.
    """
    n = len(df_original)
    ae = models.get("ae")
    ae_threshold = float(models.get("ae_threshold", 1.753))
    ref = df_mapped if df_mapped is not None else df_original

    # ── Scores AE ──────────────────────────────────────────────────────────
    ae_scores: Optional[np.ndarray] = None
    if schema_result.use_ae and ae is not None and X_arr is not None:
        try:
            ae_scores = ae.predict_score(X_arr).astype(np.float32)
        except Exception:
            ae_scores = None
            schema_result.warnings.append("Autoencoder : erreur lors du calcul des scores.")

    # ── Scores IsoForest ────────────────────────────────────────────────────
    iso_scores: Optional[np.ndarray] = None
    iso_predictions: Optional[np.ndarray] = None
    if schema_result.use_isoforest and schema_result.numeric_cols_for_iso:
        try:
            iso_scores, iso_predictions = _fit_isoforest(
                df_original, schema_result.numeric_cols_for_iso
            )
        except Exception as exc:
            schema_result.warnings.append(f"IsolationForest : erreur — {exc}")

    # ── Extraction vectorisée des métadonnées (évite iloc[i] en boucle) ────
    tx_ids   = df_original.index.tolist()
    tx_types = ref["type"].tolist()   if "type"   in ref.columns else ["UNKNOWN"] * n
    if "amount" in ref.columns:
        try:
            amounts = [float(v) for v in ref["amount"].tolist()]
        except (ValueError, TypeError):
            amounts = [0.0] * n
    else:
        amounts = [0.0] * n

    # ── Assemblage par transaction ──────────────────────────────────────────
    results: list[dict] = []

    for i in range(n):
        ae_s = float(ae_scores[i]) if ae_scores is not None else None
        iso_s = float(iso_scores[i]) if iso_scores is not None else None
        iso_p = int(iso_predictions[i]) if iso_predictions is not None else None

        # Niveau de risque selon le mode
        if ae_s is not None and iso_s is not None:
            risk, is_anomaly = _risk_combined(ae_s, ae_threshold, iso_s, iso_p)
        elif ae_s is not None:
            risk, is_anomaly = _risk_ae_only(ae_s, ae_threshold)
        elif iso_s is not None:
            risk, is_anomaly = _risk_iso_only(iso_s, iso_p)
        else:
            risk, is_anomaly = "FAIBLE", False

        tx_type = str(tx_types[i])
        amount  = amounts[i]

        entry: dict = {
            "tx_id": int(tx_ids[i]),
            "type": tx_type,
            "amount": amount,
            "risk_level": risk,
            "is_fraud_predicted": is_anomaly,
            "prediction_mode": schema_result.mode,
        }
        if ae_s is not None:
            entry["ae_score"] = round(ae_s, 6)
        if iso_s is not None:
            entry["isoforest_score"] = round(iso_s, 6)

        results.append(entry)

    # ── Tri : anomalies en tête, puis par score le plus alarmant ───────────
    def _sort_key(r: dict) -> tuple:
        primary = 1 if r["is_fraud_predicted"] else 0
        ae_v = r.get("ae_score", 0.0)
        iso_v = -r.get("isoforest_score", 0.0)   # plus négatif = plus suspect
        return (primary, ae_v + iso_v)

    results.sort(key=_sort_key, reverse=True)
    return results
