"""
app/services/predictor.py
Chargement des modèles et logique de prédiction.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes preprocessing (identiques NB02)
# ---------------------------------------------------------------------------

HIGH_RISK_HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 23]

FEATURE_COLS = [
    "step", "hour", "day", "week", "high_risk_hour", "is_transfer_or_cashout",
    "balance_diff_orig", "dest_zero_balance",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "log_amount",
]

SCALE_COLS = ["step", "hour", "day", "week", "log_amount", "balance_diff_orig"]

REQUIRED_COLS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(path: Path):
    """Charge un modèle joblib; extrait "model" si c'est un dict wrapper."""
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get("model", list(obj.values())[0])
    return obj


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les 14 features attendues par XGBoost et l'Autoencoder.

    Transformations appliquées :
      - step → heure (% 24), jour (// 24), semaine (// 168)
      - high_risk_hour : 1 si l'heure est entre 0h-9h ou 23h (horaires suspects)
      - is_transfer_or_cashout : 1 si le type est TRANSFER ou CASH_OUT (opérations à risque)
      - balance_diff_orig : solde avant - solde après (détecte les vidages de compte)
      - dest_zero_balance : 1 si le compte destinataire était à zéro avant la transaction
      - log_amount : log1p(montant) pour réduire l'effet d'échelle des très gros montants
      - type_* : encodage one-hot des 5 types de transactions (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
    """
    df = df.copy()
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["week"] = df["step"] // 168
    df["high_risk_hour"] = df["hour"].isin(HIGH_RISK_HOURS).astype(int)
    df["is_transfer_or_cashout"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_zero_balance"] = (df["oldbalanceDest"] == 0).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        df[f"type_{t}"] = (df["type"] == t).astype(int)
    return df


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------

def load_all_models(project_root: Path) -> dict:
    """Charge tous les artefacts une seule fois au démarrage de FastAPI."""
    models_dir = project_root / "outputs" / "models"
    reports_dir = project_root / "outputs" / "reports"

    from ml_core.models.autoencoder import FraudAutoEncoder

    with open(models_dir / "features.json") as f:
        features_meta = json.load(f)
    with open(models_dir / "optimal_thresholds.json") as f:
        thresholds_raw = json.load(f)
    with open(reports_dir / "baseline_report.json") as f:
        baseline_report = json.load(f)
    with open(reports_dir / "autoencoder_report.json") as f:
        ae_report = json.load(f)

    # Injecte l'entrée AutoEncoder dans baseline_report si absente
    ae_names = {m.get("name") for m in baseline_report.get("models", [])}
    if "AutoEncoder" not in ae_names:
        baseline_report.setdefault("models", []).append({
            "name": "AutoEncoder",
            "optimal_threshold": ae_report["threshold"]["optimal"],
            "train_time_s": ae_report["training"]["train_time_s"],
            "test_metrics": ae_report["test_metrics"],
        })

    # Normalise les thresholds: accepte les deux formats {"thresholds": {...}} ou direct
    thresholds = thresholds_raw.get("thresholds", thresholds_raw)

    ae = FraudAutoEncoder.load(models_dir / "autoencoder")

    return {
        "scaler": joblib.load(models_dir / "scaler.pkl"),
        "features": features_meta["all_features"],
        "thresholds": thresholds,
        "xgb": _load_model(models_dir / "xgb_smote.pkl"),
        "lr_balanced": _load_model(models_dir / "lr_balanced.pkl"),
        "lr_smote": _load_model(models_dir / "lr_smote.pkl"),
        "rf_balanced": _load_model(models_dir / "rf_balanced.pkl"),
        "rf_smote": _load_model(models_dir / "rf_smote.pkl"),
        "iso_forest": _load_model(models_dir / "iso_forest.pkl"),
        "iso_scaler": joblib.load(models_dir / "iso_forest_scaler.pkl"),
        "ae": ae,
        "ae_threshold": float(ae.threshold),
        "baseline_report": baseline_report,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, models: dict) -> np.ndarray:
    """
    Applique le preprocessing complet sur un DataFrame de transactions brut.
    Retourne np.ndarray shape (N, 14) dtype float32.
    """
    df_feat = build_features(df)
    X = df_feat[FEATURE_COLS].copy()

    scaler = models["scaler"]
    X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])

    return X.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------

def get_risk_level(xgb_score: float, threshold: float) -> str:
    """
    Convertit le score XGBoost en niveau de risque lisible.

    Logique :
      - score ≥ threshold (défaut 0.355) → CRITIQUE  (fraude probable selon le seuil optimisé)
      - score ≥ 0.5                       → ELEVE     (seuil naturel de classification binaire)
      - sinon                             → FAIBLE
    Le threshold est calibré sur le jeu de test pour maximiser le F1-score.
    """
    if xgb_score >= threshold:
        return "CRITIQUE"
    if xgb_score >= 0.5:
        return "ELEVE"
    return "FAIBLE"


def predict_batch(
    X_arr: np.ndarray,
    models: dict,
    original_df: pd.DataFrame,
) -> list[dict]:
    """
    Prédit sur un batch de transactions. Retourne une liste triée par xgb_score décroissant.
    """
    xgb = models["xgb"]
    ae = models["ae"]
    threshold = models["thresholds"].get("XGB_smote", 0.355)

    xgb_proba = xgb.predict_proba(X_arr)
    if xgb_proba.ndim == 2:
        xgb_scores = xgb_proba[:, 1]
    else:
        xgb_scores = xgb_proba

    ae_scores = ae.predict_score(X_arr)

    # Extraire les colonnes en listes Python avant la boucle :
    # pandas iloc[i] dans une boucle est O(N) × overhead Python — très lent sur gros batch.
    tx_ids   = original_df.index.tolist()
    tx_types = original_df["type"].tolist()   if "type"   in original_df.columns else ["UNKNOWN"] * len(X_arr)
    amounts  = original_df["amount"].tolist() if "amount" in original_df.columns else [0.0]        * len(X_arr)

    results = []
    for i in range(len(X_arr)):
        xgb_score = float(xgb_scores[i])
        ae_score  = float(ae_scores[i])
        risk = get_risk_level(xgb_score, threshold)
        results.append({
            "tx_id": int(tx_ids[i]),
            "type":  str(tx_types[i]),
            "amount": float(amounts[i]),
            "xgb_score": round(xgb_score, 6),
            "ae_score":  round(ae_score, 6),
            "risk_level": risk,
            "is_fraud_predicted": xgb_score >= threshold,
        })

    results.sort(key=lambda x: x["xgb_score"], reverse=True)
    return results
