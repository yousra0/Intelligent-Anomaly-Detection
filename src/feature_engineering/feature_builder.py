"""
feature_builder.py
==================
Feature engineering pour la détection de fraude PaySim.

7 features dérivées, toutes validées dans 01_data_understanding.ipynb :

  Temporelles  : hour, day, week  (depuis step)
  Binaires     : high_risk_hour, is_transfer_or_cashout, dest_zero_balance
  Comportement : balance_diff_orig  (corrélation 0.3662 vs isFraud — signal dominant)

Référence EDA :
  - Cell 53  : hour/day/week
  - Cell 57  : HIGH_RISK_HOURS = [0,1,2,3,4,5,6,7,8,9,23]  ratio 10.63×
  - Cell 70  : balance_diff_orig  corr=0.3662
  - Cell 71  : is_transfer_or_cashout  — TRANSFER+CASH_OUT seuls types avec fraudes
  - Cell 64,72 : dest_zero_balance  taux fraude 2.52 % vs 0.13 % global
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
    from .behavioral_features import add_balance_diff_orig, add_dest_zero_balance
    from .temporal_features import create_temporal_features
except ImportError:
    # Allow running this file directly: python src/feature_engineering/feature_builder.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.feature_engineering.behavioral_features import (
        add_balance_diff_orig,
        add_dest_zero_balance,
    )
    from src.feature_engineering.temporal_features import create_temporal_features

# ---------------------------------------------------------------------------
# Constante — heures à risque mesurées empiriquement dans l'EDA (Cell 57)
# Ne pas modifier sans ré-exécuter l'analyse temporelle.
# ---------------------------------------------------------------------------
HIGH_RISK_HOURS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 23]


# ---------------------------------------------------------------------------
# Features individuelles
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée hour, day, week depuis la colonne ``step``.

    PaySim : 1 step = 1 heure  →  simulation sur 31 jours (743 steps max).

    Returns:
        DataFrame avec 3 nouvelles colonnes : hour (0-23), day (0-30), week (0-4).
    """
    # Keep feature set stable for current baseline (hour/day/week only).
    return create_temporal_features(df, include_is_weekend=False)


def add_high_risk_hour(
    df: pd.DataFrame,
    high_risk_hours: list[int] | None = None,
) -> pd.DataFrame:
    """
    Feature binaire : 1 si l'heure de la transaction appartient aux heures
    à risque élevé déterminées dans l'EDA.

    Ratio fraude mesuré : 10.63× vs heures normales (EDA Cell 57).

    Args:
        df: DataFrame contenant déjà la colonne ``hour``.
        high_risk_hours: Liste des heures à risque. Par défaut : HIGH_RISK_HOURS.

    Returns:
        DataFrame avec la colonne ``high_risk_hour`` ajoutée.
    """
    if high_risk_hours is None:
        high_risk_hours = HIGH_RISK_HOURS
    df = df.copy()
    df["high_risk_hour"] = df["hour"].isin(high_risk_hours).astype(int)
    return df


def add_transfer_cashout_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature binaire : 1 si le type de transaction est TRANSFER ou CASH_OUT.

    Justification (EDA Cell 43 & 71) :
        - TRANSFER  : taux fraude 0.7745%
        - CASH_OUT  : taux fraude 0.1807%
        - Autres types (CASH_IN, DEBIT, PAYMENT) : 0 fraudes dans le sample

    Returns:
        DataFrame avec la colonne ``is_transfer_or_cashout`` ajoutée.
    """
    df = df.copy()
    df["is_transfer_or_cashout"] = (
        df["type"].isin(["TRANSFER", "CASH_OUT"])
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    high_risk_hours: list[int] | None = None,
) -> pd.DataFrame:
    """
    Applique toutes les transformations de feature engineering dans l'ordre
    défini dans ``02_data_preparation.ipynb``.

    Pipeline :
        1. add_temporal_features       → hour, day, week
        2. add_high_risk_hour          → high_risk_hour
        3. add_transfer_cashout_flag   → is_transfer_or_cashout
        4. add_balance_diff_orig       → balance_diff_orig
        5. add_dest_zero_balance       → dest_zero_balance

    Args:
        df: DataFrame PaySim brut (colonnes originales attendues).
        high_risk_hours: Si None, utilise HIGH_RISK_HOURS de l'EDA.

    Returns:
        DataFrame avec les 7 nouvelles features ajoutées.
        Les colonnes originales sont conservées (la suppression des colonnes
        à risque de leakage est réalisée dans ``preprocessor.py``).
    """
    df = add_temporal_features(df)
    df = add_high_risk_hour(df, high_risk_hours)
    df = add_transfer_cashout_flag(df)
    df = add_balance_diff_orig(df)
    df = add_dest_zero_balance(df)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_features(df: pd.DataFrame, target_col: str = "isFraud") -> dict:
    """
    Vérifie que les features engineerées reproduisent les valeurs mesurées
    dans l'EDA. Retourne un dict {feature: {value, expected, ok}}.

    Utilisé dans le notebook pour la cellule de validation.
    """
    results = {}

    # Plages temporelles
    for feat, (lo, hi) in [("hour", (0, 23)), ("day", (0, 30)), ("week", (0, 4))]:
        ok = int(df[feat].min()) == lo and int(df[feat].max()) == hi
        results[feat] = {
            "range": [int(df[feat].min()), int(df[feat].max())],
            "expected": [lo, hi],
            "ok": ok,
        }

    # high_risk_hour — ratio fraude > 5× (tolérance car sample peut varier)
    if target_col in df.columns and df[target_col].sum() > 0:
        rate_risk   = df[df["high_risk_hour"] == 1][target_col].mean()
        rate_normal = df[df["high_risk_hour"] == 0][target_col].mean()
        ratio = rate_risk / rate_normal if rate_normal > 0 else 0
        results["high_risk_hour_ratio"] = {
            "value": round(ratio, 2),
            "expected": "~10.63",
            "ok": ratio > 5.0,
        }

        # dest_zero_balance — taux fraude > 2%
        dzb_rate = df[df["dest_zero_balance"] == 1][target_col].mean() * 100
        results["dest_zero_balance_fraud_pct"] = {
            "value": round(dzb_rate, 4),
            "expected": "~2.52",
            "ok": dzb_rate > 1.5,
        }

        # balance_diff_orig — corrélation > 0.30
        corr = df["balance_diff_orig"].corr(df[target_col])
        results["balance_diff_orig_corr"] = {
            "value": round(corr, 4),
            "expected": "~0.3662",
            "ok": corr > 0.30,
        }

    return results
