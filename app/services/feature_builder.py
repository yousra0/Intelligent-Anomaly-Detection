"""
app/services/feature_builder.py

Constructeur de features adaptatif.

Produit les 14 features attendues par le modèle XGB/AE en s'adaptant
au dataset d'entrée :

  1. Inspecte le profil du dataset pour connaître les types réels
  2. Convertit les types si nécessaire (datetime → step numérique, etc.)
  3. Construit chaque feature et documente la stratégie utilisée
  4. Pour les colonnes manquantes, applique des valeurs par défaut + avertissement

Résultat :
  X_arr        : np.ndarray (N, 14) — prêt pour scaler.transform()
  build_report : dict avec détails sur chaque feature

Relation avec predictor.py :
  feature_builder.build() remplace l'appel à preprocess() en intégrant
  la détection de type dans le pipeline. predictor.preprocess() reste
  disponible pour les cas où le dataset est déjà en format canonique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.services.dataset_profiler import DatasetProfile
from app.services.predictor import HIGH_RISK_HOURS, FEATURE_COLS, SCALE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de construction des features
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureDetail:
    feature: str
    source_col: str
    method: str          # description humaine de la dérivation
    status: str          # "ok" | "adapted" | "missing_fallback" | "error"
    fallback_value: Optional[float] = None
    note: Optional[str] = None


@dataclass
class FeatureBuildReport:
    n_features: int
    features_ok: list[str]
    features_adapted: list[str]
    features_fallback: list[str]
    details: dict[str, FeatureDetail]
    adaptations: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "features_ok": self.features_ok,
            "features_adapted": self.features_adapted,
            "features_fallback": self.features_fallback,
            "details": {
                name: {
                    "source_col": d.source_col,
                    "method": d.method,
                    "status": d.status,
                    "fallback_value": d.fallback_value,
                    "note": d.note,
                }
                for name, d in self.details.items()
            },
            "adaptations": self.adaptations,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Constructeur de features
# ─────────────────────────────────────────────────────────────────────────────

class DynamicFeatureBuilder:
    """
    Construit les 14 features de manière adaptative à partir
    d'un DataFrame déjà normalisé (colonnes canoniques via ColumnMapper).

    Stratégies d'adaptation par feature :
      step          : int direct | datetime → secondes depuis epoch / 3600 | fallback=0
      hour          : step % 24 | datetime.hour direct | fallback=12
      day           : step // 24 | datetime.day_of_year | fallback=0
      week          : step // 168 | datetime.isocalendar().week | fallback=0
      high_risk_hour: hour isin [0..9, 23]
      is_transfer.. : type isin ['TRANSFER','CASH_OUT']
      balance_diff..: oldbalanceOrg - newbalanceOrig | fallback=0
      dest_zero_bal.: (oldbalanceDest == 0) | fallback=0
      type_CASH_IN..: OHE sur colonne type | fallback=0
      log_amount    : log1p(amount) | fallback=0
    """

    TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

    def build(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        scaler,
    ) -> tuple[np.ndarray, FeatureBuildReport]:
        """
        Paramètres
        ----------
        df      : DataFrame avec colonnes canoniques (issu de ColumnMapper)
        profile : profil du dataset original (avant mapping)
        scaler  : StandardScaler déjà entraîné sur le jeu d'entraînement

        Retourne
        --------
        X_arr        : (N, 14) float32
        build_report : rapport détaillé
        """
        n = len(df)
        result: dict[str, pd.Series] = {}
        details: dict[str, FeatureDetail] = {}
        adaptations: list[str] = []
        warnings: list[str] = []

        # ── step ─────────────────────────────────────────────────────────
        if "step" in df.columns:
            step_col = df["step"]
            if pd.api.types.is_numeric_dtype(step_col):
                result["step"] = step_col.fillna(0).astype(float)
                details["step"] = FeatureDetail("step", "step", "valeur directe", "ok")
            elif pd.api.types.is_datetime64_any_dtype(step_col):
                # Convertir en heures depuis l'epoch
                hours = (step_col - pd.Timestamp("1970-01-01")).dt.total_seconds() / 3600
                result["step"] = hours.fillna(0)
                msg = "Colonne 'step' de type datetime convertie en heures depuis l'epoch (1970-01-01)"
                adaptations.append(msg)
                details["step"] = FeatureDetail("step", "step", "datetime → heures depuis epoch", "adapted", note=msg)
            else:
                # Tenter une conversion numérique
                try:
                    parsed = pd.to_numeric(step_col, errors="coerce")
                    result["step"] = parsed.fillna(0)
                    adaptations.append("Colonne 'step' convertie en numérique depuis string")
                    details["step"] = FeatureDetail("step", "step", "string → numérique", "adapted")
                except Exception:
                    result["step"] = pd.Series(np.arange(n, dtype=float))
                    warnings.append("Colonne 'step' illisible — remplacée par index séquentiel 0..N-1")
                    details["step"] = FeatureDetail("step", "step", "fallback séquentiel", "missing_fallback", 0.0)
        else:
            result["step"] = pd.Series(np.arange(n, dtype=float))
            warnings.append("Colonne 'step' absente — remplacée par index séquentiel 0..N-1")
            details["step"] = FeatureDetail("step", "—", "index séquentiel", "missing_fallback", 0.0)

        # ── hour, day, week ───────────────────────────────────────────────
        step_vals = result["step"]

        # Tenter une extraction depuis datetime si la colonne d'origine était datetime
        _orig_step = df.get("step")
        _step_is_dt = _orig_step is not None and pd.api.types.is_datetime64_any_dtype(_orig_step)

        if _step_is_dt:
            dt = pd.to_datetime(_orig_step, errors="coerce")
            result["hour"] = dt.dt.hour.fillna(12).astype(float)
            result["day"]  = dt.dt.day_of_year.fillna(0).astype(float)
            result["week"] = dt.dt.isocalendar().week.astype(float).fillna(0)
            for feat in ("hour", "day", "week"):
                details[feat] = FeatureDetail(feat, "step", f"datetime.{feat} direct", "adapted")
        else:
            result["hour"] = (step_vals % 24)
            result["day"]  = (step_vals // 24)
            result["week"] = (step_vals // 168)
            for feat in ("hour", "day", "week"):
                details[feat] = FeatureDetail(feat, "step", f"step {'% 24' if feat=='hour' else '// 24' if feat=='day' else '// 168'}", "ok")

        # ── high_risk_hour ─────────────────────────────────────────────────
        result["high_risk_hour"] = result["hour"].isin(HIGH_RISK_HOURS).astype(float)
        details["high_risk_hour"] = FeatureDetail(
            "high_risk_hour", "hour",
            f"hour isin {HIGH_RISK_HOURS[:3]}...", "ok"
        )

        # ── type + dummies ────────────────────────────────────────────────
        if "type" in df.columns:
            type_col = df["type"].fillna("UNKNOWN").astype(str).str.upper()
            result["is_transfer_or_cashout"] = type_col.isin(["TRANSFER", "CASH_OUT"]).astype(float)
            details["is_transfer_or_cashout"] = FeatureDetail(
                "is_transfer_or_cashout", "type",
                "type isin ['TRANSFER','CASH_OUT']", "ok"
            )
            for t in self.TRANSACTION_TYPES:
                feat = f"type_{t}"
                result[feat] = (type_col == t).astype(float)
                details[feat] = FeatureDetail(feat, "type", f"type == '{t}' (OHE)", "ok")
            # Valeurs inconnues
            known = set(self.TRANSACTION_TYPES)
            unknown = set(type_col.unique()) - known - {"UNKNOWN"}
            if unknown:
                warnings.append(
                    f"Valeurs de type inconnues ignorées (non comprises dans les types reconnus) : {unknown}"
                )
        else:
            # Fallback : toutes les dummies à 0
            result["is_transfer_or_cashout"] = pd.Series(np.zeros(n))
            for t in self.TRANSACTION_TYPES:
                result[f"type_{t}"] = pd.Series(np.zeros(n))
            warnings.append("Colonne 'type' absente — OHE et is_transfer_or_cashout mis à 0")
            details["is_transfer_or_cashout"] = FeatureDetail(
                "is_transfer_or_cashout", "—", "fallback=0", "missing_fallback", 0.0
            )
            for t in self.TRANSACTION_TYPES:
                details[f"type_{t}"] = FeatureDetail(f"type_{t}", "—", "fallback=0", "missing_fallback", 0.0)

        # ── balance_diff_orig ─────────────────────────────────────────────
        if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
            result["balance_diff_orig"] = (
                df["oldbalanceOrg"].fillna(0) - df["newbalanceOrig"].fillna(0)
            ).astype(float)
            details["balance_diff_orig"] = FeatureDetail(
                "balance_diff_orig", "oldbalanceOrg - newbalanceOrig",
                "soustraction directe", "ok"
            )
        elif "oldbalanceOrg" in df.columns:
            result["balance_diff_orig"] = df["oldbalanceOrg"].fillna(0).astype(float)
            adaptations.append("newbalanceOrig absent — balance_diff_orig = oldbalanceOrg")
            details["balance_diff_orig"] = FeatureDetail(
                "balance_diff_orig", "oldbalanceOrg",
                "newbalanceOrig absent — valeur approchée", "adapted", note="newbalanceOrig absent"
            )
        else:
            result["balance_diff_orig"] = pd.Series(np.zeros(n))
            warnings.append("Colonnes oldbalanceOrg / newbalanceOrig absentes — balance_diff_orig=0")
            details["balance_diff_orig"] = FeatureDetail(
                "balance_diff_orig", "—", "fallback=0", "missing_fallback", 0.0
            )

        # ── dest_zero_balance ─────────────────────────────────────────────
        if "oldbalanceDest" in df.columns:
            result["dest_zero_balance"] = (df["oldbalanceDest"].fillna(0) == 0).astype(float)
            details["dest_zero_balance"] = FeatureDetail(
                "dest_zero_balance", "oldbalanceDest", "(oldbalanceDest == 0)", "ok"
            )
        else:
            result["dest_zero_balance"] = pd.Series(np.zeros(n))
            warnings.append("Colonne 'oldbalanceDest' absente — dest_zero_balance=0")
            details["dest_zero_balance"] = FeatureDetail(
                "dest_zero_balance", "—", "fallback=0", "missing_fallback", 0.0
            )

        # ── log_amount ────────────────────────────────────────────────────
        if "amount" in df.columns:
            amt = df["amount"].fillna(0).astype(float)
            # Gérer les négatifs (rare mais possible après mapping)
            if (amt < 0).any():
                amt = amt.abs()
                adaptations.append("Valeurs négatives détectées dans 'amount' — abs() appliqué")
            result["log_amount"] = np.log1p(amt)
            method = "log1p(amount)"
            if profile is not None:
                col_profile = profile.columns.get("amount") or profile.columns.get(
                    next((k for k in profile.columns if k.lower() in ("amount", "montant", "amt")), "")
                )
                if col_profile and col_profile.num_skewness and abs(col_profile.num_skewness) > EXTREME_SKEW_THRESHOLD:
                    method += f" [skewness={col_profile.num_skewness:.1f} → transformation justifiée]"
            details["log_amount"] = FeatureDetail("log_amount", "amount", method, "ok")
        else:
            result["log_amount"] = pd.Series(np.zeros(n))
            warnings.append("Colonne 'amount' absente — log_amount=0")
            details["log_amount"] = FeatureDetail("log_amount", "—", "fallback=0", "missing_fallback", 0.0)

        # ── Assemblage dans l'ordre FEATURE_COLS ─────────────────────────
        X_df = pd.DataFrame(index=df.index)
        for feat in FEATURE_COLS:
            if feat in result:
                X_df[feat] = result[feat].values
            else:
                X_df[feat] = 0.0
                warnings.append(f"Feature '{feat}' non construite — fallback=0")

        # ── Scaling (uniquement SCALE_COLS) ───────────────────────────────
        X_df[SCALE_COLS] = scaler.transform(X_df[SCALE_COLS])

        X_arr = X_df.values.astype(np.float32)

        # ── Rapport ────────────────────────────────────────────────────────
        all_feat_names = list(details.keys())
        features_ok       = [f for f, d in details.items() if d.status == "ok"]
        features_adapted  = [f for f, d in details.items() if d.status == "adapted"]
        features_fallback = [f for f, d in details.items() if d.status == "missing_fallback"]

        report = FeatureBuildReport(
            n_features=len(FEATURE_COLS),
            features_ok=features_ok,
            features_adapted=features_adapted,
            features_fallback=features_fallback,
            details=details,
            adaptations=adaptations,
            warnings=warnings,
        )
        return X_arr, report


# Singleton
_builder = DynamicFeatureBuilder()

EXTREME_SKEW_THRESHOLD = 10.0  # exposé pour import dans report


def build_features_dynamic(
    df: pd.DataFrame,
    profile: DatasetProfile,
    scaler,
) -> tuple[np.ndarray, FeatureBuildReport]:
    """Point d'entrée public."""
    return _builder.build(df, profile, scaler)
