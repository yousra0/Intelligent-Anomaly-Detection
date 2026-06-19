"""
app/services/schema_detector.py

Détecte si le dataset correspond au schéma transactionnel standard et sélectionne
le mode de prédiction optimal.

Logique de décision
───────────────────
 mapping_result.success = True          → mode "standard" (XGB + AE)
 mapping_result.success = False :
   "amount" mappé ET ≥ 1 col numérique   → mode "ae_isoforest"
   "amount" mappé, 0 cols numériques     → mode "ae_only"
   "amount" non mappé ET ≥ 1 col num.    → mode "isoforest"
   aucune col num. → essai encodage catégoriel → mode "isoforest"
   aucune colonne utilisable              → mode dégradé (tout FAIBLE)

Notes
─────
 • L'AE pré-entraîné fonctionne en mode générique car DynamicFeatureBuilder
   remplit les features manquantes par des fallbacks (0).
 • L'IsoForest est toujours fitté on-the-fly sur les colonnes numériques
   du dataset brut (avant mapping) — pas le modèle iso_forest.pkl entraîné
   sur le schéma standard qui ne serait pas pertinent sur un schéma inconnu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Minimum de colonnes pour lancer l'IsoForest — 1 suffit (toute info vaut mieux que rien)
_ISO_MIN_COLS = 1
# Minimum de valeurs uniques pour qu'une colonne soit informative
_ISO_MIN_UNIQUE = 2


@dataclass
class SchemaDetectionResult:
    mode: str              # "standard" | "ae_isoforest" | "ae_only" | "isoforest"
    n_mapped: int          # nombre de champs canoniques trouvés
    n_required: int        # nombre de champs requis du schéma transactionnel
    avg_confidence: float  # confiance moyenne du mapping
    use_xgb: bool
    use_ae: bool
    use_isoforest: bool
    numeric_cols_for_iso: list[str]  # colonnes à utiliser pour IsoForest (vide si non applicable)
    reason: str
    warnings: list[str]

    @property
    def model_label(self) -> str:
        """Construit l'étiquette lisible des modèles actifs, ex. 'XGBoost + Autoencoder'."""
        parts = []
        if self.use_xgb:
            parts.append("XGBoost")
        if self.use_ae:
            parts.append("Autoencoder")
        if self.use_isoforest:
            parts.append("IsolationForest")
        return " + ".join(parts) if parts else "Aucun"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n_mapped": self.n_mapped,
            "n_required": self.n_required,
            "avg_confidence": round(self.avg_confidence, 3),
            "use_xgb": self.use_xgb,
            "use_ae": self.use_ae,
            "use_isoforest": self.use_isoforest,
            "models_used": self.model_label,
            "reason": self.reason,
            "warnings": self.warnings,
        }


def detect_schema_mode(
    mapping_result,
    df_original: pd.DataFrame,
) -> SchemaDetectionResult:
    """
    Paramètres
    ----------
    mapping_result : MappingResult du ColumnMapper
    df_original    : DataFrame avant tout renommage (colonnes brutes)

    Retourne
    --------
    SchemaDetectionResult — mode sélectionné + métadonnées.
    Ne lève jamais d'exception : en dernier recours, retourne un mode dégradé
    où toutes les transactions sont classées FAIBLE.
    """
    mapped_fields = set(mapping_result.mapping.keys())
    conf_values = list(mapping_result.confidence.values())
    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
    n_mapped = len(mapped_fields)

    # Nombre de champs requis du schéma transactionnel (dérivé des unmapped + mapped required)
    # On compte les champs qui ont un required=True dans SEMANTIC_FIELDS
    from app.services.column_mapper import SEMANTIC_FIELDS
    n_required = sum(1 for f in SEMANTIC_FIELDS.values() if f.required)

    warnings: list[str] = list(mapping_result.warnings)

    # ── Mode STANDARD (schéma complet) ───────────────────────────────────────
    if mapping_result.success:
        return SchemaDetectionResult(
            mode="standard",
            n_mapped=n_mapped,
            n_required=n_required,
            avg_confidence=avg_conf,
            use_xgb=True,
            use_ae=True,
            use_isoforest=False,
            numeric_cols_for_iso=[],
            reason=(
                f"Schéma transactionnel complet ({n_mapped}/{n_required} colonnes requises, "
                f"confiance moy. {avg_conf:.0%}) → XGBoost + Autoencoder"
            ),
            warnings=warnings,
        )

    # ── Mode générique (schéma inconnu ou incomplet) ─────────────────────────
    if n_mapped < n_required:
        warnings.append(
            f"Schéma incomplet : {mapping_result.unmapped} non détectés "
            f"({n_mapped}/{n_required} colonnes requises trouvées)."
        )

    # Sélection des colonnes numériques brutes pour IsoForest
    numeric_cols = [
        c for c in df_original.select_dtypes(include=np.number).columns
        if df_original[c].nunique() >= _ISO_MIN_UNIQUE
    ]
    can_iso = len(numeric_cols) >= _ISO_MIN_COLS

    # L'AE est pertinent si "amount" est mappé ET que la colonne d'origine
    # est bien numérique (évite qu'un mapping fuzzy incorrect active l'AE
    # sur une colonne texte, ex. "transaction_status" → "amount")
    can_ae = False
    if "amount" in mapped_fields:
        orig_amount = mapping_result.mapping["amount"]
        if orig_amount in df_original.columns:
            can_ae = pd.api.types.is_numeric_dtype(df_original[orig_amount])

    # Fallback : encoder les colonnes catégorielles si aucune colonne numérique
    # informative n'est disponible (label encoding effectué dans _fit_isoforest)
    if not can_iso:
        cat_cols = [
            c for c in df_original.select_dtypes(include="object").columns
            if 2 <= df_original[c].nunique() <= 200
        ]
        if cat_cols:
            numeric_cols = cat_cols
            can_iso = True
            warnings.append(
                f"Aucune colonne numérique informative — {len(cat_cols)} colonne(s) "
                "catégorielle(s) seront encodées (label encoding) pour IsolationForest."
            )

    # Mode ultime dégradé : aucune colonne utilisable → tout classé FAIBLE
    if not can_ae and not can_iso:
        warnings.append(
            "Aucune colonne utilisable pour la détection d'anomalies. "
            "Toutes les transactions sont classées FAIBLE par défaut."
        )
        return SchemaDetectionResult(
            mode="isoforest",
            n_mapped=n_mapped,
            n_required=n_required,
            avg_confidence=avg_conf,
            use_xgb=False,
            use_ae=False,
            use_isoforest=False,
            numeric_cols_for_iso=[],
            reason="Aucune colonne numérique ni catégorielle utilisable — mode dégradé (tout FAIBLE).",
            warnings=warnings,
        )

    if can_ae and can_iso:
        mode = "ae_isoforest"
        reason = (
            f"Schéma partiel ({n_mapped}/{n_required} colonnes requises). "
            f"Autoencoder (features avec fallbacks) + "
            f"IsolationForest on-the-fly sur {len(numeric_cols)} colonnes numériques."
        )
    elif can_ae:
        mode = "ae_only"
        reason = (
            f"Schéma partiel ({n_mapped}/{n_required} colonnes). "
            f"Colonnes numériques insuffisantes ({len(numeric_cols)}) → Autoencoder seul."
        )
        warnings.append(
            f"Seulement {len(numeric_cols)} colonne(s) numérique(s) informative(s) "
            f"(minimum {_ISO_MIN_COLS} pour IsolationForest) — mode AE seul."
        )
    else:
        mode = "isoforest"
        reason = (
            f"Colonne 'amount' non détectée. "
            f"IsolationForest on-the-fly sur {len(numeric_cols)} colonnes numériques."
        )
        warnings.append(
            "Colonne 'amount' non détectée — Autoencoder désactivé. "
            "Les scores d'anomalie sont calculés sur les colonnes numériques brutes."
        )

    return SchemaDetectionResult(
        mode=mode,
        n_mapped=n_mapped,
        n_required=n_required,
        avg_confidence=avg_conf,
        use_xgb=False,
        use_ae=can_ae,
        use_isoforest=can_iso,
        numeric_cols_for_iso=numeric_cols if can_iso else [],
        reason=reason,
        warnings=warnings,
    )
