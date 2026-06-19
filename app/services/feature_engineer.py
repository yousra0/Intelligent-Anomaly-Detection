"""
app/services/feature_engineer.py

Moteur de feature engineering générique et adaptatif.

Principe :
  Inspecte le DataFrame (avec colonnes canoniques après ColumnMapper) et génère
  automatiquement trois familles de features selon ce qui est disponible.

  Temporelles  (si colonne temporelle détectée) :
    eng_is_weekend        : transaction le week-end (sam/dim)
    eng_is_business_hour  : heure ouvrable (lun-ven, 9h-18h)
    eng_tx_day_of_week    : jour de la semaine 0=lun … 6=dim

  Soldes (si paires old/new balance détectées) :
    eng_amount_ratio_src  : amount / max(oldbalanceOrg, 1)       → part du solde envoyée
    eng_drain_pct_src     : (oldOrg - newOrg) / max(oldOrg, 1)  → % du solde source drainé
    eng_balance_gap       : |amount - (oldOrg - newOrg)| / max(amount, ε)  → cohérence comptable
    eng_dest_gain_ratio   : (newDest - oldDest) / max(amount, ε)            → le dest a-t-il reçu?
    eng_drain_pct_dest    : (newDest - oldDest) / max(amount, ε) - 1        → variation dest

  Comportementales (si identifiants de comptes détectés) :
    eng_orig_tx_count     : nb transactions de ce compte source dans le batch
    eng_orig_unique_dests : nb de destinataires uniques de ce compte source
    eng_orig_avg_amount   : montant moyen des transactions de ce compte source
    eng_orig_total_amount : montant total des transactions de ce compte source
    eng_dest_tx_count     : nb fois ce compte destination est ciblé dans le batch
    eng_dest_avg_received : montant moyen reçu par ce compte destination
    eng_orig_is_high_freq : 1 si le compte source est dans le top 5% de fréquence

Intégration :
    df_enriched, report = engineer_features(df_mapped, profile, mapping)
    report.to_dict()  → JSON-sérialisable pour la réponse API

Notes :
  - Toutes les features générées portent le préfixe `eng_` (pas de collision possible)
  - Les features comportementales sont des statistiques INTRA-BATCH (pas historiques)
  - Le DataFrame enrichi conserve TOUTES les colonnes d'origine + les nouvelles
  - En cas d'erreur sur une feature, elle est skippée et documentée dans `report.skipped`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.services.dataset_profiler import DatasetProfile


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS_HOUR_START = 9
BUSINESS_HOUR_END = 18   # exclusive → [9, 18[
HIGH_FREQ_PERCENTILE = 95

# Colonnes canoniques attendues dans le df mappé
_TEMPORAL_COLS = {"step"}
_BALANCE_SRC_COLS = {"oldbalanceOrg", "newbalanceOrig"}
_BALANCE_DST_COLS = {"oldbalanceDest", "newbalanceDest"}
_AMOUNT_COL = "amount"
_ACCT_SRC_COL = "nameOrig"
_ACCT_DST_COL = "nameDest"

EPS = 1e-9  # division guard


# ─────────────────────────────────────────────────────────────────────────────
# Rapport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureSpec:
    """Description d'une feature engineerée."""
    name: str
    category: str        # "temporal" | "balance" | "behavioral"
    source_cols: list[str]
    formula: str         # description lisible humainement
    dtype: str           # "float32" | "int8"


@dataclass
class EngineeringReport:
    temporal_features: list[str]
    balance_features: list[str]
    behavioral_features: list[str]
    n_generated: int
    specs: dict[str, FeatureSpec]
    warnings: list[str]
    skipped: list[str]           # features non générées (colonnes manquantes ou erreur)

    @property
    def all_new_features(self) -> list[str]:
        return self.temporal_features + self.balance_features + self.behavioral_features

    def to_dict(self) -> dict:
        return {
            "n_generated": self.n_generated,
            "temporal_features": self.temporal_features,
            "balance_features": self.balance_features,
            "behavioral_features": self.behavioral_features,
            "all_new_features": self.all_new_features,
            "specs": {
                name: {
                    "category": spec.category,
                    "source_cols": spec.source_cols,
                    "formula": spec.formula,
                    "dtype": spec.dtype,
                }
                for name, spec in self.specs.items()
            },
            "warnings": self.warnings,
            "skipped": self.skipped,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Moteur principal
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Génère automatiquement des features enrichies selon les colonnes disponibles.

    Le moteur est stateless : chaque appel à `engineer()` est indépendant.
    """

    # ── Interface publique ────────────────────────────────────────────────────

    def engineer(
        self,
        df: pd.DataFrame,
        profile: Optional[DatasetProfile] = None,
        mapping: Optional[dict[str, str]] = None,
    ) -> tuple[pd.DataFrame, EngineeringReport]:
        """
        Paramètres
        ----------
        df      : DataFrame avec colonnes canoniques (après ColumnMapper)
        profile : DatasetProfile optionnel — enrichit le rapport mais non requis
        mapping : dict {canonical: original_col} du ColumnMapper — optionnel

        Retourne
        --------
        df_enriched : DataFrame original + nouvelles colonnes `eng_*`
        report      : EngineeringReport JSON-sérialisable
        """
        n = len(df)
        out = df.copy()
        specs: dict[str, FeatureSpec] = {}
        temporal: list[str] = []
        balance: list[str] = []
        behavioral: list[str] = []
        warnings: list[str] = []
        skipped: list[str] = []

        # ── Détection des sources disponibles ─────────────────────────────
        temporal_info = self._detect_temporal(df)
        has_balance_src = _BALANCE_SRC_COLS.issubset(df.columns)
        has_balance_dst = _BALANCE_DST_COLS.issubset(df.columns)
        has_amount = _AMOUNT_COL in df.columns
        has_acct_src = _ACCT_SRC_COL in df.columns
        has_acct_dst = _ACCT_DST_COL in df.columns

        # ── A. Features temporelles ────────────────────────────────────────
        if temporal_info is not None:
            t_col, t_mode, hour_s, dow_s = temporal_info
            t_res, t_specs, t_warn, t_skip = self._build_temporal(
                out, t_col, t_mode, hour_s, dow_s
            )
            for name, series in t_res.items():
                out[name] = series
                temporal.append(name)
            specs.update(t_specs)
            warnings.extend(t_warn)
            skipped.extend(t_skip)
        else:
            skipped.append("eng_is_weekend (source temporelle non détectée)")
            skipped.append("eng_is_business_hour (source temporelle non détectée)")
            skipped.append("eng_tx_day_of_week (source temporelle non détectée)")

        # ── B. Features de soldes ──────────────────────────────────────────
        if has_amount and (has_balance_src or has_balance_dst):
            b_res, b_specs, b_warn, b_skip = self._build_balance(
                out, has_balance_src, has_balance_dst
            )
            for name, series in b_res.items():
                out[name] = series
                balance.append(name)
            specs.update(b_specs)
            warnings.extend(b_warn)
            skipped.extend(b_skip)
        else:
            if not has_amount:
                skipped.append("features soldes (colonne 'amount' manquante)")
            else:
                skipped.append("features soldes (colonnes de balance manquantes)")

        # ── C. Features comportementales ───────────────────────────────────
        if has_acct_src or has_acct_dst:
            beh_res, beh_specs, beh_warn, beh_skip = self._build_behavioral(
                df, out, has_acct_src, has_acct_dst, has_amount, n
            )
            for name, series in beh_res.items():
                out[name] = series
                behavioral.append(name)
            specs.update(beh_specs)
            warnings.extend(beh_warn)
            skipped.extend(beh_skip)
        else:
            skipped.append(
                "features comportementales (colonnes nameOrig/nameDest manquantes)"
            )

        n_generated = len(temporal) + len(balance) + len(behavioral)

        report = EngineeringReport(
            temporal_features=temporal,
            balance_features=balance,
            behavioral_features=behavioral,
            n_generated=n_generated,
            specs=specs,
            warnings=warnings,
            skipped=skipped,
        )
        return out, report

    # ── Détection de la source temporelle ────────────────────────────────────

    def _detect_temporal(
        self, df: pd.DataFrame
    ) -> tuple[str, str, pd.Series, pd.Series] | None:
        """
        Cherche la source temporelle dans le df canonique.

        Retourne (col_name, mode, hour_series, day_of_week_series) ou None.
        mode = "datetime" | "numeric_step" | "generic_numeric"
        """
        # Priorité 1 : colonne `step` (canonique)
        if "step" in df.columns:
            col = df["step"]
            if pd.api.types.is_datetime64_any_dtype(col):
                dt = col
                hour_s = dt.dt.hour.astype(float)
                dow_s = dt.dt.dayofweek.astype(float)  # 0=lun
                return "step", "datetime", hour_s, dow_s
            else:
                step_num = pd.to_numeric(col, errors="coerce").fillna(0)
                hour_s = (step_num % 24)
                dow_s = ((step_num // 24) % 7)
                return "step", "numeric_step", hour_s, dow_s

        # Priorité 2 : colonne datetime native autre que step
        for col_name in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col_name]):
                dt = df[col_name]
                hour_s = dt.dt.hour.astype(float)
                dow_s = dt.dt.dayofweek.astype(float)
                return col_name, "datetime", hour_s, dow_s

        # Priorité 3 : colonne numérique avec nom temporel
        _time_hints = {"time", "ts", "timestamp", "hour", "period", "tempo"}
        for col_name in df.columns:
            if any(h in col_name.lower() for h in _time_hints):
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    step_num = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
                    hour_s = (step_num % 24)
                    dow_s = ((step_num // 24) % 7)
                    return col_name, "generic_numeric", hour_s, dow_s

        return None

    # ── Construction features temporelles ────────────────────────────────────

    def _build_temporal(
        self,
        df: pd.DataFrame,
        t_col: str,
        mode: str,
        hour_s: pd.Series,
        dow_s: pd.Series,
    ) -> tuple[dict, dict, list, list]:
        res: dict[str, pd.Series] = {}
        specs: dict[str, FeatureSpec] = {}
        warnings: list[str] = []
        skipped: list[str] = []

        mode_label = {
            "datetime"       : f"datetime.{t_col}",
            "numeric_step"   : f"step % 24 / step // 24",
            "generic_numeric": f"{t_col} % 24 / {t_col} // 24",
        }.get(mode, mode)

        # eng_tx_day_of_week  ─── 0=lundi … 6=dimanche
        res["eng_tx_day_of_week"] = dow_s.astype(np.float32)
        specs["eng_tx_day_of_week"] = FeatureSpec(
            name="eng_tx_day_of_week",
            category="temporal",
            source_cols=[t_col],
            formula=f"jour de la semaine 0=lun … 6=dim ({mode_label})",
            dtype="float32",
        )

        # eng_is_weekend  ─── sam=5 ou dim=6
        res["eng_is_weekend"] = (dow_s >= 5).astype(np.float32)
        specs["eng_is_weekend"] = FeatureSpec(
            name="eng_is_weekend",
            category="temporal",
            source_cols=[t_col],
            formula=f"day_of_week >= 5 ({mode_label})",
            dtype="float32",
        )

        # eng_is_business_hour  ─── lun-ven, 9h ≤ hour < 18h
        is_bh = (
            (hour_s >= BUSINESS_HOUR_START) &
            (hour_s < BUSINESS_HOUR_END) &
            (dow_s < 5)
        ).astype(np.float32)
        res["eng_is_business_hour"] = is_bh
        specs["eng_is_business_hour"] = FeatureSpec(
            name="eng_is_business_hour",
            category="temporal",
            source_cols=[t_col],
            formula=f"(hour in [{BUSINESS_HOUR_START}, {BUSINESS_HOUR_END}[) AND day_of_week < 5",
            dtype="float32",
        )

        if mode in ("numeric_step", "generic_numeric"):
            warnings.append(
                f"Features temporelles dérivées d'un step numérique ('{t_col}') — "
                f"les jours/semaines sont cycliques (0..6), pas calendaires."
            )

        return res, specs, warnings, skipped

    # ── Construction features de soldes ─────────────────────────────────────

    def _build_balance(
        self,
        df: pd.DataFrame,
        has_src: bool,
        has_dst: bool,
    ) -> tuple[dict, dict, list, list]:
        res: dict[str, pd.Series] = {}
        specs: dict[str, FeatureSpec] = {}
        warnings: list[str] = []
        skipped: list[str] = []

        amount = df[_AMOUNT_COL].fillna(0).astype(float)

        # ── Source balance features ───────────────────────────────────────
        if has_src:
            old_src = df["oldbalanceOrg"].fillna(0).astype(float)
            new_src = df["newbalanceOrig"].fillna(0).astype(float)
            diff_src = old_src - new_src

            # Ratio montant / solde source
            res["eng_amount_ratio_src"] = (amount / (old_src + EPS)).astype(np.float32)
            specs["eng_amount_ratio_src"] = FeatureSpec(
                name="eng_amount_ratio_src",
                category="balance",
                source_cols=["amount", "oldbalanceOrg"],
                formula="amount / max(oldbalanceOrg, ε)",
                dtype="float32",
            )

            # % du solde source drainé
            res["eng_drain_pct_src"] = (
                (diff_src / (old_src + EPS)) * 100
            ).clip(-200, 200).astype(np.float32)
            specs["eng_drain_pct_src"] = FeatureSpec(
                name="eng_drain_pct_src",
                category="balance",
                source_cols=["oldbalanceOrg", "newbalanceOrig"],
                formula="(oldbalanceOrg - newbalanceOrig) / oldbalanceOrg × 100",
                dtype="float32",
            )

            # Incohérence comptable : |amount - (old - new)| / amount
            # → proche de 0 pour transactions légitimes, élevé pour anomalies
            res["eng_balance_gap"] = (
                (amount - diff_src).abs() / (amount + EPS)
            ).clip(0, 100).astype(np.float32)
            specs["eng_balance_gap"] = FeatureSpec(
                name="eng_balance_gap",
                category="balance",
                source_cols=["amount", "oldbalanceOrg", "newbalanceOrig"],
                formula="|amount − (oldbalanceOrg − newbalanceOrig)| / amount",
                dtype="float32",
            )

        # ── Destination balance features ──────────────────────────────────
        if has_dst:
            old_dst = df["oldbalanceDest"].fillna(0).astype(float)
            new_dst = df["newbalanceDest"].fillna(0).astype(float)
            delta_dst = new_dst - old_dst

            # Le destinataire a-t-il réellement reçu le montant ?
            res["eng_dest_gain_ratio"] = (
                delta_dst / (amount + EPS)
            ).clip(-10, 10).astype(np.float32)
            specs["eng_dest_gain_ratio"] = FeatureSpec(
                name="eng_dest_gain_ratio",
                category="balance",
                source_cols=["oldbalanceDest", "newbalanceDest", "amount"],
                formula="(newbalanceDest − oldbalanceDest) / amount  [≈1 pour transfer légit]",
                dtype="float32",
            )

            # % de variation du solde destinataire (signé)
            res["eng_drain_pct_dest"] = (
                (delta_dst / (old_dst + EPS)) * 100
            ).clip(-500, 500).astype(np.float32)
            specs["eng_drain_pct_dest"] = FeatureSpec(
                name="eng_drain_pct_dest",
                category="balance",
                source_cols=["oldbalanceDest", "newbalanceDest"],
                formula="(newbalanceDest − oldbalanceDest) / oldbalanceDest × 100",
                dtype="float32",
            )

        if not has_src and not has_dst:
            warnings.append("Aucune paire de soldes détectée — features balance ignorées.")

        return res, specs, warnings, skipped

    # ── Construction features comportementales ───────────────────────────────

    def _build_behavioral(
        self,
        df_orig: pd.DataFrame,   # df original non enrichi (pour les groupbys)
        df_out: pd.DataFrame,    # df en cours d'enrichissement
        has_src: bool,
        has_dst: bool,
        has_amount: bool,
        n: int,
    ) -> tuple[dict, dict, list, list]:
        res: dict[str, pd.Series] = {}
        specs: dict[str, FeatureSpec] = {}
        warnings: list[str] = []
        skipped: list[str] = []

        # ── Compte source ─────────────────────────────────────────────────
        if has_src:
            src = df_orig[_ACCT_SRC_COL]
            index = df_orig.index

            # Fréquence dans le batch
            freq_src = src.map(src.value_counts())
            res["eng_orig_tx_count"] = freq_src.fillna(0).astype(np.float32)
            specs["eng_orig_tx_count"] = FeatureSpec(
                name="eng_orig_tx_count",
                category="behavioral",
                source_cols=[_ACCT_SRC_COL],
                formula="COUNT(nameOrig) dans le batch courant",
                dtype="float32",
            )

            # Nombre de destinataires uniques par source
            unique_dests = df_orig.groupby(_ACCT_SRC_COL)[_ACCT_DST_COL].nunique() \
                if has_dst and _ACCT_DST_COL in df_orig.columns \
                else None
            if unique_dests is not None:
                res["eng_orig_unique_dests"] = src.map(unique_dests).fillna(0).astype(np.float32)
                specs["eng_orig_unique_dests"] = FeatureSpec(
                    name="eng_orig_unique_dests",
                    category="behavioral",
                    source_cols=[_ACCT_SRC_COL, _ACCT_DST_COL],
                    formula="COUNT(DISTINCT nameDest) pour nameOrig dans le batch",
                    dtype="float32",
                )
            else:
                skipped.append("eng_orig_unique_dests (nameDest absent)")

            if has_amount:
                avg_amount_src = df_orig.groupby(_ACCT_SRC_COL)[_AMOUNT_COL].mean()
                res["eng_orig_avg_amount"] = src.map(avg_amount_src).fillna(0).astype(np.float32)
                specs["eng_orig_avg_amount"] = FeatureSpec(
                    name="eng_orig_avg_amount",
                    category="behavioral",
                    source_cols=[_ACCT_SRC_COL, _AMOUNT_COL],
                    formula="MEAN(amount) pour nameOrig dans le batch",
                    dtype="float32",
                )

                total_amount_src = df_orig.groupby(_ACCT_SRC_COL)[_AMOUNT_COL].sum()
                res["eng_orig_total_amount"] = src.map(total_amount_src).fillna(0).astype(np.float32)
                specs["eng_orig_total_amount"] = FeatureSpec(
                    name="eng_orig_total_amount",
                    category="behavioral",
                    source_cols=[_ACCT_SRC_COL, _AMOUNT_COL],
                    formula="SUM(amount) pour nameOrig dans le batch",
                    dtype="float32",
                )

            # Flag haute fréquence (top 5% des comptes les plus actifs)
            tx_counts = freq_src.fillna(0)
            if n > 0:
                threshold_freq = tx_counts.quantile(HIGH_FREQ_PERCENTILE / 100)
                res["eng_orig_is_high_freq"] = (tx_counts >= threshold_freq).astype(np.float32)
                specs["eng_orig_is_high_freq"] = FeatureSpec(
                    name="eng_orig_is_high_freq",
                    category="behavioral",
                    source_cols=[_ACCT_SRC_COL],
                    formula=f"eng_orig_tx_count >= percentile_{HIGH_FREQ_PERCENTILE}(batch)",
                    dtype="float32",
                )

        # ── Compte destination ────────────────────────────────────────────
        if has_dst:
            dst = df_orig[_ACCT_DST_COL]

            freq_dst = dst.map(dst.value_counts())
            res["eng_dest_tx_count"] = freq_dst.fillna(0).astype(np.float32)
            specs["eng_dest_tx_count"] = FeatureSpec(
                name="eng_dest_tx_count",
                category="behavioral",
                source_cols=[_ACCT_DST_COL],
                formula="COUNT(nameDest) dans le batch courant",
                dtype="float32",
            )

            if has_amount:
                avg_recv = df_orig.groupby(_ACCT_DST_COL)[_AMOUNT_COL].mean()
                res["eng_dest_avg_received"] = dst.map(avg_recv).fillna(0).astype(np.float32)
                specs["eng_dest_avg_received"] = FeatureSpec(
                    name="eng_dest_avg_received",
                    category="behavioral",
                    source_cols=[_ACCT_DST_COL, _AMOUNT_COL],
                    formula="MEAN(amount) pour nameDest dans le batch",
                    dtype="float32",
                )

        if not has_src and not has_dst:
            warnings.append(
                "Aucun identifiant de compte détecté — features comportementales ignorées."
            )

        return res, specs, warnings, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + point d'entrée public
# ─────────────────────────────────────────────────────────────────────────────

_engineer = FeatureEngineer()


def engineer_features(
    df: pd.DataFrame,
    profile: Optional[DatasetProfile] = None,
    mapping: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, EngineeringReport]:
    """
    Point d'entrée public du moteur de feature engineering.

    Paramètres
    ----------
    df      : DataFrame canonique (après ColumnMapper)
    profile : DatasetProfile optionnel (enrichit les warnings/report)
    mapping : dict {canonical: original_col} optionnel

    Retourne
    --------
    (df_enriched, EngineeringReport)
    """
    return _engineer.engineer(df, profile=profile, mapping=mapping)
