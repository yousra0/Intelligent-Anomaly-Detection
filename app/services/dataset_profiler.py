"""
app/services/dataset_profiler.py

Moteur de profilage de dataset.

Pour chaque colonne il détecte :
  - Type sémantique  : numeric | categorical | datetime | identifier | boolean | constant
  - Type de cardinalité : constant | binary | low | medium | high | identifier
  - Valeurs manquantes (n + %)
  - Quasi-constance  (valeur dominante > seuil)
  - Stats numériques : min, max, mean, std, skewness, kurtosis, % zéros, % négatifs
  - Stats catégorielles : top-5 values + fréquences
  - Détection de dates dans les colonnes string
  - Flags qualité   : high_missing | quasi_constant | extreme_skewness | zero_heavy |
                      negative_values | identifier | possible_date | high_cardinality |
                      all_unique

Le score qualité global (0-100) agrège les pénalités par colonne et guide
les recommandations générées automatiquement.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# Seuils configurables
# ─────────────────────────────────────────────────────────────────────────────

QUASI_CONSTANT_THRESHOLD  = 0.95   # fraction dominante → quasi-constante
HIGH_MISSING_THRESHOLD    = 0.30   # 30 % manquants → alerte
IDENTIFIER_THRESHOLD      = 0.90   # >90 % valeurs uniques → identifiant probable
HIGH_CARDINALITY_THRESHOLD = 0.50  # >50 % valeurs uniques
EXTREME_SKEW_THRESHOLD    = 10.0   # |skewness| > 10 → très asymétrique
ZERO_HEAVY_THRESHOLD      = 0.50   # >50 % zéros
DATE_SAMPLE_SIZE          = 200    # taille de l'échantillon pour détecter les dates

# Patterns de dates reconnus dans les colonnes string
DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),           # 2024-06-12
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),            # 12/06/2024
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),            # 12-06-2024
    re.compile(r"^\d{4}/\d{2}/\d{2}$"),            # 2024/06/12
    re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),  # ISO datetime
    re.compile(r"^\d{10}$"),                        # Unix timestamp (sec)
    re.compile(r"^\d{13}$"),                        # Unix timestamp (ms)
]


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    dtype_raw: str
    semantic_type: str          # numeric | categorical | datetime | identifier | boolean | constant
    cardinality_type: str       # constant | binary | low | medium | high | identifier
    n_missing: int
    pct_missing: float
    n_unique: int
    dominant_value: Any
    dominant_pct: float
    is_quasi_constant: bool
    # Numeric stats
    num_min: Optional[float] = None
    num_max: Optional[float] = None
    num_mean: Optional[float] = None
    num_median: Optional[float] = None
    num_std: Optional[float] = None
    num_skewness: Optional[float] = None
    num_kurtosis: Optional[float] = None
    num_zeros_pct: Optional[float] = None
    num_negatives_pct: Optional[float] = None
    num_p25: Optional[float] = None
    num_p75: Optional[float] = None
    # Categorical stats
    top_values: Optional[dict[str, int]] = None  # value → count (top 5)
    # Datetime stats
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    # Quality
    quality_flags: list[str] = field(default_factory=list)
    quality_score: float = 100.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype_raw,
            "semantic_type": self.semantic_type,
            "cardinality_type": self.cardinality_type,
            "missing": {
                "n": self.n_missing,
                "pct": round(self.pct_missing, 4),
            },
            "uniqueness": {
                "n_unique": self.n_unique,
                "dominant_value": str(self.dominant_value) if self.dominant_value is not None else None,
                "dominant_pct": round(self.dominant_pct, 4),
                "is_quasi_constant": self.is_quasi_constant,
            },
            "numeric_stats": {
                "min": self.num_min,
                "max": self.num_max,
                "mean": round(self.num_mean, 4) if self.num_mean is not None else None,
                "median": round(self.num_median, 4) if self.num_median is not None else None,
                "std": round(self.num_std, 4) if self.num_std is not None else None,
                "skewness": round(self.num_skewness, 4) if self.num_skewness is not None else None,
                "kurtosis": round(self.num_kurtosis, 4) if self.num_kurtosis is not None else None,
                "p25": self.num_p25,
                "p75": self.num_p75,
                "zeros_pct": round(self.num_zeros_pct, 4) if self.num_zeros_pct is not None else None,
                "negatives_pct": round(self.num_negatives_pct, 4) if self.num_negatives_pct is not None else None,
            } if self.semantic_type == "numeric" else None,
            "categorical_stats": {
                "top_values": self.top_values,
            } if self.semantic_type in ("categorical", "boolean") else None,
            "datetime_stats": {
                "min": self.date_min,
                "max": self.date_max,
            } if self.semantic_type == "datetime" else None,
            "quality_flags": self.quality_flags,
            "quality_score": round(self.quality_score, 1),
        }


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    n_missing_total: int
    pct_missing_total: float
    columns: dict[str, ColumnProfile]
    # Groupements par type
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    identifier_cols: list[str]
    boolean_cols: list[str]
    constant_cols: list[str]
    quasi_constant_cols: list[str]
    high_missing_cols: list[str]
    # Score global
    global_quality_score: float
    recommendations: list[str]
    profiling_time_ms: float

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing_total": {
                "n": self.n_missing_total,
                "pct": round(self.pct_missing_total, 4),
            },
            "column_categories": {
                "numeric": self.numeric_cols,
                "categorical": self.categorical_cols,
                "datetime": self.datetime_cols,
                "identifier": self.identifier_cols,
                "boolean": self.boolean_cols,
                "constant": self.constant_cols,
                "quasi_constant": self.quasi_constant_cols,
                "high_missing": self.high_missing_cols,
            },
            "global_quality_score": round(self.global_quality_score, 1),
            "recommendations": self.recommendations,
            "profiling_time_ms": round(self.profiling_time_ms, 1),
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Profiler principal
# ─────────────────────────────────────────────────────────────────────────────

class DatasetProfiler:
    """
    Analyse statistique complète d'un DataFrame pandas.

    Usage :
        profiler = DatasetProfiler()
        profile  = profiler.profile(df)
        report   = profile.to_dict()
    """

    def __init__(
        self,
        quasi_constant_threshold: float = QUASI_CONSTANT_THRESHOLD,
        high_missing_threshold: float = HIGH_MISSING_THRESHOLD,
        identifier_threshold: float = IDENTIFIER_THRESHOLD,
        high_cardinality_threshold: float = HIGH_CARDINALITY_THRESHOLD,
    ):
        self.quasi_constant_threshold = quasi_constant_threshold
        self.high_missing_threshold = high_missing_threshold
        self.identifier_threshold = identifier_threshold
        self.high_cardinality_threshold = high_cardinality_threshold

    # ── Interface publique ────────────────────────────────────────────────

    def profile(self, df: pd.DataFrame) -> DatasetProfile:
        t0 = time.perf_counter()
        n_rows, n_cols = df.shape

        col_profiles: dict[str, ColumnProfile] = {}
        for col in df.columns:
            col_profiles[col] = self._profile_column(df[col], n_rows)

        # Groupements
        numeric_cols      = [c for c, p in col_profiles.items() if p.semantic_type == "numeric"]
        categorical_cols  = [c for c, p in col_profiles.items() if p.semantic_type == "categorical"]
        datetime_cols     = [c for c, p in col_profiles.items() if p.semantic_type == "datetime"]
        identifier_cols   = [c for c, p in col_profiles.items() if p.semantic_type == "identifier"]
        boolean_cols      = [c for c, p in col_profiles.items() if p.semantic_type == "boolean"]
        constant_cols     = [c for c, p in col_profiles.items() if p.semantic_type == "constant"]
        quasi_constant_cols = [c for c, p in col_profiles.items() if p.is_quasi_constant and p.semantic_type != "constant"]
        high_missing_cols = [c for c, p in col_profiles.items() if p.pct_missing >= self.high_missing_threshold]

        # Manquants globaux
        n_missing_total = sum(p.n_missing for p in col_profiles.values())
        pct_missing_total = n_missing_total / max(n_rows * n_cols, 1)

        # Score qualité global
        scores = [p.quality_score for p in col_profiles.values()]
        global_quality = float(np.mean(scores)) if scores else 100.0

        recommendations = self._generate_recommendations(col_profiles, df)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DatasetProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            n_missing_total=int(n_missing_total),
            pct_missing_total=float(pct_missing_total),
            columns=col_profiles,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            identifier_cols=identifier_cols,
            boolean_cols=boolean_cols,
            constant_cols=constant_cols,
            quasi_constant_cols=quasi_constant_cols,
            high_missing_cols=high_missing_cols,
            global_quality_score=global_quality,
            recommendations=recommendations,
            profiling_time_ms=elapsed_ms,
        )

    # ── Analyse d'une colonne ─────────────────────────────────────────────

    def _profile_column(self, series: pd.Series, n_rows: int) -> ColumnProfile:
        name = str(series.name)
        dtype_raw = str(series.dtype)

        n_missing = int(series.isna().sum())
        pct_missing = n_missing / max(n_rows, 1)
        series_clean = series.dropna()
        n_clean = len(series_clean)

        n_unique = int(series_clean.nunique())
        cardinality_ratio = n_unique / max(n_clean, 1)

        # Valeur dominante
        if n_clean > 0:
            vc = series_clean.value_counts(normalize=True)
            dominant_value = vc.index[0]
            dominant_pct = float(vc.iloc[0])
        else:
            dominant_value = None
            dominant_pct = 0.0

        is_quasi_constant = dominant_pct >= self.quasi_constant_threshold

        # Cardinalité
        cardinality_type = self._classify_cardinality(n_unique, cardinality_ratio)

        # Type sémantique
        semantic_type = self._detect_semantic_type(series_clean, dtype_raw, n_unique, cardinality_ratio)

        # Stats spécifiques au type
        num_stats: dict = {}
        cat_stats: dict = {}
        date_stats: dict = {}

        if semantic_type == "numeric":
            num_stats = self._numeric_stats(series_clean)
        elif semantic_type in ("categorical", "boolean", "identifier"):
            cat_stats = self._categorical_stats(series_clean)
        elif semantic_type == "datetime":
            date_stats = self._datetime_stats(series_clean)

        cp = ColumnProfile(
            name=name,
            dtype_raw=dtype_raw,
            semantic_type=semantic_type,
            cardinality_type=cardinality_type,
            n_missing=n_missing,
            pct_missing=float(pct_missing),
            n_unique=n_unique,
            dominant_value=dominant_value,
            dominant_pct=dominant_pct,
            is_quasi_constant=is_quasi_constant,
            **num_stats,
            top_values=cat_stats.get("top_values"),
            date_min=date_stats.get("date_min"),
            date_max=date_stats.get("date_max"),
        )

        flags, score = self._quality_flags_and_score(cp)
        cp.quality_flags = flags
        cp.quality_score = score
        return cp

    # ── Détection du type sémantique ─────────────────────────────────────

    def _detect_semantic_type(
        self,
        series: pd.Series,
        dtype_raw: str,
        n_unique: int,
        cardinality_ratio: float,
    ) -> str:
        if len(series) == 0:
            return "constant"

        # Constant
        if n_unique <= 1:
            return "constant"

        # Datetime natif pandas
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Booléen natif
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        # Numérique natif — reste toujours "numeric" (pas d'identifier pour les dtypes numériques)
        if pd.api.types.is_numeric_dtype(series):
            # Binaire numérique (0/1)
            if n_unique == 2:
                vals = set(series.unique())
                if vals <= {0, 1, True, False}:
                    return "boolean"
            return "numeric"

        # Colonnes object/string
        sample = series.dropna().head(DATE_SAMPLE_SIZE).astype(str)

        # Tenter la conversion datetime
        if self._looks_like_datetime(sample):
            return "datetime"

        # Identifier : haute cardinalité + ressemble à un ID (pattern alphanum)
        if cardinality_ratio >= self.identifier_threshold:
            return "identifier"

        # Booléen textuel
        if n_unique <= 2:
            low = set(str(v).lower() for v in series.unique())
            if low <= {"true", "false", "yes", "no", "oui", "non", "1", "0", "y", "n"}:
                return "boolean"

        return "categorical"

    def _looks_like_datetime(self, sample: pd.Series) -> bool:
        if len(sample) == 0:
            return False
        matches = sum(1 for v in sample if any(p.match(str(v)) for p in DATE_PATTERNS))
        return matches / len(sample) >= 0.80

    def _classify_cardinality(self, n_unique: int, ratio: float) -> str:
        if n_unique <= 1:
            return "constant"
        if n_unique == 2:
            return "binary"
        if ratio >= self.identifier_threshold:
            return "identifier"
        if ratio >= self.high_cardinality_threshold:
            return "high"
        if n_unique <= 10:
            return "low"
        return "medium"

    # ── Stats numériques ──────────────────────────────────────────────────

    def _numeric_stats(self, series: pd.Series) -> dict:
        arr = series.astype(float).values
        if len(arr) == 0:
            return {}
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        try:
            skew = float(scipy_stats.skew(arr))
            kurt = float(scipy_stats.kurtosis(arr))
        except Exception:
            skew = kurt = 0.0
        return {
            "num_min": float(arr.min()),
            "num_max": float(arr.max()),
            "num_mean": float(arr.mean()),
            "num_median": float(np.median(arr)),
            "num_std": float(arr.std()),
            "num_skewness": skew,
            "num_kurtosis": kurt,
            "num_p25": q25,
            "num_p75": q75,
            "num_zeros_pct": float((arr == 0).mean()),
            "num_negatives_pct": float((arr < 0).mean()),
        }

    # ── Stats catégorielles ───────────────────────────────────────────────

    def _categorical_stats(self, series: pd.Series) -> dict:
        vc = series.value_counts().head(5)
        return {"top_values": {str(k): int(v) for k, v in vc.items()}}

    # ── Stats datetime ────────────────────────────────────────────────────

    def _datetime_stats(self, series: pd.Series) -> dict:
        try:
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors="coerce").dropna()
            if len(series) == 0:
                return {}
            return {
                "date_min": str(series.min()),
                "date_max": str(series.max()),
            }
        except Exception:
            return {}

    # ── Flags qualité ─────────────────────────────────────────────────────

    def _quality_flags_and_score(self, cp: ColumnProfile) -> tuple[list[str], float]:
        flags: list[str] = []
        score = 100.0

        if cp.pct_missing >= self.high_missing_threshold:
            flags.append("high_missing")
            score -= 20.0 + min(cp.pct_missing * 30, 30)

        if cp.is_quasi_constant and cp.semantic_type != "constant":
            flags.append("quasi_constant")
            score -= 15.0

        if cp.semantic_type == "constant":
            flags.append("constant_column")
            score -= 25.0

        if cp.semantic_type == "identifier":
            flags.append("identifier")
            score -= 5.0  # léger — c'est souvent intentionnel

        if cp.cardinality_type == "high" and cp.semantic_type not in ("numeric", "identifier"):
            flags.append("high_cardinality")
            score -= 10.0

        if cp.semantic_type == "numeric":
            if cp.num_skewness is not None and abs(cp.num_skewness) > EXTREME_SKEW_THRESHOLD:
                flags.append("extreme_skewness")
                score -= 5.0
            if cp.num_zeros_pct is not None and cp.num_zeros_pct > ZERO_HEAVY_THRESHOLD:
                flags.append("zero_heavy")
                score -= 5.0
            if cp.num_negatives_pct is not None and cp.num_negatives_pct > 0:
                flags.append("has_negative_values")
                # Pas de pénalité systématique — dépend du contexte

        if cp.n_unique == cp.n_missing == 0 and cp.semantic_type != "constant":
            flags.append("all_null")
            score -= 40.0

        return flags, max(0.0, score)

    # ── Recommandations ───────────────────────────────────────────────────

    def _generate_recommendations(
        self,
        profiles: dict[str, ColumnProfile],
        df: pd.DataFrame,
    ) -> list[str]:
        recs: list[str] = []

        # Valeurs manquantes
        high_missing = [(c, p) for c, p in profiles.items() if "high_missing" in p.quality_flags]
        if high_missing:
            for c, p in high_missing:
                recs.append(
                    f"Colonne '{c}' : {p.pct_missing*100:.1f}% de valeurs manquantes — "
                    f"vérifier la source ou envisager une imputation."
                )

        # Colonnes constantes / quasi-constantes
        for c, p in profiles.items():
            if "constant_column" in p.quality_flags:
                recs.append(f"Colonne '{c}' est constante (valeur unique = {p.dominant_value}) — peut être supprimée.")
            elif "quasi_constant" in p.quality_flags:
                recs.append(
                    f"Colonne '{c}' quasi-constante : '{p.dominant_value}' représente "
                    f"{p.dominant_pct*100:.1f}% des valeurs — apport informatif limité."
                )

        # Asymétrie forte sur colonnes numériques
        for c, p in profiles.items():
            if "extreme_skewness" in p.quality_flags:
                recs.append(
                    f"Colonne '{c}' très asymétrique (skewness={p.num_skewness:.1f}) — "
                    f"une transformation log ou Box-Cox est recommandée."
                )

        # Colonnes identifiants
        id_cols = [c for c, p in profiles.items() if "identifier" in p.quality_flags]
        if id_cols:
            recs.append(
                f"Colonnes probablement identifiants (très haute cardinalité) : {id_cols} — "
                f"à exclure des features de prédiction."
            )

        # Haute cardinalité
        for c, p in profiles.items():
            if "high_cardinality" in p.quality_flags and p.semantic_type == "categorical":
                recs.append(
                    f"Colonne catégorielle '{c}' à haute cardinalité ({p.n_unique} valeurs uniques) — "
                    f"envisager un encodage cible ou une réduction de cardinalité."
                )

        # Valeurs négatives sur colonnes susceptibles d'être des montants
        for c, p in profiles.items():
            if "has_negative_values" in p.quality_flags:
                if any(kw in c.lower() for kw in ["amount", "montant", "balance", "solde", "value"]):
                    recs.append(
                        f"Colonne '{c}' contient des valeurs négatives "
                        f"({p.num_negatives_pct*100:.1f}%) — à valider pour un montant financier."
                    )

        if not recs:
            recs.append("Aucun problème majeur détecté — dataset de bonne qualité.")

        return recs


# ─────────────────────────────────────────────────────────────────────────────
# Singleton public
# ─────────────────────────────────────────────────────────────────────────────

_profiler = DatasetProfiler()


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Point d'entrée public."""
    return _profiler.profile(df)
