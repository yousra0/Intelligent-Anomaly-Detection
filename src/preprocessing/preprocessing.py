"""
preprocessor.py
===============
Pipeline de préparation des données PaySim pour la modélisation.

Responsabilités :
  1. Suppression des colonnes à risque de leakage / non-encodables
  2. One-Hot Encoding de ``type`` (5 modalités, drop_first=False)
  3. Transformation log1p de ``amount`` → ``log_amount``
  4. Split stratifié 70 / 15 / 15
  5. Normalisation StandardScaler  (fit sur train uniquement — anti-leakage)
  6. Gestion du déséquilibre : class_weight + SMOTE

Toutes les valeurs attendues proviennent de ``01_data_understanding.ipynb`` :
  - 200 000 lignes, 258 fraudes (0.1290%), ratio 1:774
  - skewness amount = 30.80 → log1p nécessaire
  - SCALE_COLS = ['step', 'hour', 'day', 'week', 'log_amount', 'balance_diff_orig']
  - TYPE_COLS = 5 colonnes type_*
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Constantes — inchangées depuis l'EDA
# ---------------------------------------------------------------------------

TARGET = "isFraud"

# Colonnes retirées (EDA section 17 — Data Leakage)
COLS_TO_DROP: list[str] = [
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]

# Colonnes numériques continues → StandardScaler
SCALE_COLS: list[str] = [
    "step",
    "hour",
    "day",
    "week",
    "log_amount",
    "balance_diff_orig",
]

BINARY_COLS: list[str] = [
    "high_risk_hour",
    "is_transfer_or_cashout",
    "dest_zero_balance",
]

TYPE_COLS: list[str] = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


# ---------------------------------------------------------------------------
# Dataclass résultat
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    """
    Conteneur de tous les datasets issus du pipeline de préparation.

    Attributs principaux :
        X_train_sc / y_train    : train scalé (modèles supervisés)
        X_val_sc   / y_val      : validation scalée
        X_test_sc  / y_test     : test scalé (évaluation finale — ne pas toucher)
        X_norm_sc               : train sans fraudes (AutoEncoder non-supervisé)
        X_smote    / y_smote    : train augmenté SMOTE (modèles + rééchantillonnage)
        feature_cols            : liste ordonnée des 14 features
        scaler                  : StandardScaler fitté sur train
        class_weights           : {0: w0, 1: w1} pour class_weight='balanced'
    """

    # Splits scalés
    X_train_sc:  pd.DataFrame = field(repr=False)
    y_train:     pd.Series    = field(repr=False)
    X_val_sc:    pd.DataFrame = field(repr=False)
    y_val:       pd.Series    = field(repr=False)
    X_test_sc:   pd.DataFrame = field(repr=False)
    y_test:      pd.Series    = field(repr=False)

    # AutoEncoder
    X_norm_sc:   pd.DataFrame = field(repr=False)
    y_train_normal: pd.Series = field(repr=False)

    # SMOTE
    X_smote:     pd.DataFrame = field(repr=False)
    y_smote:     pd.Series    = field(repr=False)

    # Artefacts
    scaler:         StandardScaler = field(repr=False)
    class_weights:  dict           = field(default_factory=dict)
    feature_cols:   list[str]      = field(default_factory=list)
    skew_before:    float          = 0.0
    skew_after:     float          = 0.0

    def summary(self) -> str:
        lines = [
            "PreparedData — résumé",
            f"  Features       : {len(self.feature_cols)}  →  {self.feature_cols}",
            f"  Train          : {self.X_train_sc.shape}  "
            f"fraudes={int(self.y_train.sum())}  ({self.y_train.mean()*100:.4f}%)",
            f"  Val            : {self.X_val_sc.shape}  "
            f"fraudes={int(self.y_val.sum())}",
            f"  Test           : {self.X_test_sc.shape}  "
            f"fraudes={int(self.y_test.sum())}",
            f"  Train normal   : {self.X_norm_sc.shape}  (AutoEncoder)",
            f"  Train SMOTE    : {self.X_smote.shape}  "
            f"fraudes={int(self.y_smote.sum())}",
            f"  class_weights  : {self.class_weights}",
            f"  log1p skew     : {self.skew_before:.2f} → {self.skew_after:.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Étapes individuelles (réutilisables)
# ---------------------------------------------------------------------------

def drop_leakage_columns(
    df: pd.DataFrame,
    cols_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """
    Supprime les colonnes identifiées comme leakage ou non-utilisables
    en production (EDA section 17).

    Args:
        df: DataFrame avec les features engineerées déjà ajoutées.
        cols_to_drop: Si None, utilise COLS_TO_DROP.

    Returns:
        DataFrame sans les colonnes à risque.
    """
    if cols_to_drop is None:
        cols_to_drop = COLS_TO_DROP
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


def encode_type(
    df: pd.DataFrame,
    type_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    One-Hot Encoding de la colonne ``type`` (5 modalités).

    ``drop_first=False`` : l'AutoEncoder bénéficie d'une représentation
    explicite des 5 types ; la multicolinéarité est moins critique que
    pour la régression linéaire.

    Distribution attendue (EDA Cell 28) :
        CASH_OUT : 70 282  PAYMENT : 67 821  CASH_IN : 43 677
        TRANSFER : 16 915  DEBIT   :  1 305

    Args:
        df: DataFrame contenant la colonne ``type``.
        type_cols: Liste des colonnes attendues après encodage.

    Returns:
        DataFrame avec 5 colonnes ``type_*`` et sans la colonne ``type``.
    """
    if type_cols is None:
        type_cols = TYPE_COLS
    df = pd.get_dummies(df, columns=["type"], drop_first=False)
    missing = set(type_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes type manquantes après OHE : {missing}")
    return df


def log_transform_amount(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Applique ``log1p`` sur ``amount`` → crée ``log_amount`` et supprime ``amount``.

    Justification (EDA Cell 47-48) :
        - Skewness brute = 30.80
        - Ratio max/médiane ≈ 931×
        - log1p gère les valeurs nulles et compresse la queue droite

    Returns:
        (df_transformed, skew_before, skew_after)
    """
    skew_before = float(df["amount"].skew())
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df = df.drop(columns=["amount"])
    skew_after = float(df["log_amount"].skew())
    return df, skew_before, skew_after


def stratified_split(
    df: pd.DataFrame,
    target: str = TARGET,
    train_ratio: float = 0.70,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split stratifié 70 / 15 / 15 en deux étapes.

    Étape 1 : train (70%) | temp (30%)
    Étape 2 : val  (50% temp) | test (50% temp)  →  15% / 15% du total

    ``stratify=target`` : chaque split préserve le taux de fraude ~0.1290%.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    # Étape 1
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1.0 - train_ratio),
        stratify=y,
        random_state=random_state,
    )
    # Étape 2 : val/test 50/50 du temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(
    X_train: pd.DataFrame,
    scale_cols: list[str] | None = None,
) -> StandardScaler:
    """
    Entraîne un StandardScaler **uniquement sur X_train**.

    Anti-leakage : X_val et X_test n'interviennent jamais dans le fit.

    Args:
        X_train: Split d'entraînement.
        scale_cols: Colonnes à normaliser. Si None, utilise SCALE_COLS.

    Returns:
        StandardScaler fitté.
    """
    if scale_cols is None:
        scale_cols = SCALE_COLS
    scaler = StandardScaler()
    scaler.fit(X_train[scale_cols])
    return scaler


def apply_scaler(
    X: pd.DataFrame,
    scaler: StandardScaler,
    scale_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Applique un scaler déjà fitté sur les colonnes ``scale_cols``.
    Les colonnes binaires et one-hot restent inchangées.

    Returns:
        DataFrame scalé (copie).
    """
    if scale_cols is None:
        scale_cols = SCALE_COLS
    X_s = X.copy()
    X_s[scale_cols] = scaler.transform(X[scale_cols])
    return X_s


def compute_class_weights(y_train: pd.Series) -> dict[int, float]:
    """
    Calcule les poids de classe pour ``class_weight='balanced'``.

    Formule sklearn : n_samples / (n_classes * bincount(y))
    Avec ratio 1:774 → poids fraude ≈ 387×.

    Returns:
        {0: w_non_fraud, 1: w_fraud}
    """
    weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    return {int(c): float(w) for c, w in zip([0, 1], weights)}


def apply_smote(
    X_train_sc: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.1,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Applique SMOTE sur le train scalé uniquement.

    ``sampling_strategy=0.1`` : ratio cible minoritaire/majoritaire = 1:10
    (pas 1:1 pour éviter l'overfitting sur des exemples synthétiques).

    Val et Test restent **inchangés** à 1:774 pour l'évaluation réaliste.

    Returns:
        (X_smote_df, y_smote_series) avec les mêmes noms de colonnes que X_train_sc.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_arr, y_arr = smote.fit_resample(X_train_sc, y_train)
    X_df = pd.DataFrame(X_arr, columns=X_train_sc.columns)
    y_s  = pd.Series(y_arr, name=y_train.name)
    return X_df, y_s


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_preparation_pipeline(
    df: pd.DataFrame,
    target: str = TARGET,
    cols_to_drop: list[str] | None = None,
    scale_cols: list[str] | None = None,
    type_cols: list[str] | None = None,
    train_ratio: float = 0.70,
    smote_strategy: float = 0.1,
    smote_k: int = 5,
    random_state: int = 42,
) -> PreparedData:
    """
    Exécute le pipeline complet de préparation en 8 étapes.

    Le DataFrame d'entrée doit déjà contenir les 7 features engineerées
    (produites par ``feature_builder.build_features()``).

    Pipeline :
        1. Suppression colonnes leakage / identifiants
        2. One-Hot Encoding de ``type``
        3. log1p sur ``amount``  →  ``log_amount``
        4. Split stratifié 70/15/15
        5. Extraction X_train_normal (pour AutoEncoder)
        6. StandardScaler — fit sur train uniquement
        7. Calcul class_weights
        8. SMOTE sur train scalé

    Args:
        df: DataFrame avec les features engineerées (sortie de build_features).
        target: Nom de la colonne cible.
        cols_to_drop: Colonnes à supprimer avant la modélisation.
        scale_cols: Colonnes à normaliser.
        type_cols: Colonnes one-hot attendues.
        train_ratio: Part du train (défaut 0.70).
        smote_strategy: Ratio cible après SMOTE.
        smote_k: Nombre de voisins SMOTE.
        random_state: Graine aléatoire.

    Returns:
        PreparedData contenant tous les splits et artefacts.
    """
    if cols_to_drop is None:
        cols_to_drop = COLS_TO_DROP
    if scale_cols is None:
        scale_cols = SCALE_COLS
    if type_cols is None:
        type_cols = TYPE_COLS

    # ── Étape 1 : suppression colonnes ────────────────────────────────────────
    df = drop_leakage_columns(df, cols_to_drop)

    # ── Étape 2 : One-Hot Encoding ────────────────────────────────────────────
    df = encode_type(df, type_cols)

    # ── Étape 3 : log1p(amount) ───────────────────────────────────────────────
    df, skew_before, skew_after = log_transform_amount(df)

    # ── Étape 4 : Split 70/15/15 ──────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        df, target=target, train_ratio=train_ratio, random_state=random_state
    )
    feature_cols = list(X_train.columns)

    # ── Étape 5 : X_train_normal (AutoEncoder) ────────────────────────────────
    X_train_normal = X_train[y_train == 0].copy()
    y_train_normal = y_train[y_train == 0].copy()

    # ── Étape 6 : StandardScaler — fit sur train uniquement ──────────────────
    scaler = fit_scaler(X_train, scale_cols)
    X_train_sc = apply_scaler(X_train,        scaler, scale_cols)
    X_val_sc   = apply_scaler(X_val,          scaler, scale_cols)
    X_test_sc  = apply_scaler(X_test,         scaler, scale_cols)
    X_norm_sc  = apply_scaler(X_train_normal, scaler, scale_cols)

    # ── Étape 7 : class_weights ───────────────────────────────────────────────
    cw = compute_class_weights(y_train)

    # ── Étape 8 : SMOTE ───────────────────────────────────────────────────────
    X_smote, y_smote = apply_smote(
        X_train_sc, y_train,
        sampling_strategy=smote_strategy,
        k_neighbors=smote_k,
        random_state=random_state,
    )

    return PreparedData(
        X_train_sc=X_train_sc,
        y_train=y_train,
        X_val_sc=X_val_sc,
        y_val=y_val,
        X_test_sc=X_test_sc,
        y_test=y_test,
        X_norm_sc=X_norm_sc,
        y_train_normal=y_train_normal,
        X_smote=X_smote,
        y_smote=y_smote,
        scaler=scaler,
        class_weights=cw,
        feature_cols=feature_cols,
        skew_before=skew_before,
        skew_after=skew_after,
    )


# ---------------------------------------------------------------------------
# Sauvegarde / chargement
# ---------------------------------------------------------------------------

def save_artifacts(
    data: PreparedData,
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Optional[Path] = None,
) -> None:
    """
    Sauvegarde tous les datasets (CSV + NPY) et artefacts
    (scaler.pkl, features.json, class_weights.json).

    Structure créée dans ``processed_dir`` :
        X_train.npy / csv   y_train.npy / csv
        X_val.npy   / csv   y_val.npy   / csv
        X_test.npy  / csv   y_test.npy  / csv
        X_train_normal.npy / csv
        X_train_smote.npy  / csv   y_train_smote.npy / csv

    Structure créée dans ``models_dir`` :
        scaler.pkl
        features.json
        class_weights.json
    """
    processed_dir = Path(processed_dir)
    models_dir    = Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    target = data.y_train.name or "isFraud"

    # Datasets
    splits = [
        (data.X_train_sc,   data.y_train,        "train"),
        (data.X_val_sc,     data.y_val,           "val"),
        (data.X_test_sc,    data.y_test,          "test"),
        (data.X_norm_sc,    data.y_train_normal,  "train_normal"),
        (data.X_smote,      data.y_smote,         "train_smote"),
    ]
    for X, y, prefix in splits:
        X.to_csv(processed_dir / f"X_{prefix}.csv", index=False)
        y.to_csv(processed_dir / f"y_{prefix}.csv", index=False)
        np.save(processed_dir / f"X_{prefix}.npy", X.values)
        np.save(processed_dir / f"y_{prefix}.npy", y.values)

    # Scaler
    joblib.dump(data.scaler, models_dir / "scaler.pkl")

    # features.json
    feat_meta = {
        "all_features":  data.feature_cols,
        "scale_cols":    SCALE_COLS,
        "binary_cols":   BINARY_COLS,
        "type_cols":     TYPE_COLS,
        "target":        target,
        "n_features":    len(data.feature_cols),
        "n_train_fraud": int(data.y_train.sum()),
        "n_val_fraud":   int(data.y_val.sum()),
        "n_test_fraud":  int(data.y_test.sum()),
        "imbalance_ratio": int((data.y_train == 0).sum() / data.y_train.sum()),
        "fraud_rate_train_pct": round(float(data.y_train.mean() * 100), 6),
        "smote_n_fraud": int(data.y_smote.sum()),
        "X_smote_is_scaled": True,
        "contamination_rate": float(data.y_train.mean()),
    }
    with open(models_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(feat_meta, f, indent=2)

    # class_weights.json
    with open(models_dir / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump(data.class_weights, f, indent=2)

    # prep_report.json (optionnel)
    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "split_sizes": {
                "train":        len(data.X_train_sc),
                "val":          len(data.X_val_sc),
                "test":         len(data.X_test_sc),
                "train_normal": len(data.X_norm_sc),
                "train_smote":  len(data.X_smote),
            },
            "fraud_counts": {
                "train":       int(data.y_train.sum()),
                "val":         int(data.y_val.sum()),
                "test":        int(data.y_test.sum()),
                "train_smote": int(data.y_smote.sum()),
            },
            "class_weights": data.class_weights,
            "features": feat_meta,
            "log1p_skew": {
                "before": round(data.skew_before, 2),
                "after":  round(data.skew_after, 2),
            },
        }
        with open(reports_dir / "prep_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


def load_prepared_data(
    processed_dir: Path,
    models_dir: Path,
    feature_cols: list[str] | None = None,
) -> dict:
    """
    Recharge les datasets sauvegardés (NPY) et les artefacts.

    Utilisé dans les notebooks 03+ pour charger les données préparées
    sans ré-exécuter le pipeline complet.

    Returns:
        dict avec les clés :
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_normal, X_smote, y_smote,
            scaler, class_weights, features_meta
    """
    processed_dir = Path(processed_dir)
    models_dir    = Path(models_dir)

    with open(models_dir / "features.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(models_dir / "class_weights.json", encoding="utf-8") as f:
        cw = json.load(f)

    cols = feature_cols or meta.get("all_features")

    def _load(prefix: str) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame(
            np.load(processed_dir / f"X_{prefix}.npy"),
            columns=cols,
        )
        y = pd.Series(
            np.load(processed_dir / f"y_{prefix}.npy"),
            name=meta.get("target", "isFraud"),
        )
        return X, y

    X_train, y_train = _load("train")
    X_val,   y_val   = _load("val")
    X_test,  y_test  = _load("test")

    X_normal = pd.DataFrame(
        np.load(processed_dir / "X_train_normal.npy"), columns=cols
    )
    X_smote = pd.DataFrame(
        np.load(processed_dir / "X_train_smote.npy"), columns=cols
    )
    y_smote = pd.Series(
        np.load(processed_dir / "y_train_smote.npy"),
        name=meta.get("target", "isFraud"),
    )

    scaler = joblib.load(models_dir / "scaler.pkl")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "X_normal": X_normal,
        "X_smote":  X_smote, "y_smote": y_smote,
        "scaler":         scaler,
        "class_weights":  cw,
        "features_meta":  meta,
    }
