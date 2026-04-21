"""
ml_models.py
============
Modèles baseline pour la détection de fraude PaySim.

Deux modèles supervisés classiques, utilisés comme référence avant
les approches Deep Learning (AutoEncoder, LSTM) :

  1. LogisticRegression  — baseline linéaire simple
  2. RandomForestClassifier — baseline non-linéaire robuste

Les deux exposent la même interface :
    fit(X, y)  →  predict(X)  →  predict_proba(X)

Contexte (NB02) :
  - 14 features, train=139 999, val=30 000, test=30 001
  - n_train_fraud=181  →  ratio 1:772
  - class_weights = {0: 0.5006, 1: 386.74}
  - Baseline isFlaggedFraud : Recall=0.0039, F1=0.0077
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, fbeta_score


# ---------------------------------------------------------------------------
# Hyperparamètres par défaut — justifiés pour ce dataset
# ---------------------------------------------------------------------------

LR_DEFAULTS = {
    # C faible → régularisation forte (évite l'overfitting sur 181 fraudes)
    "C":             0.1,
    "max_iter":      1000,
    "solver":        "lbfgs",
    "class_weight":  "balanced",
    "random_state":  42,
    "n_jobs":        -1,
}

RF_DEFAULTS = {
    "n_estimators":   300,
    # max_depth limité pour éviter l'overfitting sur peu de fraudes
    "max_depth":      10,
    "min_samples_leaf": 5,
    "class_weight":   "balanced",
    "random_state":   42,
    "n_jobs":         -1,
}


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class FraudLogisticRegression:
    """
    Logistic Regression wrapper pour la détection de fraude.

    Encapsule sklearn.LogisticRegression avec :
      - class_weight='balanced' par défaut (ratio 1:772)
      - interface unifiée fit / predict / predict_proba
      - sauvegarde / chargement joblib
      - méta-données d'entraînement (temps, n_features, etc.)
    """

    def __init__(self, **kwargs) -> None:
        params = {**LR_DEFAULTS, **kwargs}
        self.model = LogisticRegression(**params)
        self.params = params
        self.feature_names: list[str] = []
        self.train_time: float = 0.0
        self.is_fitted: bool = False

    # ── Entraînement ─────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "FraudLogisticRegression":
        """Entraîne le modèle et enregistre les métadonnées."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        t0 = time.time()
        self.model.fit(X, y)
        self.train_time = round(time.time() - t0, 2)
        self.is_fitted = True
        return self

    # ── Prédictions ──────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Prédictions binaires avec seuil ajustable.

        Args:
            X:         Features.
            threshold: Seuil de décision (défaut 0.5).
                       Réduire pour augmenter le Recall.

        Returns:
            Array de 0/1.
        """
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Probabilité de la classe positive (fraude)."""
        return self.model.predict_proba(X)[:, 1]

    def cv_score(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
        beta: float = 2.0,
        random_state: int = 42,
    ) -> dict:
        """Run stratified cross-validation and return mean/std for key metrics."""
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scoring = {
            "recall": "recall",
            "precision": "precision",
            "f1": "f1",
            "fbeta": make_scorer(fbeta_score, beta=beta, zero_division=0),
        }
        results = cross_validate(
            self.model,
            X,
            y,
            scoring=scoring,
            cv=splitter,
            n_jobs=self.params.get("n_jobs", -1),
        )
        return {
            "cv": cv,
            "beta": beta,
            "recall_mean": float(np.mean(results["test_recall"])),
            "recall_std": float(np.std(results["test_recall"])),
            "precision_mean": float(np.mean(results["test_precision"])),
            "precision_std": float(np.std(results["test_precision"])),
            "f1_mean": float(np.mean(results["test_f1"])),
            "f1_std": float(np.std(results["test_f1"])),
            "fbeta_mean": float(np.mean(results["test_fbeta"])),
            "fbeta_std": float(np.std(results["test_fbeta"])),
        }

    # ── Persistance ──────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Sauvegarde le modèle et ses métadonnées."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":        self.model,
            "params":       self.params,
            "feature_names": self.feature_names,
            "train_time":   self.train_time,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "FraudLogisticRegression":
        """Charge un modèle sauvegardé."""
        payload = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model         = payload["model"]
        obj.params        = payload["params"]
        obj.feature_names = payload.get("feature_names", [])
        obj.train_time    = payload.get("train_time", 0.0)
        obj.is_fitted     = True
        return obj

    def summary(self) -> str:
        lines = [
            "FraudLogisticRegression",
            f"  C={self.params['C']}  solver={self.params['solver']}",
            f"  class_weight={self.params['class_weight']}",
            f"  max_iter={self.params['max_iter']}",
            f"  train_time={self.train_time}s",
            f"  n_features={len(self.feature_names)}",
        ]
        return "\n".join(lines)


class FraudRandomForest:
    """
    Random Forest wrapper pour la détection de fraude.

    Encapsule sklearn.RandomForestClassifier avec :
      - class_weight='balanced' par défaut
      - interface unifiée fit / predict / predict_proba
      - feature importances
      - sauvegarde / chargement joblib
    """

    def __init__(self, **kwargs) -> None:
        params = {**RF_DEFAULTS, **kwargs}
        self.model = RandomForestClassifier(**params)
        self.params = params
        self.feature_names: list[str] = []
        self.train_time: float = 0.0
        self.is_fitted: bool = False

    # ── Entraînement ─────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "FraudRandomForest":
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        t0 = time.time()
        self.model.fit(X, y)
        self.train_time = round(time.time() - t0, 2)
        self.is_fitted = True
        return self

    # ── Prédictions ──────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def cv_score(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
        beta: float = 2.0,
        random_state: int = 42,
    ) -> dict:
        """Run stratified cross-validation and return mean/std for key metrics."""
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scoring = {
            "recall": "recall",
            "precision": "precision",
            "f1": "f1",
            "fbeta": make_scorer(fbeta_score, beta=beta, zero_division=0),
        }
        results = cross_validate(
            self.model,
            X,
            y,
            scoring=scoring,
            cv=splitter,
            n_jobs=self.params.get("n_jobs", -1),
        )
        return {
            "cv": cv,
            "beta": beta,
            "recall_mean": float(np.mean(results["test_recall"])),
            "recall_std": float(np.std(results["test_recall"])),
            "precision_mean": float(np.mean(results["test_precision"])),
            "precision_std": float(np.std(results["test_precision"])),
            "f1_mean": float(np.mean(results["test_f1"])),
            "f1_std": float(np.std(results["test_f1"])),
            "fbeta_mean": float(np.mean(results["test_fbeta"])),
            "fbeta_std": float(np.std(results["test_fbeta"])),
        }

    # ── Feature Importances ──────────────────────────────────────────────────

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Retourne les importances des features triées par ordre décroissant.

        Returns:
            DataFrame avec colonnes : feature, importance.
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant.")
        names = self.feature_names or [f"f{i}" for i in range(
            len(self.model.feature_importances_)
        )]
        return (
            pd.DataFrame({
                "feature":    names,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Persistance ──────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":        self.model,
            "params":       self.params,
            "feature_names": self.feature_names,
            "train_time":   self.train_time,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "FraudRandomForest":
        payload = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model         = payload["model"]
        obj.params        = payload["params"]
        obj.feature_names = payload.get("feature_names", [])
        obj.train_time    = payload.get("train_time", 0.0)
        obj.is_fitted     = True
        return obj

    def summary(self) -> str:
        lines = [
            "FraudRandomForest",
            f"  n_estimators={self.params['n_estimators']}",
            f"  max_depth={self.params['max_depth']}",
            f"  min_samples_leaf={self.params['min_samples_leaf']}",
            f"  class_weight={self.params['class_weight']}",
            f"  train_time={self.train_time}s",
            f"  n_features={len(self.feature_names)}",
        ]
        return "\n".join(lines)
