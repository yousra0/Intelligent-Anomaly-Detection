"""
autoencoder.py
==============
AutoEncoder non-supervisé pour la détection de fraude financière PaySim.

Principe :
    L'AutoEncoder est entraîné UNIQUEMENT sur des transactions légitimes
    (X_train_normal, 139 818 lignes, 0 fraude). Il apprend à reconstruire
    fidèlement les transactions normales.

    Au moment de l'inférence :
    - Transaction légitime → erreur de reconstruction FAIBLE
    - Transaction frauduleuse → erreur de reconstruction ÉLEVÉE
      (le modèle n'a jamais vu ce pattern → il ne sait pas le reconstruire)

    Seuil de détection : percentile de l'erreur sur X_val → optimisé pour F1.

Architecture (14 → 10 → 7 → 4 → 7 → 10 → 14) :
    Encodeur : Dense(10, relu) → BN → Dropout(0.2)
               Dense(7,  relu) → BN → Dropout(0.2)
               Dense(4,  relu)          ← bottleneck
    Décodeur : Dense(7,  relu) → BN → Dropout(0.2)
               Dense(10, relu) → BN → Dropout(0.2)
               Dense(14, linear)        ← reconstruction

Références :
    - X_train_normal : (139818, 14) — 0 fraudes
    - X_val          : (30000,  14) — 38 fraudes  → seuil optimal
    - X_test         : (30001,  14) — 39 fraudes  → évaluation finale
    - Baseline RF_smote : Recall=0.7949, F1=0.8052, PR-AUC=0.8405
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Hyperparamètres par défaut
# ---------------------------------------------------------------------------
AE_DEFAULTS = {
    # Architecture
    "encoder_dims":  [10, 7],        # couches encodeur avant bottleneck
    "bottleneck_dim": 4,             # dimension de l'espace latent
    "decoder_dims":  [7, 10],        # couches décodeur
    "activation":    "relu",
    "output_activation": "linear",   # reconstruction continue (features scalées)

    # Régularisation
    "dropout_rate":  0.2,
    "use_batch_norm": True,
    "l2_reg":        1e-5,

    # Entraînement
    "epochs":        100,
    "batch_size":    256,
    "learning_rate": 1e-3,
    "patience":      10,             # EarlyStopping
    "val_split":     0.1,            # validation interne sur X_train_normal

    # Seuil de détection
    "threshold_percentile": 95,      # percentile erreur sur X_val pour seuil initial
}


# ---------------------------------------------------------------------------
# Classe FraudAutoEncoder
# ---------------------------------------------------------------------------

class FraudAutoEncoder:
    """
    AutoEncoder non-supervisé pour la détection de fraude.

    Usage typique :
        ae = FraudAutoEncoder()
        ae.build(n_features=14)
        ae.fit(X_train_normal)
        threshold = ae.find_optimal_threshold(X_val, y_val)
        scores    = ae.reconstruction_error(X_test)
        y_pred    = ae.predict(X_test, threshold=threshold)
    """

    def __init__(self, **kwargs) -> None:
        self.params      = {**AE_DEFAULTS, **kwargs}
        self.model:       Optional[keras.Model] = None
        self.encoder:     Optional[keras.Model] = None
        self.threshold:   float = 0.0
        self.train_time:  float = 0.0
        self.history:     Optional[dict] = None
        self.n_features:  int = 0
        self.is_fitted:   bool = False
        self.train_mse_stats: dict = {}

    # ── Construction du modèle ───────────────────────────────────────────────

    def build(self, n_features: int = 14) -> "FraudAutoEncoder":
        """
        Construit l'architecture encodeur-décodeur.

        Architecture : 14 → 10 → 7 → [4] → 7 → 10 → 14
            - BatchNormalization après chaque Dense (sauf bottleneck et sortie)
            - Dropout(0.2) pour régularisation
            - L2 sur les poids denses

        Args:
            n_features: Nombre de features en entrée (14 pour PaySim).

        Returns:
            self (pour chaînage)
        """
        self.n_features = n_features
        p = self.params

        inp = keras.Input(shape=(n_features,), name="input")
        x   = inp

        # ── Encodeur ──
        for i, dim in enumerate(p["encoder_dims"]):
            x = layers.Dense(
                dim,
                activation=p["activation"],
                kernel_regularizer=regularizers.l2(p["l2_reg"]),
                name=f"enc_{i+1}",
            )(x)
            if p["use_batch_norm"]:
                x = layers.BatchNormalization(name=f"bn_enc_{i+1}")(x)
            x = layers.Dropout(p["dropout_rate"], name=f"drop_enc_{i+1}")(x)

        # ── Bottleneck ──
        encoded = layers.Dense(
            p["bottleneck_dim"],
            activation=p["activation"],
            name="bottleneck",
        )(x)

        # ── Décodeur ──
        x = encoded
        for i, dim in enumerate(p["decoder_dims"]):
            x = layers.Dense(
                dim,
                activation=p["activation"],
                kernel_regularizer=regularizers.l2(p["l2_reg"]),
                name=f"dec_{i+1}",
            )(x)
            if p["use_batch_norm"]:
                x = layers.BatchNormalization(name=f"bn_dec_{i+1}")(x)
            x = layers.Dropout(p["dropout_rate"], name=f"drop_dec_{i+1}")(x)

        # ── Sortie (reconstruction) ──
        decoded = layers.Dense(
            n_features,
            activation=p["output_activation"],
            name="output",
        )(x)

        # ── Modèles ──
        self.model   = keras.Model(inp, decoded,  name="FraudAutoEncoder")
        self.encoder = keras.Model(inp, encoded,  name="Encoder")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
            loss="mse",
            metrics=["mae"],
        )
        return self

    def summary(self) -> str:
        """Résumé de l'architecture."""
        if self.model is None:
            return "Modèle non construit — appeler .build() d'abord."
        p = self.params
        arch = (
            f"{self.n_features} → "
            + " → ".join(str(d) for d in p["encoder_dims"])
            + f" → [{p['bottleneck_dim']}] → "
            + " → ".join(str(d) for d in p["decoder_dims"])
            + f" → {self.n_features}"
        )
        lines = [
            "FraudAutoEncoder",
            f"  Architecture : {arch}",
            f"  BatchNorm    : {p['use_batch_norm']}",
            f"  Dropout      : {p['dropout_rate']}",
            f"  L2 reg       : {p['l2_reg']}",
            f"  Bottleneck   : {p['bottleneck_dim']} dims",
            f"  Epochs max   : {p['epochs']}  (patience={p['patience']})",
            f"  Batch size   : {p['batch_size']}",
            f"  LR           : {p['learning_rate']}",
            f"  Threshold    : {self.threshold:.6f}",
            f"  Train time   : {self.train_time:.1f}s",
        ]
        return "\n".join(lines)

    # ── Entraînement ────────────────────────────────────────────────────────

    def fit(
        self,
        X_normal: np.ndarray | pd.DataFrame,
        verbose: int = 1,
    ) -> "FraudAutoEncoder":
        """
        Entraîne l'AutoEncoder sur les transactions légitimes uniquement.

        EarlyStopping sur val_loss (patience=10) pour éviter l'overfitting.
        Le modèle n'a jamais accès aux fraudes durant l'entraînement.

        Args:
            X_normal: X_train_normal scalé — (139818, 14), 0 fraudes.
            verbose:  0=silencieux, 1=barre de progression, 2=une ligne/epoch.

        Returns:
            self
        """
        if self.model is None:
            self.build(n_features=X_normal.shape[1])

        X = np.asarray(X_normal, dtype=np.float32)
        p = self.params

        cb_list = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=p["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        t0 = time.time()
        hist = self.model.fit(
            X, X,                          # AutoEncoder : input = output
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            validation_split=p["val_split"],
            callbacks=cb_list,
            shuffle=True,
            verbose=verbose,
        )
        self.train_time = round(time.time() - t0, 1)
        self.history    = hist.history
        self.is_fitted  = True

        # Stats MSE sur train normal (référence pour le seuil)
        train_errors = self.reconstruction_error(X)
        self.train_mse_stats = {
            "mean":   float(train_errors.mean()),
            "std":    float(train_errors.std()),
            "p95":    float(np.percentile(train_errors, 95)),
            "p99":    float(np.percentile(train_errors, 99)),
            "max":    float(train_errors.max()),
        }

        print(f"\n✅ Entraînement terminé en {self.train_time}s")
        print(f"   MSE moyen (train normal) : {self.train_mse_stats['mean']:.6f}")
        print(f"   MSE p95 (train normal)   : {self.train_mse_stats['p95']:.6f}")
        print(f"   MSE p99 (train normal)   : {self.train_mse_stats['p99']:.6f}")
        return self

    # ── Scoring ─────────────────────────────────────────────────────────────

    def reconstruction_error(
        self,
        X: np.ndarray | pd.DataFrame,
        reduction: str = "mean",
    ) -> np.ndarray:
        """
        Calcule l'erreur de reconstruction (MSE par transaction).

        Plus l'erreur est élevée, plus la transaction est anormale.

        Args:
            X:         Features scalées.
            reduction: "mean" = MSE par ligne | "sum" = SSE par ligne.

        Returns:
            Array 1D des erreurs de reconstruction (shape=(n,)).
        """
        X_arr  = np.asarray(X, dtype=np.float32)
        X_rec  = self.model.predict(X_arr, verbose=0)
        errors = np.mean((X_arr - X_rec) ** 2, axis=1)
        if reduction == "sum":
            errors = np.sum((X_arr - X_rec) ** 2, axis=1)
        return errors

    def encode(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Projette les données dans l'espace latent (bottleneck)."""
        X_arr = np.asarray(X, dtype=np.float32)
        return self.encoder.predict(X_arr, verbose=0)

    # ── Seuil optimal ───────────────────────────────────────────────────────

    def find_optimal_threshold(
        self,
        X_val: np.ndarray | pd.DataFrame,
        y_val: np.ndarray | pd.Series,
        metric: str = "f1",
        n_thresholds: int = 200,
    ) -> float:
        """
        Cherche le seuil de reconstruction error qui maximise le F1
        (ou Recall) sur le set de validation.

        IMPORTANT : seuil sélectionné sur X_val uniquement.
                    X_test n'intervient jamais ici (anti data-snooping).

        Args:
            X_val:        Validation set scalé.
            y_val:        Labels validation (0/1).
            metric:       "f1" ou "recall".
            n_thresholds: Nombre de seuils à tester.

        Returns:
            Seuil optimal (float) — également stocké dans self.threshold.
        """
        errors = self.reconstruction_error(X_val)
        y_true = np.asarray(y_val)

        # Plage de seuils entre percentile 50 et 99.9 des erreurs val
        t_min = float(np.percentile(errors, 50))
        t_max = float(np.percentile(errors, 99.9))
        thresholds = np.linspace(t_min, t_max, n_thresholds)

        best_score = -1.0
        best_t     = t_min

        for t in thresholds:
            y_pred = (errors >= t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())

            if metric == "recall":
                score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:  # f1
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                score = (2 * prec * rec / (prec + rec)
                         if (prec + rec) > 0 else 0.0)

            if score > best_score:
                best_score = score
                best_t     = float(t)

        self.threshold = best_t
        print(f"Seuil optimal ({metric}) : {best_t:.6f}  "
              f"(score val={best_score:.4f})")
        return best_t

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Prédit fraude (1) ou légitime (0) selon le seuil de reconstruction.

        Args:
            X:         Features scalées.
            threshold: Seuil (défaut : self.threshold).

        Returns:
            Array binaire 0/1.
        """
        t = threshold if threshold is not None else self.threshold
        errors = self.reconstruction_error(X)
        return (errors >= t).astype(int)

    def predict_score(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """
        Retourne les scores d'anomalie (erreur de reconstruction normalisée).

        Les scores sont normalisés entre 0 et 1 par rapport au max observé
        sur le train normal pour faciliter la comparaison avec les
        probabilités des modèles ML.

        Returns:
            Array de scores dans [0, 1].
        """
        errors = self.reconstruction_error(X)
        max_err = max(self.train_mse_stats.get("p99", 1.0), errors.max())
        return np.clip(errors / max_err, 0.0, 1.0)

    # ── Persistance ─────────────────────────────────────────────────────────

    def save(self, model_dir: Path | str) -> None:
        """
        Sauvegarde le modèle Keras + métadonnées.

        Structure créée dans model_dir/ :
            autoencoder_weights.keras   → poids du modèle
            autoencoder_meta.pkl        → params, threshold, stats
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model.save(model_dir / "autoencoder_weights.keras")
        meta = {
            "params":           self.params,
            "threshold":        self.threshold,
            "train_time":       self.train_time,
            "n_features":       self.n_features,
            "train_mse_stats":  self.train_mse_stats,
            "history_keys":     list(self.history.keys()) if self.history else [],
        }
        joblib.dump(meta, model_dir / "autoencoder_meta.pkl")
        print(f"✅ AutoEncoder sauvegardé → {model_dir}")

    @classmethod
    def load(cls, model_dir: Path | str) -> "FraudAutoEncoder":
        """Charge un AutoEncoder sauvegardé."""
        model_dir = Path(model_dir)
        meta      = joblib.load(model_dir / "autoencoder_meta.pkl")

        obj = cls(**meta["params"])
        obj.model         = keras.models.load_model(
            model_dir / "autoencoder_weights.keras"
        )
        obj.encoder       = keras.Model(
            obj.model.input,
            obj.model.get_layer("bottleneck").output,
        )
        obj.threshold        = meta["threshold"]
        obj.train_time       = meta["train_time"]
        obj.n_features       = meta["n_features"]
        obj.train_mse_stats  = meta["train_mse_stats"]
        obj.is_fitted        = True
        return obj
