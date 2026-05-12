"""
autoencoder.py
==============
AutoEncoder non-supervisé pour la détection de fraude financière PaySim.
IMPLÉMENTATION PYTORCH AVEC GPU

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
    Encodeur : Linear(14, 10) → BN → Dropout(0.2) → ReLU
               Linear(10, 7)  → BN → Dropout(0.2) → ReLU
               Linear(7, 4)   → ReLU          ← bottleneck
    Décodeur : Linear(4, 7)  → BN → Dropout(0.2) → ReLU
               Linear(7, 10) → BN → Dropout(0.2) → ReLU
               Linear(10, 14) → Linear (reconstruction)

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Hyperparamètres par défaut
# ---------------------------------------------------------------------------
AE_DEFAULTS = {
    # Architecture
    "encoder_dims":  [10, 7],        # couches encodeur avant bottleneck
    "bottleneck_dim": 4,             # dimension de l'espace latent
    "decoder_dims":  [7, 10],        # couches décodeur
    "activation":    "relu",
    "output_activation": "linear",   # reconstruction continue

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
# Modèle PyTorch
# ---------------------------------------------------------------------------

class AutoEncoderModel(nn.Module):
    """Architecture AutoEncoder en PyTorch."""

    def __init__(
        self,
        n_features: int,
        encoder_dims: list,
        bottleneck_dim: int,
        decoder_dims: list,
        activation: str = "relu",
        output_activation: str = "linear",
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        l2_reg: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.bottleneck_dim = bottleneck_dim
        self.use_batch_norm = use_batch_norm
        self.activation = activation.lower()
        self.output_activation = output_activation.lower()
        self.l2_reg = l2_reg

        # ── Encodeur ──
        encoder_layers = []
        prev_dim = n_features
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            if activation.lower() == "relu":
                encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        if activation.lower() == "relu":
            encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # ── Décodeur ──
        decoder_layers = []
        prev_dim = bottleneck_dim
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(dim))
            if activation.lower() == "relu":
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # Sortie (reconstruction)
        decoder_layers.append(nn.Linear(prev_dim, n_features))
        # Pas d'activation en sortie (linear)

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (bottleneck)."""
        return self.encoder(x)

    def count_params(self) -> int:
        """Nombre total de paramètres."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Callbacks personnalisés
# ---------------------------------------------------------------------------

class EarlyStopping:
    """EarlyStopping pour PyTorch."""

    def __init__(self, patience: int = 10, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ReduceLROnPlateau:
    """ReduceLROnPlateau pour PyTorch."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                new_lr = max(
                    self.min_lr,
                    self.optimizer.param_groups[0]["lr"] * self.factor
                )
                self.optimizer.param_groups[0]["lr"] = new_lr
                if self.verbose:
                    print(f"  ReduceLROnPlateau: LR → {new_lr:.2e}")
                self.counter = 0


# ---------------------------------------------------------------------------
# Classe FraudAutoEncoder
# ---------------------------------------------------------------------------

class FraudAutoEncoder:
    """
    AutoEncoder non-supervisé pour la détection de fraude — PYTORCH GPU.

    Usage typique :
        ae = FraudAutoEncoder()
        ae.build(n_features=14)
        ae.fit(X_train_normal)
        threshold = ae.find_optimal_threshold(X_val, y_val)
        scores    = ae.reconstruction_error(X_test)
        y_pred    = ae.predict(X_test, threshold=threshold)
    """

    def __init__(self, **kwargs) -> None:
        self.params = {**AE_DEFAULTS, **kwargs}
        self.model: Optional[AutoEncoderModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold: float = 0.0
        self.train_time: float = 0.0
        self.history: Optional[dict] = None
        self.n_features: int = 0
        self.is_fitted: bool = False
        self.train_mse_stats: dict = {}

    # ── Construction du modèle ───────────────────────────────────────────────

    def build(self, n_features: int = 14) -> "FraudAutoEncoder":
        """
        Construit l'architecture encodeur-décodeur.

        Args:
            n_features: Nombre de features en entrée (14 pour PaySim).

        Returns:
            self (pour chaînage)
        """
        self.n_features = n_features
        p = self.params

        self.model = AutoEncoderModel(
            n_features=n_features,
            encoder_dims=p["encoder_dims"],
            bottleneck_dim=p["bottleneck_dim"],
            decoder_dims=p["decoder_dims"],
            activation=p["activation"],
            output_activation=p["output_activation"],
            dropout_rate=p["dropout_rate"],
            use_batch_norm=p["use_batch_norm"],
            l2_reg=p["l2_reg"],
        )
        self.model = self.model.to(self.device)
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
            "FraudAutoEncoder (PyTorch + GPU)",
            f"  Architecture : {arch}",
            f"  Device       : {self.device}",
            f"  BatchNorm    : {p['use_batch_norm']}",
            f"  Dropout      : {p['dropout_rate']}",
            f"  L2 reg       : {p['l2_reg']}",
            f"  Bottleneck   : {p['bottleneck_dim']} dims",
            f"  Total params : {self.model.count_params():,}",
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

        # Split train/val
        n = len(X)
        n_val = int(n * p["val_split"])
        n_train = n - n_val

        # Shuffle et split
        idx = np.random.permutation(n)
        idx_train, idx_val = idx[:n_train], idx[n_train:]
        X_train = X[idx_train]
        X_val = X[idx_val]

        # DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        train_loader = DataLoader(
            train_dataset, batch_size=p["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=p["batch_size"], shuffle=False
        )

        # Optimiseur & Loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=p["learning_rate"],
            weight_decay=p["l2_reg"],
        )
        criterion = nn.MSELoss()

        # Callbacks
        early_stopping = EarlyStopping(patience=p["patience"], verbose=True)
        reduce_lr = ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )

        # Historique
        self.history = {"loss": [], "val_loss": [], "mae": []}

        t0 = time.time()
        best_model_state = None
        best_val_loss = float("inf")

        for epoch in range(p["epochs"]):
            # ── Train ──
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            n_batches = 0

            for batch in train_loader:
                X_batch = batch[0].to(self.device)

                optimizer.zero_grad()
                X_recon = self.model(X_batch)
                loss = criterion(X_recon, X_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                with torch.no_grad():
                    mae = torch.abs(X_recon - X_batch).mean().item()
                    train_mae += mae
                n_batches += 1

            train_loss /= n_batches
            train_mae /= n_batches

            # ── Validation ──
            self.model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    X_batch = batch[0].to(self.device)
                    X_recon = self.model(X_batch)
                    loss = criterion(X_recon, X_batch)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= n_val_batches

            # Historique
            self.history["loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["mae"].append(float(train_mae))

            # Affichage
            if verbose == 1 and (epoch + 1) % max(1, p["epochs"] // 10) == 0:
                print(
                    f"Epoch {epoch+1:3d}/{p['epochs']}  "
                    f"loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                )

            # EarlyStopping + ReduceLROnPlateau
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }

            reduce_lr(val_loss)
            if early_stopping(val_loss):
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            for k in best_model_state:
                self.model.state_dict()[k].copy_(best_model_state[k])
            self.model.to(self.device)

        self.train_time = round(time.time() - t0, 1)
        self.is_fitted = True

        # Stats MSE sur train normal (référence pour le seuil)
        train_errors = self.reconstruction_error(X_train)
        self.train_mse_stats = {
            "mean": float(train_errors.mean()),
            "std": float(train_errors.std()),
            "p95": float(np.percentile(train_errors, 95)),
            "p99": float(np.percentile(train_errors, 99)),
            "max": float(train_errors.max()),
        }

        print(f"\n✅ Entraînement terminé en {self.train_time}s")
        print(f"   Epochs effectifs : {len(self.history['loss'])}")
        print(f"   Best val_loss    : {min(self.history['val_loss']):.6f}")
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
        X_arr = np.asarray(X, dtype=np.float32)
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_arr).to(self.device)
            X_recon = self.model(X_tensor).cpu().numpy()
            if reduction == "mean":
                errors = np.mean((X_arr - X_recon) ** 2, axis=1)
            else:  # sum
                errors = np.sum((X_arr - X_recon) ** 2, axis=1)
        return errors

    def encode(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Projette les données dans l'espace latent (bottleneck)."""
        X_arr = np.asarray(X, dtype=np.float32)
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_arr).to(self.device)
            encoded = self.model.encode(X_tensor).cpu().numpy()
        return encoded

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
        best_t = t_min

        for t in thresholds:
            y_pred = (errors >= t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())

            if metric == "recall":
                score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:  # f1
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if score > best_score:
                best_score = score
                best_t = float(t)

        self.threshold = best_t
        print(f"Seuil optimal ({metric}) : {best_t:.6f}  " f"(score val={best_score:.4f})")
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
        Sauvegarde le modèle PyTorch + métadonnées.

        Structure créée dans model_dir/ :
            autoencoder_weights.pt     → poids du modèle
            autoencoder_meta.pkl       → params, threshold, stats
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les poids (sur CPU)
        torch.save(self.model.state_dict(), model_dir / "autoencoder_weights.pt")

        meta = {
            "params": self.params,
            "threshold": self.threshold,
            "train_time": self.train_time,
            "n_features": self.n_features,
            "train_mse_stats": self.train_mse_stats,
            "history_keys": list(self.history.keys()) if self.history else [],
        }
        joblib.dump(meta, model_dir / "autoencoder_meta.pkl")
        print(f"✅ AutoEncoder (PyTorch) sauvegardé → {model_dir}")

    @classmethod
    def load(cls, model_dir: Path | str) -> "FraudAutoEncoder":
        """Charge un AutoEncoder sauvegardé."""
        model_dir = Path(model_dir)
        meta = joblib.load(model_dir / "autoencoder_meta.pkl")

        obj = cls(**meta["params"])
        obj.build(n_features=meta["n_features"])
        obj.model.load_state_dict(
            torch.load(model_dir / "autoencoder_weights.pt", map_location=obj.device)
        )
        obj.threshold = meta["threshold"]
        obj.train_time = meta["train_time"]
        obj.train_mse_stats = meta["train_mse_stats"]
        obj.is_fitted = True
        return obj

