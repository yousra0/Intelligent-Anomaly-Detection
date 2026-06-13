"""
autoencoder.py
==============
AutoEncoder non-supervisé pour la détection de fraude financière.
Implémentation PyTorch avec support GPU.

Principe :
    L'AutoEncoder est entraîné UNIQUEMENT sur des transactions légitimes
    (X_train_normal, 139 818 lignes, 0 fraude). Il apprend à reconstruire
    fidèlement les patterns normaux.

    À l'inférence :
        - Transaction légitime  → erreur de reconstruction FAIBLE
        - Transaction frauduleuse → erreur de reconstruction ÉLEVÉE
          (pattern inconnu : le modèle ne sait pas le reconstruire)

    Le seuil de détection est optimisé sur X_val (jamais X_test).

Architecture (14 → 10 → 7 → [4] → 7 → 10 → 14) :
    Encodeur : Linear(14,10) → BN → ReLU → Dropout(0.2)
               Linear(10, 7) → BN → ReLU → Dropout(0.2)
               Linear(7,  4) → ReLU          ← bottleneck
    Décodeur : Linear(4,  7) → BN → ReLU → Dropout(0.2)
               Linear(7, 10) → BN → ReLU → Dropout(0.2)
               Linear(10,14)                 ← sortie linéaire

Données :
    X_train_normal : (139 818, 14) — 0 fraudes   → entraînement
    X_val          : ( 30 000, 14) — 38 fraudes  → seuil optimal
    X_test         : ( 30 001, 14) — 39 fraudes  → évaluation finale
"""

from __future__ import annotations

import hashlib
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
    "encoder_dims":       [10, 7],   # dimensions des couches cachées de l'encodeur
    "bottleneck_dim":     4,          # dimension de l'espace latent compressé
    "decoder_dims":       [7, 10],   # dimensions des couches cachées du décodeur
    "activation":         "relu",
    "output_activation":  "linear",  # sortie linéaire : features standardisées ∈ ℝ

    # Régularisation
    "dropout_rate":   0.2,
    "use_batch_norm": True,
    "l2_reg":         1e-5,

    # Entraînement
    "epochs":        100,
    "batch_size":    256,
    "learning_rate": 1e-3,
    "patience":      10,    # patience EarlyStopping (epochs sans amélioration)
    "val_split":     0.1,   # fraction de X_train_normal réservée à la validation interne

    # Seuil initial
    "threshold_percentile": 95,  # percentile de l'erreur sur X_train_normal
}


# ---------------------------------------------------------------------------
# Modèle PyTorch
# ---------------------------------------------------------------------------

class AutoEncoderModel(nn.Module):
    """
    Architecture encodeur-décodeur symétrique.

    Construit dynamiquement à partir des listes encoder_dims / decoder_dims.
    Chaque couche cachée suit le schéma : Linear → BatchNorm → ReLU → Dropout.
    La couche bottleneck n'a pas de Dropout pour préserver la représentation comprimée.
    La couche de sortie est linéaire (pas d'activation) pour reconstruire des
    valeurs réelles positives ou négatives après StandardScaler.
    """

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
        self.n_features    = n_features
        self.bottleneck_dim = bottleneck_dim
        self.use_batch_norm = use_batch_norm
        self.activation     = activation.lower()
        self.output_activation = output_activation.lower()
        self.l2_reg = l2_reg

        # ── Encodeur ────────────────────────────────────────────────────────
        encoder_layers = []
        prev_dim = n_features
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            # ReLU avant Dropout : la normalisation porte sur les activations,
            # pas sur les valeurs masquées — ordre standard en deep learning.
            if activation.lower() == "relu":
                encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # Bottleneck : pas de Dropout pour conserver toute l'information comprimée
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        if activation.lower() == "relu":
            encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # ── Décodeur ────────────────────────────────────────────────────────
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

        # Couche de sortie linéaire : reconstruction continue sans contrainte de signe
        decoder_layers.append(nn.Linear(prev_dim, n_features))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Projette x dans l'espace latent (bottleneck)."""
        return self.encoder(x)

    def count_params(self) -> int:
        """Nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Callbacks d'entraînement
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Arrête l'entraînement si val_loss ne s'améliore pas pendant `patience` epochs.

    Retourne True dès que la condition d'arrêt est remplie ; l'appelant
    est responsable de sortir de la boucle d'entraînement.
    """

    def __init__(self, patience: int = 10, verbose: bool = True):
        self.patience   = patience
        self.verbose    = verbose
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ReduceLROnPlateau:
    """
    Divise le learning rate par `factor` si val_loss ne s'améliore pas
    pendant `patience` epochs consécutifs, jusqu'à `min_lr`.

    Le compteur est réinitialisé à chaque réduction pour laisser le
    nouvel LR s'installer avant une éventuelle nouvelle réduction.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        factor:    float = 0.5,
        patience:  int   = 5,
        min_lr:    float = 1e-6,
        verbose:   bool  = True,
    ):
        self.optimizer = optimizer
        self.factor    = factor
        self.patience  = patience
        self.min_lr    = min_lr
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                new_lr = max(
                    self.min_lr,
                    self.optimizer.param_groups[0]["lr"] * self.factor,
                )
                self.optimizer.param_groups[0]["lr"] = new_lr
                if self.verbose:
                    print(f"  ReduceLROnPlateau: LR -> {new_lr:.2e}")
                self.counter = 0  # réinitialise après chaque réduction


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class FraudAutoEncoder:
    """
    AutoEncoder non-supervisé pour la détection de fraude — PyTorch GPU.

    Usage typique :
        ae = FraudAutoEncoder()
        ae.build(n_features=14)
        ae.fit(X_train_normal)
        threshold = ae.find_optimal_threshold(X_val, y_val)
        y_pred    = ae.predict(X_test, threshold=threshold)
        scores    = ae.reconstruction_error(X_test)
    """

    def __init__(self, **kwargs) -> None:
        self.params  = {**AE_DEFAULTS, **kwargs}
        self.model:  Optional[AutoEncoderModel] = None
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.threshold:  float = 0.0
        self.train_time: float = 0.0
        self.n_features: int   = 0
        self.is_fitted:  bool  = False

        self.history: Optional[dict] = None

        # Statistiques MSE sur X_train_normal après entraînement.
        # Servent de référence pour le seuil initial (p95/p99) et la
        # normalisation des scores dans predict_score().
        self.train_mse_stats: dict = {}

        # État RNG capturé au début du fit() pour la traçabilité.
        self.training_seed:      Optional[int]          = None
        self.torch_rng_state:    Optional[torch.Tensor] = None
        self.cuda_rng_state_all: list[torch.Tensor]     = []

    # ── Utilitaires internes ────────────────────────────────────────────────

    def count_params(self) -> int:
        """Nombre de paramètres entraînables (0 si le modèle n'est pas construit)."""
        if self.model is None:
            return 0
        count_fn = getattr(self.model, "count_params", None)
        if callable(count_fn):
            try:
                return int(count_fn())
            except (TypeError, ValueError):
                pass
        return int(sum(p.numel() for p in self.model.parameters()))

    def _architecture_metadata(self) -> dict:
        """Retourne un dict JSON-serializable décrivant l'architecture."""
        p = self.params
        return {
            "class_name":       self.model.__class__.__name__ if self.model else None,
            "n_features":       self.n_features,
            "encoder_dims":     list(p["encoder_dims"]),
            "bottleneck_dim":   p["bottleneck_dim"],
            "decoder_dims":     list(p["decoder_dims"]),
            "activation":       p["activation"],
            "output_activation": p["output_activation"],
            "dropout_rate":     p["dropout_rate"],
            "use_batch_norm":   p["use_batch_norm"],
            "l2_reg":           p["l2_reg"],
            "total_params":     self.count_params(),
        }

    def export_metadata(self, weights_path: Path | str | None = None) -> dict:
        """Retourne les métadonnées complètes du modèle exporté (JSON-safe)."""
        meta = {
            "model_class":  self.__class__.__name__,
            "architecture": self._architecture_metadata(),
            "training": {
                "training_seed":          self.training_seed,
                "torch_rng_state_sha256": self._tensor_sha256(self.torch_rng_state),
                "cuda_rng_state_sha256":  [
                    self._tensor_sha256(s) for s in self.cuda_rng_state_all
                ],
            },
        }
        if weights_path is not None:
            path = Path(weights_path)
            if path.exists():
                meta["weights_sha256"] = self._file_sha256(path)
        return meta

    @staticmethod
    def _tensor_sha256(tensor: Optional[torch.Tensor]) -> Optional[str]:
        if tensor is None:
            return None
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

    @staticmethod
    def _file_sha256(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _capture_training_rng_state(self) -> None:
        """Capture l'état RNG au moment du lancement du fit() pour la traçabilité."""
        self.training_seed   = int(torch.initial_seed())
        self.torch_rng_state = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            self.cuda_rng_state_all = [s.clone() for s in torch.cuda.get_rng_state_all()]
        else:
            self.cuda_rng_state_all = []

    # ── Construction ────────────────────────────────────────────────────────

    def build(self, n_features: int = 14) -> "FraudAutoEncoder":
        """
        Instancie le modèle PyTorch et le déplace sur le device disponible.

        Args:
            n_features: Nombre de features en entrée (14 pour le jeu client).

        Returns:
            self — permet le chaînage ae.build().fit().
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
        ).to(self.device)
        return self

    def summary(self) -> str:
        """Affiche un résumé lisible de l'architecture et des hyperparamètres."""
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
            f"  Total params : {self.count_params():,}",
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

        Le modèle n'a jamais accès aux labels ni aux fraudes pendant le fit.
        Un sous-ensemble de validation (10 % de X_normal) est utilisé pour
        EarlyStopping et ReduceLROnPlateau — il ne contient aucune fraude.

        Args:
            X_normal: X_train_normal mis à l'échelle — (139 818, 14), 0 fraudes.
            verbose:  0 = silencieux | 1 = log tous les 10 % d'epochs.

        Returns:
            self
        """
        if self.model is None:
            self.build(n_features=X_normal.shape[1])

        X = np.asarray(X_normal, dtype=np.float32)
        p = self.params
        self._capture_training_rng_state()

        # ── Split interne train / val (sans fraude) ──────────────────────────
        n       = len(X)
        n_val   = int(n * p["val_split"])
        n_train = n - n_val
        idx     = np.random.permutation(n)
        X_train = X[idx[:n_train]]
        X_val   = X[idx[n_train:]]

        # ── DataLoaders ──────────────────────────────────────────────────────
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=p["batch_size"], shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)),
            batch_size=p["batch_size"], shuffle=False,
        )

        # ── Optimiseur et fonction de perte ──────────────────────────────────
        # weight_decay applique la régularisation L2 directement dans Adam.
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=p["learning_rate"],
            weight_decay=p["l2_reg"],
        )
        criterion = nn.MSELoss()

        # ── Callbacks ────────────────────────────────────────────────────────
        early_stopping = EarlyStopping(patience=p["patience"], verbose=True)
        reduce_lr      = ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True,
        )

        self.history     = {"loss": [], "val_loss": [], "mae": []}
        best_val_loss    = float("inf")
        best_model_state = None
        t0 = time.time()

        for epoch in range(p["epochs"]):

            # ── Phase entraînement ───────────────────────────────────────────
            self.model.train()
            train_loss = train_mae = 0.0
            n_batches  = 0

            for (X_batch,) in train_loader:
                X_batch = X_batch.to(self.device)

                optimizer.zero_grad()
                X_recon = self.model(X_batch)
                loss    = criterion(X_recon, X_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                with torch.no_grad():
                    train_mae += torch.abs(X_recon - X_batch).mean().item()
                n_batches += 1

            train_loss /= n_batches
            train_mae  /= n_batches

            # ── Phase validation ─────────────────────────────────────────────
            self.model.eval()
            val_loss     = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for (X_batch,) in val_loader:
                    X_batch = X_batch.to(self.device)
                    X_recon = self.model(X_batch)
                    val_loss += criterion(X_recon, X_batch).item()
                    n_val_batches += 1

            val_loss /= n_val_batches

            self.history["loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["mae"].append(float(train_mae))

            if verbose == 1 and (epoch + 1) % max(1, p["epochs"] // 10) == 0:
                print(
                    f"Epoch {epoch+1:3d}/{p['epochs']}  "
                    f"loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                )

            # ── Sauvegarde du meilleur état ───────────────────────────────────
            # Copie sur CPU pour ne pas occuper deux fois la mémoire GPU.
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

            # ReduceLROnPlateau est appelé en premier : il agit sur le LR du
            # prochain epoch. EarlyStopping est appelé ensuite pour décider
            # d'arrêter — les deux partagent la même métrique val_loss.
            reduce_lr(val_loss)
            if early_stopping(val_loss):
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restaure les poids du meilleur epoch avant l'évaluation finale
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        self.train_time = round(time.time() - t0, 1)
        self.is_fitted  = True

        # Statistiques d'erreur sur X_train (hors val interne) utilisées
        # comme référence absolue pour le seuil initial et predict_score().
        train_errors = self.reconstruction_error(X_train)
        self.train_mse_stats = {
            "mean": float(train_errors.mean()),
            "std":  float(train_errors.std()),
            "p95":  float(np.percentile(train_errors, 95)),
            "p99":  float(np.percentile(train_errors, 99)),
            "max":  float(train_errors.max()),
        }

        print(f"\nOK: Entrainement termine en {self.train_time}s")
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
        Calcule l'erreur de reconstruction par transaction (score d'anomalie brut).

        Une erreur élevée indique un pattern que le modèle n'a pas appris
        à reconstruire — caractéristique d'une transaction frauduleuse.

        Args:
            X:         Features mises à l'échelle.
            reduction: "mean" → MSE par ligne | "sum" → SSE par ligne.

        Returns:
            Array 1D de shape (n,).
        """
        X_arr = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_arr).to(self.device)
            X_recon  = self.model(X_tensor).cpu().numpy()
            if reduction == "mean":
                return np.mean((X_arr - X_recon) ** 2, axis=1)
            return np.sum((X_arr - X_recon) ** 2, axis=1)

    def encode(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Projette les données dans l'espace latent (dim = bottleneck_dim)."""
        X_arr = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_arr).to(self.device)
            return self.model.encode(X_tensor).cpu().numpy()

    # ── Seuil optimal ───────────────────────────────────────────────────────

    def find_optimal_threshold(
        self,
        X_val:        np.ndarray | pd.DataFrame,
        y_val:        np.ndarray | pd.Series,
        metric:       str = "f1",
        n_thresholds: int = 200,
    ) -> float:
        """
        Cherche le seuil de reconstruction error qui maximise le F1 (ou Recall)
        sur le set de validation.

        Le seuil est cherché entre le p50 et le p99.9 des erreurs de val :
        - p50 : en dessous, le modèle classerait la majorité comme fraudes → inutile.
        - p99.9 : au-dessus, presque aucune transaction n'est détectée → inutile.

        IMPORTANT : X_test n'intervient jamais ici (pas de data snooping).

        Args:
            X_val:        Validation set mis à l'échelle.
            y_val:        Labels validation (0 = légitime, 1 = fraude).
            metric:       "f1" (défaut) ou "recall".
            n_thresholds: Nombre de seuils candidats testés.

        Returns:
            Seuil optimal (float), également stocké dans self.threshold.
        """
        errors = self.reconstruction_error(X_val)
        y_true = np.asarray(y_val)

        t_min      = float(np.percentile(errors, 50))
        t_max      = float(np.percentile(errors, 99.9))
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
            else:
                prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if score > best_score:
                best_score = score
                best_t     = float(t)

        self.threshold = best_t
        print(f"Seuil optimal ({metric}) : {best_t:.6f}  (score val={best_score:.4f})")
        return best_t

    def predict(
        self,
        X:         np.ndarray | pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Classe chaque transaction en fraude (1) ou légitime (0).

        Args:
            X:         Features mises à l'échelle.
            threshold: Seuil de reconstruction error (défaut : self.threshold).

        Returns:
            Array binaire 0/1 de shape (n,).
        """
        t = threshold if threshold is not None else self.threshold
        return (self.reconstruction_error(X) >= t).astype(int)

    def predict_score(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """
        Retourne le score d'anomalie normalisé dans [0, 1].

        La normalisation utilise max(p99_train, max_erreur_courante, ε) comme
        dénominateur : le p99 sur X_train_normal sert de référence absolue
        pour les transactions normales, et le max de la batch courante
        évite que les fraudes extrêmes soient tronquées à 1 prématurément.
        L'epsilon (1e-10) protège contre la division par zéro si toutes les
        erreurs sont nulles (batch homogène ou modèle dégénéré).

        Returns:
            Array de scores dans [0, 1].
        """
        errors  = self.reconstruction_error(X)
        max_err = max(self.train_mse_stats.get("p99", 1.0), float(errors.max()), 1e-10)
        return np.clip(errors / max_err, 0.0, 1.0)

    def evaluate_on_test(
        self,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
    ) -> dict:
        """
        Évalue le seuil (optimisé sur val) sur X_test.

        Appelé après find_optimal_threshold() pour vérifier que le seuil
        généralisé bien au test set (pas de surestimation sur val).

        Returns:
            dict {threshold, recall, precision, f1, tp, fp, fn, tn}
        """
        errors = self.reconstruction_error(X_test)
        y_true = np.asarray(y_test)
        y_pred = (errors >= self.threshold).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        result = {
            "threshold": round(self.threshold, 6),
            "recall":    round(rec,  4),
            "precision": round(prec, 4),
            "f1":        round(f1,   4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        print(
            f"Test set — seuil={self.threshold:.6f} | "
            f"Recall={rec:.4f} | Precision={prec:.4f} | F1={f1:.4f} | "
            f"TP={tp} FP={fp} FN={fn}"
        )
        return result

    # ── Persistance ─────────────────────────────────────────────────────────

    def save(self, model_dir: Path | str) -> None:
        """
        Sauvegarde les poids et les métadonnées du modèle.

        Fichiers créés dans model_dir/ :
            autoencoder_weights.pt  → poids PyTorch (state_dict)
            autoencoder_meta.pkl    → hyperparamètres, seuil, stats MSE, RNG
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        weights_path = model_dir / "autoencoder_weights.pt"
        # Sauvegarde sur CPU pour compatibilité lors du rechargement sur autre machine
        torch.save(self.model.cpu().state_dict(), weights_path)
        self.model.to(self.device)

        meta = {
            "params":             self.params,
            "threshold":          self.threshold,
            "train_time":         self.train_time,
            "n_features":         self.n_features,
            "train_mse_stats":    self.train_mse_stats,
            "history_keys":       list(self.history.keys()) if self.history else [],
            "architecture":       self._architecture_metadata(),
            "weights_sha256":     self._file_sha256(weights_path),
            "training_seed":      self.training_seed,
            "torch_rng_state":    self.torch_rng_state,
            "cuda_rng_state_all": self.cuda_rng_state_all,
        }
        joblib.dump(meta, model_dir / "autoencoder_meta.pkl")
        print(f"OK: AutoEncoder (PyTorch) sauvegarde -> {model_dir}")

    @classmethod
    def load(cls, model_dir: Path | str) -> "FraudAutoEncoder":
        """
        Recharge un AutoEncoder depuis un répertoire sauvegardé par .save().

        Args:
            model_dir: Répertoire contenant autoencoder_weights.pt et
                       autoencoder_meta.pkl.

        Returns:
            Instance FraudAutoEncoder prête à l'inférence (is_fitted=True).
        """
        model_dir = Path(model_dir)
        meta = joblib.load(model_dir / "autoencoder_meta.pkl")

        obj = cls(**meta["params"])
        obj.build(n_features=meta["n_features"])
        obj.model.load_state_dict(
            torch.load(
                model_dir / "autoencoder_weights.pt",
                map_location=obj.device,
            )
        )
        obj.threshold          = meta["threshold"]
        obj.train_time         = meta["train_time"]
        obj.train_mse_stats    = meta["train_mse_stats"]
        obj.training_seed      = meta.get("training_seed")
        obj.torch_rng_state    = meta.get("torch_rng_state")
        obj.cuda_rng_state_all = meta.get("cuda_rng_state_all", [])
        obj.is_fitted          = True
        return obj
