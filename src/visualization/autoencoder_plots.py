"""
autoencoder_plots.py
====================
Visualisations pour ``04_autoencoder.ipynb``.

Figures générées :
  18_training_history.png      : courbes loss train/val par epoch
  19_reconstruction_error.png  : distribution MSE normal vs fraude
  20_threshold_roc_pr.png      : courbes ROC et PR de l'AutoEncoder
  21_ae_confusion_matrix.png   : matrice de confusion au seuil optimal
  22_ae_vs_baselines.png       : comparaison AutoEncoder vs 4 modèles ML
  23_latent_space.png          : espace latent (PCA 2D) normal vs fraude
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.decomposition import PCA

plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 12})

C = {
    "normal":   "#2196F3",
    "fraud":    "#F44336",
    "ae":       "#9C27B0",
    "train":    "#1565C0",
    "val":      "#00897B",
    "rf_smote": "#E65100",
    "grey":     "#78909C",
    "navy":     "1B2A4A",
}

# Référence RF_smote (NB03)
RF_SMOTE = {"recall": 0.7949, "f1": 0.8052, "pr_auc": 0.8405}
BASELINE_RECALL = 0.0039
BASELINE_F1     = 0.0077


# ---------------------------------------------------------------------------
# Figure 18 — Historique d'entraînement
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Courbes loss (MSE) et MAE train/val par epoch.

    Identifie visuellement le meilleur epoch (EarlyStopping).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, title in [
        (axes[0], "loss", "MSE Loss — Train vs Validation"),
        (axes[1], "mae",  "MAE — Train vs Validation"),
    ]:
        train_vals = history.get(key, [])
        val_vals   = history.get(f"val_{key}", [])
        epochs     = range(1, len(train_vals) + 1)

        ax.plot(epochs, train_vals, color=C["train"], lw=2, label="Train")
        if val_vals:
            ax.plot(epochs, val_vals, color=C["val"], lw=2,
                    linestyle="--", label="Validation")

            # Meilleur epoch
            best_ep = int(np.argmin(val_vals)) + 1
            ax.axvline(best_ep, color=C["fraud"], linestyle=":",
                       lw=1.5, label=f"Best epoch={best_ep}")
            ax.scatter([best_ep], [val_vals[best_ep - 1]],
                       color=C["fraud"], s=80, zorder=5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.upper())
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Historique d'entraînement — AutoEncoder",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 19 — Distribution erreur de reconstruction
# ---------------------------------------------------------------------------

def plot_reconstruction_error_dist(
    errors_normal: np.ndarray,
    errors_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Distributions MSE pour transactions normales vs frauduleuses.

    Montre la séparabilité entre les deux classes.
    Le seuil optimal est représenté par une ligne verticale.

    Args:
        errors_normal: MSE sur X_train_normal.
        errors_val:    MSE sur X_val.
        y_val:         Labels X_val.
        threshold:     Seuil de décision optimal.
    """
    errors_fraud  = errors_val[y_val == 1]
    errors_legit  = errors_val[y_val == 0]

    # Clip pour la visualisation (fraudes peuvent avoir des valeurs extrêmes)
    p999 = np.percentile(errors_val, 99.9)
    clip = max(p999, threshold * 1.5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel gauche : histogrammes superposés ──
    bins = np.linspace(0, clip, 80)
    axes[0].hist(
        np.clip(errors_normal, 0, clip), bins=bins,
        alpha=0.6, color=C["normal"], label=f"Train normal (n={len(errors_normal):,})",
        density=True,
    )
    axes[0].hist(
        np.clip(errors_legit, 0, clip), bins=bins,
        alpha=0.5, color=C["train"], label=f"Val légitime (n={len(errors_legit):,})",
        density=True,
    )
    axes[0].hist(
        np.clip(errors_fraud, 0, clip), bins=bins,
        alpha=0.8, color=C["fraud"], label=f"Val fraude (n={len(errors_fraud)})",
        density=True,
    )
    axes[0].axvline(threshold, color="black", linestyle="--", lw=2,
                    label=f"Seuil={threshold:.4f}")
    axes[0].set_xlabel("Erreur de reconstruction (MSE)")
    axes[0].set_ylabel("Densité")
    axes[0].set_title("Distribution MSE — Normal vs Fraude")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # ── Panel droit : boxplot ──
    data_box   = [errors_normal, errors_legit, errors_fraud]
    labels_box = ["Train\nnormal", "Val\nlégitime", "Val\nfraude"]
    colors_box = [C["normal"], C["train"], C["fraud"]]

    bp = axes[1].boxplot(
        [np.clip(d, 0, clip) for d in data_box],
        labels=labels_box,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].axhline(threshold, color="black", linestyle="--", lw=1.5,
                    label=f"Seuil={threshold:.4f}")
    axes[1].set_ylabel("Erreur de reconstruction (MSE)")
    axes[1].set_title("Boxplot MSE par catégorie")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Erreur de Reconstruction — AutoEncoder",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 20 — Courbes ROC et PR de l'AutoEncoder
# ---------------------------------------------------------------------------

def plot_ae_roc_pr(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Courbes ROC et Precision-Recall pour l'AutoEncoder.
    Inclut les valeurs de référence RF_smote (NB03).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── ROC ──
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=C["ae"], lw=2,
                 label=f"AutoEncoder (AUC={roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance (0.50)")
    axes[0].set_xlabel("Taux Faux Positifs (FPR)")
    axes[0].set_ylabel("Taux Vrais Positifs (Recall)")
    axes[0].set_title("Courbe ROC — AutoEncoder")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # ── PR ──
    prec, rec, _ = precision_recall_curve(y_true, scores)
    pr_auc_val   = auc(rec, prec)

    axes[1].plot(rec, prec, color=C["ae"], lw=2,
                 label=f"AutoEncoder (PR-AUC={pr_auc_val:.4f})")

    # Référence RF_smote
    axes[1].scatter(RF_SMOTE["recall"], 0.8158, marker="s", s=120,
                    color=C["rf_smote"], zorder=5,
                    label=f"RF_smote (Recall={RF_SMOTE['recall']}, F1={RF_SMOTE['f1']})")

    # Baseline isFlaggedFraud
    axes[1].scatter(BASELINE_RECALL, 1.0, marker="*", s=200,
                    color=C["grey"], zorder=5,
                    label=f"Baseline isFlaggedFraud (R={BASELINE_RECALL})")

    fraud_rate = y_true.mean()
    axes[1].axhline(fraud_rate, color="#BDBDBD", linestyle="--", lw=1,
                    label=f"Chance ({fraud_rate:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Courbe Precision-Recall — AutoEncoder")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Évaluation AutoEncoder — Test Set  "
                 f"(seuil={threshold:.4f})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 21 — Matrice de confusion AutoEncoder
# ---------------------------------------------------------------------------

def plot_ae_confusion_matrix(
    metrics: dict,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Matrice de confusion de l'AutoEncoder au seuil optimal.
    """
    tp, fp = metrics["tp"], metrics["fp"]
    fn, tn = metrics["fn"], metrics["tn"]
    cm_arr = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Purples")

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            val   = cm_arr[i, j]
            color = "white" if val > cm_arr.max() * 0.6 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{val:,}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Prédit 0\n(Légitime)", "Prédit 1\n(Fraude)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Réel 0\n(Légitime)", "Réel 1\n(Fraude)"])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(
        f"AutoEncoder — Matrice de Confusion (Test)\n"
        f"Recall={metrics['recall']:.4f}  "
        f"Precision={metrics['precision']:.4f}  "
        f"F1={metrics['f1']:.4f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 22 — AutoEncoder vs Baselines ML
# ---------------------------------------------------------------------------

def plot_ae_vs_baselines(
    ae_metrics: dict,
    baselines_metrics: list[dict],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Comparaison AutoEncoder vs 4 modèles ML (Recall, F1, PR-AUC).

    Args:
        ae_metrics:        Métriques AutoEncoder (dict compute_fraud_metrics).
        baselines_metrics: Liste des métriques ML (NB03).
    """
    all_metrics = baselines_metrics + [ae_metrics]

    names    = [m.get("model", "?") for m in all_metrics]
    recalls  = [m.get("recall",  0) for m in all_metrics]
    f1s      = [m.get("f1",      0) for m in all_metrics]
    pr_aucs  = [m.get("pr_auc",  0) or 0 for m in all_metrics]

    x = np.arange(len(names))
    w = 0.25

    # Couleur spéciale pour AutoEncoder
    bar_colors = [C["grey"]] * len(baselines_metrics) + [C["ae"]]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (vals, metric, color_base) in enumerate([
        (recalls, "Recall",  "#F44336"),
        (f1s,     "F1",      "#4CAF50"),
        (pr_aucs, "PR-AUC",  "#2196F3"),
    ]):
        colors = [color_base] * len(baselines_metrics) + [C["ae"]]
        bars = ax.bar(x + i * w, vals, w, label=metric,
                      color=colors, edgecolor="white", alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + 0.012, f"{h:.3f}",
                        ha="center", fontsize=7.5, fontweight="bold")

    # Lignes référence
    ax.axhline(BASELINE_RECALL, color="#9E9E9E",
               linestyle="--", lw=1.2,
               label=f"Baseline Recall ({BASELINE_RECALL})")
    ax.axhline(RF_SMOTE["f1"], color="#FF8F00",
               linestyle=":", lw=1.5,
               label=f"RF_smote F1 ({RF_SMOTE['f1']})")

    ax.set_xticks(x + w)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.15])
    ax.set_title("AutoEncoder vs Modèles Baseline — Test Set",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)

    # Annotation AutoEncoder
    ae_idx = len(all_metrics) - 1
    ax.annotate(
        "AutoEncoder\n(non-supervisé)",
        xy=(x[ae_idx] + w, recalls[ae_idx]),
        xytext=(x[ae_idx] + w + 0.4, recalls[ae_idx] + 0.12),
        arrowprops={"arrowstyle": "->", "color": C["ae"]},
        fontsize=9, color=C["ae"], fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 23 — Espace latent (PCA 2D)
# ---------------------------------------------------------------------------

def plot_latent_space(
    encoded_normal: np.ndarray,
    encoded_val: np.ndarray,
    y_val: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Visualisation de l'espace latent (bottleneck) via PCA 2D.

    Montre si l'AutoEncoder sépare naturellement les fraudes
    des transactions normales dans l'espace compressé.

    Args:
        encoded_normal: Représentations latentes de X_train_normal.
        encoded_val:    Représentations latentes de X_val.
        y_val:          Labels de X_val.
    """
    # PCA sur l'ensemble train_normal + val
    all_enc = np.vstack([encoded_normal, encoded_val])
    pca     = PCA(n_components=2, random_state=42)
    all_2d  = pca.fit_transform(all_enc)

    n_normal   = len(encoded_normal)
    normal_2d  = all_2d[:n_normal]
    val_2d     = all_2d[n_normal:]

    legit_2d = val_2d[y_val == 0]
    fraud_2d = val_2d[y_val == 1]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Sous-échantillonnage pour lisibilité
    n_plot = min(5000, len(normal_2d))
    idx    = np.random.choice(len(normal_2d), n_plot, replace=False)
    ax.scatter(
        normal_2d[idx, 0], normal_2d[idx, 1],
        c=C["normal"], alpha=0.15, s=8, label=f"Train normal (n={n_plot:,})",
    )
    ax.scatter(
        legit_2d[:, 0], legit_2d[:, 1],
        c=C["train"], alpha=0.4, s=15,
        label=f"Val légitime (n={len(legit_2d):,})",
    )
    ax.scatter(
        fraud_2d[:, 0], fraud_2d[:, 1],
        c=C["fraud"], alpha=0.9, s=80, marker="x", linewidths=2,
        label=f"Val fraude (n={len(fraud_2d)})",
        zorder=5,
    )

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var.)")
    ax.set_title(
        "Espace Latent (Bottleneck) — PCA 2D\n"
        "Séparabilité Normal vs Fraude",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig
