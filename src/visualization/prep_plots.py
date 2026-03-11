"""
prep_plots.py
=============
Visualisations produites dans ``02_data_preparation.ipynb``.

Figures générées :
  08_log_transform.png   : distribution amount avant/après log1p
  09_split.png           : taille et taux de fraude par split
  10_scaling.png         : effet du StandardScaler sur les 6 colonnes continues
  11_smote.png           : distribution des classes avant/après SMOTE
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams["mathtext.fontset"] = "dejavusans"

# Palette cohérente avec le notebook 01
COLORS = {
    "train":     "#2196F3",
    "val":       "#4CAF50",
    "test":      "#FF9800",
    "normal":    "#78909C",
    "fraud":     "#F44336",
    "before":    "#78909C",
    "after":     "#2196F3",
}


# ---------------------------------------------------------------------------
# Figure 08 — Transformation log1p
# ---------------------------------------------------------------------------

def plot_log_transform(
    amount_raw: pd.Series,
    log_amount: pd.Series,
    skew_before: float,
    skew_after: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Deux histogrammes : distribution brute vs log1p.

    Args:
        amount_raw:   Série ``amount`` originale (avant log1p).
        log_amount:   Série ``log_amount`` transformée.
        skew_before:  Skewness avant (EDA : 30.80).
        skew_after:   Skewness après.
        save_path:    Chemin de sauvegarde (None = pas de sauvegarde).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(amount_raw,  bins=60, color=COLORS["before"], edgecolor="white", alpha=0.85)
    axes[0].set_title(f"amount brut  (skew = {skew_before:.1f})")
    axes[0].set_xlabel("Montant")
    axes[0].set_ylabel("Fréquence")
    axes[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x:,.0f}")
    )

    axes[1].hist(log_amount, bins=60, color=COLORS["after"],  edgecolor="white", alpha=0.85)
    axes[1].set_title(f"log_amount  (skew = {skew_after:.2f})")
    axes[1].set_xlabel("log(1 + montant)")
    axes[1].set_ylabel("Fréquence")

    plt.suptitle("Transformation log1p — amount", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")

    return fig


# ---------------------------------------------------------------------------
# Figure 09 — Split Train / Val / Test
# ---------------------------------------------------------------------------

def plot_split(
    sizes: list[int],
    frauds: list[int],
    global_fraud_rate: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Deux barplots : taille des splits + taux de fraude par split.

    Args:
        sizes:             [n_train, n_val, n_test]
        frauds:            [n_fraud_train, n_fraud_val, n_fraud_test]
        global_fraud_rate: Taux global de fraude (float entre 0 et 1).
        save_path:         Chemin de sauvegarde.
    """
    names  = ["Train", "Val", "Test"]
    colors = [COLORS["train"], COLORS["val"], COLORS["test"]]
    rates  = [f / s * 100 for f, s in zip(frauds, sizes)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Taille des splits
    axes[0].bar(names, sizes, color=colors, edgecolor="white")
    axes[0].set_title("Taille des splits")
    axes[0].set_ylabel("Transactions")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    for i, s in enumerate(sizes):
        axes[0].text(i, s + max(sizes) * 0.01, f"{s:,}",
                     ha="center", fontsize=9, fontweight="bold")

    # Taux de fraude
    axes[1].bar(names, rates, color=colors, edgecolor="white")
    axes[1].axhline(
        global_fraud_rate * 100, color="red", linestyle="--", lw=1.2,
        label=f"Taux global ({global_fraud_rate*100:.4f}%)",
    )
    axes[1].set_title("Taux de fraude par split (stratification)")
    axes[1].set_ylabel("Taux fraude (%)")
    axes[1].legend(fontsize=9)
    for i, (f, r) in enumerate(zip(frauds, rates)):
        axes[1].text(i, r + max(rates) * 0.05, f"{f} fraudes",
                     ha="center", fontsize=9)

    plt.suptitle("Split Train / Val / Test — stratifié sur isFraud",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")

    return fig


# ---------------------------------------------------------------------------
# Figure 10 — Effet du StandardScaler
# ---------------------------------------------------------------------------

def plot_scaling_effect(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    scale_cols: list[str],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Grille 2×3 : histogrammes avant/après StandardScaler pour chaque
    colonne continue (6 colonnes = SCALE_COLS).

    Args:
        X_before:    DataFrame non scalé (X_train).
        X_after:     DataFrame scalé (X_train_sc).
        scale_cols:  Liste des 6 colonnes continues.
        save_path:   Chemin de sauvegarde.
    """
    n = len(scale_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(scale_cols):
        axes[i].hist(X_before[col], bins=50, alpha=0.5,
                     color=COLORS["before"], label="Avant")
        axes[i].hist(X_after[col],  bins=50, alpha=0.6,
                     color=COLORS["after"],  label="Après")
        axes[i].set_title(col)
        axes[i].legend(fontsize=8)
        axes[i].set_ylabel("Fréquence")

    # Masquer les axes vides
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Effet du StandardScaler sur X_train",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")

    return fig


# ---------------------------------------------------------------------------
# Figure 11 — Effet du SMOTE
# ---------------------------------------------------------------------------

def plot_smote_effect(
    n_normal_before: int,
    n_fraud_before: int,
    n_normal_after: int,
    n_fraud_after: int,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Deux barplots en échelle log : distribution classes avant/après SMOTE.

    Args:
        n_normal_before: Non-fraudes dans train original.
        n_fraud_before:  Fraudes dans train original.
        n_normal_after:  Non-fraudes après SMOTE.
        n_fraud_after:   Fraudes après SMOTE (réelles + synthétiques).
        save_path:       Chemin de sauvegarde.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = [
        (axes[0], [n_normal_before, n_fraud_before],
         f"Avant SMOTE  (1 : {n_normal_before // n_fraud_before})"),
        (axes[1], [n_normal_after,  n_fraud_after],
         f"Après SMOTE  (1 : {int(n_normal_after / n_fraud_after)})"),
    ]

    for ax, counts, title in datasets:
        ax.bar(
            ["Non-fraude", "Fraude"], counts,
            color=[COLORS["train"], COLORS["fraud"]], edgecolor="white",
        )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("Nombre (log)")
        for i, v in enumerate(counts):
            ax.text(i, v * 1.5, f"{v:,}", ha="center",
                    fontweight="bold", fontsize=10)

    plt.suptitle(
        "Effet de SMOTE — train uniquement (val/test inchangés)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")

    return fig


# ---------------------------------------------------------------------------
# Convenience : générer toutes les figures du notebook 02
# ---------------------------------------------------------------------------

def plot_all_preparation_figures(
    amount_raw: pd.Series,
    log_amount: pd.Series,
    skew_before: float,
    skew_after: float,
    split_sizes: list[int],
    split_frauds: list[int],
    global_fraud_rate: float,
    X_train_before: pd.DataFrame,
    X_train_after: pd.DataFrame,
    scale_cols: list[str],
    n_normal_before: int,
    n_fraud_before: int,
    n_normal_after: int,
    n_fraud_after: int,
    figures_dir: Path,
) -> None:
    """
    Génère et sauvegarde les 4 figures de ``02_data_preparation.ipynb``.

    Nommage des fichiers conforme au notebook :
        08_log_transform.png
        09_split.png
        10_scaling.png
        11_smote.png
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_log_transform(
        amount_raw, log_amount, skew_before, skew_after,
        save_path=figures_dir / "08_log_transform.png",
    )
    plt.close("all")

    plot_split(
        split_sizes, split_frauds, global_fraud_rate,
        save_path=figures_dir / "09_split.png",
    )
    plt.close("all")

    plot_scaling_effect(
        X_train_before, X_train_after, scale_cols,
        save_path=figures_dir / "10_scaling.png",
    )
    plt.close("all")

    plot_smote_effect(
        n_normal_before, n_fraud_before,
        n_normal_after,  n_fraud_after,
        save_path=figures_dir / "11_smote.png",
    )
    plt.close("all")
