"""
model_plots.py
==============
Visualisations pour ``03_baseline_models.ipynb``.

Figures générées :
  12_pr_curves.png          : courbes Precision-Recall des 4 modèles
  13_roc_curves.png         : courbes ROC
  14_confusion_matrices.png : matrices de confusion (2×2)
  15_feature_importances.png: importances RF (top 14)
  16_threshold_analysis.png : Recall/Precision vs seuil pour chaque modèle
  17_model_comparison.png   : tableau récapitulatif des métriques
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc

try:
    from src.utils.baseline_config import load_baseline_metrics
except ModuleNotFoundError:
    # Allow running this file directly: python src/visualization/prep_plots.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.utils.baseline_config import load_baseline_metrics

plt.rcParams["mathtext.fontset"] = "dejavusans"

# Palette cohérente pour les 4 modèles baseline
MODEL_COLORS = {
    "LR_balanced":    "#2196F3",   # bleu
    "LR_smote":       "#64B5F6",   # bleu clair
    "RF_balanced":    "#F44336",   # rouge
    "RF_smote":       "#EF9A9A",   # rouge clair
    "baseline":       "#9E9E9E",   # gris — règle métier isFlaggedFraud
}

# Ligne de référence baseline (chargée depuis outputs/reports si disponible)
_BASELINE = load_baseline_metrics(report_path=None)
BASELINE_RECALL = _BASELINE["recall"]
BASELINE_PRECISION = _BASELINE["precision"]
BASELINE_F1 = _BASELINE["f1"]


# ---------------------------------------------------------------------------
# Figure 12 — Courbes Precision-Recall
# ---------------------------------------------------------------------------

def plot_pr_curves(
    models_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Courbes Precision-Recall pour plusieurs modèles.

    Args:
        models_scores: {nom_modele: (y_true, y_score)}
        save_path:     Chemin de sauvegarde.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, (y_true, y_score) in models_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
        color  = MODEL_COLORS.get(name, "#607D8B")
        ax.plot(rec, prec, lw=2, color=color,
                label=f"{name}  (PR-AUC={pr_auc:.4f})")

    # Point baseline isFlaggedFraud
    ax.scatter(
        BASELINE_RECALL, BASELINE_PRECISION,
        marker="*", s=200, color=MODEL_COLORS["baseline"], zorder=5,
        label=f"Baseline isFlaggedFraud  (R={BASELINE_RECALL}, P={BASELINE_PRECISION})",
    )

    # Ligne de chance (taux de fraude global)
    if models_scores:
        y_true_any = next(iter(models_scores.values()))[0]
        fraud_rate = y_true_any.mean()
        ax.axhline(fraud_rate, color="#BDBDBD", linestyle="--", lw=1,
                   label=f"Chance ({fraud_rate:.4f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Courbes Precision-Recall — Modèles Baseline",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 13 — Courbes ROC
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Courbes ROC. Moins prioritaires que PR sous fort déséquilibre,
    mais utiles pour comparer globalement les modèles.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_true, y_score) in models_scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        color   = MODEL_COLORS.get(name, "#607D8B")
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name}  (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC=0.50)")
    ax.set_xlabel("Taux Faux Positifs (FPR)", fontsize=11)
    ax.set_ylabel("Taux Vrais Positifs (TPR / Recall)", fontsize=11)
    ax.set_title("Courbes ROC — Modèles Baseline",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 14 — Matrices de confusion
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    models_cm: dict[str, dict],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Grille de matrices de confusion (une par modèle).

    Args:
        models_cm: {nom_modele: metrics_dict}
                   metrics_dict doit contenir tp, fp, fn, tn.
        save_path: Chemin de sauvegarde.
    """
    n = len(models_cm)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, m) in enumerate(models_cm.items()):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]

        tp, fp = m["tp"], m["fp"]
        fn, tn = m["fn"], m["tn"]
        cm_arr = np.array([[tn, fp], [fn, tp]])

        im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{name}\nRecall={m['recall']:.4f}  F1={m['f1']:.4f}",
                     fontsize=10, fontweight="bold")

        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = cm_arr[i, j]
                color = "white" if val > cm_arr.max() * 0.6 else "black"
                ax.text(j, i, f"{labels[i][j]}\n{val:,}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Prédit 0\n(Non-fraude)", "Prédit 1\n(Fraude)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Réel 0\n(Non-fraude)", "Réel 1\n(Fraude)"])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Masquer axes vides
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle("Matrices de Confusion — Modèles Baseline (Test set)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 15 — Feature Importances (RF)
# ---------------------------------------------------------------------------

def plot_feature_importances(
    importances_df: pd.DataFrame,
    model_name: str = "Random Forest",
    top_n: int = 14,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Barplot horizontal des feature importances.

    Args:
        importances_df: DataFrame avec colonnes 'feature' et 'importance'.
        model_name:     Nom du modèle pour le titre.
        top_n:          Nombre de features à afficher.
        save_path:      Chemin de sauvegarde.
    """
    df = importances_df.head(top_n).copy()

    # Coloration : rouge si corrélation > 0.10 (features fortes de l'EDA)
    strong = {"balance_diff_orig", "dest_zero_balance", "log_amount"}
    colors = ["#F44336" if f in strong else "#2196F3"
              for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        df["feature"][::-1], df["importance"][::-1],
        color=colors[::-1], edgecolor="white",
    )
    ax.set_xlabel("Importance (Gini)", fontsize=11)
    ax.set_title(f"Feature Importances — {model_name}",
                 fontsize=13, fontweight="bold")

    for bar, val in zip(bars, df["importance"][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    # Légende couleurs
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F44336", label="Signal fort (EDA corr > 0.10)"),
        Patch(facecolor="#2196F3", label="Autres features"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 16 — Analyse du seuil de décision
# ---------------------------------------------------------------------------

def plot_threshold_analysis(
    models_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Recall, Precision et F1 en fonction du seuil de décision.
    Aide à choisir le seuil optimal pour chaque modèle.
    """
    n = len(models_scores)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(8 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, (y_true, y_score)) in enumerate(models_scores.items()):
        r, c  = divmod(idx, ncols)
        ax    = axes[r][c]
        color = MODEL_COLORS.get(name, "#607D8B")

        thresholds = np.linspace(0.01, 0.99, 200)
        recalls, precisions, f1s = [], [], []

        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1   = (2 * prec * rec / (prec + rec)
                    if (prec + rec) > 0 else 0)
            recalls.append(rec)
            precisions.append(prec)
            f1s.append(f1)

        ax.plot(thresholds, recalls,    color="#F44336", lw=2, label="Recall")
        ax.plot(thresholds, precisions, color="#2196F3", lw=2, label="Precision")
        ax.plot(thresholds, f1s,        color="#4CAF50", lw=2, label="F1")

        # Seuil optimal F1
        best_idx = int(np.argmax(f1s))
        ax.axvline(thresholds[best_idx], color="#4CAF50",
                   linestyle="--", lw=1.2,
                   label=f"Seuil opt. F1={f1s[best_idx]:.3f}  (t={thresholds[best_idx]:.2f})")

        ax.set_xlabel("Seuil de décision", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle("Recall / Precision / F1 vs Seuil de décision",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig


# ---------------------------------------------------------------------------
# Figure 17 — Comparaison récapitulative
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_list: list[dict],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Barplots groupés : Recall, F1, PR-AUC pour tous les modèles.

    Args:
        metrics_list: Liste de dicts (sortie de compute_fraud_metrics).
        save_path:    Chemin de sauvegarde.
    """
    df = pd.DataFrame([{
        "model":    m.get("model", "?"),
        "Recall":   m.get("recall",   0),
        "F1":       m.get("f1",       0),
        "PR-AUC":   m.get("pr_auc",   0) or 0,
        "ROC-AUC":  m.get("roc_auc",  0) or 0,
    } for m in metrics_list])

    metrics_to_plot = ["Recall", "F1", "PR-AUC"]
    x   = np.arange(len(df))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    bar_colors = ["#F44336", "#4CAF50", "#2196F3"]
    for i, (metric, color) in enumerate(zip(metrics_to_plot, bar_colors)):
        bars = ax.bar(x + i * w, df[metric], w,
                      label=metric, color=color, edgecolor="white", alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + 0.005, f"{h:.3f}",
                        ha="center", fontsize=7.5, fontweight="bold")

    # Lignes de référence baseline
    ax.axhline(BASELINE_RECALL, color="#9E9E9E",
               linestyle="--", lw=1.2, label=f"Baseline Recall ({BASELINE_RECALL})")
    ax.axhline(BASELINE_F1, color="#BDBDBD",
               linestyle=":", lw=1.2, label=f"Baseline F1 ({BASELINE_F1})")

    ax.set_xticks(x + w)
    ax.set_xticklabels(df["model"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.set_title("Comparaison des Modèles Baseline — Test set",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {Path(save_path).name}")
    return fig
