# -*- coding: utf-8 -*-
"""
src/explainability/shap_explainer.py
--------------------------------------
Calcul et visualisation des valeurs SHAP pour les modèles de détection de fraudes.

Modèles supportés :
  - RandomForestClassifier  → TreeExplainer  (rapide, natif sklearn)
  - XGBoostClassifier       → TreeExplainer  (natif XGB, plus rapide que RF)
  - LogisticRegression      → LinearExplainer
  - AutoEncoder PyTorch     → KernelExplainer (modèle-agnostique, limiter à 50-100 lignes)

Usage :
    explainer = SHAPExplainer(feature_names=FEATURE_COLS, background_samples=100)
    explainer.fit_tree('RF', rf_model, X_background)
    shap_vals = explainer.compute('RF', X_test)
    explainer.plot_bar_global('RF', save_path='figures/shap_rf.png')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP non installé. Lance : pip install shap")


# ── Constantes ───────────────────────────────────────────────────────────────

# Libellés lisibles pour les 14 features du projet (affichage dans les graphiques)
FEATURE_LABELS = {
    "step"                  : "Etape temporelle",
    "hour"                  : "Heure",
    "day"                   : "Jour",
    "week"                  : "Semaine",
    "high_risk_hour"        : "Heure a risque",
    "is_transfer_or_cashout": "TRANSFER/CASH_OUT",
    "balance_diff_orig"     : "Diff. solde emetteur",
    "dest_zero_balance"     : "Destinataire solde nul",
    "type_CASH_IN"          : "Type CASH_IN",
    "type_CASH_OUT"         : "Type CASH_OUT",
    "type_DEBIT"            : "Type DEBIT",
    "type_PAYMENT"          : "Type PAYMENT",
    "type_TRANSFER"         : "Type TRANSFER",
    "log_amount"            : "Montant (log)",
}

PALETTE = {
    "positive" : "#E74C3C",   # rouge — contribution qui augmente le risque de fraude
    "negative" : "#2980B9",   # bleu  — contribution qui réduit le risque de fraude
    "neutral"  : "#95A5A6",
}


# ── Utilitaire ────────────────────────────────────────────────────────────────

def _extract_raw_estimator(model):
    """
    Extrait l'estimateur sklearn brut depuis un wrapper ou pipeline.

    Nécessaire car TreeExplainer et LinearExplainer attendent un estimateur
    sklearn natif, pas un objet wrapper personnalisé.

    Gère :
      - sklearn Pipeline  : model[-1]
      - Wrappers custom   : attributs model / estimator / classifier / clf
      - Objet direct      : retourné tel quel
    """
    import sklearn.pipeline

    if isinstance(model, sklearn.pipeline.Pipeline):
        return model[-1]

    for attr in ('model', 'estimator', 'classifier', 'clf', '_model'):
        candidate = getattr(model, attr, None)
        if candidate is not None and (
            hasattr(candidate, 'predict_proba') or hasattr(candidate, 'estimators_')
        ):
            return candidate

    return model


# ── Classe principale ─────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Calcule et visualise les valeurs SHAP pour tous les modèles du projet.

    Parameters
    ----------
    feature_names : list[str]
        Noms des 14 features dans l'ordre.
    background_samples : int
        Taille du jeu de fond pour KernelExplainer (50 max recommandé).
    """

    def __init__(self, feature_names: list, background_samples: int = 100):
        if not SHAP_AVAILABLE:
            raise ImportError("shap non installé. Lance : pip install shap")
        self.feature_names      = feature_names
        self.feature_labels     = [FEATURE_LABELS.get(f, f) for f in feature_names]
        self.background_samples = background_samples
        self.explainers         = {}    # model_name → explainer SHAP
        self.shap_values        = {}    # model_name → array (N, n_features)
        self._explainer_types   = {}    # model_name → 'tree' | 'linear' | 'kernel'

    # ── Création des explainers ───────────────────────────────────────────────

    def fit_tree(self, model_name: str, model, X_background: np.ndarray):
        """
        TreeExplainer pour Random Forest.

        Extrait automatiquement l'estimateur sklearn brut depuis les wrappers
        personnalisés avant de passer au TreeExplainer.
        """
        raw = _extract_raw_estimator(model)
        print(f"  TreeExplainer {model_name} ({type(raw).__name__})...", end=" ", flush=True)
        explainer = shap.TreeExplainer(raw)
        self.explainers[model_name]       = explainer
        self._explainer_types[model_name] = 'tree'
        print("OK")
        return explainer

    def fit_xgb(self, model_name: str, xgb_model):
        """
        TreeExplainer pour XGBoost natif.

        XGBoost intègre nativement SHAP — pas besoin d'extraire un estimateur brut,
        et le calcul est plus rapide qu'avec un RandomForest sklearn.
        """
        print(f"  TreeExplainer XGBoost {model_name}...", end=" ", flush=True)
        explainer = shap.TreeExplainer(xgb_model)
        self.explainers[model_name]       = explainer
        self._explainer_types[model_name] = 'tree'
        print("OK")
        return explainer

    def fit_linear(self, model_name: str, model, X_background: np.ndarray):
        """
        LinearExplainer pour Logistic Regression.

        Utilise les coefficients du modèle linéaire : exact et instantané.
        X_background sert à estimer la distribution des features (corrélations).
        """
        raw = _extract_raw_estimator(model)
        print(f"  LinearExplainer {model_name}...", end=" ", flush=True)
        explainer = shap.LinearExplainer(raw, X_background)
        self.explainers[model_name]       = explainer
        self._explainer_types[model_name] = 'linear'
        print("OK")
        return explainer

    def fit_kernel(self, model_name: str, predict_fn, X_background: np.ndarray):
        """
        KernelExplainer pour l'AutoEncoder PyTorch.

        ⚠️  DeepExplainer est réservé à Keras/TensorFlow — utiliser KernelExplainer
            pour les modèles PyTorch.
        ⚠️  KernelExplainer est modèle-agnostique mais O(N²) : limiter max_samples
            à 50-100 lignes dans compute().

        predict_fn doit retourner un score scalaire par ligne (ex : ae.predict_score).
        """
        n_bg = min(self.background_samples, 50)
        print(f"  KernelExplainer {model_name} (background={n_bg})...", end=" ", flush=True)
        bg = shap.sample(X_background, n_bg)

        # Forcer float64 : SHAP peut lever des erreurs de type avec float32 PyTorch
        def _predict(X):
            return predict_fn(X.astype(np.float32)).astype(np.float64)

        explainer = shap.KernelExplainer(_predict, bg)
        self.explainers[model_name]       = explainer
        self._explainer_types[model_name] = 'kernel'
        print("OK")
        return explainer

    # ── Calcul des valeurs SHAP ───────────────────────────────────────────────

    @staticmethod
    def _parse_shap_output(raw, explainer_type: str) -> np.ndarray:
        """
        Normalise la sortie de shap_values() vers un array 2D (N, n_features).

        SHAP retourne des formats différents selon l'explainer et la version :
          - TreeExplainer RF  : list[class0_arr, class1_arr] → on prend [1] (fraude)
          - LinearExplainer   : array(N, n_features) directement
          - KernelExplainer   : array(N, n_features) ou list à un seul élément
          - 3D (multi-outputs) : (N, n_features, n_outputs) → moyenne sur l'axe outputs
        """
        if isinstance(raw, list):
            # Binaire : list[class0, class1] → prendre la classe fraude (index 1)
            vals = raw[1] if len(raw) > 1 else raw[0]
        else:
            vals = raw

        if hasattr(vals, 'ndim') and vals.ndim == 3:
            vals = vals.mean(axis=2)   # (N, features, outputs) → (N, features)

        if hasattr(vals, 'ndim') and vals.ndim == 1:
            vals = vals.reshape(1, -1)   # une seule transaction → (1, features)

        return np.array(vals, dtype=np.float64)

    def compute(
        self,
        model_name      : str,
        X               : np.ndarray,
        max_samples     : int  = 500,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """
        Calcule les valeurs SHAP sur un sous-ensemble de X et les stocke en cache.

        Parameters
        ----------
        max_samples : int
            Nombre de lignes à traiter. Automatiquement plafonné à 100 pour
            KernelExplainer (complexité quadratique).
        check_additivity : bool
            Vérifie que sum(SHAP) ≈ f(x) - E[f(x)]. Désactivé par défaut car coûteux.

        Returns
        -------
        np.ndarray, shape (N, n_features)
        """
        if model_name not in self.explainers:
            raise ValueError(f"Explainer '{model_name}' non créé.")

        exp_type = self._explainer_types.get(model_name, 'unknown')

        # KernelExplainer : complexité O(N × nsamples²) — plafonner strictement
        if exp_type == 'kernel':
            max_samples = min(max_samples, 100)
            print(f"  ⚠️ KernelExplainer : max_samples plafonné à {max_samples}")

        X_sub     = X[:max_samples].astype(np.float32)
        explainer = self.explainers[model_name]
        print(f"  Calcul SHAP {model_name} ({len(X_sub)} lignes)...", end=" ", flush=True)

        if exp_type == 'tree':
            try:
                raw = explainer.shap_values(X_sub, check_additivity=check_additivity)
            except TypeError:
                # Versions SHAP antérieures ne supportent pas check_additivity
                raw = explainer.shap_values(X_sub)
        elif exp_type == 'kernel':
            # nsamples=200 : bon compromis vitesse/précision pour 14 features
            raw = explainer.shap_values(X_sub, nsamples=200)
        else:
            raw = explainer.shap_values(X_sub)

        vals     = self._parse_shap_output(raw, exp_type)
        mean_abs = float(np.abs(vals).mean())
        self.shap_values[model_name] = vals
        print(f"OK  shape={vals.shape}  |SHAP| moyen={mean_abs:.4f}")

        if mean_abs < 1e-10:
            print(f"  ⚠️ Valeurs SHAP nulles pour '{model_name}' — vérifier le modèle.")

        return vals

    # ── Explication locale ────────────────────────────────────────────────────

    def explain_single(self, model_name: str, transaction: np.ndarray) -> pd.DataFrame:
        """
        Calcule les valeurs SHAP pour une seule transaction.

        Returns
        -------
        pd.DataFrame
            Colonnes : feature, feature_label, shap_value, feature_value.
            Trié par |shap_value| décroissant.
        """
        if model_name not in self.explainers:
            raise ValueError(f"Explainer '{model_name}' non créé.")

        explainer = self.explainers[model_name]
        exp_type  = self._explainer_types.get(model_name, 'unknown')
        tx        = transaction.reshape(1, -1).astype(np.float32)

        if exp_type == 'tree':
            try:
                raw = explainer.shap_values(tx, check_additivity=False)
            except TypeError:
                raw = explainer.shap_values(tx)
        else:
            raw = explainer.shap_values(tx)

        vals = self._parse_shap_output(raw, exp_type)[0]   # shape (n_features,)

        return pd.DataFrame({
            'feature'      : self.feature_names,
            'feature_label': self.feature_labels,
            'shap_value'   : vals,
            'feature_value': transaction,
        }).sort_values('shap_value', key=abs, ascending=False)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_bar_global(
        self,
        model_name : str,
        top_k      : int   = 10,
        figsize    : tuple = (9, 6),
        save_path  : str   = None,
    ) -> plt.Figure:
        """Graphique en barres des importances SHAP globales (|SHAP| moyen sur tout le jeu)."""
        vals = self.shap_values.get(model_name)
        if vals is None:
            raise ValueError(f"Pas de valeurs SHAP pour '{model_name}'. Appeler compute() d'abord.")

        mean_abs = np.abs(vals).mean(axis=0)
        order    = np.argsort(mean_abs)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(
            [self.feature_labels[i] for i in order][::-1],
            mean_abs[order][::-1],
            color=PALETTE["positive"], alpha=0.85, edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
        ax.set_xlabel("Importance SHAP moyenne (|valeur|)", fontsize=11)
        ax.set_title(f"Importances SHAP — {model_name}", fontsize=13, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, mean_abs[order].max() * 1.18)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardée : {save_path}")
        plt.show()
        return fig

    def plot_beeswarm(
        self,
        model_name  : str,
        max_display : int   = 14,
        figsize     : tuple = (10, 7),
        save_path   : str   = None,
    ) -> plt.Figure:
        """
        Beeswarm SHAP : distribution des valeurs par feature sur l'ensemble du jeu.

        Chaque point représente une transaction ; la couleur indique la valeur de la feature
        (rouge = haute, bleu = basse), la position X indique l'impact sur la prédiction.
        """
        vals = self.shap_values.get(model_name)
        if vals is None:
            raise ValueError(f"Pas de valeurs SHAP pour '{model_name}'. Appeler compute() d'abord.")

        fig, _ = plt.subplots(figsize=figsize)
        shap.summary_plot(vals, feature_names=self.feature_labels,
                          max_display=max_display, show=False, plot_size=None)
        plt.title(f"SHAP Beeswarm — {model_name}", fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardée : {save_path}")
        plt.show()
        return fig

    def plot_waterfall_single(
        self,
        model_name  : str,
        transaction : np.ndarray,
        top_k       : int   = 10,
        figsize     : tuple = (10, 6),
        save_path   : str   = None,
        title_extra : str   = "",
    ) -> plt.Figure:
        """
        Waterfall SHAP pour une transaction : contributions individuelles de chaque feature.

        Rouge → feature qui pousse vers la fraude.
        Bleu  → feature qui atténue le score de fraude.
        """
        df     = self.explain_single(model_name, transaction)
        df_top = df.head(top_k)
        colors = [PALETTE["positive"] if v > 0 else PALETTE["negative"]
                  for v in df_top["shap_value"]]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df_top["feature_label"][::-1], df_top["shap_value"][::-1],
                       color=colors[::-1], edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
        ax.set_xlabel("Valeur SHAP (contribution au score de fraude)", fontsize=11)
        title = f"SHAP Waterfall — {model_name}"
        if title_extra:
            title += f"\n{title_extra}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(color=PALETTE["positive"], label="Hausse risque fraude"),
            mpatches.Patch(color=PALETTE["negative"], label="Baisse risque fraude"),
        ], fontsize=9, loc="lower right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardée : {save_path}")
        plt.show()
        return fig

    def plot_comparison_all_models(
        self,
        figsize   : tuple = (14, 6),
        save_path : str   = None,
        top_k     : int   = 8,
    ) -> plt.Figure:
        """
        Comparaison côte-à-côte des importances SHAP pour tous les modèles calculés.

        Permet d'identifier si les mêmes features dominent selon les algorithmes,
        ce qui renforce la robustesse des explications.
        """
        models_computed = list(self.shap_values.keys())
        if not models_computed:
            raise ValueError("Aucune valeur SHAP calculée. Appeler compute() d'abord.")

        palette = ["#E74C3C", "#2980B9", "#8E44AD", "#27AE60", "#F39C12"]
        fig, axes = plt.subplots(1, len(models_computed), figsize=figsize, sharey=False)
        if len(models_computed) == 1:
            axes = [axes]
        fig.suptitle("Comparaison SHAP — Tous les modèles", fontsize=14, fontweight='bold')

        for ax, model_name, color in zip(axes, models_computed, palette):
            mean_abs = np.abs(self.shap_values[model_name]).mean(axis=0)
            order    = np.argsort(mean_abs)[::-1][:top_k]
            labels   = [self.feature_labels[i] for i in order][::-1]
            values   = mean_abs[order][::-1]

            bars = ax.barh(labels, values, color=color, alpha=0.82, edgecolor="white")
            ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
            ax.set_title(model_name, fontsize=11, fontweight='bold', color=color)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel("|SHAP| moyen", fontsize=9)
            ax.set_xlim(0, max(values.max() * 1.22, 0.001))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardée : {save_path}")
        plt.show()
        return fig

    # ── Export ────────────────────────────────────────────────────────────────

    def summary_dict(self) -> dict:
        """
        Retourne un résumé sérialisable JSON des importances SHAP par modèle.

        Format : {model_name: [{"feature": str, "mean_abs_shap": float}, ...]}
        Trié par importance décroissante pour faciliter la lecture dans les rapports.
        """
        result = {}
        for model_name, vals in self.shap_values.items():
            mean_abs = np.abs(vals).mean(axis=0)
            order    = np.argsort(mean_abs)[::-1]
            result[model_name] = [
                {"feature": self.feature_names[i], "mean_abs_shap": float(mean_abs[i])}
                for i in order
            ]
        return result
