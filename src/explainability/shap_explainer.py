# -*- coding: utf-8 -*-
"""
src/explainability/shap_explainer.py
======================================
Module SHAP pour l'explicabilite des modeles de detection de fraudes.
Supporte : Logistic Regression, Random Forest, AutoEncoder.

Fix v2 :
- TreeExplainer : extraction de l'estimateur RF brut depuis les wrappers
- check_additivity uniquement pour TreeExplainer
- Gestion robuste du format de retour shap_values (list vs array)
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
    print("SHAP non installe. Lance : pip install shap")


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
    "positive" : "#E74C3C",
    "negative" : "#2980B9",
    "neutral"  : "#95A5A6",
}


def _extract_raw_estimator(model):
    """
    Extrait l'estimateur sklearn brut depuis un wrapper ou pipeline.
    
    Gere :
      - sklearn Pipeline  : model[-1] ou model.steps[-1][1]
      - Wrappers custom   : model.model, model.estimator, model.classifier
      - Objet direct      : retourne tel quel
    """
    import sklearn.pipeline

    # Pipeline sklearn
    if isinstance(model, sklearn.pipeline.Pipeline):
        return model[-1]

    # Attributs courants des wrappers custom (FraudRandomForest, etc.)
    for attr in ('model', 'estimator', 'classifier', 'clf', '_model'):
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if hasattr(candidate, 'predict_proba') or hasattr(candidate, 'estimators_'):
                return candidate

    return model


class SHAPExplainer:
    """
    Calcule et visualise les valeurs SHAP pour tous les modeles du projet.

    Parameters
    ----------
    feature_names : list[str]
        Noms des 14 features dans l'ordre.
    background_samples : int
        Nombre d'echantillons de fond pour KernelExplainer et DeepExplainer.
    """

    def __init__(self, feature_names: list, background_samples: int = 100):
        if not SHAP_AVAILABLE:
            raise ImportError("shap non installe. Lance : pip install shap")
        self.feature_names      = feature_names
        self.feature_labels     = [FEATURE_LABELS.get(f, f) for f in feature_names]
        self.background_samples = background_samples
        self.explainers         = {}
        self.shap_values        = {}
        self._explainer_types   = {}   # 'tree' | 'linear' | 'deep'

    # ── TreeExplainer -- Random Forest ----------------------------------------
    def fit_tree(self, model_name: str, model, X_background: np.ndarray):
        """
        Cree un TreeExplainer pour Random Forest.
        Extrait automatiquement l'estimateur brut depuis les wrappers.
        """
        raw = _extract_raw_estimator(model)
        raw_type = type(raw).__name__
        print(f"  Creation TreeExplainer pour {model_name} (type brut: {raw_type})...",
              end=" ", flush=True)
        explainer = shap.TreeExplainer(raw)
        self.explainers[model_name]       = explainer
        self._explainer_types[model_name] = 'tree'
        print("OK")
        return explainer

    # ── LinearExplainer -- Logistic Regression --------------------------------
    def fit_linear(self, model_name: str, model, X_background: np.ndarray):
        """
        Cree un LinearExplainer pour Logistic Regression.
        """
        raw = _extract_raw_estimator(model)
        print(f"  Creation LinearExplainer pour {model_name}...", end=" ", flush=True)
        explainer = shap.LinearExplainer(raw, X_background)
        self.explainers[model_name]       = explainer
        self._explainer_types[model_name] = 'linear'
        print("OK")
        return explainer

    # ── DeepExplainer -- AutoEncoder ------------------------------------------
    def fit_deep(self, model_name: str, keras_model, X_background: np.ndarray):
        """
        Cree un DeepExplainer pour AutoEncoder Keras.
        """
        print(f"  Creation DeepExplainer pour {model_name}...", end=" ", flush=True)
        bg = X_background[:self.background_samples].astype(np.float32)
        explainer = shap.DeepExplainer(keras_model, bg)
        self.explainers[model_name]       = explainer
        self._explainer_types[model_name] = 'deep'
        print("OK")
        return explainer

    # ── Parsing robuste des shap_values ---------------------------------------
    @staticmethod
    def _parse_shap_output(raw, explainer_type: str) -> np.ndarray:
        """
        Normalise la sortie de shap_values() vers un array 2D (N, n_features).
        
        SHAP retourne des formats differents selon la version et l'explainer :
          - TreeExplainer RF  : list([class0_array, class1_array]) -> prendre [1]
          - LinearExplainer   : array(N, n_features) directement
          - DeepExplainer AE  : list([output0, output1, ...]) ou array(N, n_features, n_outputs)
        """
        # Cas liste (TreeExplainer multi-classes, DeepExplainer multi-outputs)
        if isinstance(raw, list):
            if len(raw) > 1:
                vals = raw[1]   # classe 1 = fraude pour classification binaire
            else:
                vals = raw[0]
        else:
            vals = raw

        # Cas 3D (DeepExplainer : N x n_features x n_outputs) -> moyenne sur outputs
        if vals.ndim == 3:
            vals = vals.mean(axis=2)

        # Cas 1D (une seule transaction)
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        return vals.astype(np.float64)

    # ── Calcul des valeurs SHAP -----------------------------------------------
    def compute(
        self,
        model_name      : str,
        X               : np.ndarray,
        max_samples     : int  = 500,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """
        Calcule les valeurs SHAP pour un modele.

        Parameters
        ----------
        model_name : str
        X : np.ndarray, shape (N, n_features)
        max_samples : int
        check_additivity : bool
            Seulement supporte par TreeExplainer.
        """
        if model_name not in self.explainers:
            raise ValueError(f"Explainer '{model_name}' non cree.")

        X_sub      = X[:max_samples].astype(np.float32)
        explainer  = self.explainers[model_name]
        exp_type   = self._explainer_types.get(model_name, 'unknown')

        print(f"  Calcul SHAP pour {model_name} ({len(X_sub)} lignes)...",
              end=" ", flush=True)

        # check_additivity seulement pour TreeExplainer
        if exp_type == 'tree':
            try:
                raw = explainer.shap_values(X_sub, check_additivity=check_additivity)
            except TypeError:
                raw = explainer.shap_values(X_sub)
        else:
            raw = explainer.shap_values(X_sub)

        vals = self._parse_shap_output(raw, exp_type)

        # Verification : valeurs non nulles
        mean_abs = float(np.abs(vals).mean())
        if mean_abs < 1e-10:
            print(f"\n  AVERTISSEMENT : toutes les valeurs SHAP sont nulles pour {model_name}.")
            print(f"  Verifiez que le bon estimateur est passe (pas un wrapper).")
        
        self.shap_values[model_name] = vals
        print(f"OK  shape={vals.shape}  |SHAP| moyen={mean_abs:.4f}")
        return vals

    # ── Valeurs SHAP pour une transaction ------------------------------------
    def explain_single(
        self,
        model_name : str,
        transaction: np.ndarray,
    ) -> pd.DataFrame:
        """Retourne les valeurs SHAP pour une transaction sous forme DataFrame."""
        if model_name not in self.explainers:
            raise ValueError(f"Explainer '{model_name}' non cree.")

        explainer = self.explainers[model_name]
        exp_type  = self._explainer_types.get(model_name, 'unknown')
        tx = transaction.reshape(1, -1).astype(np.float32)

        if exp_type == 'tree':
            try:
                raw = explainer.shap_values(tx, check_additivity=False)
            except TypeError:
                raw = explainer.shap_values(tx)
        else:
            raw = explainer.shap_values(tx)

        vals = self._parse_shap_output(raw, exp_type)[0]  # premiere (et seule) ligne

        df = pd.DataFrame({
            'feature'      : self.feature_names,
            'feature_label': self.feature_labels,
            'shap_value'   : vals,
            'feature_value': transaction,
        }).sort_values('shap_value', key=abs, ascending=False)

        return df

    # ── Visualisation : Bar global -------------------------------------------
    def plot_bar_global(
        self,
        model_name : str,
        top_k      : int   = 10,
        figsize    : tuple = (9, 6),
        save_path  : str   = None,
    ) -> plt.Figure:
        vals     = self.shap_values.get(model_name)
        if vals is None:
            raise ValueError(f"Pas de valeurs SHAP pour '{model_name}'.")

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
        ax.set_title(f"Importances SHAP - {model_name}", fontsize=13, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, mean_abs[order].max() * 1.18)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardee : {save_path}")
        plt.show()
        return fig

    # ── Visualisation : Beeswarm --------------------------------------------
    def plot_beeswarm(
        self,
        model_name  : str,
        max_display : int   = 14,
        figsize     : tuple = (10, 7),
        save_path   : str   = None,
    ) -> plt.Figure:
        vals = self.shap_values.get(model_name)
        if vals is None:
            raise ValueError(f"Pas de valeurs SHAP pour '{model_name}'.")

        fig, ax = plt.subplots(figsize=figsize)
        shap.summary_plot(vals, feature_names=self.feature_labels,
                          max_display=max_display, show=False, plot_size=None)
        plt.title(f"SHAP Beeswarm - {model_name}", fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardee : {save_path}")
        plt.show()
        return fig

    # ── Visualisation : Waterfall single ------------------------------------
    def plot_waterfall_single(
        self,
        model_name  : str,
        transaction : np.ndarray,
        top_k       : int   = 10,
        figsize     : tuple = (10, 6),
        save_path   : str   = None,
        title_extra : str   = "",
    ) -> plt.Figure:
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
        title = f"SHAP Waterfall - {model_name}"
        if title_extra:
            title += f"\n{title_extra}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

        pos_patch = mpatches.Patch(color=PALETTE["positive"], label="Hausse risque fraude")
        neg_patch = mpatches.Patch(color=PALETTE["negative"], label="Baisse risque fraude")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardee : {save_path}")
        plt.show()
        return fig

    # ── Visualisation : Comparaison tous modeles ----------------------------
    def plot_comparison_all_models(
        self,
        figsize   : tuple = (14, 6),
        save_path : str   = None,
        top_k     : int   = 8,
    ) -> plt.Figure:
        models_computed = list(self.shap_values.keys())
        if not models_computed:
            raise ValueError("Aucune valeur SHAP calculee.")

        n_models = len(models_computed)
        fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=False)
        if n_models == 1:
            axes = [axes]

        fig.suptitle("Comparaison SHAP - Tous les modeles", fontsize=14, fontweight='bold')
        palette = ["#E74C3C", "#2980B9", "#8E44AD", "#27AE60", "#F39C12"]

        for ax, model_name, color in zip(axes, models_computed, palette):
            vals     = self.shap_values[model_name]
            mean_abs = np.abs(vals).mean(axis=0)
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
            print(f"  Figure sauvegardee : {save_path}")
        plt.show()
        return fig

    # ── Export resume SHAP --------------------------------------------------
    def summary_dict(self) -> dict:
        result = {}
        for model_name, vals in self.shap_values.items():
            mean_abs = np.abs(vals).mean(axis=0)
            order    = np.argsort(mean_abs)[::-1]
            result[model_name] = [
                {"feature": self.feature_names[i], "mean_abs_shap": float(mean_abs[i])}
                for i in order
            ]
        return result
