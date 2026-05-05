# -*- coding: utf-8 -*-
"""
src/explainability/lime_explainer.py
======================================
Module LIME pour l'explicabilite locale des modeles de detection de fraudes.

Fix v2 :
- IndexError local_pred[label] : gestion robuste de l'index selon la taille de local_pred
- Retrait des caracteres tiret em (U+2014) dans les f-strings pour compatibilite Windows
- predict_fn wrapping automatique pour les modeles avec wrapper custom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("lime non installe. Lance : pip install lime")


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

CATEGORICAL_FEATURES = [
    "high_risk_hour", "is_transfer_or_cashout", "dest_zero_balance",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
]

PALETTE = {
    "positive": "#E74C3C",
    "negative": "#2980B9",
}


def _make_predict_fn(model, mode: str):
    """
    Cree une fonction de prediction compatible LIME depuis n'importe quel modele.
    Gere les wrappers custom qui ont une methode predict_proba ou predict_score.

    Parameters
    ----------
    model : objet sklearn-like ou wrapper custom
    mode : 'classification' ou 'regression'

    Returns
    -------
    callable : X (np.ndarray) -> probas ou scores
    """
    # Wrapper custom avec predict_score (AutoEncoder)
    if hasattr(model, 'predict_score') and mode == 'regression':
        return lambda X: model.predict_score(X.astype(np.float32))

    # Modele sklearn ou wrapper avec predict_proba
    if hasattr(model, 'predict_proba'):
        return model.predict_proba

    # Fallback regression
    if hasattr(model, 'predict'):
        return model.predict

    raise ValueError(f"Le modele {type(model).__name__} n'a pas de methode compatible LIME.")


class LIMEExplainer:
    """
    Wrapper LIME pour l'explication locale de transactions frauduleuses.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, n_features)
        Donnees d'entrainement.
    feature_names : list[str]
        Noms des features.
    mode : str
        'classification' pour LR/RF, 'regression' pour AE.
    random_state : int
    """

    def __init__(
        self,
        X_train      : np.ndarray,
        feature_names: list,
        mode         : str = "classification",
        random_state : int = 42,
    ):
        if not LIME_AVAILABLE:
            raise ImportError("lime non installe. Lance : pip install lime")

        self.feature_names  = feature_names
        self.feature_labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
        self.mode           = mode

        cat_idx = [i for i, f in enumerate(feature_names) if f in CATEGORICAL_FEATURES]

        self.explainer = LimeTabularExplainer(
            training_data         = X_train,
            feature_names         = feature_names,
            categorical_features  = cat_idx,
            mode                  = mode,
            random_state          = random_state,
            discretize_continuous = True,
        )
        print(f"  LIMEExplainer cree (mode={mode}, {len(cat_idx)} features categorielles)")

    # ── Explication d'une transaction ----------------------------------------
    def explain_instance(
        self,
        transaction  : np.ndarray,
        predict_fn   = None,
        model        = None,
        num_features : int = 10,
        num_samples  : int = 1000,
        label        : int = 1,
    ) -> dict:
        """
        Genere une explication LIME pour une transaction.

        Parameters
        ----------
        transaction : np.ndarray, shape (n_features,)
        predict_fn : callable, optionnel
            Fonction de prediction (X -> probas ou scores).
            Si None, utilise 'model' pour construire automatiquement.
        model : objet, optionnel
            Modele depuis lequel construire predict_fn automatiquement.
        num_features : int
        num_samples : int
        label : int
            Classe a expliquer (1=fraude). Ignore en mode regression.

        Returns
        -------
        dict : feature_rules, weights, intercept, local_pred, df
        """
        # Construction automatique de predict_fn si non fournie
        if predict_fn is None:
            if model is None:
                raise ValueError("Fournir soit predict_fn soit model.")
            predict_fn = _make_predict_fn(model, self.mode)

        exp = self.explainer.explain_instance(
            data_row     = transaction,
            predict_fn   = predict_fn,
            num_features = num_features,
            num_samples  = num_samples,
        )

        if self.mode == "classification":
            # ── Gestion robuste de l'index local_pred ────────────────────────
            # LIME peut retourner local_pred de taille 1 ou 2 selon les classes
            # presentes dans les perturbations locales.
            available_labels = list(exp.local_pred.keys()) if hasattr(exp.local_pred, 'keys') \
                               else list(range(len(exp.local_pred)))

            if label in available_labels:
                use_label = label
            else:
                # Si label=1 n'est pas disponible, prendre le dernier label
                use_label = available_labels[-1]

            pairs      = exp.as_list(label=use_label)
            intercept  = exp.intercept.get(use_label, 0.0) \
                         if hasattr(exp.intercept, 'get') else float(exp.intercept[0])
            # local_pred peut etre un array ou un dict
            if hasattr(exp.local_pred, 'get'):
                local_pred = exp.local_pred.get(use_label, 0.0)
            elif len(exp.local_pred) > use_label:
                local_pred = float(exp.local_pred[use_label])
            else:
                local_pred = float(exp.local_pred[-1])

        else:
            # mode regression
            pairs = exp.as_list()
            if hasattr(exp.intercept, '__len__'):
                intercept = float(exp.intercept[1]) if len(exp.intercept) > 1 \
                            else float(exp.intercept[0])
            else:
                intercept = float(exp.intercept)

            if hasattr(exp.local_pred, '__len__'):
                local_pred = float(exp.local_pred[1]) if len(exp.local_pred) > 1 \
                             else float(exp.local_pred[0])
            else:
                local_pred = float(exp.local_pred)

        feat_names = [p[0] for p in pairs]
        weights    = [p[1] for p in pairs]

        df = pd.DataFrame({
            "feature_rule": feat_names,
            "weight"      : weights,
        }).sort_values("weight", key=abs, ascending=False)

        return {
            "feature_rules": feat_names,
            "weights"      : weights,
            "intercept"    : float(intercept),
            "local_pred"   : float(local_pred),
            "df"           : df,
        }

    # ── Visualisation waterfall LIME -----------------------------------------
    def plot_waterfall(
        self,
        lime_result : dict,
        model_name  : str,
        title_extra : str   = "",
        figsize     : tuple = (11, 6),
        save_path   : str   = None,
    ) -> plt.Figure:
        df         = lime_result["df"]
        local_pred = lime_result["local_pred"]

        colors = [PALETTE["positive"] if w > 0 else PALETTE["negative"]
                  for w in df["weight"]]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df["feature_rule"][::-1], df["weight"][::-1],
                       color=colors[::-1], edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)

        title = f"LIME - {model_name} | Prediction locale : {local_pred:.4f}"
        if title_extra:
            title += f"\n{title_extra}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Poids LIME (contribution locale au score de fraude)", fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)

        pos_patch = mpatches.Patch(color=PALETTE["positive"], label="Hausse risque")
        neg_patch = mpatches.Patch(color=PALETTE["negative"], label="Baisse risque")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardee : {save_path}")
        plt.show()
        return fig

    # ── SHAP vs LIME cote-a-cote --------------------------------------------
    @staticmethod
    def plot_shap_vs_lime(
        shap_df    : pd.DataFrame,
        lime_result: dict,
        model_name : str,
        top_k      : int   = 8,
        figsize    : tuple = (14, 6),
        save_path  : str   = None,
    ) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(
            f"SHAP (global) vs LIME (local) - {model_name}",
            fontsize=13, fontweight='bold'
        )

        # SHAP
        df_shap   = shap_df.head(top_k)
        colors_s  = [PALETTE["positive"] if v > 0 else PALETTE["negative"]
                     for v in df_shap["shap_value"]]
        bars1 = ax1.barh(df_shap["feature_label"][::-1], df_shap["shap_value"][::-1],
                         color=colors_s[::-1], edgecolor="white")
        ax1.bar_label(bars1, fmt="%.4f", fontsize=8, padding=2)
        ax1.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax1.set_title("SHAP (transaction specifique)", fontsize=11, color=PALETTE["positive"])
        ax1.set_xlabel("Valeur SHAP")
        ax1.spines[['top', 'right']].set_visible(False)

        # LIME
        df_lime   = lime_result["df"].head(top_k)
        colors_l  = [PALETTE["positive"] if w > 0 else PALETTE["negative"]
                     for w in df_lime["weight"]]
        bars2 = ax2.barh(df_lime["feature_rule"][::-1], df_lime["weight"][::-1],
                         color=colors_l[::-1], edgecolor="white")
        ax2.bar_label(bars2, fmt="%.4f", fontsize=8, padding=2)
        ax2.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax2.set_title(
            f"LIME (local, pred={lime_result['local_pred']:.3f})",
            fontsize=11, color=PALETTE["negative"]
        )
        ax2.set_xlabel("Poids LIME")
        ax2.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardee : {save_path}")
        plt.show()
        return fig
