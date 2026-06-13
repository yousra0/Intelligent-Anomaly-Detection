# -*- coding: utf-8 -*-
"""
src/explainability/lime_explainer.py
--------------------------------------
Explication locale des prédictions via LIME (Local Interpretable Model-agnostic Explanations).

Principe : LIME approxime localement le modèle boîte-noire par un modèle linéaire simple,
en perturbant la transaction à expliquer et en observant comment la prédiction varie.
Les poids LIME indiquent la contribution de chaque règle (feature) à la décision locale.

Modes supportés :
  - 'classification' → LR, RF, XGBoost : predict_fn retourne (N, 2) probas
  - 'regression'     → AutoEncoder      : predict_fn retourne (N,) scores d'anomalie

Usage :
    lime_exp = LIMEExplainer(X_train, feature_names=FEATURE_COLS, mode='classification')
    result   = lime_exp.explain_instance(transaction, predict_fn=model.predict_proba)
    lime_exp.plot_waterfall(result, model_name='RF')
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
    print("lime non installé. Lance : pip install lime")


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

# Features binaires ou à faible cardinalité connues de ce projet.
# Utilisé comme fallback ; préférer _get_categorical_indices() qui lit features.json.
_DEFAULT_CATEGORICAL_FEATURES = [
    "high_risk_hour", "is_transfer_or_cashout", "dest_zero_balance",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]
# Alias public gardé pour la rétrocompatibilité des notebooks
CATEGORICAL_FEATURES = _DEFAULT_CATEGORICAL_FEATURES


def _get_categorical_indices(feature_names: list[str], features_json_path=None) -> list[int]:
    """
    Retourne les indices des features catégorielles (binaires/OHE) dans feature_names.

    Priorité :
      1. features.json (binary_cols + type_cols) si le chemin est fourni et le fichier existe.
      2. Fallback sur _DEFAULT_CATEGORICAL_FEATURES hardcodées.

    Cette approche garantit que si le feature set change (ajout/renommage), les indices
    catégoriels restent corrects sans modifier le code.
    """
    if features_json_path is not None:
        try:
            import json
            from pathlib import Path
            with open(features_json_path, encoding="utf-8") as f:
                meta = json.load(f)
            cat_names = set(meta.get("binary_cols", [])) | set(meta.get("type_cols", []))
            return [i for i, f in enumerate(feature_names) if f in cat_names]
        except Exception:
            pass
    # Fallback
    return [i for i, f in enumerate(feature_names) if f in _DEFAULT_CATEGORICAL_FEATURES]

PALETTE = {
    "positive": "#E74C3C",   # rouge — contribution qui augmente le risque de fraude
    "negative": "#2980B9",   # bleu  — contribution qui réduit le risque de fraude
}


# ── Utilitaire ────────────────────────────────────────────────────────────────

def _make_predict_fn(model, mode: str):
    """
    Construit une fonction de prédiction compatible LIME depuis n'importe quel modèle.

    LIME exige une fonction X → array NumPy. Cette fonction gère les wrappers
    personnalisés (AutoEncoder avec predict_score) et les modèles sklearn standards.

    Parameters
    ----------
    model : objet sklearn-like ou wrapper custom
    mode  : 'classification' → (N, 2) | 'regression' → (N,)

    Returns
    -------
    callable
    """
    if hasattr(model, 'predict_score') and mode == 'regression':
        # AutoEncoder : retourne un score d'anomalie scalaire par transaction
        return lambda X: model.predict_score(X.astype(np.float32))

    if hasattr(model, 'predict_proba'):
        return model.predict_proba

    if hasattr(model, 'predict'):
        return model.predict

    raise ValueError(f"Le modèle {type(model).__name__} n'a pas de méthode compatible LIME.")


# ── Classe principale ─────────────────────────────────────────────────────────

class LIMEExplainer:
    """
    Wrapper LIME pour l'explication locale de transactions frauduleuses.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, n_features)
        Données d'entraînement — sert de distribution de référence pour générer
        les perturbations locales autour de la transaction à expliquer.
    feature_names : list[str]
        Noms des features dans l'ordre.
    mode : str
        'classification' pour LR/RF/XGBoost, 'regression' pour AutoEncoder.
    random_state : int
    """

    def __init__(
        self,
        X_train           : np.ndarray,
        feature_names     : list,
        mode              : str = "classification",
        random_state      : int = 42,
        features_json_path: str | None = None,
    ):
        if not LIME_AVAILABLE:
            raise ImportError("lime non installé. Lance : pip install lime")

        self.feature_names  = feature_names
        self.feature_labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
        self.mode           = mode

        # Indices catégoriels dérivés dynamiquement depuis features.json si disponible
        cat_idx = _get_categorical_indices(feature_names, features_json_path)

        self.explainer = LimeTabularExplainer(
            training_data         = X_train,
            feature_names         = feature_names,
            categorical_features  = cat_idx,   # pas de discrétisation pour ces features
            mode                  = mode,
            random_state          = random_state,
            discretize_continuous = True,       # les features continues sont discrétisées en intervalles
        )
        print(f"  LIMEExplainer créé (mode={mode}, {len(cat_idx)} features catégorielles)")

    # ── Explication d'une transaction ─────────────────────────────────────────

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
        Génère une explication LIME pour une transaction.

        Parameters
        ----------
        transaction  : np.ndarray, shape (n_features,)
        predict_fn   : callable, optionnel
            Fonction X → probas ou scores. Si None, construit automatiquement depuis `model`.
        model        : objet, optionnel
            Alternative à predict_fn — la fonction est déduite du type de modèle.
        num_features : int
            Nombre de features (règles) à afficher dans l'explication.
        num_samples  : int
            Nombre de perturbations générées autour de la transaction.
            1000 suffit pour 14 features (stabilité correcte, ~3× plus rapide que 3000).
        label        : int
            Classe à expliquer (1 = fraude). Ignoré en mode regression.

        Returns
        -------
        dict
            feature_rules : list[str]  — règles LIME (ex. "balance_diff_orig > 0.20")
            weights       : list[float] — poids de chaque règle
            intercept     : float       — terme de biais du modèle linéaire local
            local_pred    : float       — prédiction locale du modèle linéaire LIME
            df            : pd.DataFrame — résumé trié par |weight|
        """
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
            pairs, intercept, local_pred, use_label = self._parse_classification(exp, label)
        else:
            pairs, intercept, local_pred = self._parse_regression(exp)
            use_label = None

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
            "use_label"    : use_label,
            "df"           : df,
        }

    @staticmethod
    def _parse_classification(exp, label: int):
        """
        Extrait (pairs, intercept, local_pred) depuis un résultat LIME classification.

        LIME ne produit pas toujours une explication pour le label demandé
        (ex. transaction trop clairement non-frauduleuse). Dans ce cas on prend
        le dernier label disponible plutôt que de lever une KeyError.
        """
        available = list(exp.local_exp.keys())
        if not available:
            available = list(range(len(exp.local_pred)))

        use_label = label if label in available else available[-1]
        pairs     = exp.as_list(label=use_label)

        intercept = (exp.intercept.get(use_label, 0.0)
                     if hasattr(exp.intercept, 'get')
                     else float(exp.intercept[0]))

        if hasattr(exp.local_pred, 'get'):
            local_pred = exp.local_pred.get(use_label, 0.0)
        elif len(exp.local_pred) > use_label:
            local_pred = float(exp.local_pred[use_label])
        else:
            local_pred = float(exp.local_pred[-1])

        return pairs, intercept, local_pred, use_label

    @staticmethod
    def _parse_regression(exp):
        """
        Extrait (pairs, intercept, local_pred) depuis un résultat LIME regression.

        Le format de exp.intercept et exp.local_pred varie selon la version de LIME
        (scalaire ou array à 1-2 éléments) — on normalise ici vers un float.
        """
        pairs = exp.as_list()

        if hasattr(exp.intercept, '__len__'):
            intercept = float(exp.intercept[1]) if len(exp.intercept) > 1 else float(exp.intercept[0])
        else:
            intercept = float(exp.intercept)

        if hasattr(exp.local_pred, '__len__'):
            local_pred = float(exp.local_pred[1]) if len(exp.local_pred) > 1 else float(exp.local_pred[0])
        else:
            local_pred = float(exp.local_pred)

        return pairs, intercept, local_pred

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_waterfall(
        self,
        lime_result : dict,
        model_name  : str,
        title_extra : str   = "",
        figsize     : tuple = (11, 6),
        save_path   : str   = None,
    ) -> plt.Figure:
        """
        Waterfall LIME : poids par règle pour une transaction donnée.

        Rouge → règle qui pousse vers la fraude (poids positif).
        Bleu  → règle qui atténue le score de fraude (poids négatif).
        """
        df         = lime_result["df"]
        local_pred = lime_result["local_pred"]
        colors     = [PALETTE["positive"] if w > 0 else PALETTE["negative"] for w in df["weight"]]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df["feature_rule"][::-1], df["weight"][::-1],
                       color=colors[::-1], edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)

        title = f"LIME — {model_name} | Prédiction locale : {local_pred:.4f}"
        if title_extra:
            title += f"\n{title_extra}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Poids LIME (contribution locale au score de fraude)", fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(color=PALETTE["positive"], label="Hausse risque"),
            mpatches.Patch(color=PALETTE["negative"], label="Baisse risque"),
        ], fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardée : {save_path}")
        plt.show()
        return fig

    @staticmethod
    def plot_shap_vs_lime(
        shap_df    : pd.DataFrame,
        lime_result: dict,
        model_name : str,
        top_k      : int   = 8,
        figsize    : tuple = (14, 6),
        save_path  : str   = None,
    ) -> plt.Figure:
        """
        Comparaison côte-à-côte SHAP et LIME pour le même modèle et la même transaction.

        Permet de valider la cohérence entre les deux méthodes d'explicabilité :
        si les mêmes features ressortent en tête, les explications sont robustes.

        Parameters
        ----------
        shap_df : pd.DataFrame
            Sortie de SHAPExplainer.explain_single() — colonnes feature_label et shap_value.
        lime_result : dict
            Sortie de LIMEExplainer.explain_instance().
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"SHAP vs LIME — {model_name}", fontsize=13, fontweight='bold')

        # --- Panel SHAP ---
        df_shap  = shap_df.head(top_k)
        colors_s = [PALETTE["positive"] if v > 0 else PALETTE["negative"]
                    for v in df_shap["shap_value"]]
        bars1 = ax1.barh(df_shap["feature_label"][::-1], df_shap["shap_value"][::-1],
                         color=colors_s[::-1], edgecolor="white")
        ax1.bar_label(bars1, fmt="%.4f", fontsize=8, padding=2)
        ax1.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax1.set_title("SHAP (transaction spécifique)", fontsize=11, color=PALETTE["positive"])
        ax1.set_xlabel("Valeur SHAP")
        ax1.spines[['top', 'right']].set_visible(False)

        # --- Panel LIME ---
        df_lime  = lime_result["df"].head(top_k)
        colors_l = [PALETTE["positive"] if w > 0 else PALETTE["negative"]
                    for w in df_lime["weight"]]
        bars2 = ax2.barh(df_lime["feature_rule"][::-1], df_lime["weight"][::-1],
                         color=colors_l[::-1], edgecolor="white")
        ax2.bar_label(bars2, fmt="%.4f", fontsize=8, padding=2)
        ax2.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax2.set_title(
            f"LIME (local, pred={lime_result['local_pred']:.3f})",
            fontsize=11, color=PALETTE["negative"],
        )
        ax2.set_xlabel("Poids LIME")
        ax2.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure sauvegardée : {save_path}")
        plt.show()
        return fig

