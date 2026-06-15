"""
app/services/explainer.py
SHAP et LIME implémentés directement (sans dépendance sur src/explainability).
"""

from __future__ import annotations

import numpy as np


def compute_ae_feature_errors(
    tx_arr: np.ndarray,
    ae,
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Proxy AE : |x - AE(x)| par feature.

    Remplace KernelExplainer (trop lent pour la production) pour expliquer
    l'AutoEncoder. Chaque valeur représente l'erreur de reconstruction de
    l'AutoEncoder sur cette feature — plus elle est grande, plus la feature
    a contribué au score d'anomalie.
    """
    import torch

    tx_tensor = torch.FloatTensor(tx_arr.reshape(1, -1)).to(ae.device)
    ae.model.eval()
    with torch.no_grad():
        recon = ae.model(tx_tensor).cpu().numpy()[0]
    errors = np.abs(tx_arr - recon)
    return {feat: round(float(e), 6) for feat, e in zip(feature_cols, errors)}


def compute_shap(
    tx_arr: np.ndarray,
    xgb_raw,
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Calcule les valeurs SHAP pour une transaction via TreeExplainer.
    tx_arr : shape (14,) ou (1, 14)
    """
    import shap

    explainer = shap.TreeExplainer(xgb_raw)
    x = tx_arr.reshape(1, -1)
    sv = explainer.shap_values(x, check_additivity=False)

    # TreeExplainer peut retourner une liste [neg_class, pos_class] ou un array (1,14)
    if isinstance(sv, list):
        values = sv[1][0] if len(sv) > 1 else sv[0][0]
    else:
        values = sv[0]

    return {feat: round(float(v), 6) for feat, v in zip(feature_cols, values)}


def compute_lime(
    tx_arr: np.ndarray,
    X_train_arr: np.ndarray,
    xgb,
    feature_cols: list[str],
    num_features: int = 6,
) -> list[str]:
    """
    Calcule l'explication LIME pour une transaction.
    tx_arr : shape (14,)
    Retourne une liste de strings "rule: +0.xxx"
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train_arr,
        feature_names=feature_cols,
        mode="classification",
        random_state=42,
    )

    def predict_fn(X: np.ndarray) -> np.ndarray:
        p = xgb.predict_proba(X)
        if p.ndim == 1:
            return np.column_stack([1 - p, p])
        return p

    exp = explainer.explain_instance(
        tx_arr,
        predict_fn,
        num_features=num_features,
    )

    rules = []
    for feat, weight in exp.as_list():
        sign = "+" if weight >= 0 else ""
        rules.append(f"{feat}: {sign}{weight:.3f}")
    return rules
