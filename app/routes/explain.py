"""
app/routes/explain.py
GET /api/explain/{tx_id} — SHAP + LIME + LLM pour une transaction
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.services.explainer import compute_shap, compute_lime
from app.services.predictor import FEATURE_COLS

router = APIRouter()


@router.get("/explain/{tx_id}")
def explain(tx_id: int, request: Request):
    cache = request.app.state.results_cache
    models = request.app.state.models
    llm_helper = request.app.state.llm_helper

    if not cache:
        raise HTTPException(
            status_code=400,
            detail="Aucune prédiction disponible. Lancez POST /api/predict d'abord.",
        )

    X_arr = cache["X_arr"]
    df = cache["df"]
    transactions = cache["transactions"]

    # Trouver la transaction par tx_id (index dans df original)
    tx_meta = next((t for t in transactions if t["tx_id"] == tx_id), None)
    if tx_meta is None:
        raise HTTPException(status_code=404, detail=f"Transaction tx_id={tx_id} introuvable.")

    # Position dans X_arr (tx_id est l'index df)
    try:
        pos = df.index.get_loc(tx_id)
    except KeyError:
        # Fallback : chercher dans transactions list
        tx_ids = [t["tx_id"] for t in transactions]
        if tx_id not in tx_ids:
            raise HTTPException(status_code=404, detail=f"tx_id={tx_id} hors limites.")
        pos = tx_ids.index(tx_id)

    tx_arr = X_arr[pos]
    xgb = models["xgb"]
    ae = models["ae"]
    ae_threshold = models["ae_threshold"]

    # Feature values (normalisées)
    feature_values = {feat: round(float(tx_arr[i]), 6) for i, feat in enumerate(FEATURE_COLS)}

    # SHAP
    try:
        shap_values = compute_shap(tx_arr, xgb, FEATURE_COLS)
    except Exception as e:
        shap_values = {"error": str(e)}

    # LIME (utilise les données en cache comme training set de référence)
    try:
        lime_rules = compute_lime(tx_arr, X_arr, xgb, FEATURE_COLS)
    except Exception as e:
        lime_rules = [f"LIME non disponible : {e}"]

    # LLM — calcul des erreurs de reconstruction feature par feature
    try:
        import torch
        tx_tensor = torch.FloatTensor(tx_arr.reshape(1, -1)).to(ae.device)
        ae.model.eval()
        with torch.no_grad():
            recon = ae.model(tx_tensor).cpu().numpy()[0]
        feature_errors = np.abs(tx_arr - recon)

        transaction_dict = dict(zip(FEATURE_COLS, tx_arr.tolist()))
        llm_result = llm_helper.explain_fraud(
            transaction=transaction_dict,
            feature_errors=feature_errors,
            ae_score=tx_meta["ae_score"],
            threshold=ae_threshold,
        )
    except Exception as e:
        llm_result = {
            "risk_level": tx_meta["risk_level"],
            "resume": f"Explication LLM indisponible : {e}",
            "raisons": [],
            "actions_recommandees": [],
            "status": "error",
        }

    return {
        "tx_id": tx_id,
        "xgb_score": tx_meta["xgb_score"],
        "ae_score": tx_meta["ae_score"],
        "risk_level": tx_meta["risk_level"],
        "feature_values": feature_values,
        "shap_values": shap_values,
        "lime_rules": lime_rules,
        "llm": llm_result,
    }
