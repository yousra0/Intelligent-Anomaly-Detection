"""
app/routes/explain.py
GET  /api/explain/{tx_id}     — SHAP XGB + AE proxy errors + LIME + LLM pour une transaction
POST /api/explain/batch        — même pipeline (sans LIME) pour un ensemble de tx_ids
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.explainer import compute_shap, compute_lime, compute_ae_feature_errors
from app.services.predictor import FEATURE_COLS

router = APIRouter()


def _explain_one(
    tx_id: int,
    cache: dict,
    models: dict,
    llm_helper,
    include_lime: bool = True,
) -> dict:
    """Logique commune à l'endpoint single et batch."""
    X_arr = cache["X_arr"]
    df = cache["df"]
    transactions = cache["transactions"]

    tx_meta = next((t for t in transactions if t["tx_id"] == tx_id), None)
    if tx_meta is None:
        raise HTTPException(status_code=404, detail=f"Transaction tx_id={tx_id} introuvable.")

    # Position dans X_arr
    try:
        pos = df.index.get_loc(tx_id)
    except KeyError:
        tx_ids = [t["tx_id"] for t in transactions]
        if tx_id not in tx_ids:
            raise HTTPException(status_code=404, detail=f"tx_id={tx_id} hors limites.")
        pos = tx_ids.index(tx_id)

    if X_arr is None:
        raise HTTPException(
            status_code=400,
            detail=f"tx_id={tx_id} : pas de features AE disponibles (mode générique sans AE).",
        )

    tx_arr = X_arr[pos]
    xgb = models["xgb"]
    ae = models["ae"]
    ae_threshold = models["ae_threshold"]

    feature_values = {feat: round(float(tx_arr[i]), 6) for i, feat in enumerate(FEATURE_COLS)}

    # SHAP XGB — nommé explicitement pour éviter toute ambiguïté
    try:
        shap_values_xgb = compute_shap(tx_arr, xgb, FEATURE_COLS)
    except Exception as e:
        shap_values_xgb = {"error": str(e)}

    # Proxy AE : |x - AE(x)| par feature (remplace KernelExplainer, trop lent)
    try:
        ae_feature_errors = compute_ae_feature_errors(tx_arr, ae, FEATURE_COLS)
        # Top-3 features par erreur de reconstruction décroissante
        top3_ae = sorted(ae_feature_errors.items(), key=lambda kv: kv[1], reverse=True)[:3]
        ae_top_features = [{"feature": k, "error": v} for k, v in top3_ae]
    except Exception as e:
        ae_feature_errors = {"error": str(e)}
        ae_top_features = []

    # LIME (optionnel — désactivé en batch pour la performance)
    lime_rules: Optional[list] = None
    if include_lime:
        try:
            lime_rules = compute_lime(tx_arr, X_arr, xgb, FEATURE_COLS)
        except Exception as e:
            lime_rules = [f"LIME non disponible : {e}"]

    # LLM — utilise les erreurs de reconstruction calculées ci-dessus
    try:
        import torch
        tx_tensor = torch.FloatTensor(tx_arr.reshape(1, -1)).to(ae.device)
        ae.model.eval()
        with torch.no_grad():
            recon = ae.model(tx_tensor).cpu().numpy()[0]
        feature_errors_arr = np.abs(tx_arr - recon)

        transaction_dict = dict(zip(FEATURE_COLS, tx_arr.tolist()))
        ae_score = tx_meta.get("ae_score", 0.0)
        llm_result = llm_helper.explain_fraud(
            transaction=transaction_dict,
            feature_errors=feature_errors_arr,
            ae_score=ae_score,
            threshold=ae_threshold,
        ) if llm_helper is not None else None
    except Exception as e:
        llm_result = {
            "risk_level": tx_meta["risk_level"],
            "resume": f"Explication LLM indisponible : {e}",
            "raisons": [],
            "actions_recommandees": [],
            "status": "error",
        }

    result: dict = {
        "tx_id": tx_id,
        "risk_level": tx_meta["risk_level"],
        "feature_values": feature_values,
        # SHAP porte explicitement sur XGB uniquement
        "shap_values_xgb": shap_values_xgb,
        # Proxy AE : erreurs de reconstruction feature par feature
        "ae_feature_errors": ae_feature_errors,
        "ae_top_features": ae_top_features,
        "llm": llm_result,
    }

    if "xgb_score" in tx_meta:
        result["xgb_score"] = tx_meta["xgb_score"]
    if "ae_score" in tx_meta:
        result["ae_score"] = tx_meta["ae_score"]
    if include_lime and lime_rules is not None:
        result["lime_rules"] = lime_rules

    return result


@router.get("/explain/{tx_id}")
def explain(tx_id: int, request: Request):
    cache = request.app.state.results_cache
    if not cache:
        raise HTTPException(
            status_code=400,
            detail="Aucune prédiction disponible. Lancez POST /api/predict d'abord.",
        )
    return _explain_one(
        tx_id=tx_id,
        cache=cache,
        models=request.app.state.models,
        llm_helper=request.app.state.llm_helper,
        include_lime=True,
    )


class BatchExplainRequest(BaseModel):
    tx_ids: list[int] = Field(..., description="Liste des tx_id à expliquer.")
    max_explain: int = Field(20, ge=1, le=100, description="Limite max d'explications.")


@router.post("/explain/batch")
def explain_batch(body: BatchExplainRequest, request: Request):
    """
    Explication batch pour un ensemble de tx_ids.
    LIME est désactivé pour des raisons de performance.
    Limite à max_explain transactions (défaut : 20, max : 100).
    """
    cache = request.app.state.results_cache
    if not cache:
        raise HTTPException(
            status_code=400,
            detail="Aucune prédiction disponible. Lancez POST /api/predict d'abord.",
        )

    tx_ids = body.tx_ids[: body.max_explain]
    results = []
    errors = []

    for tx_id in tx_ids:
        try:
            expl = _explain_one(
                tx_id=tx_id,
                cache=cache,
                models=request.app.state.models,
                llm_helper=request.app.state.llm_helper,
                include_lime=False,
            )
            results.append(expl)
        except HTTPException as exc:
            errors.append({"tx_id": tx_id, "error": exc.detail})

    return {
        "n_requested": len(body.tx_ids),
        "n_explained": len(results),
        "n_errors": len(errors),
        "explanations": results,
        "errors": errors,
    }
