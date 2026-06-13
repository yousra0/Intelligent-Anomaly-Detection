"""
app/routes/report.py
POST /api/report      — Rapport PDF PwC
POST /api/report/docx — Rapport Word (template exemple_rapport.docx)
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.services.report_gen      import generate_pwc_report
from app.services.report_gen_docx import generate_pwc_docx_report

router = APIRouter()


@router.post("/report")
async def generate_report(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Fallback : utiliser le cache si le corps est vide
    if not body:
        cache = request.app.state.results_cache
        if not cache:
            raise HTTPException(
                status_code=400,
                detail="Aucun résultat de prédiction disponible. Lancez POST /api/predict d'abord.",
            )
        predict_result = {
            "n_transactions": len(cache.get("transactions", [])),
            "n_fraud": sum(1 for t in cache.get("transactions", []) if t.get("is_fraud_predicted")),
            "fraud_rate_pct": 0.0,
            "amount_at_risk": sum(
                t.get("amount", 0) for t in cache.get("transactions", []) if t.get("is_fraud_predicted")
            ),
            "model_used": "XGB_smote",
            "threshold": request.app.state.models["thresholds"].get("XGB_smote", 0.355),
            "transactions": cache.get("transactions", []),
        }
        transactions = predict_result["transactions"]
        n = max(len(transactions), 1)
        predict_result["fraud_rate_pct"] = round(predict_result["n_fraud"] / n * 100, 4)
    else:
        predict_result = body

    try:
        pdf_bytes = generate_pwc_report(predict_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération PDF : {e}")

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"rapport_pwc_{date_str}.pdf"

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _build_predict_result(request: Request, body: dict) -> dict:
    """Construit predict_result depuis le body ou le cache."""
    if body:
        return body
    cache = request.app.state.results_cache
    if not cache:
        raise HTTPException(
            status_code=400,
            detail="Aucun resultat disponible. Lancez POST /api/predict d'abord.",
        )
    transactions = cache.get("transactions", [])
    n_fraud = sum(1 for t in transactions if t.get("is_fraud_predicted"))
    amount  = sum(t.get("amount", 0) for t in transactions if t.get("is_fraud_predicted"))
    n       = max(len(transactions), 1)
    return {
        "n_transactions": len(transactions),
        "n_fraud":        n_fraud,
        "fraud_rate_pct": round(n_fraud / n * 100, 4),
        "amount_at_risk": amount,
        "model_used":     "XGB_smote",
        "threshold":      request.app.state.models["thresholds"].get("XGB_smote", 0.355),
        "transactions":   transactions,
    }


@router.post("/report/docx")
async def generate_report_docx(request: Request):
    """Génère le rapport Word en remplissant le template PwC (exemple_rapport.docx)."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    predict_result = _build_predict_result(request, body)

    try:
        docx_bytes = generate_pwc_docx_report(predict_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur generation DOCX : {e}")

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"rapport_pwc_{date_str}.docx"

    return StreamingResponse(
        iter([docx_bytes]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
