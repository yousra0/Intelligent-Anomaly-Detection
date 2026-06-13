"""
app/main.py
Point d'entrée FastAPI — backend détection de fraude PFE.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.routes import predict, explain, report, models as models_route, profile
from app.services.predictor import load_all_models
from app.services.llm_service import get_llm_helper


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Chargement des modèles…")
    app.state.models = load_all_models(PROJECT_ROOT)
    print(f"[startup] {len(app.state.models)} artefacts chargés.")

    print("[startup] Initialisation LLMHelper…")
    try:
        app.state.llm_helper = get_llm_helper(PROJECT_ROOT)
        print(f"[startup] LLM provider : {app.state.llm_helper.provider}")
    except Exception as e:
        print(f"[startup] LLM non disponible ({e}) — mode fallback rule-based activé.")
        app.state.llm_helper = None

    app.state.results_cache = {}

    yield

    print("[shutdown] Nettoyage.")


app = FastAPI(
    title="API Détection de Fraude — PFE",
    description="Backend FastAPI pour la détection de fraude financière (PaySim).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
app.include_router(report.router, prefix="/api")
app.include_router(models_route.router, prefix="/api")
app.include_router(profile.router, prefix="/api")


@app.get("/api/health")
async def health_check(request: Request):
    return {
        "status": "ok",
        "models_loaded": hasattr(request.app.state, "models") and request.app.state.models is not None,
        "llm_available": (
            hasattr(request.app.state, "llm_helper")
            and request.app.state.llm_helper is not None
            and request.app.state.llm_helper.is_available()
        ),
    }
