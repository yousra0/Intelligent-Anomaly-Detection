"""
app/main.py
Point d'entrée FastAPI — backend détection de fraude PFE.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from app.routes import predict, explain, report, models as models_route, profile
from app.routes import audit as audit_route
from app.routes import datasets as datasets_route
from app.routes import model_registry
from app.services.predictor import load_all_models
from app.services.llm_service import get_llm_helper


def _run_alembic_upgrade() -> None:
    """Run `alembic upgrade head` synchronously (called once at startup in dev)."""
    from alembic.config import Config
    from alembic import command

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie FastAPI (ASGI lifespan).

    Exécuté une seule fois au démarrage et à l'arrêt du serveur.
    Ordre d'initialisation :
      1. Base de données  : migration Alembic si AUTO_MIGRATE=true, sinon ping de vérification
      2. Modèles ML       : chargement de tous les artefacts depuis outputs/models/
      3. LLM              : initialisation du helper (provider Anthropic ou fallback rule-based)
      4. Cache résultats  : dictionnaire en mémoire keyed by run_id (max 20 entrées, LRU)
    """
    # ── Database ──────────────────────────────────────────────────────────────
    if os.getenv("DATABASE_URL"):
        auto_migrate = os.getenv("AUTO_MIGRATE", "false").lower() == "true"
        try:
            if auto_migrate:
                import asyncio
                await asyncio.to_thread(_run_alembic_upgrade)
                print("[startup] Migrations Alembic appliquées.")
            else:
                from app.db.database import engine
                import sqlalchemy
                async with engine.connect() as conn:
                    await conn.execute(sqlalchemy.text("SELECT 1"))
                print("[startup] Connexion PostgreSQL OK. Lancez 'alembic upgrade head' si besoin.")
        except Exception as e:
            print(f"[startup] DB non disponible ({e}) — les routes DB seront désactivées.")
    else:
        print("[startup] DATABASE_URL non définie — fonctionnement sans persistance DB.")

    # ── ML models ─────────────────────────────────────────────────────────────
    print("[startup] Chargement des modèles…")
    app.state.models = load_all_models(PROJECT_ROOT)
    print(f"[startup] {len(app.state.models)} artefacts chargés.")

    # ── LLM ───────────────────────────────────────────────────────────────────
    print("[startup] Initialisation LLMHelper…")
    try:
        app.state.llm_helper = get_llm_helper(PROJECT_ROOT)
        print(f"[startup] LLM provider : {app.state.llm_helper.provider}")
    except Exception as e:
        print(f"[startup] LLM non disponible ({e}) — mode fallback rule-based activé.")
        app.state.llm_helper = None

    # Per-run cache keyed by run_id — prevents cross-user contamination.
    # Max 20 entries to cap memory (LRU eviction handled in predict route).
    app.state.results_cache = {}

    yield

    print("[shutdown] Nettoyage.")
    if os.getenv("DATABASE_URL"):
        try:
            from app.db.database import engine
            await engine.dispose()
        except Exception:
            pass


app = FastAPI(
    title="API Détection de Fraude — PwC",
    description="Backend FastAPI pour la détection d'anomalies financières — PwC Audit Analytics",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        os.getenv("NEXT_PUBLIC_API_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ML routes ─────────────────────────────────────────────────────────────────
app.include_router(predict.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
app.include_router(report.router, prefix="/api")
app.include_router(models_route.router, prefix="/api")
app.include_router(profile.router, prefix="/api")

# ── Audit trail route (only when DB is available) ─────────────────────────────
app.include_router(audit_route.router, prefix="/api")

# ── Dataset file download ──────────────────────────────────────────────────────
app.include_router(datasets_route.router, prefix="/api")

# ── Model registry + monitoring ───────────────────────────────────────────────
app.include_router(model_registry.router, prefix="/api")


@app.get("/api/health")
async def health_check(request: Request):
    """
    Endpoint de supervision utilisé par le frontend (AnalysisWizard) pour afficher
    l'état du backend en temps réel. Vérifie trois composants indépendants :
      - models_loaded    : les artefacts ML sont bien en mémoire (app.state.models)
      - llm_available    : le LLM répond (ou mode fallback actif)
      - database_connected : un SELECT 1 sur PostgreSQL passe sans erreur
    Retourne toujours HTTP 200 même si un composant est KO — le frontend lit les champs booléens.
    """
    db_ok = False
    if os.getenv("DATABASE_URL"):
        try:
            from app.db.database import engine
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False

    return {
        "status": "ok",
        "models_loaded": hasattr(request.app.state, "models") and request.app.state.models is not None,
        "llm_available": (
            hasattr(request.app.state, "llm_helper")
            and request.app.state.llm_helper is not None
            and request.app.state.llm_helper.is_available()
        ),
        "database_connected": db_ok,
    }
