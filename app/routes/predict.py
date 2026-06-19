"""
app/routes/predict.py
POST /api/predict — Analyse + prédiction de fraude sur un CSV

Pipeline
────────
1. Lecture CSV
2. Profilage sémantique (DatasetProfiler)
3. Mapping des colonnes (ColumnMapper)
4. Détection du mode de prédiction (SchemaDetector)
   → "standard"    : XGBoost + Autoencoder  (schéma transactionnel complet)
   → "ae_isoforest": Autoencoder + IsoForest (schéma partiel, amount détecté)
   → "ae_only"     : Autoencoder seul        (amount détecté, peu de numériques)
   → "isoforest"   : IsoForest seul          (schéma inconnu, colonnes numériques)
5. Construction dynamique des features (DynamicFeatureBuilder, avec fallbacks)
6. Feature engineering générique (FeatureEngineer)
7. Prédiction selon le mode détecté

L'intégralité du pipeline CPU est exécutée dans asyncio.to_thread() pour
ne pas bloquer le worker event-loop FastAPI.
"""

from __future__ import annotations

import asyncio
import io
import os
import uuid
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File

from app.auth.dependencies import CurrentUser, require_roles
from app.services.column_mapper import map_columns
from app.services.dataset_profiler import profile_dataset
from app.services.feature_builder import build_features_dynamic
from app.services.feature_engineer import engineer_features
from app.services.file_storage import save_upload
from app.services.generic_predictor import predict_generic_batch
from app.services.predictor import predict_batch
from app.services.schema_detector import detect_schema_mode

router = APIRouter()

# Max entries in the per-process run cache before LRU eviction
_CACHE_MAX = 20


async def _update_dataset_storage_path(dataset_id: str, storage_path: str) -> None:
    """Write storage_path to the datasets table in PostgreSQL."""
    try:
        from sqlalchemy import update
        from app.db.database import AsyncSessionLocal
        from app.db.models import Dataset
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Dataset)
                .where(Dataset.id == dataset_id)
                .values(storage_path=storage_path)
            )
            await session.commit()
    except Exception:
        pass


async def _log_monitoring(
    run_id: str,
    model_name: str,
    prediction_mode: str,
    n_transactions: int,
    n_fraud: int,
    fraud_rate_pct: float,
    amount_at_risk: float,
    latency_ms: float,
) -> None:
    """Persist prediction statistics for online monitoring."""
    try:
        from app.db.database import AsyncSessionLocal
        from app.db.models import PredictionMonitoringLog
        async with AsyncSessionLocal() as session:
            log = PredictionMonitoringLog(
                run_id=run_id,
                model_name=model_name,
                prediction_mode=prediction_mode,
                n_transactions=n_transactions,
                n_fraud=n_fraud,
                fraud_rate_pct=fraud_rate_pct,
                amount_at_risk=amount_at_risk,
                latency_ms=latency_ms,
            )
            session.add(log)
            await session.commit()
    except Exception:
        pass


def _run_pipeline(df: pd.DataFrame, models: dict) -> dict:
    """All CPU-heavy work — runs inside asyncio.to_thread."""
    # ── Profilage ─────────────────────────────────────────────────────────────
    dataset_profile = None
    try:
        dataset_profile = profile_dataset(df)
    except Exception:
        pass

    # ── Mapping sémantique ────────────────────────────────────────────────────
    mapping_result = map_columns(df)

    # ── Détection du mode ─────────────────────────────────────────────────────
    schema_result = detect_schema_mode(mapping_result, df)

    df_mapped = mapping_result.df

    # ── Construction des features ─────────────────────────────────────────────
    X_arr = None
    build_report = None
    if schema_result.use_ae or schema_result.use_xgb:
        try:
            X_arr, build_report = build_features_dynamic(
                df_mapped, dataset_profile, models["scaler"]
            )
        except Exception as e:
            if schema_result.mode == "standard":
                raise RuntimeError(f"Erreur construction features : {e}") from e
            schema_result.warnings.append(f"Construction features AE échouée : {e}")
            schema_result.use_ae = False

    # ── Feature engineering générique ────────────────────────────────────────
    df_enriched = df_mapped
    eng_report = None
    try:
        df_enriched, eng_report = engineer_features(
            df_mapped, profile=dataset_profile, mapping=mapping_result.mapping
        )
    except Exception as e:
        schema_result.warnings.append(f"Feature engineering ignoré : {e}")

    # ── Prédiction ────────────────────────────────────────────────────────────
    if schema_result.mode == "standard":
        transactions = predict_batch(X_arr, models, df_mapped)
    else:
        transactions = predict_generic_batch(
            df_original=df,
            X_arr=X_arr,
            models=models,
            schema_result=schema_result,
            df_mapped=df_mapped,
        )

    return {
        "transactions": transactions,
        "X_arr": X_arr,
        "df_enriched": df_enriched,
        "dataset_profile": dataset_profile,
        "mapping_result": mapping_result,
        "schema_result": schema_result,
        "eng_report": eng_report,
        "build_report": build_report,
    }


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    run_id: Optional[str] = Query(None, description="ID de l'AnalysisRun pour lier le cache"),
    mission_id: Optional[str] = Query(None, description="ID de la mission pour organiser le stockage"),
    dataset_id: Optional[str] = Query(None, description="ID du dataset pour écrire storage_path en DB"),
    current_user: CurrentUser = Depends(require_roles("auditor", "manager", "admin")),
):
    """
    Point d'entrée principal de la détection d'anomalies.

    Flux complet :
      1. Lit le CSV uploadé (async I/O)
      2. Persiste le fichier sur disque si mission_id + dataset_id sont fournis
      3. Exécute le pipeline ML dans un thread séparé (asyncio.to_thread)
         pour ne pas bloquer l'event-loop FastAPI pendant les calculs CPU
      4. Stocke le résultat dans results_cache (clé = run_id) pour les appels
         ultérieurs à /explain et /report sans ré-uploader le fichier
      5. Journalise les métriques de monitoring en base (fire-and-forget)
      6. Retourne les transactions triées par score décroissant + métadonnées

    Accès : rôles auditor, manager, admin uniquement.
    """
    models = request.app.state.models

    # ── Lecture CSV (I/O async) ───────────────────────────────────────────────
    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Impossible de lire le CSV : {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Le CSV ne contient aucune ligne de données.")

    # ── Persistance du fichier uploadé ───────────────────────────────────────
    storage_path: Optional[str] = None
    if mission_id and dataset_id and file.filename:
        try:
            storage_path = await save_upload(
                content=content,
                mission_id=mission_id,
                dataset_id=dataset_id,
                filename=file.filename,
            )
            # Update storage_path in DB (fire-and-forget, non-blocking)
            if os.getenv("DATABASE_URL"):
                asyncio.create_task(_update_dataset_storage_path(dataset_id, storage_path))
        except Exception:
            pass  # storage failure must not break the prediction

    # ── Pipeline CPU dans un thread séparé ───────────────────────────────────
    _t0 = asyncio.get_event_loop().time()
    try:
        result = await asyncio.to_thread(_run_pipeline, df, models)
    except ValueError as e:
        # Raised by detect_schema_mode when no numeric columns found
        mapping_result = map_columns(df)
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(e),
                "columns_in_csv": list(df.columns),
                "partial_mapping": {
                    k: {"original": v, "confidence": round(mapping_result.confidence.get(k, 0), 2)}
                    for k, v in mapping_result.mapping.items()
                },
            },
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {e}")

    transactions = result["transactions"]
    schema_result = result["schema_result"]
    mapping_result = result["mapping_result"]
    dataset_profile = result["dataset_profile"]
    eng_report = result["eng_report"]
    build_report = result["build_report"]

    latency_ms = (asyncio.get_event_loop().time() - _t0) * 1000

    # ── Cache keyed by run_id (prevents cross-user contamination) ─────────────
    cache_key = run_id or str(uuid.uuid4())
    cache = request.app.state.results_cache
    if len(cache) >= _CACHE_MAX:
        # Evict oldest entry
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    cache[cache_key] = {
        "X_arr": result["X_arr"],
        "df": result["df_enriched"],
        "transactions": transactions,
    }

    # ── Résumé ────────────────────────────────────────────────────────────────
    frauds = [t for t in transactions if t["is_fraud_predicted"]]
    amount_at_risk = sum(t["amount"] for t in frauds)
    fraud_rate_pct = round(len(frauds) / max(len(transactions), 1) * 100, 4)

    threshold = (
        models["thresholds"].get("XGB_smote", 0.355) if schema_result.mode == "standard" else None
    )

    column_mapping_info = {
        k: {
            "original_name": v,
            "confidence": round(mapping_result.confidence.get(k, 0), 2),
        }
        for k, v in mapping_result.mapping.items()
    }

    # ── Monitoring log (fire-and-forget) ─────────────────────────────────────
    if os.getenv("DATABASE_URL") and run_id:
        asyncio.create_task(_log_monitoring(
            run_id=cache_key,
            model_name=schema_result.model_label,
            prediction_mode=schema_result.mode,
            n_transactions=len(transactions),
            n_fraud=len(frauds),
            fraud_rate_pct=fraud_rate_pct,
            amount_at_risk=round(amount_at_risk, 2),
            latency_ms=round(latency_ms, 1),
        ))

    response: dict = {
        "run_id": cache_key,
        "storage_path": storage_path,
        "n_transactions": len(transactions),
        "n_fraud": len(frauds),
        "fraud_rate_pct": fraud_rate_pct,
        "amount_at_risk": round(amount_at_risk, 2),
        "model_used": schema_result.model_label,
        "prediction_mode": schema_result.mode,
        "schema_detection": schema_result.to_dict(),
        "column_mapping": column_mapping_info,
        "mapping_warnings": mapping_result.warnings,
        "feature_engineering": eng_report.to_dict() if eng_report else {},
        "transactions": transactions,
    }

    if threshold is not None:
        response["threshold"] = threshold

    if build_report is not None:
        response["feature_build"] = build_report.to_dict()

    if dataset_profile is not None:
        response["dataset_profile"] = {
            "n_rows": dataset_profile.n_rows,
            "n_cols": dataset_profile.n_cols,
            "global_quality_score": round(dataset_profile.global_quality_score, 1),
            "numeric_cols": dataset_profile.numeric_cols,
            "categorical_cols": dataset_profile.categorical_cols,
            "datetime_cols": dataset_profile.datetime_cols,
            "identifier_cols": dataset_profile.identifier_cols,
            "quasi_constant_cols": dataset_profile.quasi_constant_cols,
            "high_missing_cols": dataset_profile.high_missing_cols,
            "recommendations": dataset_profile.recommendations,
            "profiling_time_ms": round(dataset_profile.profiling_time_ms, 1),
        }

    return response
