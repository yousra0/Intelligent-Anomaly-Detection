"""
app/routes/model_registry.py
Model versioning, registry, and monitoring endpoints.

GET  /api/registry/models                     — list all model versions
POST /api/registry/models                     — register a new version
GET  /api/registry/models/{model_name}/latest — latest production version
PATCH /api/registry/models/{version_id}       — promote/archive a version
GET  /api/registry/monitoring                 — online monitoring stats (recent runs)
GET  /api/registry/monitoring/drift           — offline drift summary
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser, get_current_user, require_roles
from app.db.database import get_db
from app.db.models import ModelVersion, ModelVersionStatus, PredictionMonitoringLog

router = APIRouter(prefix="/registry", tags=["Model Registry"])


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class ModelVersionOut(BaseModel):
    id: str
    model_name: str
    version: int
    status: str
    storage_path: Optional[str]
    metrics: Optional[dict]
    tags: Optional[dict]
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class RegisterModelVersionIn(BaseModel):
    model_name: str
    storage_path: Optional[str] = None
    metrics: Optional[dict] = None
    tags: Optional[dict] = None


class PatchModelVersionIn(BaseModel):
    status: ModelVersionStatus


class MonitoringLogOut(BaseModel):
    id: str
    run_id: str
    model_name: str
    prediction_mode: str
    n_transactions: int
    n_fraud: int
    fraud_rate_pct: float
    amount_at_risk: Optional[float]
    latency_ms: Optional[float]
    created_at: str

    model_config = {"from_attributes": True}


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/models", response_model=list[ModelVersionOut])
async def list_model_versions(
    model_name: Optional[str] = Query(None),
    status: Optional[ModelVersionStatus] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    q = select(ModelVersion).order_by(desc(ModelVersion.created_at))
    if model_name:
        q = q.where(ModelVersion.model_name == model_name)
    if status:
        q = q.where(ModelVersion.status == status)
    result = await db.execute(q)
    versions = result.scalars().all()
    return [
        ModelVersionOut(
            id=v.id,
            model_name=v.model_name,
            version=v.version,
            status=v.status.value,
            storage_path=v.storage_path,
            metrics=v.metrics,
            tags=v.tags,
            created_at=v.created_at.isoformat(),
            updated_at=v.updated_at.isoformat(),
        )
        for v in versions
    ]


@router.post("/models", response_model=ModelVersionOut, status_code=201)
async def register_model_version(
    body: RegisterModelVersionIn,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_roles("manager", "admin")),
):
    # Auto-increment version number
    result = await db.execute(
        select(func.max(ModelVersion.version)).where(ModelVersion.model_name == body.model_name)
    )
    max_v = result.scalar() or 0

    v = ModelVersion(
        model_name=body.model_name,
        version=max_v + 1,
        status=ModelVersionStatus.staged,
        storage_path=body.storage_path,
        metrics=body.metrics,
        tags=body.tags,
        promoted_by_id=current_user.id,
    )
    db.add(v)
    await db.commit()
    await db.refresh(v)
    return ModelVersionOut(
        id=v.id, model_name=v.model_name, version=v.version,
        status=v.status.value, storage_path=v.storage_path,
        metrics=v.metrics, tags=v.tags,
        created_at=v.created_at.isoformat(), updated_at=v.updated_at.isoformat(),
    )


@router.get("/models/{model_name}/latest", response_model=ModelVersionOut)
async def get_latest_production_version(
    model_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.model_name == model_name, ModelVersion.status == ModelVersionStatus.production)
        .order_by(desc(ModelVersion.version))
        .limit(1)
    )
    v = result.scalar_one_or_none()
    if not v:
        raise HTTPException(status_code=404, detail=f"Aucune version en production pour '{model_name}'.")
    return ModelVersionOut(
        id=v.id, model_name=v.model_name, version=v.version,
        status=v.status.value, storage_path=v.storage_path,
        metrics=v.metrics, tags=v.tags,
        created_at=v.created_at.isoformat(), updated_at=v.updated_at.isoformat(),
    )


@router.patch("/models/{version_id}", response_model=ModelVersionOut)
async def update_model_version_status(
    version_id: str,
    body: PatchModelVersionIn,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_roles("manager", "admin")),
):
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == version_id))
    v = result.scalar_one_or_none()
    if not v:
        raise HTTPException(status_code=404, detail="Version introuvable.")

    # When promoting to production, archive any existing production version
    if body.status == ModelVersionStatus.production:
        await db.execute(
            text(
                "UPDATE model_versions SET status='archived', updated_at=now() "
                "WHERE model_name=:name AND status='production' AND id != :id"
            ),
            {"name": v.model_name, "id": version_id},
        )

    v.status = body.status
    v.promoted_by_id = current_user.id
    await db.commit()
    await db.refresh(v)
    return ModelVersionOut(
        id=v.id, model_name=v.model_name, version=v.version,
        status=v.status.value, storage_path=v.storage_path,
        metrics=v.metrics, tags=v.tags,
        created_at=v.created_at.isoformat(), updated_at=v.updated_at.isoformat(),
    )


# ─── Monitoring ───────────────────────────────────────────────────────────────

@router.get("/monitoring", response_model=list[MonitoringLogOut])
async def list_monitoring_logs(
    model_name: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Recent online monitoring statistics per analysis run."""
    q = select(PredictionMonitoringLog).order_by(desc(PredictionMonitoringLog.created_at)).limit(limit)
    if model_name:
        q = q.where(PredictionMonitoringLog.model_name == model_name)
    result = await db.execute(q)
    logs = result.scalars().all()
    return [
        MonitoringLogOut(
            id=log.id, run_id=log.run_id, model_name=log.model_name,
            prediction_mode=log.prediction_mode, n_transactions=log.n_transactions,
            n_fraud=log.n_fraud, fraud_rate_pct=log.fraud_rate_pct,
            amount_at_risk=log.amount_at_risk, latency_ms=log.latency_ms,
            created_at=log.created_at.isoformat(),
        )
        for log in logs
    ]


@router.get("/monitoring/drift")
async def get_drift_summary(
    model_name: Optional[str] = Query(None),
    window: int = Query(30, description="Nombre de dernières exécutions à comparer"),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Offline drift detection: compare recent window vs. baseline.
    Returns avg fraud_rate_pct, stddev, and a drift flag if the recent
    average exceeds 2 standard deviations above the baseline mean.
    """
    q = (
        select(
            func.avg(PredictionMonitoringLog.fraud_rate_pct).label("avg_rate"),
            func.stddev_pop(PredictionMonitoringLog.fraud_rate_pct).label("std_rate"),
            func.count().label("n_runs"),
        )
    )
    if model_name:
        q = q.where(PredictionMonitoringLog.model_name == model_name)
    result = await db.execute(q)
    row = result.one()

    recent_q = (
        select(func.avg(PredictionMonitoringLog.fraud_rate_pct).label("recent_avg"))
        .order_by(desc(PredictionMonitoringLog.created_at))
        .limit(window)
    )
    if model_name:
        recent_q = recent_q.where(PredictionMonitoringLog.model_name == model_name)
    recent_result = await db.execute(recent_q)
    recent_row = recent_result.one()

    baseline_avg = float(row.avg_rate or 0)
    baseline_std = float(row.std_rate or 0)
    recent_avg = float(recent_row.recent_avg or 0)
    drift_detected = baseline_std > 0 and abs(recent_avg - baseline_avg) > 2 * baseline_std

    return {
        "model_name": model_name,
        "n_runs_total": row.n_runs,
        "baseline_avg_fraud_rate": round(baseline_avg, 4),
        "baseline_std_fraud_rate": round(baseline_std, 4),
        "recent_avg_fraud_rate": round(recent_avg, 4),
        "recent_window": window,
        "drift_detected": drift_detected,
        "alert": (
            f"DRIFT DÉTECTÉ : taux de fraude récent ({recent_avg:.2f}%) s'écarte "
            f"de plus de 2σ de la baseline ({baseline_avg:.2f}% ± {baseline_std:.2f}%)."
            if drift_detected else None
        ),
    }
