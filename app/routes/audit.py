"""
app/routes/audit.py
FastAPI routes for audit trail read-only access.
Serves audit logs stored in PostgreSQL by the Next.js API layer.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser, get_current_user, require_roles
from app.db.database import get_db
from app.db.models import AuditLog, AuditAction, UserRole

router = APIRouter(prefix="/audit", tags=["Audit Trail"])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class AuditLogOut(BaseModel):
    id: str
    action: str
    user_id: str
    user_name: str
    user_role: str
    mission_id: Optional[str]
    mission_name: Optional[str]
    details: str
    timestamp: str

    model_config = {"from_attributes": True}


class AuditLogIn(BaseModel):
    action: str
    user_id: str
    user_name: str
    user_role: str
    mission_id: Optional[str] = None
    mission_name: Optional[str] = None
    details: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/logs", response_model=list[AuditLogOut])
async def list_audit_logs(
    mission_id: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Return audit logs, optionally filtered by mission. Read-only."""
    query = select(AuditLog).order_by(desc(AuditLog.timestamp)).limit(limit)
    if mission_id:
        query = query.where(AuditLog.mission_id == mission_id)
    result = await db.execute(query)
    logs = result.scalars().all()
    return [
        AuditLogOut(
            id=log.id,
            action=log.action.value if hasattr(log.action, "value") else str(log.action),
            user_id=log.user_id,
            user_name=log.user_name,
            user_role=log.user_role.value if hasattr(log.user_role, "value") else str(log.user_role),
            mission_id=log.mission_id,
            mission_name=log.mission_name,
            details=log.details,
            timestamp=log.timestamp.isoformat(),
        )
        for log in logs
    ]


@router.post("/logs", response_model=AuditLogOut, status_code=201)
async def create_audit_log(
    body: AuditLogIn,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_roles("manager", "admin")),
):
    """Write an audit log entry from the ML pipeline (analysis start/complete)."""
    import uuid
    from datetime import datetime, timezone

    log = AuditLog(
        id=str(uuid.uuid4()),
        action=AuditAction(body.action),
        user_id=body.user_id,
        user_name=body.user_name,
        user_role=UserRole(body.user_role),
        mission_id=body.mission_id,
        mission_name=body.mission_name,
        details=body.details,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    return AuditLogOut(
        id=log.id,
        action=log.action.value,
        user_id=log.user_id,
        user_name=log.user_name,
        user_role=log.user_role.value,
        mission_id=log.mission_id,
        mission_name=log.mission_name,
        details=log.details,
        timestamp=log.timestamp.isoformat(),
    )
