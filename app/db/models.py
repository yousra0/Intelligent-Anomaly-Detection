"""
app/db/models.py
SQLAlchemy ORM models mirroring the Prisma schema.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Enums ────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    auditor = "auditor"
    manager = "manager"
    partner = "partner"
    admin = "admin"


class UserStatus(str, enum.Enum):
    active = "active"
    inactive = "inactive"
    suspended = "suspended"


class MissionStatus(str, enum.Enum):
    active = "active"
    in_progress = "in_progress"
    completed = "completed"
    archived = "archived"


class MissionType(str, enum.Enum):
    financial_audit = "financial_audit"
    fraud_detection = "fraud_detection"
    compliance_review = "compliance_review"
    risk_assessment = "risk_assessment"
    internal_audit = "internal_audit"


class DatasetCategory(str, enum.Enum):
    transactions = "transactions"
    general_ledger = "general_ledger"
    trial_balance = "trial_balance"


class DatasetStatus(str, enum.Enum):
    pending = "pending"
    uploaded = "uploaded"
    analyzing = "analyzing"
    analyzed = "analyzed"
    error = "error"


class AnalysisStatus(str, enum.Enum):
    running = "running"
    completed = "completed"
    failed = "failed"


class RiskLevel(str, enum.Enum):
    CRITIQUE = "CRITIQUE"
    ELEVE = "ELEVE"
    FAIBLE = "FAIBLE"


class AuditAction(str, enum.Enum):
    login = "login"
    logout = "logout"
    mission_create = "mission_create"
    mission_update = "mission_update"
    mission_delete = "mission_delete"
    mission_assign = "mission_assign"
    dataset_upload = "dataset_upload"
    dataset_delete = "dataset_delete"
    dataset_replace = "dataset_replace"
    analysis_start = "analysis_start"
    analysis_complete = "analysis_complete"
    report_generate = "report_generate"
    report_download = "report_download"
    anomaly_comment = "anomaly_comment"
    anomaly_status_change = "anomaly_status_change"
    user_create = "user_create"
    user_update = "user_update"
    user_deactivate = "user_deactivate"
    role_modify = "role_modify"


# ─── Models ───────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    status: Mapped[UserStatus] = mapped_column(Enum(UserStatus), default=UserStatus.active, nullable=False)
    phone: Mapped[str | None] = mapped_column(String)
    position: Mapped[str | None] = mapped_column(String)
    department: Mapped[str | None] = mapped_column(String)
    password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    audit_logs: Mapped[list["AuditLog"]] = relationship("AuditLog", back_populates="user")


class Mission(Base):
    __tablename__ = "missions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    mission_type: Mapped[MissionType] = mapped_column(Enum(MissionType), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[MissionStatus] = mapped_column(Enum(MissionStatus), default=MissionStatus.active, nullable=False)
    start_date: Mapped[str] = mapped_column(String, nullable=False)
    end_date: Mapped[str | None] = mapped_column(String)
    created_by_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    assigned_to_id: Mapped[str | None] = mapped_column(String, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    audit_logs: Mapped[list["AuditLog"]] = relationship("AuditLog", back_populates="mission")
    datasets: Mapped[list["Dataset"]] = relationship("Dataset", back_populates="mission")
    analysis_runs: Mapped[list["AnalysisRun"]] = relationship("AnalysisRun", back_populates="mission")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mission_id: Mapped[str] = mapped_column(String, ForeignKey("missions.id", ondelete="CASCADE"), nullable=False, index=True)
    uploaded_by_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    original_name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[DatasetCategory] = mapped_column(Enum(DatasetCategory), default=DatasetCategory.transactions)
    status: Mapped[DatasetStatus] = mapped_column(Enum(DatasetStatus), default=DatasetStatus.uploaded)
    row_count: Mapped[int | None] = mapped_column(Integer)
    column_count: Mapped[int | None] = mapped_column(Integer)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    storage_path: Mapped[str | None] = mapped_column(String)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    mission: Mapped["Mission"] = relationship("Mission", back_populates="datasets")
    versions: Mapped[list["DatasetVersion"]] = relationship("DatasetVersion", back_populates="dataset")
    analysis_runs: Mapped[list["AnalysisRun"]] = relationship("AnalysisRun", back_populates="dataset")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    row_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="versions")


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mission_id: Mapped[str] = mapped_column(String, ForeignKey("missions.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.id"), nullable=False)
    run_by_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    mission_name: Mapped[str] = mapped_column(String, nullable=False)
    dataset_name: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[AnalysisStatus] = mapped_column(Enum(AnalysisStatus), default=AnalysisStatus.running)
    result: Mapped[dict | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    mission: Mapped["Mission"] = relationship("Mission", back_populates="analysis_runs")
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="analysis_runs")
    anomalies: Mapped[list["Anomaly"]] = relationship("Anomaly", back_populates="analysis_run")


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_run_id: Mapped[str] = mapped_column(String, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    transaction_id: Mapped[str] = mapped_column(String, nullable=False)
    risk_level: Mapped[RiskLevel] = mapped_column(Enum(RiskLevel), nullable=False)
    fraud_score: Mapped[float] = mapped_column(nullable=False)
    amount: Mapped[float | None] = mapped_column()
    features: Mapped[dict | None] = mapped_column(JSON)
    explanation: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis_run: Mapped["AnalysisRun"] = relationship("AnalysisRun", back_populates="anomalies")
    comments: Mapped[list["AnomalyComment"]] = relationship("AnomalyComment", back_populates="anomaly")


class AnomalyComment(Base):
    __tablename__ = "anomaly_comments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    anomaly_id: Mapped[str] = mapped_column(String, ForeignKey("anomalies.id", ondelete="CASCADE"), nullable=False)
    author_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    anomaly: Mapped["Anomaly"] = relationship("Anomaly", back_populates="comments")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mission_id: Mapped[str] = mapped_column(String, ForeignKey("missions.id", ondelete="CASCADE"), nullable=False, index=True)
    generated_by_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    format: Mapped[str] = mapped_column(String, nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String)
    meta: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ModelVersionStatus(str, enum.Enum):
    training = "training"
    staged = "staged"
    production = "production"
    archived = "archived"


class ModelVersion(Base):
    """Tracks every trained version of every model artefact."""
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ModelVersionStatus] = mapped_column(
        Enum(ModelVersionStatus), default=ModelVersionStatus.staged, nullable=False
    )
    storage_path: Mapped[str | None] = mapped_column(String)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    tags: Mapped[dict | None] = mapped_column(JSON)
    promoted_by_id: Mapped[str | None] = mapped_column(String, ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (UniqueConstraint("model_name", "version", name="uq_model_version"),)


class PredictionMonitoringLog(Base):
    """Per-run statistics for online monitoring (drift, anomaly rate, latency)."""
    __tablename__ = "prediction_monitoring_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    prediction_mode: Mapped[str] = mapped_column(String, nullable=False)
    n_transactions: Mapped[int] = mapped_column(Integer, nullable=False)
    n_fraud: Mapped[int] = mapped_column(Integer, nullable=False)
    fraud_rate_pct: Mapped[float] = mapped_column(nullable=False)
    amount_at_risk: Mapped[float | None] = mapped_column()
    latency_ms: Mapped[float | None] = mapped_column()
    feature_stats: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_mission_id", "mission_id"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_timestamp_desc", "timestamp"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    action: Mapped[AuditAction] = mapped_column(Enum(AuditAction), nullable=False)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    user_name: Mapped[str] = mapped_column(String, nullable=False)
    user_role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    mission_id: Mapped[str | None] = mapped_column(String, ForeignKey("missions.id", ondelete="SET NULL"))
    mission_name: Mapped[str | None] = mapped_column(String)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON)  # DB column is "metadata" (Prisma name)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    user: Mapped["User"] = relationship("User", back_populates="audit_logs")
    mission: Mapped["Mission | None"] = relationship("Mission", back_populates="audit_logs")
