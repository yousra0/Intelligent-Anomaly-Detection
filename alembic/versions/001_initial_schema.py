"""Initial schema — all tables

Revision ID: 001
Revises:
Create Date: 2026-06-17

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enums
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE userrole AS ENUM ('auditor','manager','partner','admin');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE userstatus AS ENUM ('active','inactive','suspended');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE missionstatus AS ENUM ('active','in_progress','completed','archived');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE missiontype AS ENUM ('financial_audit','fraud_detection','compliance_review','risk_assessment','internal_audit');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE datasetcategory AS ENUM ('transactions','general_ledger','trial_balance');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE datasetstatus AS ENUM ('pending','uploaded','analyzing','analyzed','error');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE analysisstatus AS ENUM ('running','completed','failed');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE risklevel AS ENUM ('CRITIQUE','ELEVE','FAIBLE');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE auditaction AS ENUM (
                'login','logout','mission_create','mission_update','mission_delete','mission_assign',
                'dataset_upload','dataset_delete','dataset_replace',
                'analysis_start','analysis_complete',
                'report_generate','report_download',
                'anomaly_comment','anomaly_status_change',
                'user_create','user_update','user_deactivate','role_modify'
            );
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)

    # users
    op.create_table("users",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("email", sa.String, nullable=False, unique=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("role", sa.Enum("auditor","manager","partner","admin", name="userrole"), nullable=False),
        sa.Column("status", sa.Enum("active","inactive","suspended", name="userstatus"), nullable=False, server_default="active"),
        sa.Column("phone", sa.String),
        sa.Column("position", sa.String),
        sa.Column("department", sa.String),
        sa.Column("password", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # missions
    op.create_table("missions",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("company_name", sa.String, nullable=False),
        sa.Column("mission_type", sa.Enum("financial_audit","fraud_detection","compliance_review","risk_assessment","internal_audit", name="missiontype"), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("status", sa.Enum("active","in_progress","completed","archived", name="missionstatus"), nullable=False, server_default="active"),
        sa.Column("start_date", sa.String, nullable=False),
        sa.Column("end_date", sa.String),
        sa.Column("created_by_id", sa.String, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("assigned_to_id", sa.String, sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # mission_assignments
    op.create_table("mission_assignments",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("mission_id", sa.String, sa.ForeignKey("missions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("mission_id", "user_id", name="uq_mission_user"),
    )

    # datasets
    op.create_table("datasets",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("mission_id", sa.String, sa.ForeignKey("missions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("uploaded_by_id", sa.String, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("original_name", sa.String, nullable=False),
        sa.Column("category", sa.Enum("transactions","general_ledger","trial_balance", name="datasetcategory"), nullable=False, server_default="transactions"),
        sa.Column("status", sa.Enum("pending","uploaded","analyzing","analyzed","error", name="datasetstatus"), nullable=False, server_default="uploaded"),
        sa.Column("row_count", sa.Integer),
        sa.Column("column_count", sa.Integer),
        sa.Column("file_size_bytes", sa.BigInteger),
        sa.Column("storage_path", sa.String),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_datasets_mission_id", "datasets", ["mission_id"])

    # dataset_versions
    op.create_table("dataset_versions",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("dataset_id", sa.String, sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("storage_path", sa.String, nullable=False),
        sa.Column("row_count", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
    )

    # analysis_runs
    op.create_table("analysis_runs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("mission_id", sa.String, sa.ForeignKey("missions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", sa.String, sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("run_by_id", sa.String, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("mission_name", sa.String, nullable=False),
        sa.Column("dataset_name", sa.String, nullable=False),
        sa.Column("model", sa.String, nullable=False),
        sa.Column("status", sa.Enum("running","completed","failed", name="analysisstatus"), nullable=False, server_default="running"),
        sa.Column("result", postgresql.JSON),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_analysis_runs_mission_id", "analysis_runs", ["mission_id"])

    # anomalies
    op.create_table("anomalies",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("analysis_run_id", sa.String, sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("transaction_id", sa.String, nullable=False),
        sa.Column("risk_level", sa.Enum("CRITIQUE","ELEVE","FAIBLE", name="risklevel"), nullable=False),
        sa.Column("fraud_score", sa.Float, nullable=False),
        sa.Column("amount", sa.Float),
        sa.Column("features", postgresql.JSON),
        sa.Column("explanation", sa.Text),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_anomalies_run_id", "anomalies", ["analysis_run_id"])

    # anomaly_comments
    op.create_table("anomaly_comments",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("anomaly_id", sa.String, sa.ForeignKey("anomalies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("author_id", sa.String, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # reports
    op.create_table("reports",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("mission_id", sa.String, sa.ForeignKey("missions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("generated_by_id", sa.String, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("format", sa.String, nullable=False),
        sa.Column("file_name", sa.String, nullable=False),
        sa.Column("storage_path", sa.String),
        sa.Column("meta", postgresql.JSON),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_reports_mission_id", "reports", ["mission_id"])

    # audit_logs
    op.create_table("audit_logs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("action", sa.Enum(
            "login","logout","mission_create","mission_update","mission_delete","mission_assign",
            "dataset_upload","dataset_delete","dataset_replace",
            "analysis_start","analysis_complete",
            "report_generate","report_download",
            "anomaly_comment","anomaly_status_change",
            "user_create","user_update","user_deactivate","role_modify",
            name="auditaction"
        ), nullable=False),
        sa.Column("user_id", sa.String, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("user_name", sa.String, nullable=False),
        sa.Column("user_role", sa.Enum("auditor","manager","partner","admin", name="userrole"), nullable=False),
        sa.Column("mission_id", sa.String, sa.ForeignKey("missions.id", ondelete="SET NULL")),
        sa.Column("mission_name", sa.String),
        sa.Column("details", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSON),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_mission_id", "audit_logs", ["mission_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("reports")
    op.drop_table("anomaly_comments")
    op.drop_table("anomalies")
    op.drop_table("analysis_runs")
    op.drop_table("dataset_versions")
    op.drop_table("datasets")
    op.drop_table("mission_assignments")
    op.drop_table("missions")
    op.drop_table("users")
