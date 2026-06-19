"""Add model_versions and prediction_monitoring_logs tables

Revision ID: 002
Revises: 001
Create Date: 2026-06-17

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE modelversion_status AS ENUM ('training','staged','production','archived');
            EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("training", "staged", "production", "archived",
                            name="modelversion_status", create_type=False),
            nullable=False,
            server_default="staged",
        ),
        sa.Column("storage_path", sa.String(), nullable=True),
        sa.Column("metrics", postgresql.JSON(), nullable=True),
        sa.Column("tags", postgresql.JSON(), nullable=True),
        sa.Column("promoted_by_id", sa.String(),
                  sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("model_name", "version", name="uq_model_version"),
    )
    op.create_index("ix_model_versions_model_name", "model_versions", ["model_name"])

    op.create_table(
        "prediction_monitoring_logs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(),
                  sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("prediction_mode", sa.String(), nullable=False),
        sa.Column("n_transactions", sa.Integer(), nullable=False),
        sa.Column("n_fraud", sa.Integer(), nullable=False),
        sa.Column("fraud_rate_pct", sa.Float(), nullable=False),
        sa.Column("amount_at_risk", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("feature_stats", postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False, index=True),
    )
    op.create_index("ix_monitoring_run_id", "prediction_monitoring_logs", ["run_id"])
    op.create_index("ix_monitoring_model_name", "prediction_monitoring_logs", ["model_name"])


def downgrade() -> None:
    op.drop_table("prediction_monitoring_logs")
    op.drop_table("model_versions")
    op.execute("DROP TYPE IF EXISTS modelversion_status;")
