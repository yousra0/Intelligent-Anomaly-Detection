"""Alembic environment configuration — uses synchronous psycopg2 for CLI migrations.

Note: the FastAPI runtime uses asyncpg (async). Alembic CLI uses psycopg2 (sync).
Both connect to the same PostgreSQL database; they just use different drivers.
The DATABASE_URL env var uses +asyncpg — this file rewrites it to +psycopg2
so asyncio compatibility issues on Windows (ProactorEventLoop) are avoided.
"""

from __future__ import annotations

import os
import re
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection

from alembic import context

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.db.database import Base
from app.db import models  # noqa: F401 — register all models with Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Convert asyncpg URL → psycopg2 URL for synchronous Alembic CLI
_raw_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/pwcaudit",
)
# Replace +asyncpg driver with +psycopg2 (or plain postgresql://)
SYNC_URL = re.sub(r"\+asyncpg", "", _raw_url)

config.set_main_option("sqlalchemy.url", SYNC_URL)


def run_migrations_offline() -> None:
    context.configure(
        url=SYNC_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        do_run_migrations(connection)
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
