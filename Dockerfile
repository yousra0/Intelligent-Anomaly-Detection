# ── FastAPI ML Backend ────────────────────────────────────────────────────────
# Multi-stage build: builder installs deps, runner is lean.

FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed by some Python packages (e.g. psycopg2, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_app.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_app.txt

# Install ml_core package (pyproject.toml at repo root)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS runner

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .
COPY config/ config/
COPY outputs/ outputs/

# Persistent uploads volume mount point
RUN mkdir -p /app/uploads
VOLUME ["/app/uploads"]

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Run migrations then start the server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
