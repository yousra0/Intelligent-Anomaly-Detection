-- ============================================================
-- PwC Audit Analytics Platform — PostgreSQL Migration
-- Version: 001 — Initial Schema
-- Description: Full enterprise schema replacing in-memory stores
-- ============================================================

-- ─────────────────────────────────────────────
-- EXTENSIONS
-- ─────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─────────────────────────────────────────────
-- ENUM TYPES
-- ─────────────────────────────────────────────

CREATE TYPE user_role AS ENUM ('auditor', 'manager', 'partner', 'admin');

CREATE TYPE mission_status AS ENUM ('active', 'in_progress', 'completed', 'archived');

CREATE TYPE mission_type AS ENUM (
  'financial_audit',
  'fraud_detection',
  'compliance_review',
  'risk_assessment',
  'internal_audit'
);

CREATE TYPE dataset_category AS ENUM ('transactions', 'general_ledger', 'trial_balance');
CREATE TYPE dataset_status   AS ENUM ('pending', 'uploaded', 'analyzing', 'analyzed', 'error');

CREATE TYPE analysis_status AS ENUM ('idle', 'running', 'completed', 'error');

CREATE TYPE audit_log_action AS ENUM (
  'login', 'logout',
  'mission.create', 'mission.update', 'mission.delete', 'mission.assign',
  'dataset.upload', 'dataset.delete', 'dataset.replace',
  'analysis.start', 'analysis.complete',
  'report.generate', 'report.download',
  'anomaly.comment', 'anomaly.status_change'
);

-- ─────────────────────────────────────────────
-- USERS & ROLES
-- ─────────────────────────────────────────────

CREATE TABLE users (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email       TEXT UNIQUE NOT NULL,
  name        TEXT NOT NULL,
  password_hash TEXT NOT NULL,              -- bcrypt hash, never store plaintext
  role        user_role NOT NULL DEFAULT 'auditor',
  is_active   BOOLEAN NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert initial admin (password must be hashed with bcrypt at application startup)
-- UPDATE users SET password_hash = crypt('your_password', gen_salt('bf')) WHERE email = 'admin@pwc.com';
INSERT INTO users (id, email, name, role, password_hash) VALUES
  ('u1', 'auditeur@pwc.com', 'Sophie Aubert',  'auditor',  ''),
  ('u2', 'manager@pwc.com',  'Marc Martin',    'manager',  ''),
  ('u3', 'partner@pwc.com',  'Pierre Dupont',  'partner',  ''),
  ('u4', 'admin@pwc.com',    'Admin PwC',      'admin',    '');

-- ─────────────────────────────────────────────
-- MISSIONS
-- ─────────────────────────────────────────────

CREATE TABLE missions (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name         TEXT NOT NULL,
  company_name TEXT NOT NULL,
  mission_type mission_type NOT NULL,
  description  TEXT,
  start_date   DATE NOT NULL,
  end_date     DATE NOT NULL,
  status       mission_status NOT NULL DEFAULT 'active',
  assigned_to  UUID REFERENCES users(id) ON DELETE SET NULL,
  created_by   UUID NOT NULL REFERENCES users(id),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT dates_valid CHECK (end_date >= start_date)
);

CREATE INDEX idx_missions_status     ON missions(status);
CREATE INDEX idx_missions_assigned   ON missions(assigned_to);
CREATE INDEX idx_missions_created_by ON missions(created_by);

-- ─────────────────────────────────────────────
-- MISSION ASSIGNMENTS (many-to-many for multi-auditor missions)
-- ─────────────────────────────────────────────

CREATE TABLE mission_assignments (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  mission_id  UUID NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
  user_id     UUID NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
  assigned_by UUID NOT NULL REFERENCES users(id),
  assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (mission_id, user_id)
);

-- ─────────────────────────────────────────────
-- DATASETS
-- ─────────────────────────────────────────────

CREATE TABLE datasets (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  mission_id  UUID NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  category    dataset_category NOT NULL DEFAULT 'transactions',
  file_size   BIGINT NOT NULL,
  file_type   TEXT NOT NULL,
  storage_key TEXT,                          -- S3 / filesystem key
  status      dataset_status NOT NULL DEFAULT 'uploaded',
  row_count   INTEGER,
  uploaded_by UUID REFERENCES users(id),
  uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  analyzed_at TIMESTAMPTZ,
  version     INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_datasets_mission ON datasets(mission_id);
CREATE INDEX idx_datasets_status  ON datasets(status);

-- ─────────────────────────────────────────────
-- DATASET VERSIONS
-- ─────────────────────────────────────────────

CREATE TABLE dataset_versions (
  id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  dataset_id     UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  version        INTEGER NOT NULL,
  name           TEXT NOT NULL,
  file_size      BIGINT NOT NULL,
  storage_key    TEXT,
  uploaded_by    UUID REFERENCES users(id),
  uploaded_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  change_summary TEXT,
  UNIQUE (dataset_id, version)
);

-- ─────────────────────────────────────────────
-- MODEL VERSIONS
-- ─────────────────────────────────────────────

CREATE TABLE model_versions (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  model_name       TEXT NOT NULL,               -- 'xgboost', 'autoencoder'
  version          TEXT NOT NULL,               -- e.g. 'v1.2.0'
  artifact_path    TEXT NOT NULL,
  recall           FLOAT,
  precision        FLOAT,
  f1               FLOAT,
  pr_auc           FLOAT,
  roc_auc          FLOAT,
  optimal_threshold FLOAT,
  is_in_production BOOLEAN NOT NULL DEFAULT FALSE,
  trained_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  trained_by       UUID REFERENCES users(id),
  notes            TEXT,
  UNIQUE (model_name, version)
);

-- ─────────────────────────────────────────────
-- ANALYSIS HISTORY
-- ─────────────────────────────────────────────

CREATE TABLE analysis_runs (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  mission_id       UUID NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
  dataset_id       UUID REFERENCES datasets(id) ON DELETE SET NULL,
  model_version_id UUID REFERENCES model_versions(id) ON DELETE SET NULL,
  model_mode       TEXT NOT NULL,               -- 'combined', 'xgboost', 'autoencoder'
  status           analysis_status NOT NULL DEFAULT 'running',
  n_transactions   INTEGER,
  n_fraud          INTEGER,
  fraud_rate_pct   FLOAT,
  amount_at_risk   FLOAT,
  result_json      JSONB,                       -- Full PredictResponse stored for replay
  error_message    TEXT,
  started_by       UUID REFERENCES users(id),
  started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at     TIMESTAMPTZ
);

CREATE INDEX idx_analysis_runs_mission ON analysis_runs(mission_id);
CREATE INDEX idx_analysis_runs_status  ON analysis_runs(status);
CREATE INDEX idx_analysis_runs_started ON analysis_runs(started_at DESC);

-- ─────────────────────────────────────────────
-- ANOMALY STATUS (per-transaction review workflow)
-- ─────────────────────────────────────────────

CREATE TABLE anomaly_reviews (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  analysis_run_id UUID NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  tx_id           BIGINT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'pending',   -- 'pending', 'confirmed', 'dismissed', 'escalated'
  risk_level      TEXT NOT NULL,                     -- 'CRITIQUE', 'ELEVE', 'FAIBLE'
  reviewed_by     UUID REFERENCES users(id),
  reviewed_at     TIMESTAMPTZ,
  UNIQUE (analysis_run_id, tx_id)
);

CREATE INDEX idx_anomaly_reviews_run ON anomaly_reviews(analysis_run_id);

-- ─────────────────────────────────────────────
-- ANOMALY STATUS HISTORY
-- ─────────────────────────────────────────────

CREATE TABLE anomaly_status_history (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  anomaly_id      UUID NOT NULL REFERENCES anomaly_reviews(id) ON DELETE CASCADE,
  old_status      TEXT,
  new_status      TEXT NOT NULL,
  changed_by      UUID NOT NULL REFERENCES users(id),
  changed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  reason          TEXT
);

-- ─────────────────────────────────────────────
-- ANOMALY COMMENTS
-- ─────────────────────────────────────────────

CREATE TABLE anomaly_comments (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  anomaly_id      UUID NOT NULL REFERENCES anomaly_reviews(id) ON DELETE CASCADE,
  author_id       UUID NOT NULL REFERENCES users(id),
  content         TEXT NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_anomaly_comments_anomaly ON anomaly_comments(anomaly_id);

-- ─────────────────────────────────────────────
-- REPORTS
-- ─────────────────────────────────────────────

CREATE TABLE reports (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  mission_id      UUID NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
  analysis_run_id UUID REFERENCES analysis_runs(id) ON DELETE SET NULL,
  format          TEXT NOT NULL CHECK (format IN ('pdf', 'docx')),
  name            TEXT NOT NULL,
  storage_key     TEXT,                           -- S3 / filesystem key
  generated_by    UUID REFERENCES users(id),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────
-- AUDIT LOGS
-- ─────────────────────────────────────────────

CREATE TABLE audit_logs (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  action       audit_log_action NOT NULL,
  user_id      UUID REFERENCES users(id) ON DELETE SET NULL,
  user_name    TEXT NOT NULL,                    -- denormalized for historical records
  user_role    user_role NOT NULL,
  mission_id   UUID REFERENCES missions(id) ON DELETE SET NULL,
  mission_name TEXT,                             -- denormalized for historical records
  details      TEXT NOT NULL,
  ip_address   INET,
  user_agent   TEXT,
  timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_user      ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_mission   ON audit_logs(mission_id);
CREATE INDEX idx_audit_logs_action    ON audit_logs(action);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp DESC);

-- ─────────────────────────────────────────────
-- AUTOMATIC updated_at TRIGGER
-- ─────────────────────────────────────────────

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_missions_updated_at
  BEFORE UPDATE ON missions
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_anomaly_comments_updated_at
  BEFORE UPDATE ON anomaly_comments
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ─────────────────────────────────────────────
-- ROW-LEVEL SECURITY (RLS) — enable in production
-- ─────────────────────────────────────────────

-- Uncomment after configuring auth.uid() from JWT in pg_session
-- ALTER TABLE missions ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY missions_auditor ON missions FOR SELECT
--   USING (assigned_to::text = current_setting('app.user_id', true)
--          OR created_by::text = current_setting('app.user_id', true));
-- CREATE POLICY missions_manager ON missions FOR ALL
--   USING (current_setting('app.user_role', true) IN ('manager', 'admin'));
