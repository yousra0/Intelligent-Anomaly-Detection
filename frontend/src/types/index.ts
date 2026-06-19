// ─────────────────────────────────────────────
// Auth / Users
// ─────────────────────────────────────────────

export type UserRole = "auditor" | "manager" | "partner" | "admin";

export type UserStatus = "active" | "inactive";

export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  phone?: string;
  position?: string;
  department?: string;
  status?: UserStatus;
  created_at?: string;
}

export interface CreateUserPayload {
  first_name: string;
  last_name: string;
  email: string;
  phone?: string;
  position?: string;
  department?: string;
  password: string;
  role: UserRole;
}

export interface UpdateUserPayload {
  first_name?: string;
  last_name?: string;
  email?: string;
  phone?: string;
  position?: string;
  department?: string;
  role?: UserRole;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}

// ─────────────────────────────────────────────
// Missions
// ─────────────────────────────────────────────

export type MissionStatus = "active" | "in_progress" | "completed" | "archived";
export type MissionType =
  | "financial_audit"
  | "fraud_detection"
  | "compliance_review"
  | "risk_assessment"
  | "internal_audit";

export interface Mission {
  id: string;
  name: string;
  company_name: string;
  mission_type: MissionType;
  description?: string;
  start_date: string;
  end_date: string;
  status: MissionStatus;
  assigned_to?: string;
  assigned_auditors?: string[];
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface CreateMissionPayload {
  name: string;
  company_name: string;
  mission_type: MissionType;
  description?: string;
  start_date: string;
  end_date: string;
  assigned_to?: string;
  assigned_auditors?: string[];
}

// ─────────────────────────────────────────────
// Datasets
// ─────────────────────────────────────────────

export type DatasetCategory = "transactions" | "general_ledger" | "trial_balance";
export type DatasetStatus = "pending" | "uploaded" | "analyzing" | "analyzed" | "error";

export interface Dataset {
  id: string;
  mission_id: string;
  name: string;
  category: DatasetCategory;
  file_size: number;
  file_type: string;
  status: DatasetStatus;
  row_count?: number;
  uploaded_at: string;
  analyzed_at?: string;
}

// ─────────────────────────────────────────────
// Analysis / Prediction  (mirrors FastAPI response)
// ─────────────────────────────────────────────

export type AnalysisModel = "standard" | "ae_isoforest" | "ae_only" | "isoforest";
export type RiskLevel = "CRITIQUE" | "ELEVE" | "FAIBLE";

export interface ColumnMapping {
  original_name: string;
  confidence: number;
}

export interface SchemaDetection {
  mode: AnalysisModel;
  n_mapped: number;
  n_required: number;
  avg_confidence: number;
  use_xgb: boolean;
  use_ae: boolean;
  use_isoforest: boolean;
  models_used: string;
  reason: string;
  warnings: string[];
}

export interface TransactionResult {
  tx_id: number;
  type?: string;
  amount?: number;
  xgb_score?: number;
  ae_score?: number;
  risk_level: RiskLevel;
  is_fraud_predicted: boolean;
}

export interface DatasetProfile {
  n_rows: number;
  n_cols: number;
  global_quality_score: number;
  numeric_cols: string[];
  categorical_cols: string[];
  datetime_cols: string[];
  identifier_cols: string[];
  quasi_constant_cols: string[];
  high_missing_cols: string[];
  recommendations: string[];
  profiling_time_ms: number;
}

export interface FeatureEngineering {
  n_generated: number;
}

export interface FeatureBuild {
  n_features: number;
}

export interface PredictResponse {
  n_transactions: number;
  n_fraud: number;
  fraud_rate_pct: number;
  amount_at_risk: number;
  model_used: string;
  prediction_mode: AnalysisModel;
  schema_detection: SchemaDetection;
  column_mapping: Record<string, ColumnMapping>;
  mapping_warnings: string[];
  feature_engineering: FeatureEngineering;
  transactions: TransactionResult[];
  threshold: number;
  feature_build: FeatureBuild;
  dataset_profile: DatasetProfile;
}

// ─────────────────────────────────────────────
// Explanations
// ─────────────────────────────────────────────

export interface AuditInfo {
  hash: string;
  timestamp_utc: string;
  hash_algo: string;
}

export interface LLMExplanation {
  risk_level: RiskLevel;
  risk_score: number;
  resume: string;
  raisons: string[];
  actions_recommandees: string[];
  status: "ok" | "fallback" | "error";
  _audit?: AuditInfo;
}

export interface AETopFeature {
  feature: string;
  error: number;
}

export interface ExplainResponse {
  tx_id: number;
  xgb_score?: number;
  ae_score?: number;
  risk_level: RiskLevel;
  feature_values: Record<string, number>;
  shap_values_xgb?: Record<string, number>;
  ae_feature_errors?: Record<string, number>;
  ae_top_features?: AETopFeature[];
  lime_rules?: string[];
  llm: LLMExplanation;
}

export interface BatchExplainRequest {
  tx_ids: number[];
  max_explain?: number;
}

export interface BatchExplainResponse {
  n_requested: number;
  n_explained: number;
  n_errors: number;
  explanations: ExplainResponse[];
  errors: Array<{ tx_id: number; error: string }>;
}

// ─────────────────────────────────────────────
// Models / Metrics
// ─────────────────────────────────────────────

export interface ModelMetrics {
  name: string;
  recall: number;
  precision: number;
  f1: number;
  pr_auc: number;
  roc_auc: number;
  train_time_s: number;
  optimal_threshold: number;
  is_best: boolean;
  is_in_production: boolean;
}

export interface ModelsResponse {
  models: ModelMetrics[];
}

// ─────────────────────────────────────────────
// Analysis Run (frontend state + persistence)
// ─────────────────────────────────────────────

export type AnalysisStatus = "idle" | "running" | "completed" | "error";

export interface AnalysisRun {
  id: string;
  mission_id: string;
  dataset_id: string;
  model_mode: AnalysisModel | "combined";
  status: AnalysisStatus;
  started_at: string;
  completed_at?: string;
  result?: PredictResponse;
  error_message?: string;
}

export interface AnalysisRunRecord extends AnalysisRun {
  dataset_name: string;
  mission_name: string;
  company_name: string;
  user_id: string;
  user_name: string;
}

// ─────────────────────────────────────────────
// Reports
// ─────────────────────────────────────────────

export interface Report {
  id: string;
  mission_id: string;
  analysis_run_id: string;
  format: "pdf" | "docx";
  name: string;
  created_at: string;
  download_url: string;
}

// ─────────────────────────────────────────────
// Audit Trail
// ─────────────────────────────────────────────

export type AuditLogAction =
  | "login"
  | "logout"
  | "mission.create"
  | "mission.update"
  | "mission.delete"
  | "mission.assign"
  | "dataset.upload"
  | "dataset.delete"
  | "dataset.replace"
  | "analysis.start"
  | "analysis.complete"
  | "report.generate"
  | "report.download"
  | "anomaly.comment"
  | "anomaly.status_change"
  | "user_create"
  | "user_update"
  | "user_disable"
  | "user_activate"
  | "user_reset_password";

export interface AuditLog {
  id: string;
  action: AuditLogAction;
  user_id: string;
  user_name: string;
  user_role: UserRole;
  mission_id?: string;
  mission_name?: string;
  details: string;
  timestamp: string;
}

// ─────────────────────────────────────────────
// Health
// ─────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  llm_available: boolean;
}

// ─────────────────────────────────────────────
// UI Helpers
// ─────────────────────────────────────────────

export interface PaginationState {
  page: number;
  pageSize: number;
  total: number;
}

export interface FilterState {
  search: string;
  riskLevel: RiskLevel | "all";
  isFraud: boolean | "all";
}
