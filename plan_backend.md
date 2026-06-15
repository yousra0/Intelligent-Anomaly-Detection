# Plan Backend FastAPI — Détection de Fraude PFE

## Statut global : COMPLÉTÉ ✓

- **Répertoire** : `C:\Users\lenovo\Desktop\anomaly_detection_project`
- **Python** : 3.10 · `.venv` · VS Code
- **GPU** : NVIDIA MX450 (PyTorch)

---

## Artefacts disponibles (produits par les notebooks)

```
outputs/models/
├── scaler.pkl                  ← StandardScaler (joblib) — 6 colonnes SCALE_COLS
├── features.json               ← 14 features + metadata
├── optimal_thresholds.json     ← {LR_balanced:0.999, RF_smote:0.629, XGB_smote:0.355, ...}
├── lr_balanced.pkl             ← dict{"model": LogisticRegression}  (joblib)
├── lr_smote.pkl
├── rf_balanced.pkl             ← dict{"model": RandomForestClassifier}
├── rf_smote.pkl
├── xgb_smote.pkl               ← XGBClassifier direct (joblib)
├── iso_forest.pkl              ← IsolationForest direct (joblib)
├── iso_forest_scaler.pkl       ← MinMaxScaler pour les scores AE
├── autoencoder/
│   ├── autoencoder_weights.pth ← poids PyTorch
│   └── autoencoder_meta.pkl    ← metadata (threshold=1.753, architecture)
├── ae_scores_test.npy          ← scores MSE sur test set (30001,)
├── ae_scores_val.npy
└── shap_values_rfsmote.npy     ← (500, 14)

outputs/reports/
├── baseline_report.json        ← métriques des 7 modèles
├── autoencoder_report.json
├── shap_report.json
└── lime_report.json

config/
├── llm_config.yaml             ← provider:groq · model:llama3-8b-8192 · api_key
└── config.yaml

src/
├── models/autoencoder.py       ← class FraudAutoEncoder (PyTorch)
├── models/ml_models.py         ← FraudLogisticRegression, FraudRandomForest
├── preprocessing/preprocessor.py
├── feature_engineering/feature_builder.py
├── utils/evaluator.py
├── llm_integration/llm_helper.py  ← class LLMHelper (Groq/Gemini/HuggingFace)
└── explainability/
    ├── shap_explainer.py
    └── lime_explainer.py

exemple_rapport.docx            ← Template Word PwC ({{placeholders}})
```

---

## Structure créée

```
app/
├── __init__.py
├── main.py                     ← lifespan, CORS (5173/3000), routeurs, health check
├── routes/
│   ├── __init__.py
│   ├── predict.py              ← POST /api/predict
│   ├── explain.py              ← GET  /api/explain/{tx_id}
│   ├── report.py               ← POST /api/report  +  POST /api/report/docx
│   ├── models.py               ← GET  /api/models
│   └── profile.py              ← POST /api/profile
└── services/
    ├── __init__.py
    ├── predictor.py            ← load_all_models(), build_features(), predict_batch()
    ├── generic_predictor.py    ← predict_generic_batch() — modes ae_isoforest/ae_only/isoforest
    ├── explainer.py            ← compute_shap(), compute_lime() (standalone, sans src/)
    ├── llm_service.py          ← get_llm_helper() → LLMHelper wrapper
    ├── column_mapper.py        ← ColumnMapper — 4 niveaux de confiance
    ├── dataset_profiler.py     ← DatasetProfiler — stats, qualité, types sémantiques
    ├── schema_detector.py      ← detect_schema_mode() → SchemaDetectionResult
    ├── feature_builder.py      ← DynamicFeatureBuilder — 14 features + fallbacks
    ├── feature_engineer.py     ← FeatureEngineer — 15 features dérivées
    ├── report_gen.py           ← PwCReport(FPDF) — PDF 8 sections
    └── report_gen_docx.py      ← injection template Word python-docx

tests/
├── conftest.py                 ← fixtures: client, test_csv (PaySim 20 lignes), test_csv_invalid
├── test_predict.py             ← 298 lignes — intégration /predict tous modes
├── test_profiler.py            ← 351 lignes — DatasetProfiler unit + intégration
├── test_feature_engineer.py    ← 463 lignes — temporel, balance, comportemental
├── test_column_mapper.py       ← 238 lignes — alias, fuzzy, normalisation
├── test_explain.py             ← 55 lignes  — SHAP, LIME, LLM, tx_id invalide
├── test_models.py              ← 42 lignes  — liste modèles, métriques, best flag
├── test_report.py              ← 30 lignes  — PDF Content-Type + headers
├── test_preprocessing.py       ← 43 lignes  — utilitaires preprocessing
├── test_llm.py                 ← placeholder (à compléter)
├── test_utils.py               ← 38 lignes  — évaluateur, anomaly_utils
└── test_visualization.py       ← placeholder
```

---

## Preprocessing exact (depuis NB02) — implémenté dans feature_builder.py

```python
HIGH_RISK_HOURS = [0,1,2,3,4,5,6,7,8,9,23]
SCALE_COLS = ['step','hour','day','week','log_amount','balance_diff_orig']

def build_features(df):
    df = df.copy()
    df['hour'] = df['step'] % 24
    df['day']  = df['step'] // 24
    df['week'] = df['step'] // 168
    df['high_risk_hour']         = df['hour'].isin(HIGH_RISK_HOURS).astype(int)
    df['is_transfer_or_cashout'] = df['type'].isin(['TRANSFER','CASH_OUT']).astype(int)
    df['balance_diff_orig']      = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['dest_zero_balance']      = (df['oldbalanceDest'] == 0).astype(int)
    df['log_amount']             = np.log1p(df['amount'])
    for t in ['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']:
        df[f'type_{t}'] = (df['type'] == t).astype(int)
    return df

FEATURE_COLS = [
    'step','hour','day','week','high_risk_hour','is_transfer_or_cashout',
    'balance_diff_orig','dest_zero_balance',
    'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER',
    'log_amount'
]
```

---

## Chargement des modèles — implémenté dans predictor.py

```python
# LR et RF : dict{"model": estimateur_sklearn}
payload = joblib.load("lr_balanced.pkl")
lr_balanced = payload["model"] if isinstance(payload, dict) else payload

# XGBoost : XGBClassifier direct
xgb_smote = joblib.load("xgb_smote.pkl")

# AutoEncoder PyTorch
from src.models.autoencoder import FraudAutoEncoder
ae = FraudAutoEncoder.load(MODELS_DIR / "autoencoder")
# ae.threshold = 1.753

# Scaler
scaler = joblib.load("scaler.pkl")
# scaler.transform(X_df[SCALE_COLS])
```

---

## Les 6 endpoints — tous implémentés

| Endpoint | Méthode | Statut | Description |
|---|---|---|---|
| `/api/health` | GET | ✓ | Vérification modèles + LLM |
| `/api/predict` | POST | ✓ | Upload CSV → fraudes détectées (4 modes auto) |
| `/api/explain/{tx_id}` | GET | ✓ | SHAP + LIME + LLM pour 1 transaction |
| `/api/report` | POST | ✓ | Génère le PDF PwC (fpdf2, streaming) |
| `/api/report/docx` | POST | ✓ | Génère le DOCX PwC (python-docx, template) |
| `/api/models` | GET | ✓ | Métriques comparatives des 7 modèles |
| `/api/profile` | POST | ✓ | Profilage dataset sans prédiction |

---

## Tests écrits vs planifiés

| Test | Statut | Fichier |
|---|---|---|
| `test_health` → GET /api/health → 200, models_loaded=True | ✓ | test_predict.py |
| `test_predict_valid_csv` → POST CSV valide → n_fraud >= 0 | ✓ | test_predict.py |
| `test_predict_missing_columns` → CSV incomplet → 422 | ✓ | test_predict.py |
| `test_predict_empty_csv` → CSV vide → 400 | ✓ | test_predict.py |
| `test_predict_generic_modes` → ae_isoforest / ae_only / isoforest | ✓ | test_predict.py |
| `test_explain_valid` → shap_values, lime_rules, llm | ✓ | test_explain.py |
| `test_explain_invalid_id` → 404 | ✓ | test_explain.py |
| `test_models_endpoint` → 7 modèles avec recall, f1 | ✓ | test_models.py |
| `test_report_generation` → Content-Type application/pdf | ✓ | test_report.py |
| `test_profiler_*` → 12 tests unitaires | ✓ | test_profiler.py |
| `test_column_mapper_*` → alias, fuzzy, normalisation | ✓ | test_column_mapper.py |
| `test_feature_engineer_*` → temporel, balance, comportemental | ✓ | test_feature_engineer.py |
| `test_llm_*` | ☐ À compléter | test_llm.py |
| `test_report_docx` → Content-Type DOCX | ☐ À compléter | test_report.py |

---

## Décisions architecturales prises

| Décision | Choix | Raison |
|---|---|---|
| Explainer | Standalone dans `app/services/explainer.py` | Évite dépendance circulaire avec `src/` |
| IsoForest générique | Refitté par batch (transductif) | Pas de données d'entraînement disponibles pour le mode générique |
| Cache résultats | `app.state.results_cache` (dict en mémoire) | Suffisant pour démo PFE ; pas de Redis nécessaire |
| Authentification | Aucune | Hors scope PFE, noter pour future prod |
| Rapports | PDF (fpdf2) + DOCX (python-docx) | Double format : PDF pour présentation, DOCX pour édition |
| Langue | Français intégral | Contexte audit PwC Tunisie, auditeurs non-techniques |
