# Plan Backend FastAPI — Détection de Fraude PFE

## Projet
- **Répertoire** : `C:\Users\lenovo\Desktop\anomaly_detection_project`
- **Python** : 3.10 · `.venv` · VS Code
- **GPU** : NVIDIA MX450 (PyTorch)

---

## Artefacts disponibles (déjà produits par les notebooks)

```
outputs/models/
├── scaler.pkl                  ← StandardScaler (joblib)
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
```

---

## Structure à créer

```
app/
├── __init__.py
├── main.py
├── routes/
│   ├── __init__.py
│   ├── predict.py
│   ├── explain.py
│   ├── report.py
│   └── models.py
└── services/
    ├── __init__.py
    ├── predictor.py
    ├── explainer.py
    ├── llm_service.py
    └── report_gen.py
requirements_app.txt
tests/
├── test_predict.py
├── test_explain.py
└── conftest.py
```

---

## Preprocessing exact (depuis NB02)

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
    # OHE type
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

## Chargement des modèles

```python
# LR et RF : dict{"model": estimateur_sklearn}
payload = joblib.load("lr_balanced.pkl")
lr_balanced = payload["model"] if isinstance(payload, dict) else payload

# XGBoost : XGBClassifier direct
xgb_smote = joblib.load("xgb_smote.pkl")
# Si dict : xgb_smote = payload["model"]

# AutoEncoder PyTorch
from src.models.autoencoder import FraudAutoEncoder
ae = FraudAutoEncoder.load(MODELS_DIR / "autoencoder")
# ae.threshold = 1.753
# ae.predict_score(X) → np.ndarray (N,)

# Scaler
scaler = joblib.load("scaler.pkl")
# scaler.transform(X_df[SCALE_COLS])
```

---

## Les 5 endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/api/health` | GET | Vérification que les modèles sont chargés |
| `/api/predict` | POST | Upload CSV → fraudes détectées |
| `/api/explain/{tx_id}` | GET | SHAP + LIME + LLM pour 1 transaction |
| `/api/report` | POST | Génère le PDF PwC |
| `/api/models` | GET | Métriques comparatives des 7 modèles |

---

## Tests à écrire

1. `test_health` → GET /api/health → status 200, models_loaded=True
2. `test_predict_valid_csv` → POST CSV valide → n_fraud >= 0, liste transactions
3. `test_predict_missing_columns` → POST CSV incomplet → status 422
4. `test_predict_empty_csv` → POST CSV vide → status 400
5. `test_explain_valid` → GET /api/explain/0 → shap_values, lime_rules, llm
6. `test_explain_invalid_id` → GET /api/explain/999999 → status 404
7. `test_models_endpoint` → GET /api/models → liste de 7 modèles avec recall, f1
8. `test_report_generation` → POST /api/report → Content-Type application/pdf