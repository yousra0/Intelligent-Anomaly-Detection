# Guide du Backend FastAPI — Système de Détection d'Anomalies

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Structure du projet](#2-structure-du-projet)
3. [Installation et configuration](#3-installation-et-configuration)
4. [Lancement du serveur](#4-lancement-du-serveur)
5. [Endpoints de l'API](#5-endpoints-de-lapi)
6. [Pipeline de prédiction](#6-pipeline-de-prédiction)
7. [Intégration LLM](#7-intégration-llm)
8. [Génération de rapports PDF](#8-génération-de-rapports-pdf)
9. [Tests](#9-tests)
10. [Variables d'environnement](#10-variables-denvironnement)
11. [Architecture des services](#11-architecture-des-services)
12. [Notes de déploiement](#12-notes-de-déploiement)

---

## 1. Vue d'ensemble

Le backend est une API REST construite avec **FastAPI** exposant un pipeline complet de détection de fraude financière :

- **Prédiction multi-modèles** : XGBoost, Random Forest, Régression Logistique, Isolation Forest, AutoEncoder (PyTorch)
- **Détection automatique de schéma CSV** : adapte le pipeline aux colonnes disponibles
- **Explicabilité** : SHAP + LIME + explication LLM en langage naturel
- **Profilage de dataset** : analyse qualité, statistiques, recommandations
- **Génération de rapports PDF** à la charte PwC
- **Intégration LLM** : Groq (llama-3.1-8b), Gemini, ou HuggingFace

---

## 2. Structure du projet

```
anomaly_detection_project/
├── app/                          ← Backend FastAPI
│   ├── main.py                   ← Point d'entrée, lifespan, CORS, routeurs
│   ├── routes/
│   │   ├── predict.py            ← POST /api/predict
│   │   ├── explain.py            ← GET  /api/explain/{tx_id}
│   │   ├── report.py             ← POST /api/report
│   │   ├── models.py             ← GET  /api/models
│   │   └── profile.py            ← POST /api/profile
│   └── services/
│       ├── predictor.py          ← Chargement modèles + pipeline PaySim
│       ├── generic_predictor.py  ← Pipeline fallback (schéma inconnu)
│       ├── llm_service.py        ← Wrapper LLMHelper
│       ├── explainer.py          ← Calcul SHAP & LIME
│       ├── dataset_profiler.py   ← Analyse qualité du dataset
│       ├── column_mapper.py      ← Détection sémantique des colonnes
│       ├── schema_detector.py    ← Sélection du mode de prédiction
│       ├── feature_builder.py    ← Construction features adaptatives (14 features)
│       ├── feature_engineer.py   ← Features dérivées (temporel, balance, comportemental)
│       └── report_gen.py         ← Génération PDF avec fpdf2
│
├── src/                          ← Bibliothèque ML (ml_core)
│   ├── models/                   ← Définitions AutoEncoder + modèles ML
│   ├── feature_engineering/      ← Features temporelles, comportementales
│   ├── llm_integration/          ← LLMHelper (appels API + retry + parsing JSON)
│   ├── preprocessing/            ← Chargement et nettoyage données
│   ├── explainability/           ← SHAP et LIME
│   └── utils/                    ← Évaluateur, utilitaires
│
├── config/
│   ├── config.yaml               ← Configuration générale du projet
│   └── llm_config.yaml           ← Provider LLM, modèles, paramètres de génération
│
├── outputs/
│   └── models/
│       ├── xgb_smote.pkl         ← XGBoost (modèle principal)
│       ├── rf_smote.pkl          ← Random Forest SMOTE
│       ├── rf_balanced.pkl       ← Random Forest balanced
│       ├── lr_smote.pkl          ← Régression Logistique SMOTE
│       ├── lr_balanced.pkl       ← Régression Logistique balanced
│       ├── iso_forest.pkl        ← Isolation Forest
│       ├── iso_forest_scaler.pkl ← Scaler pour IsoForest
│       ├── scaler.pkl            ← StandardScaler principal (6 colonnes)
│       ├── features.json         ← Liste des 14 features attendues
│       ├── optimal_thresholds.json ← Seuils optimaux par modèle
│       └── autoencoder/
│           ├── autoencoder_weights.pt  ← Poids PyTorch
│           └── autoencoder_meta.pkl    ← Métadonnées (threshold, architecture)
│
├── tests/                        ← Suite de tests pytest
├── .env                          ← Clés API (ne pas commiter)
├── requirements.txt
└── pyproject.toml
```

---

## 3. Installation et configuration

### Prérequis

- Python 3.10 ou supérieur
- pip ou uv

### Installation des dépendances

```bash
# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows)
.venv\Scripts\activate

# Activer l'environnement (Linux/Mac)
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Installer le package ml_core en mode éditable
pip install -e .
```

### Configurer les clés API

Créer un fichier `.env` à la racine du projet :

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxx
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Le provider actif est contrôlé dans [config/llm_config.yaml](config/llm_config.yaml) :

```yaml
active_provider: groq   # options: groq | gemini | huggingface
```

---

## 4. Lancement du serveur

### Développement (avec rechargement automatique)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Vérifier que le serveur fonctionne

```bash
curl http://localhost:8000/api/health
```

Réponse attendue :
```json
{
  "status": "ok",
  "models_loaded": true,
  "llm_available": true
}
```

### Documentation interactive

Une fois le serveur lancé :

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

---

## 5. Endpoints de l'API

### `POST /api/predict` — Prédiction de fraude

Upload d'un fichier CSV → détection de schéma → construction de features → prédiction.

**Requête** :
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@data/transactions.csv"
```

**Réponse** :
```json
{
  "n_transactions": 1000,
  "n_fraud": 23,
  "fraud_rate_pct": 2.3,
  "amount_at_risk": 584320.50,
  "prediction_mode": "paysim",
  "schema_detection": { "mode": "paysim", "confidence": 0.95 },
  "column_mapping": { "amount": {"source": "montant", "confidence": 0.95} },
  "transactions": [
    {
      "index": 0,
      "xgb_score": 0.87,
      "ae_score": 2.14,
      "risk_level": "CRITIQUE",
      "is_fraud_predicted": true,
      "amount": 15000.0,
      "type": "TRANSFER"
    }
  ],
  "dataset_profile": { ... },
  "feature_build": { ... }
}
```

**Modes de prédiction sélectionnés automatiquement** :

| Mode | Condition | Modèles utilisés |
|------|-----------|-----------------|
| `paysim` | Toutes les colonnes PaySim détectées | XGBoost + AutoEncoder |
| `ae_isoforest` | `amount` mappé + ≥3 colonnes numériques | AutoEncoder + IsolationForest |
| `ae_only` | `amount` mappé + <3 colonnes numériques | AutoEncoder uniquement |
| `isoforest` | Pas d'`amount` + ≥3 colonnes numériques | IsolationForest uniquement |

---

### `GET /api/explain/{tx_id}` — Explication d'une transaction

Doit être appelé **après** `/api/predict`. `tx_id` est l'index de la transaction dans le résultat.

**Requête** :
```bash
curl http://localhost:8000/api/explain/0
```

**Réponse** :
```json
{
  "tx_id": 0,
  "xgb_score": 0.87,
  "ae_score": 2.14,
  "risk_level": "CRITIQUE",
  "feature_values": {
    "log_amount": 9.62,
    "balance_diff_orig": -15000.0,
    "type_TRANSFER": 1
  },
  "shap_values": {
    "balance_diff_orig": 0.42,
    "log_amount": 0.31,
    "type_TRANSFER": 0.18
  },
  "lime_rules": [
    "balance_diff_orig <= -5000 → fraud+0.38",
    "log_amount > 9.0 → fraud+0.22"
  ],
  "llm": {
    "risk_level": "CRITIQUE",
    "risk_score": 0.87,
    "resume": "Transaction de virement suspect avec vidange totale du compte",
    "raisons": ["Solde source réduit à zéro", "Montant élevé atypique"],
    "actions_recommandees": ["Bloquer la transaction", "Contacter le titulaire"]
  }
}
```

---

### `GET /api/models` — Métriques des modèles

Liste les 7 modèles entraînés avec leurs performances sur le jeu de test PaySim.

**Requête** :
```bash
curl http://localhost:8000/api/models
```

**Réponse** :
```json
[
  {
    "name": "XGB_smote",
    "recall": 0.92,
    "precision": 0.88,
    "f1": 0.90,
    "pr_auc": 0.94,
    "roc_auc": 0.99,
    "optimal_threshold": 0.355,
    "is_best": true
  },
  { "name": "RF_smote", ... },
  { "name": "RF_balanced", ... },
  { "name": "LR_smote", ... },
  { "name": "LR_balanced", ... },
  { "name": "IsoForest", ... },
  { "name": "AutoEncoder", ... }
]
```

---

### `POST /api/profile` — Profilage de dataset (sans prédiction)

Analyse la qualité du CSV sans lancer la prédiction.

**Requête** :
```bash
curl -X POST http://localhost:8000/api/profile \
  -F "file=@data/transactions.csv"
```

**Réponse** :
```json
{
  "n_rows": 5000,
  "n_cols": 11,
  "missing_pct": 0.2,
  "quality_score": 87,
  "column_types": {
    "numeric": ["amount", "oldbalanceOrg"],
    "categorical": ["type"],
    "datetime": [],
    "identifiers": ["nameOrig", "nameDest"]
  },
  "quasi_constant_cols": [],
  "recommendations": ["Vérifier les valeurs manquantes dans 'newbalanceDest'"],
  "column_mapping": { ... }
}
```

---

### `POST /api/report` — Génération de rapport PDF

Génère un rapport PDF à la charte PwC depuis les résultats de la dernière prédiction.

**Requête** :
```bash
curl -X POST http://localhost:8000/api/report \
  -o rapport_fraude.pdf
```

Ou avec des résultats personnalisés en JSON :
```bash
curl -X POST http://localhost:8000/api/report \
  -H "Content-Type: application/json" \
  -d '{"n_transactions": 100, "n_fraud": 5, ...}' \
  -o rapport_fraude.pdf
```

---

### `GET /api/health` — Statut du serveur

```bash
curl http://localhost:8000/api/health
```

---

## 6. Pipeline de prédiction

Le pipeline complet lors d'un appel à `/api/predict` :

```
CSV Upload
    │
    ▼
1. Profilage dataset (dataset_profiler.py)
   → statistiques, qualité, types de colonnes
    │
    ▼
2. Mapping sémantique des colonnes (column_mapper.py)
   → correspondance par alias exact / fuzzy matching / signal composé
   → normalisation des types (ex: "virement" → "TRANSFER")
    │
    ▼
3. Détection de schéma (schema_detector.py)
   → sélection du mode: paysim / ae_isoforest / ae_only / isoforest
    │
    ▼
4. Construction des 14 features (feature_builder.py)
   → step, hour, day, week, high_risk_hour
   → balance_diff_orig, dest_zero_balance
   → type_CASH_IN/OUT/DEBIT/PAYMENT/TRANSFER
   → log_amount
   → Scaling avec StandardScaler sur 6 colonnes
    │
    ▼
5. Feature engineering dérivé (feature_engineer.py)
   → features temporelles (weekend, business_hour, day_of_week)
   → features de balance (drain_pct, amount_ratio, balance_gap)
   → features comportementales (tx_count, unique_dests, avg_amount)
    │
    ▼
6. Prédiction (predictor.py / generic_predictor.py)
   → XGBoost: probabilité de fraude (seuil 0.355)
   → AutoEncoder: erreur de reconstruction (seuil 1.753)
   → Classification: CRITIQUE / ELEVE / FAIBLE
    │
    ▼
7. Mise en cache des résultats (app.state.results_cache)
   → Permet l'appel ultérieur à /api/explain/{tx_id}
```

---

## 7. Intégration LLM

### Fournisseurs supportés

| Provider | Modèle par défaut | Variable d'env |
|----------|------------------|---------------|
| **Groq** (actif) | `llama-3.1-8b-instant` | `GROQ_API_KEY` |
| Gemini | `gemini-1.5-flash` | `GEMINI_API_KEY` |
| HuggingFace | `mistralai/Mistral-7B-Instruct-v0.2` | `HF_API_KEY` |

### Changer de provider

Modifier [config/llm_config.yaml](config/llm_config.yaml) :

```yaml
active_provider: gemini   # groq | gemini | huggingface
```

### Comportement du LLM

- **Retry automatique** : 3 tentatives avec backoff exponentiel
- **JSON repair** : corrige les réponses JSON malformées du LLM
- **Fallback** : si LLM indisponible, une explication basée sur les règles est retournée
- **Timeout** : 30 secondes par appel

### Format de sortie du LLM

```json
{
  "risk_level": "CRITIQUE|ELEVE|FAIBLE",
  "risk_score": 0.87,
  "confidence_score": 0.91,
  "resume": "Résumé en une ligne",
  "raisons": ["raison 1", "raison 2", "raison 3"],
  "actions_recommandees": ["action 1", "action 2"]
}
```

---

## 8. Génération de rapports PDF

Le service `report_gen.py` génère un PDF avec :

- En-tête et pied de page à la charte PwC (rouge `#D04A02`)
- Résumé exécutif (KPIs : transactions, fraudes, montant à risque, taux)
- Tableau des transactions les plus risquées
- Distribution des niveaux de risque
- Section méthodologie (modèles utilisés, seuils)
- Date et signature automatiques

Le PDF est retourné en streaming (`StreamingResponse`) — aucun fichier n'est sauvegardé sur le serveur.

---

## 9. Tests

### Lancer tous les tests

```bash
# Depuis la racine du projet
pytest tests/ -v
```

### Lancer un fichier de tests spécifique

```bash
pytest tests/test_predict.py -v
pytest tests/test_explain.py -v
pytest tests/test_column_mapper.py -v
pytest tests/test_profiler.py -v
pytest tests/test_feature_engineer.py -v
pytest tests/test_report.py -v
pytest tests/test_models.py -v
pytest tests/test_llm.py -v
```

### Lancer un test spécifique

```bash
pytest tests/test_predict.py::test_predict_paysim_csv -v
```

### Avec couverture de code

```bash
pip install pytest-cov
pytest tests/ --cov=app --cov-report=html
# Rapport HTML dans htmlcov/index.html
```

### Structure des tests

| Fichier | Ce qui est testé |
|---------|-----------------|
| `conftest.py` | Fixtures partagées : `client` (TestClient), données PaySim synthétiques |
| `test_predict.py` | `/api/predict` : mode paysim, modes génériques, CSV vide, colonnes invalides |
| `test_explain.py` | `/api/explain/{tx_id}` : SHAP, LIME, LLM, tx_id invalide |
| `test_report.py` | `/api/report` : génération PDF, contenu, headers HTTP |
| `test_models.py` | `/api/models` : liste modèles, métriques, best flag |
| `test_column_mapper.py` | Détection sémantique, alias, fuzzy, normalisation types |
| `test_profiler.py` | Statistiques dataset, score qualité, recommandations |
| `test_feature_engineer.py` | Features temporelles, balance, comportementales |
| `test_llm.py` | Appels LLM, retry, fallback, parsing JSON |
| `test_preprocessing.py` | Chargement et nettoyage des données |
| `test_visualization.py` | Fonctions de visualisation |
| `test_utils.py` | Utilitaires (évaluateur, anomaly_utils) |

### Tests manuels avec curl

#### Test complet du flow (predict → explain → report)

```bash
# 1. Prédire
curl -X POST http://localhost:8000/api/predict \
  -F "file=@data/PaySim_sample.csv" \
  -o predict_result.json

# 2. Expliquer la transaction la plus risquée (index 0)
curl http://localhost:8000/api/explain/0

# 3. Générer le rapport PDF
curl -X POST http://localhost:8000/api/report -o rapport.pdf

# 4. Ouvrir le rapport (Windows)
start rapport.pdf
```

#### Tester avec un CSV générique (colonnes non-PaySim)

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@data/bank_transactions.csv"
# Le serveur détecte automatiquement le schéma et adapte le pipeline
```

#### Tester le profilage seul

```bash
curl -X POST http://localhost:8000/api/profile \
  -F "file=@data/transactions.csv"
```

### Générer un CSV de test PaySim minimal

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'step': np.random.randint(1, 744, 100),
    'type': np.random.choice(['TRANSFER', 'CASH_OUT', 'PAYMENT'], 100),
    'amount': np.random.uniform(100, 50000, 100),
    'nameOrig': [f'C{i}' for i in range(100)],
    'oldbalanceOrg': np.random.uniform(0, 100000, 100),
    'newbalanceOrig': np.random.uniform(0, 100000, 100),
    'nameDest': [f'M{i}' for i in range(100)],
    'oldbalanceDest': np.random.uniform(0, 100000, 100),
    'newbalanceDest': np.random.uniform(0, 100000, 100),
})
df.to_csv('test_paysim.csv', index=False)
```

---

## 10. Variables d'environnement

| Variable | Requis | Description |
|----------|--------|-------------|
| `GROQ_API_KEY` | Si Groq actif | Clé API Groq (llama-3.1-8b-instant) |
| `GEMINI_API_KEY` | Si Gemini actif | Clé API Google Gemini |
| `HF_API_KEY` | Si HuggingFace actif | Token HuggingFace |

Ces variables sont lues depuis `.env` via `python-dotenv`. Si aucune clé n'est disponible, le LLM est désactivé et une explication règle-métier est utilisée en fallback.

---

## 11. Architecture des services

### Chargement des modèles au démarrage

Au lancement, `lifespan()` dans `app/main.py` charge tous les modèles en mémoire :

```
app.state.models = {
  "scaler":       StandardScaler (6 features)
  "features":     Liste des 14 noms de features
  "thresholds":   {"XGB_smote": 0.355, "RF_smote": 0.45, ...}
  "xgb":          XGBClassifier
  "lr_balanced":  LogisticRegression
  "lr_smote":     LogisticRegression
  "rf_balanced":  RandomForestClassifier
  "rf_smote":     RandomForestClassifier
  "iso_forest":   IsolationForest
  "iso_scaler":   StandardScaler (IsoForest)
  "ae":           FraudAutoEncoder (PyTorch)
  "ae_threshold": 1.753
  "baseline_report": Métriques JSON (7 modèles)
}

app.state.llm = LLMHelper (ou None si indisponible)
app.state.results_cache = {}  ← Cache en mémoire des derniers résultats
```

### CORS

Configuré pour accepter les requêtes de :
- `http://localhost:5173` (frontend React/Vite)
- `http://localhost:3000` (alternative)

Pour ajouter d'autres origines, modifier `allow_origins` dans [app/main.py](app/main.py).

### Authentification

**Aucune authentification n'est implémentée** — toutes les routes sont publiques. Pour une mise en production, ajouter un middleware JWT ou une validation de clé API.

---

## 12. Notes de déploiement

### Dépendances critiques à installer

```bash
# Si PyTorch non installé (AutoEncoder)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Si SHAP pose problème
pip install shap --no-build-isolation
```

### Vérifier que les modèles sont présents

```bash
ls outputs/models/
# Doit contenir: xgb_smote.pkl, scaler.pkl, features.json,
#                optimal_thresholds.json, autoencoder/
```

### Commandes utiles

```bash
# Vérifier la version Python
python --version

# Vérifier les packages installés
pip list | grep -E "fastapi|uvicorn|torch|xgboost|shap"

# Lancer les tests en mode silencieux
pytest tests/ -q

# Lancer avec logs détaillés
uvicorn app.main:app --reload --log-level debug

# Inspecter les routes enregistrées
python -c "from app.main import app; [print(r.path) for r in app.routes]"
```

### Performances

- **Temps de démarrage** : ~3-5 secondes (chargement des modèles)
- **Temps de réponse `/api/predict`** : ~200-500ms pour 1000 transactions (sans LLM)
- **Temps de réponse `/api/explain`** : ~2-5 secondes (avec appel LLM)
- **Mémoire** : ~500MB (tous les modèles chargés)
