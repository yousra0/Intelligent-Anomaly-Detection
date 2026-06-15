# Guide du Backend FastAPI — Système de Détection d'Anomalies

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Structure du projet](#2-structure-du-projet)
3. [Installation et configuration](#3-installation-et-configuration)
4. [Lancement du serveur](#4-lancement-du-serveur)
5. [Endpoints de l'API](#5-endpoints-de-lapi)
6. [Pipeline de prédiction](#6-pipeline-de-prédiction)
7. [Intégration LLM](#7-intégration-llm)
8. [Génération de rapports](#8-génération-de-rapports)
9. [Tests](#9-tests)
10. [Variables d'environnement](#10-variables-denvironnement)
11. [Architecture des services](#11-architecture-des-services)
12. [Notes de déploiement](#12-notes-de-déploiement)
13. [Modifications récentes du backend](#13-modifications-récentes-du-backend)

---

## 1. Vue d'ensemble

Le backend est une API REST construite avec **FastAPI** exposant un pipeline complet de détection de fraude financière.

### Modèles de production (ensemble principal)

Le système repose sur **deux modèles complémentaires** déployés ensemble en production :

| Modèle | Rôle | Type | Seuil |
|--------|------|------|-------|
| **XGBoost** (`XGB_smote`) | Modèle supervisé principal — classifie chaque transaction | Arbre boosté (XGBClassifier) | 0.355 |
| **AutoEncoder** | Détecteur d'anomalies — mesure l'écart de reconstruction | Réseau de neurones PyTorch (FraudAutoEncoder) | 1.753 (MSE) |

En mode PaySim (schéma complet), les deux modèles sont utilisés simultanément :
- **XGBoost** fournit le `xgb_score` (0–1) — c'est lui qui décide `is_fraud_predicted`
- **AutoEncoder** fournit l'`ae_score` (MSE normalisé) — signal complémentaire d'anomalie

Les autres modèles (LR, RF, IsoForest) ont été évalués lors de la phase de recherche mais **ne sont pas utilisés en production**.

### Fonctionnalités complètes

- **Prédiction multi-modes** : 4 modes selon le schéma CSV détecté (voir section 5)
- **Détection automatique de schéma** : adapte le pipeline aux colonnes disponibles
- **Explicabilité** : SHAP (XGBoost) + Proxy AE (|x−AE(x)|) + LIME + explication LLM en langage naturel
- **Profilage de dataset** : analyse qualité, statistiques, recommandations
- **Génération de rapports** : PDF à la charte PwC (fpdf2) **et** Word DOCX depuis template
- **Intégration LLM** : Groq (llama-3.3-70b), Gemini, ou HuggingFace

---

## 2. Structure du projet

```
anomaly_detection_project/
├── app/                          ← Backend FastAPI
│   ├── main.py                   ← Point d'entrée, lifespan, CORS, routeurs
│   ├── routes/
│   │   ├── predict.py            ← POST /api/predict
│   │   ├── explain.py            ← GET  /api/explain/{tx_id}  +  POST /api/explain/batch
│   │   ├── report.py             ← POST /api/report  +  POST /api/report/docx
│   │   ├── models.py             ← GET  /api/models
│   │   └── profile.py            ← POST /api/profile
│   └── services/
│       ├── predictor.py          ← Chargement modèles + pipeline PaySim (XGB + AE)
│       ├── generic_predictor.py  ← Pipeline fallback (schéma inconnu)
│       ├── llm_service.py        ← Wrapper LLMHelper
│       ├── explainer.py          ← SHAP TreeExplainer (XGB), Proxy AE (|x−AE(x)|), LIME
│       ├── dataset_profiler.py   ← Analyse qualité du dataset
│       ├── column_mapper.py      ← Détection sémantique des colonnes (4 niveaux)
│       ├── schema_detector.py    ← Sélection du mode de prédiction
│       ├── feature_builder.py    ← Construction features adaptatives (14 features)
│       ├── feature_engineer.py   ← Features dérivées (temporel, balance, comportemental)
│       ├── report_gen.py         ← Génération PDF avec fpdf2
│       └── report_gen_docx.py    ← Génération DOCX depuis template python-docx
│
├── src/                          ← Bibliothèque ML (ml_core)
│   ├── models/                   ← Définitions AutoEncoder PyTorch + modèles ML
│   ├── feature_engineering/      ← Features temporelles, comportementales
│   ├── llm_integration/          ← LLMHelper (appels API + retry + parsing JSON)
│   ├── preprocessing/            ← Chargement et nettoyage données
│   ├── explainability/           ← SHAP et LIME (non utilisé directement par l'API)
│   └── utils/                    ← Évaluateur, utilitaires
│
├── config/
│   ├── config.yaml               ← Configuration générale du projet
│   └── llm_config.yaml           ← Provider LLM, modèles, paramètres de génération
│
├── exemple_rapport.docx          ← Template Word PwC ({{placeholders}})
│
├── outputs/
│   ├── models/
│   │   ├── xgb_smote.pkl         ← XGBClassifier direct (joblib)
│   │   ├── rf_smote.pkl          ← dict{"model": RandomForestClassifier}
│   │   ├── rf_balanced.pkl       ← dict{"model": RandomForestClassifier}
│   │   ├── lr_smote.pkl          ← dict{"model": LogisticRegression}
│   │   ├── lr_balanced.pkl       ← dict{"model": LogisticRegression}
│   │   ├── iso_forest.pkl        ← IsolationForest direct (joblib)
│   │   ├── iso_forest_scaler.pkl ← MinMaxScaler pour la normalisation des scores AE
│   │   ├── scaler.pkl            ← StandardScaler (6 colonnes SCALE_COLS)
│   │   ├── features.json         ← Liste des 14 features attendues
│   │   ├── optimal_thresholds.json ← Seuils optimaux LR/RF/XGB/IsoForest
│   │   └── autoencoder/
│   │       ├── autoencoder_weights.pt  ← Poids PyTorch
│   │       └── autoencoder_meta.pkl    ← Métadonnées (threshold=1.753, architecture)
│   └── reports/
│       ├── baseline_report.json  ← Métriques LR/RF/XGB/IsoForest (NB03)
│       └── autoencoder_report.json ← Métriques AutoEncoder (NB05)
│
├── tests/                        ← Suite de tests pytest (13 fichiers)
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
  "n_transactions": 50,
  "n_fraud": 10,
  "fraud_rate_pct": 20.0,
  "amount_at_risk": 1645470.97,
  "model_used": "XGBoost + Autoencoder",
  "prediction_mode": "paysim",
  "schema_detection": {
    "mode": "paysim",
    "n_mapped": 9,
    "n_paysim_required": 7,
    "avg_confidence": 1.0,
    "use_xgb": true,
    "use_ae": true,
    "use_isoforest": false,
    "models_used": "XGBoost + Autoencoder",
    "reason": "Schéma PaySim complet (9/7 colonnes requises, confiance moy. 100%) → XGBoost + Autoencoder",
    "warnings": []
  },
  "column_mapping": {
    "amount": { "original_name": "amount", "confidence": 1 },
    "...": "..."
  },
  "mapping_warnings": [],
  "feature_engineering": { "n_generated": 15, "..." : "..." },
  "transactions": [
    {
      "tx_id": 21,
      "type": "CASH_OUT",
      "amount": 62229.21,
      "xgb_score": 0.993882,
      "ae_score": 0.623424,
      "risk_level": "CRITIQUE",
      "is_fraud_predicted": true
    },
    {
      "tx_id": 20,
      "type": "TRANSFER",
      "amount": 14257.00,
      "xgb_score": 0.302858,
      "ae_score": 0.376791,
      "risk_level": "FAIBLE",
      "is_fraud_predicted": false
    }
  ],
  "threshold": 0.3547,
  "feature_build": { "n_features": 14, "..." : "..." },
  "dataset_profile": { "n_rows": 50, "global_quality_score": 98.9, "..." : "..." }
}
```

**Champs clés de chaque transaction** :

| Champ | Description |
|-------|-------------|
| `tx_id` | Index de la ligne dans le CSV |
| `xgb_score` | Score XGBoost (0–1) — seuil 0.355 → `is_fraud_predicted` |
| `ae_score` | Score AutoEncoder normalisé (MSE / seuil) |
| `risk_level` | `CRITIQUE` (xgb ≥ seuil) / `ELEVE` (xgb ≥ 0.5) / `FAIBLE` |
| `is_fraud_predicted` | `true` si `xgb_score ≥ threshold` |

**Modes de prédiction sélectionnés automatiquement** :

| Mode | Condition | Modèles utilisés |
|------|-----------|-----------------|
| `paysim` | Toutes les colonnes PaySim détectées | XGBoost + AutoEncoder |
| `ae_isoforest` | `amount` mappé + ≥3 colonnes numériques | AutoEncoder + IsolationForest (on-the-fly) |
| `ae_only` | `amount` mappé + <3 colonnes numériques | AutoEncoder uniquement |
| `isoforest` | Pas d'`amount` + ≥3 colonnes numériques | IsolationForest uniquement |

**Codes d'erreur** :
- `400` : fichier vide
- `422` : CSV illisible ou colonnes insuffisantes (aucun modèle applicable)
- `500` : erreur interne de traitement

---

### `GET /api/explain/{tx_id}` — Explication d'une transaction

Doit être appelé **après** `/api/predict`. `tx_id` correspond au champ `tx_id` dans la réponse de predict.

**Requête** :
```bash
curl http://localhost:8000/api/explain/21
```

**Réponse** :
```json
{
  "tx_id": 21,
  "xgb_score": 0.993882,
  "ae_score": 0.623424,
  "risk_level": "CRITIQUE",
  "feature_values": {
    "log_amount": 11.04,
    "balance_diff_orig": 62229.21,
    "type_CASH_OUT": 1.0,
    "...": "..."
  },
  "shap_values_xgb": {
    "balance_diff_orig": 0.42,
    "log_amount": 0.31,
    "type_CASH_OUT": 0.18,
    "...": "..."
  },
  "ae_feature_errors": {
    "balance_diff_orig": 65.73,
    "log_amount": 5.87,
    "type_PAYMENT": 1.66,
    "...": "..."
  },
  "ae_top_features": [
    {"feature": "balance_diff_orig", "error": 65.73},
    {"feature": "log_amount",        "error": 5.87},
    {"feature": "type_PAYMENT",      "error": 1.66}
  ],
  "lime_rules": [
    "balance_diff_orig > 0: +0.382",
    "log_amount > 9.5: +0.221"
  ],
  "llm": {
    "risk_level": "CRITIQUE",
    "risk_score": 0.993882,
    "resume": "Retrait espèces avec vidange totale du compte source",
    "raisons": ["Solde source réduit à zéro", "Montant élevé atypique"],
    "actions_recommandees": ["Bloquer la transaction", "Contacter le titulaire"],
    "status": "ok",
    "_audit": {
      "hash": "04a1c8...",
      "timestamp_utc": "2026-06-15T12:00:00+00:00",
      "hash_algo": "sha256"
    }
  }
}
```

**Note sur les champs d'explicabilité — deux modèles, deux méthodes distinctes** :

| Champ | Méthode | Modèle expliqué |
|-------|---------|-----------------|
| `shap_values_xgb` | SHAP TreeExplainer | XGBoost uniquement |
| `ae_feature_errors` | Proxy `|x − AE(x)|` par feature | AutoEncoder uniquement |
| `ae_top_features` | Top 3 features par erreur décroissante | AutoEncoder uniquement |
| `lime_rules` | LIME LimeTabularExplainer | XGBoost |
| `llm.top_features` | Proxy AE transmis au LLM | AutoEncoder |

> **Important pour les auditeurs :** `shap_values_xgb` explique uniquement la composante XGBoost (modèle supervisé). Les erreurs AE (`ae_feature_errors`) expliquent la composante AutoEncoder (modèle non supervisé). Un score `xgb_score` élevé avec un `ae_score` faible signifie que XGBoost a détecté un pattern de fraude connu, mais que la transaction n'est pas anormale du point de vue de la reconstruction.

---

### `POST /api/explain/batch` — Explication en lot

Doit être appelé **après** `/api/predict`. LIME est désactivé en mode batch pour la performance.

**Requête** :
```bash
curl -X POST http://localhost:8000/api/explain/batch \
  -H "Content-Type: application/json" \
  -d '{"tx_ids": [21, 48, 41], "max_explain": 20}'
```

**Corps de la requête** :

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `tx_ids` | `list[int]` | requis | Liste des `tx_id` à expliquer |
| `max_explain` | `int` | `20` | Limite max (1–100). Les `tx_ids` au-delà sont ignorés |

**Réponse** :
```json
{
  "n_requested": 3,
  "n_explained": 3,
  "n_errors": 0,
  "explanations": [
    {
      "tx_id": 21,
      "risk_level": "CRITIQUE",
      "shap_values_xgb": { "balance_diff_orig": 0.42, "...": "..." },
      "ae_feature_errors": { "balance_diff_orig": 65.73, "...": "..." },
      "ae_top_features": [{"feature": "balance_diff_orig", "error": 65.73}],
      "llm": { "risk_level": "CRITIQUE", "resume": "...", "status": "ok" }
    }
  ],
  "errors": []
}
```

> Workflow recommandé pour 1 000 transactions : filtrer d'abord avec `/api/predict` (`is_fraud_predicted = true`), puis appeler `/api/explain/batch` sur les `tx_id` des transactions CRITIQUE/ELEVE uniquement.

---

### `GET /api/models` — Métriques des modèles

Liste les 7 modèles évalués avec leurs performances sur le jeu de test PaySim.

**Requête** :
```bash
curl http://localhost:8000/api/models
```

**Réponse** :
```json
{
  "models": [
    {
      "name": "XGB_smote",
      "recall": 0.8462,
      "precision": 0.825,
      "f1": 0.8354,
      "pr_auc": 0.8677,
      "roc_auc": 0.9975,
      "train_time_s": 10.34,
      "optimal_threshold": 0.3547,
      "is_best": true,
      "is_in_production": true
    },
    {
      "name": "AutoEncoder",
      "recall": 0.359,
      "precision": 0.6087,
      "f1": 0.4516,
      "pr_auc": 0.382,
      "roc_auc": 0.9358,
      "train_time_s": 191.3,
      "optimal_threshold": 1.753,
      "is_best": false,
      "is_in_production": true
    },
    {
      "name": "RF_smote",
      "recall": 0.7949,
      "precision": 0.8158,
      "f1": 0.8052,
      "pr_auc": 0.8405,
      "roc_auc": 0.994,
      "train_time_s": 15.28,
      "optimal_threshold": 0.6291,
      "is_best": false,
      "is_in_production": false
    }
  ]
}
```

**Signification des deux flags booléens** :

| Champ | Signification | Qui |
|-------|---------------|-----|
| `is_best` | Meilleur modèle standalone en termes de métriques (recall, F1, PR-AUC) | XGBoost uniquement |
| `is_in_production` | Modèle actif dans l'ensemble de production | XGBoost + AutoEncoder |

> `is_best = false` pour l'AutoEncoder ne signifie pas qu'il est inutile — il est utilisé en production (`is_in_production = true`) comme détecteur d'anomalies complémentaire au XGBoost supervisé. Les deux modèles jouent des rôles différents.

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
  "global_quality_score": 87,
  "numeric_cols": ["amount", "oldbalanceOrg"],
  "categorical_cols": ["type"],
  "datetime_cols": [],
  "identifier_cols": ["nameOrig", "nameDest"],
  "quasi_constant_cols": [],
  "high_missing_cols": [],
  "recommendations": ["Vérifier les valeurs manquantes dans 'newbalanceDest'"],
  "profiling_time_ms": 33.3
}
```

---

### `POST /api/report` — Génération de rapport PDF

Génère un rapport PDF à la charte PwC depuis les résultats de la dernière prédiction.

```bash
curl -X POST http://localhost:8000/api/report -o rapport_fraude.pdf
```

**Contenu du PDF (8 sections)** :
1. Page de couverture avec jauge de risque globale
2. Résumé exécutif (4 KPI boxes : transactions, fraudes, taux, montant à risque)
3. 3 graphiques matplotlib (donut répartition, barres niveau de risque, exposition financière)
4. Tableau top-10 transactions suspectes
5. Cartes détaillées transactions CRITIQUE (max 8) avec facteurs SHAP
6. Recommandations pour auditeurs
7. Glossaire (7 termes métier)
8. Disclaimer légal

---

### `POST /api/report/docx` — Génération de rapport Word

Génère un rapport Word depuis le template `exemple_rapport.docx`.

```bash
curl -X POST http://localhost:8000/api/report/docx -o rapport_fraude.docx
```

**Contenu du DOCX** :
- Substitution des `{{placeholders}}` dans le template
- 5 graphiques matplotlib intégrés (grille 2×2 + histogramme distribution des scores)
- Tableau des top-10 transactions + cartes CRITIQUE (max 14)
- Sections recommandations et glossaire dynamiques
- Texte entièrement en français pour contexte audit PwC Tunisie

---

### `GET /api/health` — Statut du serveur

```bash
curl http://localhost:8000/api/health
# → {"status": "ok", "models_loaded": true, "llm_available": true}
```

---

## 6. Pipeline de prédiction

Le pipeline complet lors d'un appel à `/api/predict` :

```
CSV Upload
    │
    ▼
1. Lecture CSV (pandas)
   → Validation : fichier non vide, parseable
    │
    ▼
2. Profilage dataset (dataset_profiler.py)
   → statistiques par colonne, score qualité global, types sémantiques
   → identifie : numeric_cols, categorical_cols, identifier_cols, quasi_constant_cols
    │
    ▼
3. Mapping sémantique des colonnes (column_mapper.py)
   → 4 niveaux de confiance :
       alias exact       (confiance 1.00)  — "amount" → "amount"
       alias normalisé   (confiance 0.95)  — "montant" → "amount"
       signal composé    (confiance 0.80)  — corrélation statistique
       fuzzy             (confiance 0.60)  — SequenceMatcher
   → normalisation des types (ex: "virement" → "TRANSFER")
   → MappingResult.success = True si toutes les colonnes PaySim requises sont trouvées
    │
    ▼
4. Détection de schéma (schema_detector.py)
   → success=True                          → mode "paysim"       (XGB + AE)
   → "amount" mappé + ≥3 cols numériques  → mode "ae_isoforest" (AE + IsoForest)
   → "amount" mappé + <3 cols numériques  → mode "ae_only"       (AE seul)
   → "amount" non mappé + ≥3 cols num.    → mode "isoforest"     (IsoForest seul)
   → aucune colonne utilisable             → ValueError 422
    │
    ▼
5. Construction des 14 features (feature_builder.py)
   → step, hour, day, week, high_risk_hour
   → balance_diff_orig, dest_zero_balance
   → type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
   → log_amount
   → Scaling avec StandardScaler sur 6 colonnes (SCALE_COLS)
   → Fallback : colonne manquante → 0.0 + warning dans build_report
    │
    ▼
6. Feature engineering dérivé (feature_engineer.py)
   → temporel    : eng_tx_day_of_week, eng_is_weekend, eng_is_business_hour
   → balance     : eng_drain_pct_src, eng_amount_ratio_src, eng_balance_gap,
                   eng_dest_gain_ratio, eng_drain_pct_dest
   → comportemental : eng_orig_tx_count, eng_orig_unique_dests, eng_orig_avg_amount,
                      eng_orig_total_amount, eng_dest_tx_count, eng_dest_avg_received,
                      eng_orig_is_high_freq
   [Ces 15 features enrichissent la réponse JSON mais ne sont pas utilisées
    par XGBoost/AE — elles sont exposées pour le frontend et les rapports]
    │
    ▼
7. Prédiction
   Mode "paysim"  → predictor.predict_batch()
                    XGBoost.predict_proba(X_arr)  → xgb_score (0–1)
                    AE.predict_score(X_arr)        → ae_score  (MSE normalisé)
                    Classification : CRITIQUE si xgb_score ≥ 0.355
                                     ELEVE    si xgb_score ≥ 0.5
                                     FAIBLE   sinon
                    Tri par xgb_score décroissant

   Mode générique → generic_predictor.predict_generic_batch()
                    AE + IsoForest on-the-fly selon schema_result.use_ae/use_isoforest
                    IsoForest : fitté transductivement sur le batch (contamination=5%)
                    Classification : basée sur ae_score et/ou decision_function IsoForest
    │
    ▼
8. Mise en cache (app.state.results_cache)
   → {"X_arr": ..., "df": ..., "transactions": ...}
   → Permet l'appel ultérieur à /api/explain/{tx_id}
```

---

## 7. Intégration LLM

### Fournisseurs supportés

| Provider | Modèle | Variable d'env | Limite gratuite |
|----------|--------|---------------|-----------------|
| **Groq** (actif) | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | 14 400 req/jour |
| Gemini | `gemini-1.5-flash` | `GEMINI_API_KEY` | 1 500 req/jour |
| HuggingFace | `Mistral-7B-Instruct-v0.2` | `HF_API_KEY` | Variable |

### Changer de provider

Modifier [config/llm_config.yaml](config/llm_config.yaml) :

```yaml
active_provider: gemini   # groq | gemini | huggingface
```

### Comportement du LLM

- **Retry automatique** : 3 tentatives avec backoff exponentiel (tenacity)
- **JSON repair** : corrige les réponses JSON malformées (`json_repair`)
- **Fallback** : si LLM indisponible → explication basée sur les règles métier
- **Timeout** : 30 secondes par appel, `temperature=0.1`, `max_tokens=400`
- **Audit trail** : chaque réponse LLM inclut un hash SHA-256 + timestamp UTC

### Format de sortie du LLM

```json
{
  "risk_level": "CRITIQUE|ELEVE|FAIBLE",
  "risk_score": 0.87,
  "resume": "Résumé en une ligne",
  "raisons": ["raison 1", "raison 2", "raison 3"],
  "actions_recommandees": ["action 1", "action 2"],
  "status": "ok",
  "_audit": {
    "hash": "sha256_de_la_reponse",
    "timestamp_utc": "2026-06-15T12:00:00+00:00",
    "hash_algo": "sha256"
  }
}
```

---

## 8. Génération de rapports

### Rapport PDF (`report_gen.py`)

Utilise la bibliothèque **fpdf2**. Le PDF est retourné en streaming (`StreamingResponse`) — aucun fichier sauvegardé sur le serveur.

- Palette PwC : Orange `#D04A02`, Dark `#293854`, Red `#C00000`, Green `#008246`
- Graphiques matplotlib embarqués (donut risque, barres niveau, exposition par type)
- Cartes de détail avec facteurs SHAP pour les transactions CRITIQUE (max 8)
- Section recommandations + glossaire en français pour auditeurs non-techniques

### Rapport Word (`report_gen_docx.py`)

Utilise **python-docx** avec injection dans le template `exemple_rapport.docx`.

- Placeholders : `{{date}}`, `{{taux_de_fraudes}}`, `{{analyse_text}}`, etc.
- 5 graphiques matplotlib intégrés en grille 2×2 + histogramme (PNG embarqué dans DOCX)
- Table de transactions dynamique + cartes CRITIQUE (max 14)
- Traductions français des noms de features (`FEATURE_FR`) pour les auditeurs

---

## 9. Tests

### Lancer tous les tests

```bash
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

### Avec couverture de code

```bash
pip install pytest-cov
pytest tests/ --cov=app --cov-report=html
# Rapport HTML dans htmlcov/index.html
```

### Structure des tests

| Fichier | Ce qui est testé |
|---------|-----------------|
| `conftest.py` | Fixtures : `client` (TestClient), CSV PaySim synthétique, CSV invalide |
| `test_predict.py` | `/api/predict` : mode paysim, modes génériques, CSV vide, colonnes invalides |
| `test_profiler.py` | `DatasetProfiler` : stats, score qualité, types sémantiques, recommandations |
| `test_feature_engineer.py` | `FeatureEngineer` : temporel, balance, comportemental (15 features) |
| `test_column_mapper.py` | `ColumnMapper` : alias, fuzzy, normalisation, scoring confiance |
| `test_explain.py` | `/api/explain/{tx_id}` : SHAP, AE proxy, LIME, LLM, tx_id invalide |
| `test_models.py` | `/api/models` : liste 7 modèles, métriques, flags `is_best`/`is_in_production` |
| `test_report.py` | `/api/report` : génération PDF, Content-Type, headers |
| `test_preprocessing.py` | Utilitaires de preprocessing |
| `test_llm.py` | Placeholder (à compléter) |
| `test_utils.py` | Utilitaires (évaluateur, anomaly_utils) |

### Flow complet avec curl

```bash
# 1. Prédire
curl -X POST http://localhost:8000/api/predict \
  -F "file=@test_paysim.csv" \
  -o predict_result.json

# 2. Expliquer la transaction la plus risquée
curl http://localhost:8000/api/explain/21

# 3. Expliquer plusieurs transactions en lot
curl -X POST http://localhost:8000/api/explain/batch \
  -H "Content-Type: application/json" \
  -d '{"tx_ids": [21, 48, 41], "max_explain": 10}'

# 4. Rapport PDF
curl -X POST http://localhost:8000/api/report -o rapport.pdf

# 5. Rapport Word
curl -X POST http://localhost:8000/api/report/docx -o rapport.docx
```

### Générer un CSV de test PaySim minimal

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'step': np.random.randint(1, 744, 100),
    'type': np.random.choice(['TRANSFER', 'CASH_OUT', 'PAYMENT'], 100),
    'amount': np.random.uniform(100, 500000, 100),
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
| `GROQ_API_KEY` | Si Groq actif | Clé API Groq |
| `GEMINI_API_KEY` | Si Gemini actif | Clé API Google Gemini |
| `HF_API_KEY` | Si HuggingFace actif | Token HuggingFace |

Ces variables sont lues depuis `.env` via `python-dotenv`. Si aucune clé n'est disponible, le LLM est désactivé et une explication règle-métier est utilisée en fallback.

---

## 11. Architecture des services

### Chargement des modèles au démarrage

Au lancement, `lifespan()` dans `app/main.py` appelle `load_all_models()` qui charge :

1. `baseline_report.json` — métriques LR, RF, XGB, IsoForest
2. `autoencoder_report.json` — métriques AutoEncoder (fichier séparé car entraîné dans NB05)
3. Fusion en mémoire : l'entrée AutoEncoder est injectée dans `baseline_report["models"]` si absente

```python
app.state.models = {
  "scaler":          StandardScaler (6 features — SCALE_COLS)
  "features":        Liste des 14 noms de features
  "thresholds":      {"XGB_smote": 0.355, "RF_smote": 0.629, "LR_balanced": 0.999, ...}
                     [NB: AutoEncoder absent ici — son seuil vient de ae_threshold]
  "xgb":             XGBClassifier            ← production
  "ae":              FraudAutoEncoder (PyTorch) ← production
  "ae_threshold":    1.753                     ← seuil MSE AutoEncoder
  "lr_balanced":     LogisticRegression        ← évaluation uniquement
  "lr_smote":        LogisticRegression        ← évaluation uniquement
  "rf_balanced":     RandomForestClassifier    ← évaluation uniquement
  "rf_smote":        RandomForestClassifier    ← évaluation uniquement
  "iso_forest":      IsolationForest           ← mode générique (fallback)
  "iso_scaler":      MinMaxScaler (scores AE)
  "baseline_report": dict avec 7 entrées (LR×2, RF×2, XGB, IsoForest, AutoEncoder)
}

app.state.llm_helper = LLMHelper (ou None si indisponible)
app.state.results_cache = {}  ← Cache en mémoire des derniers résultats predict
```

### Flux de données entre endpoints

```
POST /api/predict
    └─> app.state.results_cache = {X_arr, df, transactions}

GET /api/explain/{tx_id}      ← lit results_cache
POST /api/explain/batch       ← lit results_cache

POST /api/report              ← lit results_cache (ou body JSON)
POST /api/report/docx         ← lit results_cache (ou body JSON)
```

> Le cache est en mémoire — il est écrasé à chaque nouveau `/api/predict`. En production multi-utilisateurs, utiliser Redis ou une base de données.

### Patterns architecturaux notables

| Pattern | Description |
|---------|-------------|
| **Singleton services** | `ColumnMapper`, `DatasetProfiler`, `FeatureEngineer`, `DynamicFeatureBuilder` — fonctions module-level |
| **Dégradation gracieuse** | Colonne manquante → 0.0 + warning ; LLM indisponible → règles métier |
| **IsoForest transductif** | Refitté par batch en mode générique (contamination=5%) — avertissement explicite dans `schema_detection.warnings` |
| **Cache session** | `app.state.results_cache` permet le chainage predict → explain |
| **Fusion de rapports JSON** | `autoencoder_report.json` injecté dans `baseline_report["models"]` au démarrage, évitant toute duplication dans les routes |

### CORS

Configuré pour :
- `http://localhost:5173` (frontend React/Vite)
- `http://localhost:3000` (alternative)

### Authentification

**Aucune authentification n'est implémentée** — toutes les routes sont publiques. Pour une mise en production, ajouter un middleware JWT ou une validation de clé API.

---

## 12. Notes de déploiement

### Dépendances critiques

```bash
# Si PyTorch non installé (AutoEncoder)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Si SHAP pose problème
pip install shap --no-build-isolation

# Pour les rapports Word
pip install python-docx

# Pour la réparation de JSON LLM
pip install json-repair tenacity
```

### Vérifier que les artefacts sont présents

```bash
# Modèles de production (obligatoires)
ls outputs/models/xgb_smote.pkl
ls outputs/models/autoencoder/autoencoder_weights.pt
ls outputs/models/scaler.pkl
ls outputs/models/features.json

# Rapports de métriques (requis par /api/models)
ls outputs/reports/baseline_report.json
ls outputs/reports/autoencoder_report.json

# Template Word (requis par /api/report/docx)
ls exemple_rapport.docx
```

### Performances estimées

| Opération | Temps |
|-----------|-------|
| Démarrage serveur | ~3–5 s (chargement modèles) |
| `/api/predict` (1 000 transactions) | ~200–500 ms (sans LLM) |
| `/api/explain/{tx_id}` | ~2–5 s (avec appel LLM) |
| `/api/report` PDF | ~1–2 s |
| `/api/report/docx` | ~2–4 s |
| Mémoire totale | ~500 MB |

---

## 13. Modifications récentes du backend

### Correction : AutoEncoder affichait des métriques à 0 dans `/api/models`

**Problème** : `baseline_report.json` (généré par NB03) ne contenait pas d'entrée "AutoEncoder" car ce modèle a été entraîné séparément dans NB05 et ses métriques sauvegardées dans `autoencoder_report.json`. La route `/api/models` cherchait "AutoEncoder" dans `baseline_report["models"]`, ne trouvait rien, et retournait `recall: 0, f1: 0, ...`.

**Correction dans `app/services/predictor.py`** :

```python
# Charge les deux fichiers de métriques
with open(reports_dir / "baseline_report.json") as f:
    baseline_report = json.load(f)
with open(reports_dir / "autoencoder_report.json") as f:
    ae_report = json.load(f)

# Injecte l'entrée AutoEncoder si absente
ae_names = {m.get("name") for m in baseline_report.get("models", [])}
if "AutoEncoder" not in ae_names:
    baseline_report["models"].append({
        "name": "AutoEncoder",
        "optimal_threshold": ae_report["threshold"]["optimal"],
        "train_time_s": ae_report["training"]["train_time_s"],
        "test_metrics": ae_report["test_metrics"],
    })
```

### Ajout : champ `is_in_production` dans `/api/models`

**Motivation** : distinguer "meilleur modèle standalone" (`is_best`) de "modèle utilisé en production" (`is_in_production`). L'AutoEncoder a des métriques standalone inférieures au XGBoost mais est bien utilisé dans l'ensemble de production.

**Correction dans `app/routes/models.py`** :

```python
BEST_MODEL = "XGB_smote"
PRODUCTION_MODELS = {"XGB_smote", "AutoEncoder"}

# Dans chaque entrée de la réponse :
"is_best": name == BEST_MODEL,            # XGBoost uniquement
"is_in_production": name in PRODUCTION_MODELS,  # XGBoost + AutoEncoder
```

**Résultat** :
```json
{ "name": "XGB_smote",   "is_best": true,  "is_in_production": true  }
{ "name": "AutoEncoder", "is_best": false, "is_in_production": true  }
{ "name": "RF_smote",    "is_best": false, "is_in_production": false }
```
