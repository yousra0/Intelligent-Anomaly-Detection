# Plateforme de Détection d'Anomalies — PwC Tunisie

Plateforme complète d'audit financier basée sur la détection d'anomalies transactionnelles.
Combine un backend FastAPI (ML) et un frontend Next.js 15, avec gestion de missions, rapports PDF et contrôle d'accès basé sur les rôles (RBAC).

## Architecture

```
Frontend  →  Next.js 15 (App Router, TypeScript, TailwindCSS + shadcn/ui)
Backend   →  FastAPI (Python) — inférence ML, rapports, explications SHAP/LLM
ML Core   →  src/ exposé comme package Python `ml_core` (autoencoder, XGBoost, IsolationForest)
Auth      →  JWT HS256 via `jose` — cookie httpOnly (middleware) + localStorage (React)
```

**Proxy API** : les routes `/ml/*` du frontend sont redirigées vers FastAPI via `next.config.ts`.

## Démarrage rapide (Windows)

### Prérequis

- Python 3.10+
- Node.js 18+
- npm ou pnpm

### 1. Backend Python

```powershell
# Créer et activer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Installer les dépendances Python
pip install -r requirements.txt
pip install -r requirements_app.txt

# Installer ml_core comme package local
pip install -e .

# Lancer le serveur FastAPI (port 8000)
uvicorn app.main:app --reload
```

### 2. Frontend Next.js

```powershell
cd frontend

# Copier et configurer les variables d'environnement
copy .env.local.example .env.local
# Éditer .env.local : renseigner JWT_SECRET et FASTAPI_URL

# Installer les dépendances Node.js
npm install

# Lancer le serveur de développement (port 3000)
npm run dev
```

L'application est accessible sur `http://localhost:3000`.

### 3. Pipeline ML (notebooks)

```powershell
# Depuis la racine, exécuter tous les notebooks dans l'ordre
python run_all.py

# Options utiles
python run_all.py --check-only      # vérifie uniquement les imports src/
python run_all.py --from 03         # reprend à partir du notebook 03
python run_all.py --only 04         # exécute uniquement le notebook 04
python run_all.py --timeout 3600    # timeout par notebook (secondes)
```

## Structure du projet

```text
anomaly_detection_project/
├── README.md
├── requirements.txt          # dépendances ML core
├── requirements_app.txt      # dépendances FastAPI
├── pyproject.toml            # packaging ml_core
├── run_all.py                # exécuteur notebooks
│
├── app/                      # Backend FastAPI
│   ├── main.py
│   ├── routes/               # predict, explain, models, profile, report
│   └── services/
│       ├── predictor.py      # inférence chemin mappé (XGBoost + AutoEncoder)
│       ├── generic_predictor.py  # inférence chemin générique (IsolationForest)
│       ├── explainer.py      # SHAP + LIME
│       ├── llm_service.py    # intégration Groq/Gemini
│       ├── report_gen.py     # rapport PDF (FPDF2 + matplotlib)
│       ├── report_gen_docx.py
│       ├── dataset_profiler.py
│       ├── schema_detector.py
│       └── column_mapper.py
│
├── frontend/                 # Frontend Next.js 15
│   ├── package.json
│   ├── .env.local.example
│   ├── migrations/
│   │   └── 001_initial_schema.sql   # schéma PostgreSQL production
│   └── src/
│       ├── app/
│       │   ├── (auth)/       # /login
│       │   ├── (dashboard)/  # dashboard, missions, reports, history, audit-trail
│       │   └── api/          # routes internes Next.js
│       ├── components/       # UI : analysis, datasets, missions, reports, layout
│       ├── lib/
│       │   ├── store/        # stores en mémoire (→ PostgreSQL en production)
│       │   └── hooks/        # usePermissions(), useMissions(), ...
│       └── types/
│
├── src/                      # ML Core (package `ml_core`)
│   ├── preprocessing/        # chargement et nettoyage
│   ├── feature_engineering/  # variables temporelles et comportementales
│   ├── models/               # autoencoder, ml_models
│   ├── pipeline/             # training_pipeline, inference_pipeline
│   ├── explainability/       # SHAP, LIME
│   ├── monitoring/           # drift_detector, model_monitor
│   ├── utils/                # évaluation, anomaly_utils, model_governance
│   ├── visualization/        # figures modèles et préparation
│   └── llm_integration/      # llm_helper (Groq / Gemini)
│
├── config/
│   ├── config.yaml
│   └── llm_config.yaml
│
├── data/
│   ├── raw/                  # dataset_orig.csv
│   └── processed/            # splits X/y train/val/test (.npy et .csv)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_autoencoder.ipynb
│   ├── 05_shap_lime.ipynb
│   └── 06_llm_integration.ipynb
│
├── outputs/
│   ├── anomalies_report.csv
│   ├── explanations.json
│   ├── figures/
│   ├── models/               # poids keras, scores AE, seuils, features.json
│   └── reports/              # rapports JSON par étape
│
└── tests/                    # 12 fichiers de tests unitaires
```

## Endpoints FastAPI

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/api/predict` | Inférence sur un dataset uploadé |
| `GET` | `/api/explain/{tx_id}` | Explication SHAP/LLM d'une transaction |
| `POST` | `/api/explain/batch` | Explications en lot |
| `GET` | `/api/models` | Liste des modèles disponibles |
| `POST` | `/api/report` | Génération rapport PDF |
| `POST` | `/api/report/docx` | Génération rapport DOCX |
| `GET` | `/api/health` | Vérification santé du service |

## Rôles & Permissions (RBAC)

| Permission | auditor | manager | partner | admin |
|---|:---:|:---:|:---:|:---:|
| Créer / supprimer une mission | ✗ | ✓ | ✗ | ✓ |
| Voir toutes les missions | ✗ | ✓ | ✓ | ✓ |
| Uploader un dataset | ✓ | ✓ | ✗ | ✓ |
| Lancer une analyse | ✓ | ✓ | ✗ | ✓ |
| Générer un rapport | ✓ | ✓ | ✗ | ✓ |
| Valider une anomalie | ✗ | ✓ | ✗ | ✓ |
| Gestion utilisateurs | ✗ | ✗ | ✗ | ✓ |

Les auditeurs ne voient que les missions qui leur sont assignées.

## Comptes de démonstration

Les mots de passe sont définis par la variable `DEMO_PASSWORD` dans `.env.local` (valeur par défaut : `pwc2024`).

| Email | Rôle | Nom |
|-------|------|-----|
| `auditeur@pwc.com` | auditor | Sophie Aubert |
| `manager@pwc.com` | manager | Marc Martin |
| `partner@pwc.com` | partner | Pierre Dupont |
| `admin@pwc.com` | admin | Admin PwC |

## Inférence ML

Deux chemins d'inférence sont disponibles :

- **Chemin mappé** (PaySim → 14 features canoniques) : XGBoost + AutoEncoder.
- **Chemin générique** (tout schéma) : IsolationForest avec détection automatique de colonnes.

## Rapport PDF

Généré par `app/services/report_gen.py` (FPDF2 + matplotlib). Structure en 7 pages :
1. Page de garde (bandeau orange PwC, bannière de risque global)
2. Résumé exécutif (4 KPIs colorés, jauge de risque)
3. Graphiques (donut, barres niveaux de risque, exposition financière, SHAP optionnel)
4. Top-10 transactions suspectes (tableau avec statuts colorés)
5. Fiches transactions critiques (indicateur de risque, analyse LLM, facteurs SHAP)
6. Recommandations numérotées + glossaire + mention légale

Language : français non technique, destiné aux auditeurs financiers.

## Tests

```powershell
pytest -q
```

Fichiers de tests dans `tests/` :
`test_preprocessing.py`, `test_models.py`, `test_visualization.py`, `test_utils.py`,
`test_llm.py`, `test_predictions.py`, `test_explanations.py`, `test_feature_engineering.py`,
`test_profiling.py`, `test_reports.py`, `test_column_mapping.py`

## Migration vers PostgreSQL

Le fichier `frontend/migrations/001_initial_schema.sql` contient le schéma PostgreSQL complet
(tables : users, missions, datasets, analysis_runs, anomaly_reviews, reports, audit_logs)
avec des politiques RLS commentées prêtes à activer.

Les stores en mémoire (`frontend/src/lib/store/`) sont à remplacer par des appels à PostgreSQL.
