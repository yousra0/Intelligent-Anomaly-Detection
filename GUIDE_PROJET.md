# Guide Complet — Plateforme de Détection d'Anomalies Financières

> Projet PFE · PwC Tunisie · Détection de fraude par ML/IA avec interface web

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture technique](#2-architecture-technique)
3. [Structure du dépôt](#3-structure-du-dépôt)
4. [Installation & configuration](#4-installation--configuration)
5. [Notebooks Jupyter (pipeline ML)](#5-notebooks-jupyter-pipeline-ml)
6. [Backend FastAPI](#6-backend-fastapi)
7. [Frontend Next.js](#7-frontend-nextjs)
8. [Docker (déploiement complet)](#8-docker-déploiement-complet)
9. [Base de données](#9-base-de-données)
10. [Tests](#10-tests)
11. [API — endpoints disponibles](#11-api--endpoints-disponibles)
12. [Variables d'environnement](#12-variables-denvironnement)
13. [Flux de travail complet](#13-flux-de-travail-complet)

---

## 1. Vue d'ensemble du projet

Plateforme full-stack d'audit financier développée pour PwC Tunisie. Elle permet à des auditeurs non-techniques de :

- **Uploader** un fichier CSV de transactions bancaires
- **Détecter automatiquement** les anomalies via deux modèles ML (XGBoost + AutoEncoder PyTorch)
- **Expliquer** chaque anomalie via SHAP, LIME et un LLM (Groq / Gemini)
- **Générer un rapport** PDF ou Word professionnel (thème PwC, en français)
- **Tracer** toutes les actions via un journal d'audit persistant

### Technologies principales

| Couche | Stack |
|--------|-------|
| ML / Data Science | Python 3.10+, PyTorch, XGBoost, scikit-learn, SHAP, LIME |
| Backend API | FastAPI, SQLAlchemy async, Alembic, PostgreSQL |
| LLM | Groq (LLaMA-3), Gemini (Google), HuggingFace (fallback) |
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS, Prisma |
| Containerisation | Docker, Docker Compose |
| Rapports | FPDF2 (PDF), python-docx (Word) |

---

## 2. Architecture technique

```
┌─────────────────────────────────────────────────────────────┐
│                     NAVIGATEUR (port 3000)                  │
│              Next.js 15 — React 19 — Tailwind CSS           │
│   Login · Dashboard · Missions · Analyse · Rapports         │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / JSON
┌────────────────────────▼────────────────────────────────────┐
│               BACKEND (port 8000)                           │
│              FastAPI + Uvicorn (async)                      │
│  /api/predict  /api/explain  /api/report  /api/profile      │
│  /api/models   /api/datasets /api/audit                     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  XGBoost +   │  │  AutoEncoder │  │  LLM (Groq /     │  │
│  │  Scaler pkl  │  │  PyTorch .pt │  │  Gemini / HF)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ asyncpg
┌────────────────────────▼────────────────────────────────────┐
│               PostgreSQL 16 (port 5432)                     │
│   users · missions · datasets · analysis_runs · audit_logs  │
└─────────────────────────────────────────────────────────────┘
```

**Modes de prédiction** (sélectionnés automatiquement selon le schéma CSV) :

| Mode | Déclencheur | Modèles utilisés |
|------|-------------|------------------|
| `paysim` | CSV PaySim standard | XGBoost + AutoEncoder |
| `ae_isoforest` | CSV générique enrichissable | AutoEncoder + IsolationForest |
| `fallback` | CSV inconnu | Règles métier uniquement |

---

## 3. Structure du dépôt

```
anomaly_detection_project/
│
├── GUIDE_PROJET.md          ← ce fichier
├── README.md
├── BACKEND_GUIDE.md         ← guide technique backend détaillé
├── TECHNICAL_REPORT.md
│
├── pyproject.toml           ← package ml_core (pip install -e .)
├── requirements.txt         ← dépendances notebooks ML
├── requirements_app.txt     ← dépendances backend FastAPI
│
├── Dockerfile               ← image backend (multi-stage)
├── docker-compose.yml       ← orchestration complète (postgres + api + frontend)
├── .env                     ← clés API locales (non versionné)
├── .env.docker              ← template pour Docker
│
├── app/                     ← Backend FastAPI
│   ├── main.py              ← point d'entrée (lifespan, CORS, routers)
│   ├── auth/                ← JWT middleware
│   ├── db/                  ← SQLAlchemy models + engine async
│   ├── routes/              ← endpoints (predict, explain, report, ...)
│   └── services/            ← logique métier (predictor, explainer, report_gen, ...)
│
├── src/                     ← Package ML réutilisable (importé sous ml_core)
│   ├── models/              ← AutoEncoder PyTorch, ML models
│   ├── feature_engineering/ ← 14 features canoniques
│   ├── preprocessing/       ← chargement et nettoyage CSV
│   ├── explainability/      ← SHAP + LIME
│   ├── llm_integration/     ← wrapper Groq / Gemini / HF
│   ├── pipeline/            ← orchestration entraînement & inférence
│   ├── monitoring/          ← drift detection
│   └── visualization/       ← graphiques matplotlib/seaborn
│
├── notebooks/               ← Pipeline ML (7 notebooks séquentiels)
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_autoencoder.ipynb
│   ├── 06_shap_lime.ipynb
│   └── 07_llm_integration.ipynb
│
├── frontend/                ← Application Next.js 15
│   ├── src/app/             ← pages (App Router)
│   ├── src/components/      ← ~30 composants React
│   ├── prisma/              ← schéma Prisma ORM
│   └── package.json
│
├── outputs/                 ← Artefacts ML générés par les notebooks
│   ├── models/
│   │   ├── xgb_smote.pkl         ← XGBoost entraîné
│   │   ├── scaler.pkl            ← StandardScaler fitté
│   │   ├── features.json         ← 14 noms de features
│   │   ├── optimal_thresholds.json
│   │   ├── iso_forest.pkl        ← IsolationForest (fallback)
│   │   └── autoencoder/
│   │       ├── autoencoder_weights.pt  ← poids PyTorch
│   │       └── autoencoder_meta.pkl    ← archi + seuil MSE
│   ├── reports/
│   └── figures/
│
├── data/
│   ├── raw/                 ← datasets originaux
│   ├── processed/           ← splits entraînement/validation/test
│   └── labels/
│
├── alembic/                 ← migrations PostgreSQL backend
├── config/                  ← config.yaml, llm_config.yaml
├── tests/                   ← tests unitaires pytest
└── test_paysim.csv          ← fichier de test rapide
```

---

## 4. Installation & configuration

### Prérequis

- Python 3.10 ou supérieur
- Node.js 18 ou supérieur
- PostgreSQL 16 (optionnel en dev, obligatoire en prod)
- Docker & Docker Compose (pour le déploiement conteneurisé)

### 4.1 — Environnement Python (backend + notebooks)

```bash
# Créer et activer l'environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Installer les dépendances notebooks
pip install -r requirements.txt

# Installer le package ml_core en mode éditable (obligatoire pour le backend)
pip install -e .

# Installer les dépendances backend
pip install -r requirements_app.txt
```

### 4.2 — Variables d'environnement (backend)

```bash
# Copier le template et remplir les clés
cp .env.docker .env
```

Fichier `.env` minimal :

```env
# Clé JWT (minimum 32 caractères)
JWT_SECRET=your-super-secret-jwt-key-at-least-32-chars

# Base de données (laisser vide pour fonctionner sans DB)
DATABASE_URL=postgresql+asyncpg://postgres:123@localhost:5432/pwcaudit

# LLM — au moins une clé requise pour les explications enrichies
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
HF_API_KEY=hf_...
```

### 4.3 — Frontend

```bash
cd frontend

# Copier le template des variables d'environnement
cp .env.local.example .env.local

# Installer les dépendances Node
npm install
```

Fichier `frontend/.env.local` :

```env
FASTAPI_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:3000
JWT_SECRET=your-super-secret-jwt-key-at-least-32-chars
JWT_EXPIRY=28800
DEMO_PASSWORD=pwc2024
```

---

## 5. Notebooks Jupyter (pipeline ML)

Les notebooks doivent être exécutés **dans l'ordre** pour générer les artefacts ML utilisés par le backend.

### Lancer Jupyter

```bash
# Depuis la racine du projet, avec le venv activé
jupyter notebook
# ou
jupyter lab
```

### Exécuter tous les notebooks automatiquement

```bash
# Script d'orchestration complet (exécute les 7 notebooks en séquence)
python run_all.py
```

### Description des notebooks

| # | Fichier | Ce qu'il fait | Sorties générées |
|---|---------|---------------|-----------------|
| 01 | `01_business_understanding.ipynb` | Contexte métier, patterns de fraude, exploration initiale | — |
| 02 | `02_data_understanding.ipynb` | EDA, distributions, corrélations, qualité des données | `outputs/figures/` |
| 03 | `03_data_preparation.ipynb` | Nettoyage, feature engineering (14 features), splits train/val/test | `data/processed/`, `outputs/models/scaler.pkl`, `outputs/models/features.json` |
| 04 | `04_baseline_models.ipynb` | Entraînement XGBoost + 6 autres modèles, SMOTE, optimisation seuils | `outputs/models/xgb_smote.pkl`, `outputs/models/optimal_thresholds.json`, `outputs/reports/baseline_report.json` |
| 05 | `05_autoencoder.ipynb` | AutoEncoder PyTorch (non-supervisé), calcul seuil MSE | `outputs/models/autoencoder/autoencoder_weights.pt`, `outputs/models/autoencoder/autoencoder_meta.pkl`, `outputs/reports/autoencoder_report.json` |
| 06 | `06_shap_lime.ipynb` | Analyse SHAP globale et LIME locale sur les prédictions | `outputs/figures/shap_*`, `outputs/figures/lime_*` |
| 07 | `07_llm_integration.ipynb` | Test intégration LLM (Groq/Gemini), génération d'explications textuelles | — |

> **Important** : Le backend charge les artefacts de `outputs/models/` au démarrage. Si ces fichiers n'existent pas, certains modes de prédiction seront désactivés.

---

## 6. Backend FastAPI

### Démarrage (développement)

```bash
# Depuis la racine du projet, avec le venv activé
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'API est accessible sur : `http://localhost:8000`  
Documentation interactive Swagger : `http://localhost:8000/docs`  
Documentation ReDoc : `http://localhost:8000/redoc`

### Démarrage sans rechargement automatique (production locale)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Variables d'environnement optionnelles au démarrage

```bash
# Activer les migrations automatiques Alembic au démarrage
AUTO_MIGRATE=true uvicorn app.main:app --reload

# Spécifier un répertoire d'upload personnalisé
UPLOADS_DIR=/tmp/uploads uvicorn app.main:app --reload
```

### Services principaux (`app/services/`)

| Fichier | Rôle |
|---------|------|
| `predictor.py` | Charge les modèles, coordonne la prédiction |
| `generic_predictor.py` | IsolationForest en mode fallback |
| `column_mapper.py` | Détection automatique des colonnes CSV |
| `schema_detector.py` | Détermine le mode de prédiction (paysim / ae_isoforest / fallback) |
| `feature_builder.py` | Construit les 14 features canoniques |
| `feature_engineer.py` | 15 features enrichies (temporelles, comportementales) |
| `dataset_profiler.py` | Analyse qualité du CSV uploadé |
| `explainer.py` | SHAP, LIME, proxy AutoEncoder |
| `llm_service.py` | Abstraction multi-provider LLM |
| `report_gen.py` | Génération PDF (FPDF2, thème PwC) |
| `report_gen_docx.py` | Génération Word (python-docx) |

---

## 7. Frontend Next.js

### Démarrage (développement)

```bash
cd frontend
npm run dev
```

L'interface est accessible sur : `http://localhost:3000`

### Build de production

```bash
cd frontend
npm run build
npm run start
```

### Autres commandes frontend

```bash
# Vérifier les erreurs TypeScript / ESLint
npm run lint

# Générer le client Prisma (après modification du schéma)
npm run db:generate

# Créer une nouvelle migration Prisma
npm run db:migrate

# Peupler la base avec des utilisateurs de démo
npm run db:seed

# Ouvrir Prisma Studio (interface visuelle de la DB)
npm run db:studio

# Synchroniser le schéma Prisma sans créer de migration (dev rapide)
npm run db:push
```

### Pages disponibles

| URL | Description | Accès |
|-----|-------------|-------|
| `/login` | Connexion | Public |
| `/dashboard` | Tableau de bord principal — KPIs | Tous les rôles |
| `/missions` | Liste des missions d'audit | Tous les rôles |
| `/missions/[id]` | Détail d'une mission | Tous les rôles |
| `/missions/[id]/analysis` | Résultats de l'analyse ML | Tous les rôles |
| `/reports` | Liste des rapports générés | Tous les rôles |
| `/history` | Historique des analyses | Tous les rôles |
| `/audit-trail` | Journal d'audit complet | Manager, Partner, Admin |
| `/admin/users` | Gestion des utilisateurs | Admin uniquement |

### Rôles utilisateurs

| Rôle | Permissions |
|------|-------------|
| `auditor` | Upload CSV, lancer analyse, voir résultats |
| `manager` | + voir journal d'audit, valider rapports |
| `partner` | + accès toutes missions, export |
| `admin` | + gestion utilisateurs, configuration |

---

## 8. Docker (déploiement complet)

### Première installation

```bash
# 1. Copier et remplir les variables d'environnement
cp .env.docker .env
# Éditer .env et renseigner JWT_SECRET, GROQ_API_KEY, GEMINI_API_KEY

# 2. Build et démarrage de tous les services
docker compose up --build

# 3. (Premier démarrage uniquement) Seeder les utilisateurs de démo
docker compose exec frontend npx prisma db seed
```

### Commandes courantes Docker

```bash
# Démarrer tous les services (sans rebuild)
docker compose up

# Démarrer en arrière-plan (daemon)
docker compose up -d

# Rebuild et redémarrer
docker compose up --build

# Arrêter tous les services
docker compose down

# Arrêter et supprimer les volumes (ATTENTION : supprime les données DB)
docker compose down -v

# Voir les logs en temps réel
docker compose logs -f

# Logs d'un seul service
docker compose logs -f fastapi
docker compose logs -f frontend
docker compose logs -f postgres

# Redémarrer un seul service
docker compose restart fastapi

# Ouvrir un shell dans un conteneur
docker compose exec fastapi bash
docker compose exec frontend sh
docker compose exec postgres psql -U postgres -d pwcaudit

# Statut des services
docker compose ps

# Rebuild un seul service sans redémarrer les autres
docker compose build fastapi
docker compose up -d --no-deps fastapi
```

### URLs après démarrage Docker

| Service | URL |
|---------|-----|
| Frontend | `http://localhost:3000` |
| Backend API | `http://localhost:8000` |
| Swagger docs | `http://localhost:8000/docs` |
| PostgreSQL | `localhost:5432` |

### Services Docker

| Service | Image | Port |
|---------|-------|------|
| `postgres` | postgres:16-alpine | 5432 |
| `fastapi` | Build local (Dockerfile) | 8000 |
| `frontend` | Build local (frontend/Dockerfile) | 3000 |

---

## 9. Base de données

### Backend — Alembic (migrations Python)

```bash
# Appliquer toutes les migrations en attente
alembic upgrade head

# Revenir à la migration précédente
alembic downgrade -1

# Voir l'état actuel
alembic current

# Voir l'historique des migrations
alembic history

# Créer une nouvelle migration automatique (après modification de app/db/models.py)
alembic revision --autogenerate -m "description_de_la_migration"
```

### Frontend — Prisma (ORM TypeScript)

```bash
cd frontend

# Générer le client TypeScript depuis le schéma
npx prisma generate

# Créer et appliquer une migration
npx prisma migrate dev --name nom_migration

# Appliquer les migrations en production
npx prisma migrate deploy

# Synchroniser sans créer de migration (développement rapide)
npx prisma db push

# Peuplement initial (utilisateurs de démo)
npx prisma db seed
# ou via npm : npm run db:seed

# Interface graphique pour explorer la DB
npx prisma studio
# ou via npm : npm run db:studio

# Réinitialiser complètement la DB (DANGER : perte de données)
npx prisma migrate reset
```

### Tables principales (PostgreSQL)

| Table | Description |
|-------|-------------|
| `users` | Comptes utilisateurs (email, rôle, hash mot de passe) |
| `missions` | Missions d'audit (client, période, statut) |
| `datasets` | Fichiers CSV uploadés (métadonnées) |
| `analysis_runs` | Exécutions d'analyse (résultats JSON, score) |
| `audit_logs` | Journal d'audit immutable |

---

## 10. Tests

### Lancer tous les tests

```bash
# Depuis la racine, avec le venv activé
pytest tests/ -v
```

### Commandes pytest détaillées

```bash
# Tests avec rapport de couverture
pytest tests/ -v --cov=app --cov-report=html

# Lancer un fichier de tests spécifique
pytest tests/test_predictions.py -v

# Lancer un test spécifique par nom
pytest tests/test_predictions.py::test_predict_paysim -v

# Tests en parallèle (plus rapide)
pytest tests/ -v -n auto

# Arrêter au premier échec
pytest tests/ -v -x

# Mode silencieux (seulement les échecs)
pytest tests/ -q
```

### Fichiers de tests

| Fichier | Ce qu'il teste |
|---------|----------------|
| `test_preprocessing.py` | Nettoyage et normalisation des données |
| `test_models.py` | Chargement et inférence des modèles ML |
| `test_predictions.py` | Endpoint `/api/predict` complet |
| `test_explanations.py` | Endpoint `/api/explain` (SHAP, LIME) |
| `test_column_mapping.py` | Détection automatique des colonnes CSV |
| `test_profiling.py` | Analyse qualité des datasets |

### Test rapide de l'API (sans pytest)

```bash
# Health check
curl http://localhost:8000/api/health

# Test de prédiction avec le fichier de démo
curl -X POST http://localhost:8000/api/predict \
  -F "file=@test_paysim.csv"

# Via Python
python -c "
import requests
with open('test_paysim.csv', 'rb') as f:
    r = requests.post('http://localhost:8000/api/predict', files={'file': f})
print(r.json())
"
```

---

## 11. API — endpoints disponibles

Tous les endpoints sont préfixés par `/api`.

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/health` | Vérification de l'état du service |
| `POST` | `/api/predict` | Prédiction sur un fichier CSV |
| `GET` | `/api/explain/{tx_id}` | Explication d'une transaction (SHAP + LIME + LLM) |
| `POST` | `/api/explain/batch` | Explication de plusieurs transactions |
| `POST` | `/api/report` | Génération rapport PDF ou DOCX |
| `GET` | `/api/models` | Métriques des modèles chargés |
| `POST` | `/api/profile` | Profilage qualité d'un fichier CSV |
| `GET/POST` | `/api/datasets` | Gestion des datasets |
| `GET` | `/api/audit` | Journal d'audit |
| `GET/POST` | `/api/model-registry` | Registre des versions de modèles |

### Exemples d'appels API

```bash
# Prédiction
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@transactions.csv"

# Explication d'une transaction
curl http://localhost:8000/api/explain/TXN_001?run_id=<RUN_ID> \
  -H "Authorization: Bearer <TOKEN>"

# Génération rapport PDF
curl -X POST http://localhost:8000/api/report \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"run_id": "<RUN_ID>", "format": "pdf", "mission_name": "Audit Q1 2024"}'

# Profilage CSV
curl -X POST http://localhost:8000/api/profile \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@transactions.csv"

# Métriques modèles
curl http://localhost:8000/api/models \
  -H "Authorization: Bearer <TOKEN>"
```

---

## 12. Variables d'environnement

### Backend (`.env`)

| Variable | Obligatoire | Description |
|----------|-------------|-------------|
| `JWT_SECRET` | Oui | Clé secrète JWT (min. 32 caractères) |
| `DATABASE_URL` | Non | URL PostgreSQL (sans DB = mode sans persistance) |
| `GROQ_API_KEY` | Non* | Clé API Groq (LLaMA-3) |
| `GEMINI_API_KEY` | Non* | Clé API Google Gemini |
| `HF_API_KEY` | Non* | Clé API HuggingFace |
| `AUTO_MIGRATE` | Non | `true` pour migrer automatiquement au démarrage |
| `UPLOADS_DIR` | Non | Répertoire de stockage des CSV uploadés |

*Au moins une clé LLM est recommandée pour les explications enrichies. Sans LLM, le système bascule en mode règles métier.

### Frontend (`frontend/.env.local`)

| Variable | Description |
|----------|-------------|
| `FASTAPI_URL` | URL interne du backend FastAPI |
| `NEXT_PUBLIC_API_URL` | URL publique de l'API (visible dans le navigateur) |
| `JWT_SECRET` | Même valeur que le backend |
| `JWT_EXPIRY` | Durée de validité du token en secondes (défaut : 28800 = 8h) |
| `DEMO_PASSWORD` | Mot de passe commun pour les comptes de démo |
| `DATABASE_URL` | URL PostgreSQL pour Prisma |

### Docker (`.env` à la racine)

| Variable | Description |
|----------|-------------|
| `POSTGRES_USER` | Nom d'utilisateur PostgreSQL |
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL |
| `POSTGRES_DB` | Nom de la base de données |
| `JWT_SECRET` | Clé secrète JWT partagée |
| `GROQ_API_KEY` | Clé Groq |
| `GEMINI_API_KEY` | Clé Gemini |
| `NEXT_PUBLIC_API_URL` | URL publique du frontend |

---

## 13. Flux de travail complet

### Développement local (recommandé pour la phase ML)

```bash
# Terminal 1 — Backend
.venv\Scripts\activate           # Windows
# source .venv/bin/activate       # Linux/macOS
uvicorn app.main:app --reload

# Terminal 2 — Frontend
cd frontend
npm run dev

# Terminal 3 — Notebooks (optionnel)
jupyter lab
```

### Déploiement Docker (recommandé pour la démo)

```bash
# Une seule commande pour tout démarrer
docker compose up --build

# Accéder à l'application
# Frontend : http://localhost:3000
# API docs : http://localhost:8000/docs
```

### Ordre de lancement recommandé (développement)

1. **PostgreSQL** — démarrer la DB (ou utiliser Docker uniquement pour la DB)
2. **Alembic** — `alembic upgrade head` (première fois)
3. **Backend FastAPI** — `uvicorn app.main:app --reload`
4. **Prisma seed** — `cd frontend && npm run db:seed` (première fois)
5. **Frontend Next.js** — `cd frontend && npm run dev`

### Ordre de régénération des modèles ML (après nouvelles données)

```bash
# Activer le venv
.venv\Scripts\activate

# Exécuter les notebooks dans l'ordre
python run_all.py

# OU manuellement notebook par notebook via Jupyter
# Les artefacts sont automatiquement sauvegardés dans outputs/models/

# Redémarrer le backend pour recharger les nouveaux modèles
# (le backend charge les modèles une seule fois au démarrage)
```

---

## Récapitulatif des commandes essentielles

```bash
# ── SETUP ───────────────────────────────────────────────────────────────────
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . && pip install -r requirements_app.txt
cd frontend && npm install && cd ..

# ── NOTEBOOKS ───────────────────────────────────────────────────────────────
jupyter lab                          # Ouvrir Jupyter
python run_all.py                    # Exécuter tous les notebooks

# ── BACKEND ─────────────────────────────────────────────────────────────────
uvicorn app.main:app --reload        # Démarrer l'API (dev)
alembic upgrade head                 # Appliquer les migrations DB
alembic revision --autogenerate -m "msg"  # Créer une migration

# ── FRONTEND ────────────────────────────────────────────────────────────────
cd frontend
npm run dev                          # Démarrer le frontend (dev)
npm run build && npm run start       # Build de production
npm run db:seed                      # Peuplement initial
npm run db:studio                    # Interface visuelle DB

# ── DOCKER ──────────────────────────────────────────────────────────────────
docker compose up --build            # Build + démarrage complet
docker compose up -d                 # Démarrage en arrière-plan
docker compose down                  # Arrêt
docker compose logs -f               # Logs en temps réel

# ── TESTS ───────────────────────────────────────────────────────────────────
pytest tests/ -v                     # Tous les tests
pytest tests/ -v --cov=app           # Avec couverture de code
curl http://localhost:8000/api/health  # Test rapide de l'API
```
