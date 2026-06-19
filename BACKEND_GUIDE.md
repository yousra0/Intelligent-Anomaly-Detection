# Guide Complet du Backend FastAPI — Système de Détection d'Anomalies
### (Document technique détaillé — de A à Z, adapté à une validation par expert)

---

## Table des matières

1. [Qu'est-ce qu'un backend et pourquoi FastAPI ?](#1-quest-ce-quun-backend-et-pourquoi-fastapi)
2. [Vue d'ensemble du système](#2-vue-densemble-du-système)
3. [Structure du projet](#3-structure-du-projet)
4. [Comment les modèles des notebooks ont été intégrés](#4-comment-les-modèles-des-notebooks-ont-été-intégrés)
5. [Démarrage de l'application — le fichier main.py](#5-démarrage-de-lapplication--le-fichier-mainpy)
6. [Pipeline de prédiction — de bout en bout](#6-pipeline-de-prédiction--de-bout-en-bout)
7. [Les routes (endpoints) — explication complète](#7-les-routes-endpoints--explication-complète)
8. [Services internes — comment ils fonctionnent](#8-services-internes--comment-ils-fonctionnent)
9. [Intégration LLM](#9-intégration-llm)
10. [Génération de rapports](#10-génération-de-rapports)
11. [La documentation interactive Swagger](#11-la-documentation-interactive-swagger)
12. [Variables d'environnement et configuration](#12-variables-denvironnement-et-configuration)
13. [Installation et déploiement pas à pas](#13-installation-et-déploiement-pas-à-pas)
14. [Tests](#14-tests)
15. [Architecture des services et flux de données](#15-architecture-des-services-et-flux-de-données)
16. [Détail des artefacts ML produits par les notebooks](#16-détail-des-artefacts-ml-produits-par-les-notebooks)
17. [Modifications récentes et corrections](#17-modifications-récentes-et-corrections)
18. [Questions fréquentes d'un expert technique](#18-questions-fréquentes-dun-expert-technique)

---

## 1. Qu'est-ce qu'un backend et pourquoi FastAPI ?

### Définition simple

Un **backend** est la partie invisible d'une application web — le cerveau. Quand vous appuyez sur un bouton sur le site, c'est le backend qui reçoit la demande, effectue les calculs (ici : la détection de fraude), et renvoie le résultat.

Le frontend (interface React) est la partie visible (les graphiques, tableaux, boutons). Le frontend et le backend communiquent via des **requêtes HTTP**, exactement comme un navigateur qui charge une page web.

### Pourquoi FastAPI ?

**FastAPI** est un framework Python moderne pour construire des APIs REST. Il a été choisi pour trois raisons :

| Raison | Explication |
|--------|-------------|
| **Performance** | L'un des frameworks Python les plus rapides — comparable à Node.js grâce à `asyncio` |
| **Documentation automatique** | Génère Swagger UI et ReDoc automatiquement depuis le code |
| **Typage strict** | Utilise Pydantic pour valider automatiquement les données entrantes/sortantes |

### Qu'est-ce qu'une API REST ?

Une API REST est un ensemble de "points d'entrée" (routes/endpoints) que le frontend peut appeler. Chaque route correspond à une action :

```
POST /api/predict   → envoyer un fichier CSV, recevoir les prédictions de fraude
GET  /api/models    → obtenir les métriques des modèles ML
GET  /api/explain/5 → obtenir l'explication de la transaction numéro 5
POST /api/report    → générer et télécharger un rapport PDF
```

`POST` signifie "j'envoie des données au serveur". `GET` signifie "je demande des données au serveur".

---

## 2. Vue d'ensemble du système

```
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACE UTILISATEUR                       │
│              (React + Next.js, http://localhost:3000)           │
└─────────────────────────────┬───────────────────────────────────┘
                              │ Requêtes HTTP (JSON/multipart)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND FastAPI                               │
│              (Python, http://localhost:8000)                     │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  /api/predict│  │/api/explain  │  │  /api/report (PDF)   │  │
│  │  /api/profile│  │/api/models   │  │  /api/report/docx    │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────────────────┘  │
│         │                │                                       │
│         ▼                ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    SERVICES INTERNES                        │ │
│  │  ColumnMapper → SchemaDetector → FeatureBuilder            │ │
│  │  Predictor (XGBoost + AutoEncoder)                         │ │
│  │  Explainer (SHAP + AE Proxy + LIME) → LLMHelper           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              MODÈLES ML (chargés au démarrage)             │ │
│  │  xgb_smote.pkl │ autoencoder_weights.pt │ scaler.pkl       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Modèles de production (les deux modèles qui tournent réellement)

Le système utilise **deux modèles complémentaires** simultanément quand le CSV est au format PaySim :

| Modèle | Ce qu'il fait | Fichier |
|--------|---------------|---------|
| **XGBoost** (`XGB_smote`) | Modèle supervisé : a été entraîné avec des exemples de fraudes connues. Prédit la probabilité de fraude (0→1). C'est lui qui décide si c'est une fraude. | `outputs/models/xgb_smote.pkl` |
| **AutoEncoder** | Réseau de neurones non supervisé : a été entraîné uniquement sur des transactions légitimes. Mesure à quel point une transaction est "anormale". | `outputs/models/autoencoder/` |

**Analogie** : XGBoost est comme un expert qui a vu des milliers de fraudes et reconnaît les patterns. L'AutoEncoder est comme une personne qui connaît parfaitement les transactions normales et signale tout ce qui lui semble inhabituel.

---

## 3. Structure du projet

```
anomaly_detection_project/
│
├── app/                          ← Tout le code du backend FastAPI
│   ├── main.py                   ← Point d'entrée : lance le serveur, charge les modèles
│   │
│   ├── routes/                   ← Les endpoints (ce que le frontend peut appeler)
│   │   ├── predict.py            ← POST /api/predict
│   │   ├── explain.py            ← GET  /api/explain/{tx_id}
│   │   │                            POST /api/explain/batch
│   │   ├── report.py             ← POST /api/report  (PDF)
│   │   │                            POST /api/report/docx (Word)
│   │   ├── models.py             ← GET  /api/models
│   │   └── profile.py            ← POST /api/profile
│   │
│   └── services/                 ← La logique métier (les "cerveaux" de chaque action)
│       ├── predictor.py          ← Charge les modèles + exécute XGBoost + AutoEncoder
│       ├── generic_predictor.py  ← Prédiction pour les CSV non-PaySim
│       ├── column_mapper.py      ← Détecte automatiquement les colonnes du CSV
│       ├── schema_detector.py    ← Choisit quel(s) modèle(s) utiliser
│       ├── feature_builder.py    ← Construit les 14 variables pour les modèles
│       ├── feature_engineer.py   ← Variables enrichies supplémentaires
│       ├── dataset_profiler.py   ← Analyse la qualité du CSV
│       ├── explainer.py          ← SHAP + Proxy AE + LIME
│       ├── llm_service.py        ← Connexion au modèle de langage (LLM)
│       ├── report_gen.py         ← Génère le PDF style PwC
│       └── report_gen_docx.py    ← Génère le Word depuis template
│
├── src/                          ← Bibliothèque ML réutilisable (installée comme package)
│   ├── models/
│   │   ├── autoencoder.py        ← Définition PyTorch de l'AutoEncoder + save/load
│   │   └── ml_models.py          ← Définitions des autres modèles ML
│   ├── llm_integration/
│   │   └── llm_helper.py         ← Wrapper LLM (Groq, Gemini, HuggingFace)
│   └── ...
│
├── outputs/                      ← Artefacts produits par les notebooks (JAMAIS touchés manuellement)
│   ├── models/
│   │   ├── xgb_smote.pkl         ← XGBoost entraîné (Notebook NB02/NB03)
│   │   ├── scaler.pkl            ← StandardScaler pour normaliser les 6 features
│   │   ├── features.json         ← Liste exacte des 14 features attendues
│   │   ├── optimal_thresholds.json ← Seuils optimaux calculés sur validation
│   │   └── autoencoder/
│   │       ├── autoencoder_weights.pt  ← Poids PyTorch (Notebook NB05)
│   │       └── autoencoder_meta.pkl    ← Architecture, seuil, stats MSE
│   └── reports/
│       ├── baseline_report.json  ← Métriques LR/RF/XGB/IsoForest (NB03)
│       └── autoencoder_report.json ← Métriques AutoEncoder (NB05)
│
├── config/
│   ├── config.yaml               ← Configuration générale
│   └── llm_config.yaml           ← Choix du provider LLM et paramètres
│
├── exemple_rapport.docx          ← Template Word avec {{placeholders}}
├── .env                          ← Clés API (ne jamais commiter sur Git)
├── requirements.txt              ← Dépendances Python
└── pyproject.toml                ← Configuration du package src/ (ml_core)
```

---

## 4. Comment les modèles des notebooks ont été intégrés

C'est une question centrale pour un expert technique. Voici le processus complet.

### Le problème fondamental

Les notebooks Jupyter sont excellents pour explorer et entraîner des modèles. Mais un notebook ne peut pas être "appelé" par une API web. Il faut **sauvegarder** les modèles entraînés dans des fichiers, puis **recharger** ces fichiers dans le backend.

### Étape 1 — Les notebooks entraînent et sauvegardent

**Notebook NB02 (Preprocessing)** produit :
- `outputs/models/scaler.pkl` — le StandardScaler ajusté sur les données d'entraînement
- `outputs/models/features.json` — la liste exacte des 14 features dans l'ordre attendu

```python
# Dans NB02 (simplifié)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train[SCALE_COLS])
joblib.dump(scaler, "outputs/models/scaler.pkl")

features_meta = {
    "all_features": FEATURE_COLS,  # ["step", "hour", "day", ..., "log_amount"]
    "scale_cols": SCALE_COLS,
    ...
}
with open("outputs/models/features.json", "w") as f:
    json.dump(features_meta, f)
```

**Notebook NB03 (Modèles supervisés)** produit :
- `outputs/models/xgb_smote.pkl` — XGBClassifier entraîné avec SMOTE
- `outputs/models/rf_smote.pkl`, `lr_balanced.pkl`, etc. — autres modèles évalués
- `outputs/models/optimal_thresholds.json` — seuil 0.355 calculé sur validation
- `outputs/reports/baseline_report.json` — métriques recall, precision, F1, ROC-AUC

```python
# Dans NB03 (simplifié)
xgb = XGBClassifier(...)
xgb.fit(X_smote, y_smote)
joblib.dump(xgb, "outputs/models/xgb_smote.pkl")

# Recherche du seuil optimal sur validation (PAS sur test)
threshold = find_optimal_threshold(xgb, X_val, y_val, metric="f1")
# → 0.355

baseline_report = {
    "models": [
        {"name": "XGB_smote", "optimal_threshold": 0.355,
         "test_metrics": {"recall": 0.8462, "f1": 0.8354, ...}}
    ]
}
```

**Notebook NB05 (AutoEncoder)** produit :
- `outputs/models/autoencoder/autoencoder_weights.pt` — poids PyTorch
- `outputs/models/autoencoder/autoencoder_meta.pkl` — architecture + seuil 1.753
- `outputs/reports/autoencoder_report.json` — métriques de l'AE

```python
# Dans NB05 (simplifié)
ae = FraudAutoEncoder()
ae.build(n_features=14)
ae.fit(X_train_normal)           # Entraîné UNIQUEMENT sur les transactions légitimes
threshold = ae.find_optimal_threshold(X_val, y_val)  # → 1.753
ae.save("outputs/models/autoencoder")
# Crée : autoencoder_weights.pt (torch.save) + autoencoder_meta.pkl (joblib)
```

### Étape 2 — Le backend charge tout au démarrage

Quand on lance le serveur, la fonction `load_all_models()` dans `app/services/predictor.py` charge **tous les artefacts en mémoire vive (RAM)** une seule fois :

```python
# app/services/predictor.py — load_all_models()
def load_all_models(project_root: Path) -> dict:
    models_dir = project_root / "outputs" / "models"
    reports_dir = project_root / "outputs" / "reports"

    # 1. Charger la liste des features et les seuils
    with open(models_dir / "features.json") as f:
        features_meta = json.load(f)   # → {"all_features": ["step", "hour", ...]}
    with open(models_dir / "optimal_thresholds.json") as f:
        thresholds_raw = json.load(f)  # → {"XGB_smote": 0.355, ...}

    # 2. Charger les métriques
    with open(reports_dir / "baseline_report.json") as f:
        baseline_report = json.load(f)
    with open(reports_dir / "autoencoder_report.json") as f:
        ae_report = json.load(f)

    # 3. Fusionner les métriques AE dans baseline_report
    # (Les deux fichiers sont séparés car l'AE a été entraîné dans un NB différent)
    ae_names = {m.get("name") for m in baseline_report.get("models", [])}
    if "AutoEncoder" not in ae_names:
        baseline_report["models"].append({
            "name": "AutoEncoder",
            "optimal_threshold": ae_report["threshold"]["optimal"],  # 1.753
            "train_time_s": ae_report["training"]["train_time_s"],
            "test_metrics": ae_report["test_metrics"],
        })

    # 4. Charger les modèles ML (joblib pour sklearn, PyTorch pour l'AE)
    ae = FraudAutoEncoder.load(models_dir / "autoencoder")
    # FraudAutoEncoder.load() :
    #   1. lit autoencoder_meta.pkl (architecture, seuil, hyperparamètres)
    #   2. reconstruit le réseau de neurones PyTorch en mémoire
    #   3. charge les poids depuis autoencoder_weights.pt

    return {
        "scaler": joblib.load(models_dir / "scaler.pkl"),
        "features": features_meta["all_features"],
        "thresholds": thresholds_raw,
        "xgb": joblib.load(models_dir / "xgb_smote.pkl"),
        "ae": ae,
        "ae_threshold": float(ae.threshold),  # 1.753
        "iso_forest": joblib.load(models_dir / "iso_forest.pkl"),
        "baseline_report": baseline_report,
        ...
    }
```

### Étape 3 — Les modèles sont accessibles via app.state

FastAPI permet de stocker des objets globaux accessibles dans toutes les routes via `app.state` :

```python
# Au démarrage (lifespan)
app.state.models = load_all_models(PROJECT_ROOT)

# Dans n'importe quelle route
@router.post("/predict")
async def predict(request: Request, ...):
    models = request.app.state.models   # accès direct au dict
    xgb = models["xgb"]                # XGBClassifier prêt à l'emploi
    ae = models["ae"]                  # FraudAutoEncoder prêt à l'emploi
```

### Pourquoi ne pas réentraîner les modèles à chaque requête ?

Un expert posera cette question. La réponse : l'entraînement prend **3–5 minutes** pour XGBoost et **~3 minutes** pour l'AutoEncoder. On les entraîne une fois dans les notebooks, on sauvegarde, et le backend se contente de charger et d'utiliser. C'est le principe du "train offline, serve online".

### L'AutoEncoder — Architecture technique

```
Entrée (14 features)
    │
    ▼
Linear(14 → 10) → BatchNorm → ReLU → Dropout(0.2)  ← Encodeur couche 1
    │
    ▼
Linear(10 → 7)  → BatchNorm → ReLU → Dropout(0.2)  ← Encodeur couche 2
    │
    ▼
Linear(7 → 4)   → ReLU                              ← Bottleneck (espace latent)
    │
    ▼
Linear(4 → 7)   → BatchNorm → ReLU → Dropout(0.2)  ← Décodeur couche 1
    │
    ▼
Linear(7 → 10)  → BatchNorm → ReLU → Dropout(0.2)  ← Décodeur couche 2
    │
    ▼
Linear(10 → 14)                                      ← Sortie (reconstruction)
```

**Principe de détection** :
- Entraîné uniquement sur 139 818 transactions légitimes → il apprend à reconstruire les patterns normaux
- Sur une transaction légitime → erreur de reconstruction faible (≤ 1.753)
- Sur une transaction frauduleuse → erreur de reconstruction élevée (> 1.753) — le modèle ne "reconnaît" pas le pattern

**Score normalisé** (`predict_score`) : l'erreur MSE brute est divisée par le 99ème percentile de l'erreur sur les données d'entraînement → score entre 0 et 1 affiché dans l'API.

---

## 5. Démarrage de l'application — le fichier main.py

Le fichier `app/main.py` est le point d'entrée. Voici ce qu'il fait ligne par ligne :

```python
# app/main.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ═══ PHASE DÉMARRAGE (avant d'accepter des requêtes) ═══
    
    # 1. Charger tous les modèles ML en RAM
    app.state.models = load_all_models(PROJECT_ROOT)
    # → dict avec xgb, ae, scaler, thresholds, baseline_report...
    
    # 2. Initialiser le LLM
    try:
        app.state.llm_helper = get_llm_helper(PROJECT_ROOT)
        # → LLMHelper connecté à Groq/Gemini/HuggingFace selon llm_config.yaml
    except Exception as e:
        app.state.llm_helper = None
        # → mode fallback : explications basées sur des règles, sans IA
    
    # 3. Initialiser le cache en mémoire
    app.state.results_cache = {}
    # → stockera les résultats du dernier /api/predict pour /api/explain
    
    yield   # ← Le serveur tourne ici, accept les requêtes
    
    # ═══ PHASE ARRÊT ═══
    # (nettoyage si nécessaire)
```

```python
# Création de l'application FastAPI
app = FastAPI(
    title="API Détection de Fraude — PFE",
    version="1.0.0",
    lifespan=lifespan,  # ← exécute lifespan() au démarrage/arrêt
)

# Configuration CORS : autorise le frontend à appeler le backend
# (par défaut les navigateurs bloquent les appels cross-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],    # GET, POST, DELETE...
    allow_headers=["*"],    # Content-Type, Authorization...
)

# Enregistrement des routeurs avec le préfixe /api
app.include_router(predict.router, prefix="/api")   # → /api/predict
app.include_router(explain.router, prefix="/api")   # → /api/explain/{tx_id}
app.include_router(report.router,  prefix="/api")   # → /api/report
app.include_router(models_route.router, prefix="/api")  # → /api/models
app.include_router(profile.router, prefix="/api")   # → /api/profile
```

**Point clé** : chaque `router` est défini dans son propre fichier dans `app/routes/`. Cela permet de **séparer les responsabilités** (chaque fichier gère un domaine fonctionnel).

---

## 6. Pipeline de prédiction — de bout en bout

Voici exactement ce qui se passe quand le frontend envoie un fichier CSV :

```
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1 : Lecture du CSV                                          │
│  Le fichier arrivé en multipart/form-data est lu par pandas        │
│  → df = pd.read_csv(io.BytesIO(content))                          │
│  → Validation : fichier non vide, parseable                        │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2 : Profilage du dataset (dataset_profiler.py)              │
│  Analyse chaque colonne : type (numérique/catégoriel/datetime),    │
│  valeurs manquantes, asymétrie statistique, cardinalité            │
│  → Produit un DatasetProfile utilisé par le feature builder        │
│  → Calcule un score qualité global (0-100)                         │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3 : Mapping sémantique des colonnes (column_mapper.py)      │
│  Le CSV peut avoir n'importe quel nom de colonne.                  │
│  Le mapper détecte automatiquement à quoi correspond chaque colonne│
│  en utilisant 4 niveaux de matching :                              │
│                                                                     │
│    Niveau 1 — Alias exact (confiance 1.00)                         │
│      "amount", "montant", "amt" → amount                           │
│    Niveau 2 — Alias normalisé (confiance 0.95)                     │
│      "AMOUNT", "Amount" → amount (insensible à la casse)           │
│    Niveau 3 — Signaux composés (confiance 0.80)                    │
│      colonne contenant "old" + "balance" + "orig" → oldbalanceOrg  │
│    Niveau 4 — Fuzzy matching (confiance 0.60)                      │
│      "amnt" → amount via SequenceMatcher                           │
│                                                                     │
│  → MappingResult.success = True si TOUTES les colonnes             │
│    requises PaySim sont trouvées                                    │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 4 : Détection du mode (schema_detector.py)                  │
│                                                                     │
│  mapping.success = True ?                                           │
│    → OUI  : mode "paysim"      → XGBoost + AutoEncoder            │
│    → NON  : "amount" mappé ?                                        │
│               → OUI + ≥1 col num. : mode "ae_isoforest"           │
│               → OUI + 0 col num.  : mode "ae_only"                │
│               → NON + ≥1 col num. : mode "isoforest"              │
│               → Rien              : mode dégradé (tout FAIBLE)     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 5 : Construction des 14 features (feature_builder.py)       │
│                                                                     │
│  Feature               Source                  Calcul              │
│  ─────────────────────────────────────────────────────────────     │
│  step                  colonne "step"           direct (numérique) │
│  hour                  step                     step % 24          │
│  day                   step                     step // 24         │
│  week                  step                     step // 168        │
│  high_risk_hour        hour                     hour ∈ [0-9, 23]  │
│  is_transfer_or_cashout type                    type ∈ [TRANSFER,  │
│                                                  CASH_OUT]         │
│  balance_diff_orig     oldbalanceOrg,           oldBalance -        │
│                        newbalanceOrig           newBalance          │
│  dest_zero_balance     oldbalanceDest           oldbalanceDest == 0│
│  type_CASH_IN          type                     OHE (0 ou 1)       │
│  type_CASH_OUT         type                     OHE (0 ou 1)       │
│  type_DEBIT            type                     OHE (0 ou 1)       │
│  type_PAYMENT          type                     OHE (0 ou 1)       │
│  type_TRANSFER         type                     OHE (0 ou 1)       │
│  log_amount            amount                   log(1 + amount)    │
│                                                                     │
│  Puis : StandardScaler sur 6 colonnes (step, hour, day, week,     │
│         log_amount, balance_diff_orig) — même scaler que NB02      │
│  → X_arr : numpy array (N, 14) dtype float32                       │
│                                                                     │
│  Si une colonne est absente → valeur 0 + avertissement dans la     │
│  réponse JSON (dégradation gracieuse, jamais d'erreur fatale)      │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 6 : Feature engineering enrichi (feature_engineer.py)       │
│  15 features supplémentaires calculées pour le frontend/rapports   │
│  (NON utilisées par XGBoost/AE — informatives uniquement) :        │
│  temporel : eng_tx_day_of_week, eng_is_weekend, eng_is_business_h  │
│  balance  : eng_drain_pct_src, eng_amount_ratio_src, ...           │
│  comportemental : eng_orig_tx_count, eng_orig_unique_dests, ...    │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 7 : Prédiction                                              │
│                                                                     │
│  Mode "paysim" → predictor.predict_batch(X_arr, models, df)        │
│                                                                     │
│    xgb_scores = xgb.predict_proba(X_arr)[:, 1]  → [0.99, 0.02...] │
│    ae_scores  = ae.predict_score(X_arr)          → [0.62, 0.10...] │
│                                                                     │
│    Pour chaque transaction :                                        │
│      xgb_score ≥ 0.355 → is_fraud_predicted = True                 │
│      xgb_score ≥ 0.355 → risk_level = "CRITIQUE"                  │
│      0.5 ≤ xgb_score < 0.355 → risk_level = "ELEVE"               │
│      xgb_score < 0.5   → risk_level = "FAIBLE"                    │
│                                                                     │
│  Mode générique → generic_predictor.predict_generic_batch()        │
│    IsoForest fitté ON-THE-FLY sur ce batch (contamination 5%)      │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 8 : Mise en cache + réponse JSON                            │
│                                                                     │
│  app.state.results_cache = {X_arr, df_enriched, transactions}      │
│  → permet à /api/explain de retrouver les features sans recalcul   │
│                                                                     │
│  Réponse JSON retournée au frontend :                               │
│  { n_transactions, n_fraud, fraud_rate_pct, amount_at_risk,        │
│    model_used, prediction_mode, schema_detection, column_mapping,   │
│    transactions: [{tx_id, type, amount, xgb_score, ae_score,       │
│                    risk_level, is_fraud_predicted}...],             │
│    dataset_profile, feature_build }                                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## 7. Les routes (endpoints) — explication complète

### `POST /api/predict` — Prédiction de fraude

**Qui appelle ça ?** Le frontend quand l'utilisateur glisse un fichier CSV.

**Comment ?** En `multipart/form-data` (comme un formulaire HTML avec un fichier attaché).

**Ce qui se passe** : pipeline complet des 8 étapes décrites ci-dessus.

**Requête** :
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@transactions.csv"
```

**Réponse** (exemple réel) :
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
    "reason": "Schéma PaySim complet → XGBoost + Autoencoder",
    "warnings": []
  },
  "column_mapping": {
    "amount": {"original_name": "amount", "confidence": 1.0},
    "step":   {"original_name": "step",   "confidence": 1.0}
  },
  "transactions": [
    {
      "tx_id": 21,
      "type": "CASH_OUT",
      "amount": 62229.21,
      "xgb_score": 0.993882,
      "ae_score": 0.623424,
      "risk_level": "CRITIQUE",
      "is_fraud_predicted": true
    }
  ],
  "threshold": 0.3547,
  "dataset_profile": {
    "n_rows": 50,
    "global_quality_score": 98.9
  }
}
```

**Champs clés** :

| Champ | Type | Signification |
|-------|------|---------------|
| `tx_id` | int | Index de la ligne dans le CSV (0, 1, 2...) |
| `xgb_score` | float [0-1] | Probabilité de fraude selon XGBoost. 0 = très légitime, 1 = très suspect |
| `ae_score` | float [0-1] | Niveau d'anomalie selon l'AutoEncoder. 0 = normal, >1 = très anormal |
| `risk_level` | string | "CRITIQUE" (xgb ≥ 0.355), "ELEVE" (xgb ≥ 0.5 seulement), "FAIBLE" |
| `is_fraud_predicted` | bool | **La décision finale** : true si xgb_score ≥ seuil (0.355) |

**Modes de prédiction** :

| Mode | Condition déclenchante | Modèles |
|------|------------------------|---------|
| `paysim` | Toutes les colonnes PaySim reconnues | XGBoost + AutoEncoder |
| `ae_isoforest` | `amount` reconnu + colonnes numériques présentes | AE + IsoForest on-the-fly |
| `ae_only` | `amount` reconnu + pas assez de colonnes numériques | AutoEncoder seul |
| `isoforest` | Pas d'`amount` + colonnes numériques présentes | IsoForest on-the-fly |

**Codes d'erreur** :
- `400` : fichier vide ou aucun résultat en cache
- `422` : CSV illisible ou aucun modèle applicable
- `500` : erreur interne pendant le traitement

---

### `GET /api/explain/{tx_id}` — Explication d'une transaction

**Quand ?** Après `/api/predict`. L'utilisateur clique sur une transaction suspecte pour comprendre POURQUOI elle est signalée.

**Paramètre** : `tx_id` = le `tx_id` dans la réponse de predict (ex: `/api/explain/21`)

**Ce que fait le backend** :
1. Retrouve la transaction dans `app.state.results_cache`
2. Récupère le vecteur de features `tx_arr` (les 14 valeurs)
3. Calcule **trois types d'explication** :

**SHAP (SHapley Additive exPlanations)** pour XGBoost :
```python
# explainer.py
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(tx_arr.reshape(1, -1))
# → {"balance_diff_orig": +0.42, "log_amount": +0.31, "type_CASH_OUT": +0.18, ...}
# Interprétation : chaque valeur est la contribution de cette feature
# à la prédiction finale. + = pousse vers "fraude", - = pousse vers "légitime"
```

**Proxy AE** pour l'AutoEncoder :
```python
# explainer.py
recon = ae.model(tx_tensor)                   # L'AE reconstruit la transaction
errors = |tx_arr - recon|                     # Erreur par feature
# → {"balance_diff_orig": 65.73, "log_amount": 5.87, ...}
# Interprétation : les features avec la plus grande erreur sont celles
# qui ont le plus contribué au score d'anomalie
```

> **Note technique** : On n'utilise pas SHAP KernelExplainer pour l'AE (beaucoup trop lent en production — plusieurs dizaines de secondes par transaction). Le Proxy AE est une approximation directe, rapide, et interprétable.

**LIME (Local Interpretable Model-agnostic Explanations)** pour XGBoost :
```python
# explainer.py
explainer = LimeTabularExplainer(training_data=X_arr, ...)
exp = explainer.explain_instance(tx_arr, xgb.predict_proba)
# → ["balance_diff_orig > 0: +0.382", "log_amount > 9.5: +0.221", ...]
# Interprétation : règles locales qui expliquent CETTE transaction spécifique
```

4. Appelle le **LLM** pour une explication en langage naturel (si disponible)

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
    "type_CASH_OUT": 1.0
  },
  "shap_values_xgb": {
    "balance_diff_orig": 0.42,
    "log_amount": 0.31,
    "type_CASH_OUT": 0.18
  },
  "ae_feature_errors": {
    "balance_diff_orig": 65.73,
    "log_amount": 5.87
  },
  "ae_top_features": [
    {"feature": "balance_diff_orig", "error": 65.73},
    {"feature": "log_amount", "error": 5.87}
  ],
  "lime_rules": [
    "balance_diff_orig > 0: +0.382",
    "log_amount > 9.5: +0.221"
  ],
  "llm": {
    "risk_level": "CRITIQUE",
    "resume": "Retrait espèces avec vidange totale du compte source",
    "raisons": ["Solde réduit à zéro", "Montant élevé atypique"],
    "actions_recommandees": ["Bloquer la transaction", "Contacter le titulaire"],
    "status": "ok",
    "_audit": {
      "hash": "04a1c8...",
      "timestamp_utc": "2026-06-17T09:00:00+00:00",
      "hash_algo": "sha256"
    }
  }
}
```

**Tableau des méthodes d'explicabilité** :

| Champ | Méthode | Modèle expliqué | Vitesse |
|-------|---------|-----------------|---------|
| `shap_values_xgb` | SHAP TreeExplainer | XGBoost | ~10ms |
| `ae_feature_errors` | Proxy \|x - AE(x)\| | AutoEncoder | ~5ms |
| `lime_rules` | LIME LimeTabularExplainer | XGBoost | ~200ms |
| `llm.*` | Appel API (Groq/Gemini) | Explication globale | ~2-5s |

---

### `POST /api/explain/batch` — Explication en lot

**Quand ?** Pour expliquer plusieurs transactions en une seule requête (sans faire 50 appels séparés).

**LIME est désactivé** en mode batch (trop lent pour >1 transaction).

**Requête** :
```bash
curl -X POST http://localhost:8000/api/explain/batch \
  -H "Content-Type: application/json" \
  -d '{"tx_ids": [21, 48, 41], "max_explain": 20}'
```

**Corps** :

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `tx_ids` | `list[int]` | requis | Liste des identifiants à expliquer |
| `max_explain` | `int` | 20 | Maximum de 100. Limite de sécurité pour éviter les timeouts |

**Réponse** :
```json
{
  "n_requested": 3,
  "n_explained": 3,
  "n_errors": 0,
  "explanations": [{"tx_id": 21, ...}, {"tx_id": 48, ...}],
  "errors": []
}
```

**Workflow recommandé** : filtrer avec `/api/predict` → extraire les `tx_id` des transactions `is_fraud_predicted = true` → appeler `/api/explain/batch` uniquement sur ceux-là.

---

### `GET /api/models` — Métriques comparatives des modèles

**Quand ?** Le frontend charge le tableau de comparaison des modèles.

**Ce que fait le backend** :
- Lit `baseline_report.json` et `autoencoder_report.json` (déjà fusionnés en RAM)
- Formate les métriques pour les 7 modèles évalués

**Réponse** :
```json
{
  "models": [
    {
      "name": "XGB_smote",
      "recall": 0.8462,
      "precision": 0.8250,
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
      "recall": 0.3590,
      "f1": 0.4516,
      "roc_auc": 0.9358,
      "optimal_threshold": 1.753,
      "is_best": false,
      "is_in_production": true
    },
    {
      "name": "RF_smote",
      "is_best": false,
      "is_in_production": false
    }
  ]
}
```

**Deux flags importants** :

| Flag | Signification |
|------|---------------|
| `is_best: true` | Meilleur modèle standalone en métriques (XGBoost uniquement) |
| `is_in_production: true` | Modèle actif dans le système (XGBoost + AutoEncoder) |

> L'AutoEncoder a `is_best: false` car ses métriques standalone (recall 0.36) sont inférieures à XGBoost (recall 0.85). Mais il est `is_in_production: true` car il apporte une **détection complémentaire** basée sur l'anomalie structurelle, indépendante des labels.

---

### `POST /api/profile` — Profilage d'un CSV (sans prédiction)

**Quand ?** L'utilisateur veut analyser la qualité d'un fichier sans lancer la détection.

**Ce que fait le backend** : uniquement les étapes 1 et 2 du pipeline (lecture + profilage), sans construction de features ni prédiction.

**Requête** :
```bash
curl -X POST http://localhost:8000/api/profile -F "file=@transactions.csv"
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
  "high_missing_cols": [],
  "recommendations": ["Vérifier les valeurs manquantes dans 'newbalanceDest'"],
  "profiling_time_ms": 33.3
}
```

---

### `POST /api/report` — Rapport PDF

**Quand ?** L'utilisateur clique sur "Télécharger le rapport PDF".

**Ce que fait le backend** :
1. Lit `app.state.results_cache` (ou le corps JSON envoyé)
2. Appelle `generate_pwc_report()` dans `report_gen.py`
3. Retourne le PDF en streaming (pas de fichier sauvegardé sur le serveur)

**Réponse** : Fichier binaire PDF avec headers :
```
Content-Type: application/pdf
Content-Disposition: attachment; filename="rapport_pwc_20260617_0900.pdf"
```

**Structure du PDF (8 sections)** :
1. Page de couverture + jauge de risque globale
2. Résumé exécutif (4 KPI : transactions, fraudes, taux, montant)
3. 3 graphiques matplotlib (donut, barres, exposition)
4. Tableau top-10 transactions suspectes
5. Cartes détaillées CRITIQUE (max 8) avec SHAP
6. Recommandations pour auditeurs
7. Glossaire (7 termes métier)
8. Disclaimer légal

---

### `POST /api/report/docx` — Rapport Word

**Quand ?** L'utilisateur télécharge le rapport en format Word éditable.

**Comment fonctionne le template** :
- Le fichier `exemple_rapport.docx` contient des `{{placeholders}}`
- `python-docx` parcourt tous les paragraphes et remplace les placeholders
- Les graphiques matplotlib sont intégrés comme images PNG dans le document

**Placeholders remplacés** : `{{date}}`, `{{taux_de_fraudes}}`, `{{n_transactions}}`, `{{n_fraudes}}`, `{{montant_a_risque}}`, `{{analyse_text}}`, etc.

---

### `GET /api/health` — Vérification du statut

**Quand ?** Le monitoring ou le frontend vérifie que le serveur est opérationnel.

```bash
curl http://localhost:8000/api/health
```

**Réponse** :
```json
{
  "status": "ok",
  "models_loaded": true,
  "llm_available": true
}
```

---

## 8. Services internes — comment ils fonctionnent

### `column_mapper.py` — Détection automatique des colonnes

**Problème résolu** : un client peut envoyer un CSV avec des colonnes nommées "montant", "Montant", "amt", "transaction_value" — toutes signifient "amount". Le mapper gère tous ces cas.

**Mécanisme** (affectation greedy) :
1. Pour chaque feature canonique PaySim (9 au total), on calcule un score pour chaque colonne du CSV
2. On trie toutes les paires (score, feature, colonne) par score décroissant
3. On affecte en priorité les paires avec le plus grand score
4. Une colonne ne peut être affectée qu'à une seule feature

```python
# Exemple de scoring pour la colonne "ancien_solde_compte_source"
# → Niveau 3 (signaux composés) :
#   contient "solde" (=balance), "ancien" (=old/before), "source" (=orig)
#   → oldbalanceOrg, confiance 0.88
```

**Normalisation des valeurs de `type`** : "virement" → "TRANSFER", "retrait" → "CASH_OUT", etc.

### `dataset_profiler.py` — Profilage sémantique

Pour chaque colonne, le profiler détecte :
- **Type sémantique** : numérique, catégoriel, datetime, identifiant, booléen, constante
- **Qualité** : valeurs manquantes, quasi-constance (>95% même valeur), asymétrie extrême
- **Stats numériques** : min, max, moyenne, écart-type, skewness, kurtosis

**Score qualité** : commence à 100, des pénalités sont déduites :
- Colonne quasi-constante : -5 points
- >30% de valeurs manquantes : -10 points
- Asymétrie extrême (|skewness| > 10) : -3 points

### `feature_builder.py` — Construction adaptative des features

**Dégradation gracieuse** : si une colonne est absente, la feature correspondante est mise à 0 avec un avertissement — jamais une erreur fatale.

```python
# Exemple : colonne "oldbalanceDest" absente
if "oldbalanceDest" in df.columns:
    result["dest_zero_balance"] = (df["oldbalanceDest"] == 0).astype(float)
else:
    result["dest_zero_balance"] = pd.Series(np.zeros(n))
    warnings.append("Colonne 'oldbalanceDest' absente — dest_zero_balance=0")
```

**Rapport détaillé** : chaque feature retourne son statut ("ok", "adapted", "missing_fallback") et la stratégie utilisée.

### `schema_detector.py` — Sélection du mode de prédiction

**Logique de décision** :

```
mapping.success = True
    → mode "paysim" (XGB + AE)

mapping.success = False :
    "amount" mappé + colonne numérique valide ?
        OUI + ≥1 col num. → mode "ae_isoforest"
        OUI + 0 col num.  → mode "ae_only"
        NON + ≥1 col num. → mode "isoforest"
    Aucune colonne utilisable ?
        → mode dégradé (tout classé FAIBLE + avertissement)
```

**Avertissement transductif IsoForest** :
```
"IsolationForest transductif : ajusté sur ce batch uniquement
(contamination=5%). Le modèle pré-entraîné PaySim n'est pas utilisé.
Si le batch ne contient pas de fraude réelle, des faux positifs sont inévitables."
```

### `generic_predictor.py` — Mode générique

**IsoForest on-the-fly** : contrairement au XGBoost/AE pré-entraînés, l'IsoForest en mode générique est ajusté directement sur le batch entrant. Pourquoi ? Le modèle `iso_forest.pkl` entraîné sur PaySim ne serait pas pertinent sur des données de structure inconnue.

**Classification en mode combiné (ae_isoforest)** :
```python
ae_flag = ae_score >= ae_threshold        # AE dit "anormal"
iso_flag = iso_prediction == -1           # IsoForest dit "anomalie"

if ae_flag and iso_flag:  → "CRITIQUE"   # Les deux modèles sont d'accord
if ae_flag or iso_flag:   → "ELEVE"      # Un seul modèle signale
else:                     → "FAIBLE"     # Aucun modèle ne signale
```

---

## 9. Intégration LLM

### Providers supportés

| Provider | Modèle | Variable d'env | Limite gratuite |
|----------|--------|----------------|-----------------|
| **Groq** (par défaut) | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | 14 400 req/jour |
| Gemini | `gemini-1.5-flash` | `GEMINI_API_KEY` | 1 500 req/jour |
| HuggingFace | `Mistral-7B-Instruct-v0.2` | `HF_API_KEY` | Variable |

### Changer de provider

Dans `config/llm_config.yaml` :
```yaml
active_provider: groq   # groq | gemini | huggingface
generation:
  temperature: 0.1      # Basse pour des réponses cohérentes et déterministes
  max_tokens: 400       # Réponse courte pour la performance
  timeout: 30           # Secondes avant abandon
```

### Ce que fait le LLM

Le LLM reçoit en entrée :
- Les valeurs des features de la transaction
- Les erreurs de reconstruction par feature (de l'AE)
- Le score de risque XGBoost
- Le seuil

Il retourne en JSON :
```json
{
  "risk_level": "CRITIQUE",
  "risk_score": 0.994,
  "resume": "Retrait espèces avec vidange totale du compte source",
  "raisons": ["Solde source réduit à zéro", "Montant élevé atypique"],
  "actions_recommandees": ["Bloquer la transaction", "Contacter le titulaire"],
  "status": "ok",
  "_audit": {
    "hash": "sha256_de_la_réponse",
    "timestamp_utc": "2026-06-17T09:00:00+00:00",
    "hash_algo": "sha256"
  }
}
```

### Mécanismes de robustesse

- **Retry automatique** : 3 tentatives avec backoff exponentiel (via `tenacity`)
- **JSON repair** : corrige les JSONs malformés via `json_repair`
- **Fallback** : si le LLM est indisponible → explication basée sur des règles métier (pas d'erreur pour l'utilisateur)
- **Audit trail** : chaque réponse LLM porte un hash SHA-256 + timestamp UTC pour la traçabilité

---

## 10. Génération de rapports

### PDF (`report_gen.py`)

Utilise **fpdf2** (génération de PDF pur Python, sans LaTeX ni dépendances externes).

**Palette PwC** :
- Orange PwC : `#D04A02`
- Bleu foncé : `#293854`
- Rouge critique : `#C00000`
- Vert légal : `#008246`

**Graphiques** : générés avec matplotlib, convertis en PNG en mémoire (`io.BytesIO`), embarqués dans le PDF. Aucun fichier temporaire sur le disque.

**Streaming** : le PDF est retourné via `StreamingResponse(iter([pdf_bytes]))` — zéro fichier sauvegardé sur le serveur.

### Word (`report_gen_docx.py`)

Utilise **python-docx** avec injection dans le template `exemple_rapport.docx`.

**Substitution des placeholders** :
```python
for paragraph in doc.paragraphs:
    for key, value in replacements.items():
        if f"{{{{{key}}}}}" in paragraph.text:
            for run in paragraph.runs:
                run.text = run.text.replace(f"{{{{{key}}}}}", str(value))
```

**Traductions des features** (pour les auditeurs non-techniques) :
```python
FEATURE_FR = {
    "log_amount":           "Montant (log)",
    "balance_diff_orig":    "Différence de solde (compte source)",
    "type_CASH_OUT":        "Type : Retrait espèces",
    "high_risk_hour":       "Heure à risque élevé",
    "dest_zero_balance":    "Compte destinataire vide",
    ...
}
```

---

## 11. La documentation interactive Swagger

FastAPI génère **automatiquement** une documentation interactive depuis le code Python.

**Accès** (une fois le serveur lancé) :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

**Ce qu'on peut faire dans Swagger UI** :
- Voir tous les endpoints avec leurs paramètres
- Tester une requête directement depuis le navigateur (sans Postman ni curl)
- Voir les schémas de réponse avec exemples

**Comment FastAPI génère la doc** : depuis les annotations Python et les modèles Pydantic :

```python
class BatchExplainRequest(BaseModel):
    tx_ids: list[int] = Field(..., description="Liste des tx_id à expliquer.")
    max_explain: int  = Field(20, ge=1, le=100, description="Limite max.")
```

Cette classe Pydantic génère automatiquement dans Swagger :
- Un formulaire JSON avec les champs `tx_ids` et `max_explain`
- La validation : `max_explain` doit être entre 1 et 100
- La description de chaque champ

---

## 12. Variables d'environnement et configuration

### Fichier `.env` (à la racine du projet)

```env
# Fournisseur LLM (au moins un)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxx
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Ce fichier ne doit **jamais** être commité sur Git (il est dans `.gitignore`).

### `config/llm_config.yaml`

```yaml
active_provider: groq    # ← Changer ici pour basculer de provider

providers:
  groq:
    model: llama-3.3-70b-versatile
  gemini:
    model: gemini-1.5-flash
  huggingface:
    model: mistralai/Mistral-7B-Instruct-v0.2

generation:
  temperature: 0.1
  max_tokens: 400
  timeout: 30
```

### `config/config.yaml`

Configuration générale du projet (chemins, paramètres globaux).

---

## 13. Installation et déploiement pas à pas

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Environ 2 Go d'espace disque (PyTorch + modèles)

### Étape 1 — Créer l'environnement virtuel

```bash
# Un environnement virtuel isole les dépendances du projet
python -m venv .venv

# Activer l'environnement (Windows PowerShell)
.venv\Scripts\activate

# Activer l'environnement (Linux/Mac)
source .venv/bin/activate

# Le prompt doit afficher (.venv) devant
```

### Étape 2 — Installer les dépendances

```bash
# Installer toutes les dépendances listées dans requirements.txt
pip install -r requirements.txt

# Installer le package src/ (ml_core) en mode éditable
# Nécessaire pour que "from src.models.autoencoder import FraudAutoEncoder" fonctionne
pip install -e .
```

**Si PyTorch pose problème** (CPU uniquement, version légère) :
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Si SHAP pose problème** :
```bash
pip install shap --no-build-isolation
```

### Étape 3 — Configurer les clés API

```bash
# Créer le fichier .env
echo "GROQ_API_KEY=votre_clé_ici" > .env
```

### Étape 4 — Vérifier que les artefacts ML sont présents

```bash
# Modèles obligatoires
ls outputs/models/xgb_smote.pkl
ls outputs/models/autoencoder/autoencoder_weights.pt
ls outputs/models/autoencoder/autoencoder_meta.pkl
ls outputs/models/scaler.pkl
ls outputs/models/features.json
ls outputs/models/optimal_thresholds.json

# Rapports de métriques
ls outputs/reports/baseline_report.json
ls outputs/reports/autoencoder_report.json

# Template Word
ls exemple_rapport.docx
```

### Étape 5 — Lancer le serveur

**Mode développement** (avec rechargement automatique) :
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Mode production** (4 workers en parallèle) :
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Étape 6 — Vérifier que tout fonctionne

```bash
# Test de santé
curl http://localhost:8000/api/health
# → {"status": "ok", "models_loaded": true, "llm_available": true}

# Documentation interactive
# Ouvrir http://localhost:8000/docs dans le navigateur
```

### Performances estimées

| Opération | Temps |
|-----------|-------|
| Démarrage du serveur (chargement des modèles) | 3–5 secondes |
| `/api/predict` sur 1 000 transactions | 200–500 ms |
| `/api/explain/{tx_id}` avec LLM | 2–5 secondes |
| `/api/report` PDF | 1–2 secondes |
| `/api/report/docx` Word | 2–4 secondes |
| Mémoire RAM totale | ~500 MB |

---

## 14. Tests

### Lancer tous les tests

```bash
pytest tests/ -v
```

### Tests spécifiques

```bash
pytest tests/test_predict.py -v       # Pipeline de prédiction
pytest tests/test_explain.py -v       # SHAP, AE Proxy, LIME, LLM
pytest tests/test_column_mapper.py -v # Détection sémantique des colonnes
pytest tests/test_profiler.py -v      # Profilage dataset
pytest tests/test_models.py -v        # Métriques des modèles
pytest tests/test_report.py -v        # Génération PDF
```

### Avec rapport de couverture

```bash
pip install pytest-cov
pytest tests/ --cov=app --cov-report=html
# Rapport HTML dans htmlcov/index.html
```

### Ce que teste chaque fichier

| Fichier | Ce qui est testé |
|---------|-----------------|
| `conftest.py` | Fixtures partagées : client TestClient, CSV PaySim synthétique |
| `test_predict.py` | Mode paysim, modes génériques, CSV vide, colonnes invalides |
| `test_column_mapper.py` | Alias exacts, fuzzy, normalisation type, conflits de mapping |
| `test_profiler.py` | Stats, score qualité, types sémantiques, recommandations |
| `test_explain.py` | SHAP, AE proxy, LIME, LLM, tx_id invalide |
| `test_models.py` | 7 modèles, métriques, flags is_best/is_in_production |
| `test_report.py` | PDF : Content-Type, headers, taille non nulle |
| `test_feature_engineer.py` | 15 features enrichies, valeurs limites |

### Flux de test complet (curl)

```bash
# 1. Prédire sur un CSV
curl -X POST http://localhost:8000/api/predict \
  -F "file=@test_paysim.csv" -o predict_result.json

# 2. Voir le résultat
cat predict_result.json | python -m json.tool | head -50

# 3. Expliquer la transaction la plus suspecte (tx_id 21 dans l'exemple)
curl http://localhost:8000/api/explain/21

# 4. Expliquer plusieurs en lot
curl -X POST http://localhost:8000/api/explain/batch \
  -H "Content-Type: application/json" \
  -d '{"tx_ids": [21, 48, 41], "max_explain": 10}'

# 5. Télécharger le rapport PDF
curl -X POST http://localhost:8000/api/report -o rapport.pdf

# 6. Télécharger le rapport Word
curl -X POST http://localhost:8000/api/report/docx -o rapport.docx
```

---

## 15. Architecture des services et flux de données

### État global (app.state)

FastAPI maintient un état global partagé entre toutes les requêtes via `app.state` :

```python
app.state.models = {
    "scaler":          StandardScaler     # normalisation des features
    "features":        list[str]          # ordre exact des 14 features
    "thresholds":      {"XGB_smote": 0.355, "RF_smote": 0.629, ...}
    "xgb":             XGBClassifier      # modèle de PRODUCTION
    "ae":              FraudAutoEncoder   # modèle de PRODUCTION
    "ae_threshold":    1.753             # seuil MSE de l'AE
    "lr_balanced":     LogisticRegression # évaluation seulement
    "lr_smote":        LogisticRegression # évaluation seulement
    "rf_balanced":     RandomForestClassifier  # évaluation seulement
    "rf_smote":        RandomForestClassifier  # évaluation seulement
    "iso_forest":      IsolationForest    # mode générique fallback
    "iso_scaler":      MinMaxScaler       # normalisation scores AE
    "baseline_report": dict              # métriques des 7 modèles
}

app.state.llm_helper = LLMHelper | None   # None si LLM indisponible

app.state.results_cache = {              # Résultats du DERNIER /api/predict
    "X_arr":         np.ndarray (N, 14)  # features prêtes pour l'explication
    "df":            pd.DataFrame        # données enrichies
    "transactions":  list[dict]          # résultats de prédiction
}
```

### Flux de données entre endpoints

```
POST /api/predict
    → calcule X_arr, df_enriched, transactions
    → stocke dans app.state.results_cache

GET  /api/explain/{tx_id}
    → lit results_cache["X_arr"][position]
    → appelle SHAP, AE Proxy, LIME, LLM

POST /api/explain/batch
    → lit results_cache en boucle
    → appelle SHAP, AE Proxy, LLM (pas LIME)

POST /api/report
    → lit results_cache OU le body JSON
    → génère PDF en mémoire

POST /api/report/docx
    → lit results_cache OU le body JSON
    → génère DOCX en mémoire
```

> **Limitation connue** : le cache est en mémoire — il est **écrasé** à chaque nouveau `/api/predict`. En production multi-utilisateurs, il faudrait Redis ou une base de données. En l'état, l'application est mono-utilisateur (un seul jeu de résultats à la fois).

### Patterns architecturaux

| Pattern | Description |
|---------|-------------|
| **Lifespan** | Les modèles ML sont chargés une fois au démarrage via `@asynccontextmanager lifespan()`, pas à chaque requête |
| **Singleton services** | `ColumnMapper`, `DatasetProfiler`, `DynamicFeatureBuilder` — instances créées une seule fois au niveau module (`_mapper = ColumnMapper()`) |
| **Dégradation gracieuse** | Colonne manquante → 0.0 + warning ; LLM hors ligne → règles métier ; AE impossible → IsoForest seul |
| **Streaming responses** | Les PDF et DOCX sont streamés sans jamais écrire sur le disque du serveur |
| **Fusion de rapports** | `autoencoder_report.json` est injecté dans `baseline_report["models"]` au démarrage — les routes `/api/models` voient un dict unifié |
| **Cache de session** | `app.state.results_cache` permet le chaînage predict → explain sans recalcul des features |

### CORS (Cross-Origin Resource Sharing)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Pourquoi ?** Les navigateurs refusent par sécurité les requêtes entre deux origines différentes (frontend sur port 3000 → backend sur port 8000). Le middleware CORS dit explicitement "c'est autorisé".

### Authentification

**Aucune authentification n'est implémentée** — toutes les routes sont publiques. Pour une mise en production réelle, il faudrait ajouter :
- Un middleware JWT (JSON Web Tokens)
- Ou une validation de clé API dans les headers

---

## 16. Détail des artefacts ML produits par les notebooks

### `features.json` — La liste canonique des features

```json
{
  "all_features": ["step", "hour", "day", "week", "high_risk_hour",
                   "is_transfer_or_cashout", "balance_diff_orig",
                   "dest_zero_balance", "type_CASH_IN", "type_CASH_OUT",
                   "type_DEBIT", "type_PAYMENT", "type_TRANSFER", "log_amount"],
  "scale_cols": ["step", "hour", "day", "week", "log_amount", "balance_diff_orig"],
  "binary_cols": ["high_risk_hour", "is_transfer_or_cashout", "dest_zero_balance"],
  "type_cols": ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"],
  "n_features": 14,
  "high_risk_hours": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 23],
  "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
  "imbalance_ratio": 772
}
```

**`imbalance_ratio: 772`** : il y a 772 transactions légitimes pour 1 fraude dans les données d'entraînement — justifie l'utilisation de SMOTE pour XGBoost et l'entraînement non supervisé pour l'AE.

### `optimal_thresholds.json`

```json
{
  "thresholds": {
    "XGB_smote":    0.355,
    "RF_smote":     0.629,
    "RF_balanced":  0.602,
    "LR_balanced":  0.999,
    "LR_smote":     0.991,
    "IsoForest":    -0.01
  }
}
```

Ces seuils ont été calculés sur le **validation set** (jamais sur le test set) pour maximiser le F1-score. C'est un point important pour un expert : l'utilisation correcte des trois splits (train/val/test).

### `autoencoder_meta.pkl` — Métadonnées complètes

```python
{
    "params": {
        "encoder_dims": [10, 7],
        "bottleneck_dim": 4,
        "decoder_dims": [7, 10],
        "dropout_rate": 0.2,
        "use_batch_norm": True,
        "l2_reg": 1e-5,
        "epochs": 100,
        "patience": 10,
        "learning_rate": 1e-3,
    },
    "threshold": 1.753,
    "train_time": 191.3,        # secondes
    "n_features": 14,
    "train_mse_stats": {
        "mean": 0.312,
        "p95": 1.753,            # ← c'est le seuil initial
        "p99": 3.218,
    },
    "weights_sha256": "abc123...",  # intégrité du fichier .pt
    "training_seed": 42,
}
```

---

## 17. Modifications récentes et corrections

### Correction — AutoEncoder affichait des métriques à 0 dans `/api/models`

**Problème** : `baseline_report.json` (généré par NB03) ne contenait pas d'entrée "AutoEncoder" car ce modèle a été entraîné dans NB05 séparément. La route `/api/models` cherchait "AutoEncoder" dans `baseline_report["models"]`, ne le trouvait pas, et retournait des métriques nulles.

**Correction** dans `app/services/predictor.py` — fusion au chargement :
```python
ae_names = {m.get("name") for m in baseline_report.get("models", [])}
if "AutoEncoder" not in ae_names:
    baseline_report["models"].append({
        "name": "AutoEncoder",
        "optimal_threshold": ae_report["threshold"]["optimal"],
        "train_time_s":      ae_report["training"]["train_time_s"],
        "test_metrics":      ae_report["test_metrics"],
    })
```

### Ajout — champ `is_in_production` dans `/api/models`

**Motivation** : distinguer "meilleur modèle en métriques" de "modèle utilisé en production". L'AutoEncoder est moins performant standalone mais essentiel dans l'ensemble.

```python
# app/routes/models.py
BEST_MODEL = "XGB_smote"
PRODUCTION_MODELS = {"XGB_smote", "AutoEncoder"}

"is_best":          name == BEST_MODEL,
"is_in_production": name in PRODUCTION_MODELS,
```

---

## 18. Questions fréquentes d'un expert technique

**Q : Pourquoi le seuil XGBoost est 0.355 et pas 0.5 (valeur par défaut) ?**
R : Le seuil par défaut de 0.5 est optimisé pour l'accuracy, pas pour le recall. Dans la détection de fraude, un faux négatif (fraude non détectée) est beaucoup plus coûteux qu'un faux positif (transaction légitime bloquée). Le seuil 0.355 maximise le F1-score sur le validation set, ce qui donne un meilleur compromis recall/précision.

**Q : Pourquoi SMOTE pour XGBoost et entraînement non supervisé pour l'AE ?**
R : Le ratio fraude/légitime est de 1/772. XGBoost étant supervisé, il a besoin d'exemples de fraudes. SMOTE génère des exemples synthétiques de fraudes (oversampling). L'AutoEncoder étant non supervisé, il n'a pas besoin de labels — il apprend uniquement sur les transactions légitimes et détecte les fraudes par leur "étrangeté".

**Q : Comment le backend évite-t-il le data snooping (contamination du test set) ?**
R : Les seuils optimaux sont calculés sur le validation set dans les notebooks. Le test set n'est utilisé qu'une seule fois pour l'évaluation finale. Le backend charge des seuils déjà calculés — il ne fait aucune optimisation.

**Q : Que se passe-t-il si PyTorch n'est pas disponible ?**
R : L'import PyTorch est lazily evaluated (dans `load_all_models`, pas à l'import du module). Si PyTorch manque, le serveur démarre mais le chargement de l'AutoEncoder échoue et lève une erreur claire au démarrage.

**Q : Le cache mémoire est-il thread-safe ?**
R : `app.state.results_cache` est un dict Python simple. En mode `--workers 1` (développement), il n'y a aucun risque. En mode multi-workers (`--workers 4`), chaque worker a son propre processus Python avec son propre cache — le frontend doit toujours appeler predict et explain sur le même worker. Pour un déploiement production multi-utilisateurs, Redis est la solution.

**Q : L'AE est-il ré-entraîné à chaque prédiction ?**
R : Non. L'AutoEncoder est chargé une fois au démarrage depuis `autoencoder_weights.pt` et utilisé en inférence uniquement (mode `eval()`, `torch.no_grad()`). Seul l'IsoForest en mode générique est fitté on-the-fly sur chaque batch.

**Q : Quelle est la sécurité du système LLM ?**
R : Les clés API sont dans `.env`, jamais dans le code. Chaque réponse LLM est hashée (SHA-256 + timestamp UTC) pour garantir l'intégrité et la traçabilité dans un contexte d'audit. Les appels ont un timeout de 30 secondes et 3 tentatives avec backoff exponentiel.
