# Scénario Complet d'Utilisation — Plateforme de Détection d'Anomalies PwC

Ce document décrit un scénario de bout en bout, depuis la phase de recherche data science (notebooks) jusqu'au déploiement et à l'utilisation quotidienne de la plateforme (backend + frontend).

---

## Contexte de la Mission

**Client :** Groupe Retail SA — grande chaîne de distribution  
**Mission PwC :** Audit des transactions financières sur l'exercice 2024 (12 millions de transactions)  
**Objectif :** Identifier les transactions suspectes et produire un rapport d'audit structuré pour les commissaires aux comptes  
**Équipe :**
- **Sara** — Manager PwC, supervise la mission, valide les anomalies
- **Karim** — Auditeur junior, exécute les analyses, interprète les résultats
- **Leila** — Partenaire PwC, lit les rapports exécutifs
- **Admin** — Responsable informatique PwC, gère les utilisateurs et le suivi des modèles

---

## PHASE 1 — Notebooks : Entraînement et Validation des Modèles

> Cette phase est réalisée **une seule fois** par l'équipe data science avant le déploiement de la plateforme. Elle produit les modèles ML utilisés par le backend.

### Étape 1.1 — Compréhension Métier (`01_business_understanding.ipynb`)

Le data scientist ouvre le premier notebook pour cadrer le problème :

```bash
cd c:\Users\lenovo\Desktop\anomaly_detection_project
jupyter notebook notebooks/01_business_understanding.ipynb
```

**Ce qui se passe dans le notebook :**
- Définition des typologies de fraude ciblées (transactions anormalement élevées, virements vers comptes inhabituels, patterns temporels suspects)
- Identification des features métier pertinentes : `amount`, `type`, `oldbalanceOrg`, `newbalanceOrig`
- Définition des seuils de risque : `HIGH` (score > 0.7), `MEDIUM` (0.4–0.7), `LOW` (< 0.4)
- Calcul des coûts métier : faux négatif = 10× plus coûteux qu'un faux positif

**Sortie :** Cahier des charges technique pour les notebooks suivants

---

### Étape 1.2 — Exploration des Données (`02_data_understanding.ipynb`)

```bash
jupyter notebook notebooks/02_data_understanding.ipynb
```

**Ce qui se passe :**
- Chargement du dataset PaySim (`data/raw/`) — 6,3 millions de transactions simulées
- Analyse de la distribution des montants (log-normale, queue épaisse)
- Détection du déséquilibre de classes : **0,13% de fraudes** sur l'ensemble du dataset
- Matrice de corrélation : `amount` et `oldbalanceOrg` sont les features les plus discriminantes
- Visualisations exportées dans `outputs/figures/` (histogrammes, heatmap, boxplots par type de transaction)

**Sortie :** Rapport d'exploration, `outputs/figures/class_imbalance.png`, `outputs/figures/correlations.png`

---

### Étape 1.3 — Préparation des Données (`03_data_preparation.ipynb`)

```bash
jupyter notebook notebooks/03_data_preparation.ipynb
```

**Ce qui se passe :**
- Nettoyage : suppression des doublons, gestion des valeurs manquantes (imputation médiane sur `oldbalanceDest`)
- Feature engineering : création de `balance_diff_orig`, `balance_diff_dest`, `is_high_risk_hour`, `amount_log`
- Normalisation : `StandardScaler` entraîné sur le train set → sauvegardé dans `outputs/models/scaler.pkl`
- Split stratifié : 70% train / 15% validation / 15% test
- SMOTE appliqué sur le train set pour équilibrer les classes (ratio 1:10)
- Sauvegarde : `data/processed/X_train_smote.npy`, `X_val.npy`, `X_test.npy`, etc.

**Sortie :** Datasets prêts pour l'entraînement, scaler sérialisé

---

### Étape 1.4 — Entraînement des Modèles Baseline (`04_baseline_models.ipynb`)

```bash
jupyter notebook notebooks/04_baseline_models.ipynb
```

**Ce qui se passe :**
- Entraînement de 7 modèles : Logistic Regression (balanced/SMOTE), Random Forest (balanced/SMOTE), XGBoost (balanced/SMOTE/sans)
- Évaluation sur le test set : F1-score, AUCPR, recall, précision, F2-score
- **Meilleur modèle : XGBoost SMOTE** — F1=0.91, AUCPR=0.97, Recall=0.88
- Calibration des seuils de décision par optimisation du F2-score (favoriser le recall)
- Sauvegarde : `outputs/models/xgb_smote.pkl`, `outputs/models/optimal_thresholds.json`
- Rapport de métriques : `outputs/reports/baseline_report.json`

**Sortie :** 7 modèles sérialisés + rapport de comparaison

---

### Étape 1.5 — AutoEncoder (`05_autoencoder.ipynb`)

```bash
jupyter notebook notebooks/05_autoencoder.ipynb
```

**Ce qui se passe :**
- Entraînement d'un AutoEncoder PyTorch **uniquement sur les transactions légitimes** (apprentissage non-supervisé)
- Architecture : Encoder [14→8→4] + Decoder [4→8→14], activation ReLU, dropout 0.2
- L'AutoEncoder apprend à reconstruire les transactions normales ; les anomalies ont une erreur de reconstruction élevée
- Validation : distribution des erreurs pour légitimes vs fraudes → seuil optimal = percentile 95 des erreurs légitimes
- Sauvegarde : `outputs/models/autoencoder/autoencoder_weights.pt`
- Rapport : `outputs/reports/autoencoder_report.json`

**Sortie :** Modèle AutoEncoder + seuils d'anomalie

---

### Étape 1.6 — Explicabilité (`06_shap_lime.ipynb`)

```bash
jupyter notebook notebooks/06_shap_lime.ipynb
```

**Ce qui se passe :**
- Calcul des SHAP values (TreeExplainer) sur un échantillon de 1 000 transactions de test
- Beeswarm plot global : `amount` et `balance_diff_orig` sont les 2 features les plus importantes
- Explication locale LIME pour 3 transactions suspectes représentatives
- Export des visualisations dans `outputs/figures/shap_summary.png`, `outputs/figures/lime_example.png`

**Sortie :** Visualisations d'explicabilité validées par le data scientist

---

### Étape 1.7 — Intégration LLM (`07_llm_integration.ipynb`)

```bash
jupyter notebook notebooks/07_llm_integration.ipynb
```

**Ce qui se passe :**
- Test de l'intégration avec Groq (modèle `llama-3.3-70b-versatile`) et Gemini
- Prompt engineering : génération d'explications en **français** pour les auditeurs non-techniques
- Exemple de sortie LLM pour une transaction à risque élevé :
  > *"Cette transaction présente un risque élevé car le montant transféré (247 500 €) est 18× supérieur à la moyenne des transactions de ce compte. L'écart entre le solde avant et après l'opération est anormalement élevé, ce qui suggère un possible détournement de fonds."*
- Tests de robustesse : latence, qualité des explications, gestion des timeouts

**Sortie :** Prompts validés, configuration `config/llm_config.yaml`

---

### Étape 1.8 — Exécution Séquentielle de Tous les Notebooks

Pour reproduire l'ensemble du pipeline ML en une seule commande :

```bash
python run_all.py
# Ou pour n'exécuter qu'un subset :
python run_all.py --from 4 --only 5
# Ou pour vérifier l'intégrité sans ré-exécuter :
python run_all.py --check-only
```

**Résultat final :** Le dossier `outputs/models/` contient tous les artefacts ML prêts pour le déploiement :
```
outputs/models/
├── xgb_smote.pkl              ← Modèle principal (XGBoost)
├── autoencoder/
│   └── autoencoder_weights.pt ← AutoEncoder PyTorch
├── iso_forest.pkl             ← IsolationForest (fallback)
├── scaler.pkl                 ← Normalisation des features
├── features.json              ← Métadonnées des features
├── optimal_thresholds.json    ← Seuils de décision par modèle
└── class_weights.json         ← Poids des classes
```

---

## PHASE 2 — Backend : Démarrage et API FastAPI

> L'équipe technique déploie le backend FastAPI qui expose les modèles ML via une API REST.

### Étape 2.1 — Configuration de l'Environnement

```bash
# Création et activation de l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate          # Windows

# Installation des dépendances Python
pip install -r requirements.txt

# Installation du package ml_core en mode développement
pip install -e .

# Configuration des variables d'environnement
copy .env.example .env
# Éditer .env :
# DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/anomaly_db
# SECRET_KEY=votre_cle_secrete_jwt
# GROQ_API_KEY=gsk_xxx...
# GEMINI_API_KEY=AIzaSy...
```

### Étape 2.2 — Initialisation de la Base de Données

```bash
# Appliquer les migrations Alembic
alembic upgrade head

# Vérifier que les tables sont créées
# Tables : users, missions, datasets, analysis_runs, anomaly_reviews,
#          reports, audit_logs, model_versions, prediction_monitoring_logs
```

### Étape 2.3 — Démarrage du Serveur FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Au démarrage, le backend effectue automatiquement :**
1. Application des migrations Alembic en attente
2. Chargement des modèles ML en mémoire :
   - `xgb_smote.pkl` → XGBoost classifier
   - `autoencoder_weights.pt` → AutoEncoder PyTorch
   - `iso_forest.pkl` → IsolationForest (fallback)
   - `scaler.pkl` → StandardScaler
3. Initialisation du client LLM (Groq ou Gemini selon la config)

**Vérification du démarrage :**
```bash
curl http://localhost:8000/api/health
# Réponse attendue :
# {
#   "status": "healthy",
#   "database": "connected",
#   "models": {"xgboost": "loaded", "autoencoder": "loaded"},
#   "llm": "groq:connected"
# }
```

### Étape 2.4 — Appel API : Upload et Prédiction (exemple curl)

```bash
# Authentification
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "karim@pwc.com", "password": "secure123"}'
# Réponse : {"access_token": "eyJ...", "token_type": "bearer"}

# Upload d'un CSV et prédiction
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer eyJ..." \
  -F "file=@data/transactions_q4_2024.csv" \
  -F "mission_id=1"

# Réponse (extrait) :
# {
#   "total_transactions": 45230,
#   "anomalies_detected": 312,
#   "risk_distribution": {"HIGH": 48, "MEDIUM": 127, "LOW": 137},
#   "model_used": "xgboost_autoencoder",
#   "predictions": [
#     {
#       "transaction_id": "TX-00847",
#       "amount": 247500.0,
#       "anomaly_score": 0.94,
#       "risk_level": "HIGH",
#       "flags": ["amount_outlier", "balance_anomaly"]
#     }, ...
#   ]
# }
```

### Étape 2.5 — Appel API : Explication d'une Transaction

```bash
curl -X GET "http://localhost:8000/api/explain/TX-00847" \
  -H "Authorization: Bearer eyJ..."

# Réponse :
# {
#   "transaction_id": "TX-00847",
#   "shap_values": {"amount": 0.42, "balance_diff_orig": 0.31, ...},
#   "lime_explanation": [["amount > 200000", 0.38], ...],
#   "autoencoder_error": 0.087,
#   "llm_explanation": "Cette transaction présente un risque élevé car..."
# }
```

### Étape 2.6 — Appel API : Génération d'un Rapport PDF

```bash
curl -X POST http://localhost:8000/api/report \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{
    "mission_id": 1,
    "analysis_run_id": 42,
    "include_top_n": 10,
    "language": "fr"
  }' \
  --output rapport_audit_q4_2024.pdf
```

---

## PHASE 3 — Frontend : Utilisation Quotidienne par l'Équipe d'Audit

> L'équipe PwC utilise l'interface web pour piloter les analyses et consulter les résultats sans jamais écrire une ligne de code.

### Étape 3.1 — Démarrage du Frontend

```bash
cd frontend
npm install
npm run dev
# Interface disponible sur http://localhost:3000
```

---

### Scénario Utilisateur — Sara (Manager)

#### 3.2 — Connexion

Sara ouvre `http://localhost:3000` dans son navigateur.

Elle voit la **page de login** avec le logo PwC. Elle saisit ses identifiants :
- Email : `sara@pwc.com`
- Mot de passe : `●●●●●●●●`

Après authentification JWT, elle est redirigée vers le **Dashboard** (`/dashboard`).

---

#### 3.3 — Création d'une Mission

Sur le dashboard, Sara voit un bouton **"Nouvelle Mission"**. Elle clique dessus et remplit le formulaire dans la modale `CreateMissionModal` :

| Champ | Valeur |
|-------|--------|
| Nom | Audit Groupe Retail SA — Q4 2024 |
| Client | Groupe Retail SA |
| Description | Analyse des transactions financières Q4 2024 |
| Date de début | 2024-10-01 |
| Date de clôture | 2024-12-31 |

Elle assigne **Karim** (auditeur) à la mission et clique sur **"Créer"**.

La mission apparaît dans la liste `/missions` avec le statut **"En cours"**.

---

### Scénario Utilisateur — Karim (Auditeur Junior)

#### 3.4 — Connexion

Karim se connecte avec `karim@pwc.com`. Il voit uniquement les missions qui lui ont été assignées.

---

#### 3.5 — Accès à la Mission

Karim clique sur la mission **"Audit Groupe Retail SA — Q4 2024"**. Il arrive sur la page de détail de la mission (`/missions/1`).

Il voit le composant `DatasetSection` avec un bouton **"Ajouter un dataset"**.

---

#### 3.6 — Upload du Fichier CSV

Karim glisse-dépose le fichier `transactions_q4_2024.csv` (850 MB) dans le composant `UploadDropzone`.

Une barre de progression s'affiche. Le fichier est uploadé vers le backend via l'endpoint `/api/predict`. Le backend :
1. Charge le CSV en mémoire par chunks
2. Détecte le schéma des colonnes (`schema_detector.py`) — colonnes `step`, `type`, `amount`, `nameOrig`, `nameDest`, `oldbalanceOrg`, `newbalanceOrig` détectées → mode **"standard"**
3. Mappe les colonnes vers les 14 features canoniques (`column_mapper.py`)
4. Profile le dataset (`dataset_profiler.py`) — résultat affiché dans l'interface

Karim voit le **résumé de profiling** :

```
Dataset: transactions_q4_2024.csv
Lignes: 45 230 | Colonnes: 9
Qualité: 96.3% (score global)
Valeurs manquantes: 0.2% (colonne "nameDest")
Recommandations: "2 colonnes redondantes détectées (nameOrig/nameDest)"
```

---

#### 3.7 — Lancement de l'Analyse

Karim clique sur **"Lancer l'Analyse"** dans le composant `AnalysisWizard` (étape 3/3).

L'interface affiche un indicateur de chargement. En arrière-plan, le backend exécute :
1. Ingénierie des features (`feature_builder.py`, `feature_engineer.py`)
2. Prédiction XGBoost sur les 45 230 transactions
3. Calcul des erreurs de reconstruction AutoEncoder
4. Combinaison des deux scores (ensemble)
5. Classement des anomalies par score décroissant

Après ~12 secondes, l'interface redirige automatiquement vers la page de résultats.

---

#### 3.8 — Consultation des Résultats (`/missions/1/analysis`)

Karim arrive sur la page `ResultsDashboard` avec les **KPI Cards** (`KPICards`) :

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ 45 230          │  │ 312             │  │ 0,69%           │  │ 0.87            │
│ Transactions    │  │ Anomalies       │  │ Taux d'anomalie │  │ Score de risque │
│ analysées       │  │ détectées       │  │                 │  │ moyen (HIGH)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

Il voit ensuite les **graphiques** (`charts/`) :
- `RiskPieChart` : 48 HIGH (15%) | 127 MEDIUM (41%) | 137 LOW (44%)
- `AnomalyBarChart` : distribution par type de transaction (TRANSFER, CASH_OUT dominants)
- `ScoreDistributionChart` : histogramme des scores d'anomalie

Et le tableau `AnomalyTable` des **top 10 transactions suspectes** :

| # | Transaction ID | Montant | Type | Score | Risque | Actions |
|---|---------------|---------|------|-------|--------|---------|
| 1 | TX-00847 | 247 500 € | TRANSFER | 0.94 | 🔴 HIGH | [Expliquer] |
| 2 | TX-02301 | 189 000 € | CASH_OUT | 0.91 | 🔴 HIGH | [Expliquer] |
| 3 | TX-00194 | 312 000 € | TRANSFER | 0.89 | 🔴 HIGH | [Expliquer] |
| ... | ... | ... | ... | ... | ... | |

---

#### 3.9 — Explication Détaillée d'une Transaction

Karim clique sur **"Expliquer"** pour `TX-00847`. Le composant `ExplanationCard` s'affiche :

**SHAP Values (importance des features) :**
```
amount             ████████████████████ +0.42  (montant 18× supérieur à la moyenne)
balance_diff_orig  ████████████████     +0.31  (écart de solde anormal)
type_TRANSFER      ██████               +0.12  (type à risque élevé)
is_high_risk_hour  ████                 +0.08  (transaction à 2h47 du matin)
...
```

**Erreur AutoEncoder :** `0.087` (seuil = 0.042 → anomalie confirmée)

**Explication LLM (en français) :**
> *"Cette transaction présente un risque élevé pour les raisons suivantes : (1) Le montant de 247 500 € est 18 fois supérieur à la moyenne des transactions de ce compte sur les 30 derniers jours. (2) Le solde du compte émetteur est passé de 251 340 € à 3 840 € après l'opération, ce qui représente un vidage quasi-total du compte. (3) La transaction a été effectuée à 2h47, en dehors des heures d'activité habituelles. Ces éléments combinés suggèrent un possible détournement ou une compromission du compte."*

---

#### 3.10 — Génération du Rapport PDF

Karim clique sur l'onglet **"Rapport"** → composant `ReportSection`.

Il sélectionne :
- **Format :** PDF (FPDF2)
- **Langue :** Français
- **Inclure :** Top 10 transactions + graphiques + recommandations

Il clique sur **"Générer le Rapport"**. Après ~5 secondes, un bouton **"Télécharger PDF"** apparaît.

Le rapport généré (`rapport_audit_groupe_retail_q4_2024.pdf`) contient **7 pages** :
1. **Page de couverture** — Logo PwC, nom de la mission, date, niveau de confidentialité
2. **Résumé exécutif** — KPIs clés, recommandation principale
3. **Contexte et périmètre** — Description du dataset, période d'analyse
4. **Résultats de l'analyse** — Distribution des risques, graphiques
5. **Top 10 transactions suspectes** — Tableau détaillé avec scores et flags
6. **Fiches détaillées** — 3 transactions HIGH à investiguer en priorité
7. **Recommandations** — Actions correctives suggérées par le LLM

---

### Scénario Utilisateur — Sara (Manager, validation)

#### 3.11 — Validation des Anomalies

Sara se reconnecte et navigue vers la mission. Elle voit les résultats de Karim.

En tant que Manager, elle peut **valider ou rejeter** les anomalies dans le tableau `AnomalyTable` :

- `TX-00847` → **Validé** ✓ (anomalie confirmée, à escalader)
- `TX-02301` → **Rejeté** ✗ (faux positif — virement interne connu)
- `TX-00194` → **En investigation** ⏳

Chaque action génère une entrée dans la **piste d'audit** (`audit_logs`).

---

### Scénario Utilisateur — Leila (Partenaire)

#### 3.12 — Lecture du Rapport Exécutif

Leila se connecte et navigue vers `/reports`. Elle voit la liste de tous les rapports générés.

Elle télécharge `rapport_audit_groupe_retail_q4_2024.pdf` et le parcourt en 5 minutes.

Elle n'a pas accès aux détails techniques (pas de droits sur `/missions/[id]/analysis`), mais peut lire le résumé exécutif et les recommandations.

---

### Scénario Utilisateur — Admin

#### 3.13 — Gestion des Utilisateurs (`/admin/users`)

L'admin se connecte et accède à la page d'administration.

Il voit le tableau de tous les utilisateurs avec leurs rôles. Il :
1. Crée un nouvel auditeur : `nouveau@pwc.com` — rôle `auditor`
2. Modifie le rôle de Karim : `auditor` → `manager` (promotion)

---

#### 3.14 — Monitoring des Modèles

L'admin accède à la section **Model Registry** (via l'API ou un tableau de bord dédié) :

```bash
curl http://localhost:8000/api/registry/monitoring
# Réponse :
# {
#   "recent_runs": 42,
#   "avg_anomaly_rate": 0.0071,
#   "avg_inference_time_ms": 234,
#   "model_version": "xgb_smote_v2.1"
# }

curl http://localhost:8000/api/registry/monitoring/drift
# Réponse :
# {
#   "drift_detected": false,
#   "ks_statistic": 0.041,
#   "p_value": 0.18,
#   "recommendation": "Modèle stable, pas de ré-entraînement requis"
# }
```

---

### Étape 3.15 — Consultation de la Piste d'Audit (`/audit-trail`)

N'importe quel utilisateur connecté peut consulter la piste d'audit complète :

```
┌──────────────────────────────────────────────────────────────────────────┐
│ PISTE D'AUDIT — Mission : Audit Groupe Retail SA Q4 2024                │
├────────────────┬──────────────┬──────────────────────────────────────────┤
│ Horodatage     │ Utilisateur  │ Action                                   │
├────────────────┼──────────────┼──────────────────────────────────────────┤
│ 2024-12-10     │ karim@pwc    │ UPLOAD_DATASET — transactions_q4.csv    │
│ 09:14:23       │              │ (45 230 lignes)                          │
├────────────────┼──────────────┼──────────────────────────────────────────┤
│ 2024-12-10     │ karim@pwc    │ RUN_ANALYSIS — Run #42, 312 anomalies   │
│ 09:14:35       │              │ détectées                                │
├────────────────┼──────────────┼──────────────────────────────────────────┤
│ 2024-12-10     │ karim@pwc    │ GENERATE_REPORT — rapport PDF (7 pages) │
│ 09:27:01       │              │                                          │
├────────────────┼──────────────┼──────────────────────────────────────────┤
│ 2024-12-10     │ sara@pwc     │ VALIDATE_ANOMALY — TX-00847 → CONFIRMÉ  │
│ 11:03:44       │              │                                          │
├────────────────┼──────────────┼──────────────────────────────────────────┤
│ 2024-12-10     │ sara@pwc     │ REJECT_ANOMALY — TX-02301 → FAUX POSITIF│
│ 11:05:12       │              │                                          │
└────────────────┴──────────────┴──────────────────────────────────────────┘
```

---

## Récapitulatif du Flux de Bout en Bout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FLUX COMPLET                                       │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1 — NOTEBOOKS (Data Science)
────────────────────────────────────────────────────────────────────────────
01_business_understanding → 02_data_understanding → 03_data_preparation
       ↓                                                    ↓
04_baseline_models ──────────────────────────── 05_autoencoder
       ↓                                                    ↓
06_shap_lime ──────────────────────────────── 07_llm_integration
       ↓
outputs/models/ (xgb_smote.pkl, autoencoder_weights.pt, iso_forest.pkl, scaler.pkl)

PHASE 2 — BACKEND (FastAPI, port 8000)
────────────────────────────────────────────────────────────────────────────
Démarrage → chargement des modèles en mémoire → écoute sur /api/*

CSV → /api/predict → schema_detector → column_mapper → feature_builder
                                                              ↓
                                                    predictor (XGB + AE)
                                                              ↓
                                        /api/explain/{tx_id} → explainer + LLM
                                                              ↓
                                              /api/report → report_gen (PDF 7p)

PHASE 3 — FRONTEND (Next.js, port 3000)
────────────────────────────────────────────────────────────────────────────
Login → Dashboard → Nouvelle Mission (Sara)
                         ↓
                  Upload CSV → Profiling (Karim)
                         ↓
                  Lancer l'Analyse → Résultats
                         ↓
              Expliquer TX → SHAP + LIME + LLM
                         ↓
              Générer Rapport PDF → Télécharger
                         ↓
              Valider / Rejeter anomalies (Sara)
                         ↓
              Lire rapport exécutif (Leila)
                         ↓
              Audit Trail → traçabilité complète
```

---

## Commandes de Démarrage Rapide

```bash
# 1. ML Core : exécuter tous les notebooks (1 fois)
python run_all.py

# 2. Backend FastAPI
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

# 3. Frontend Next.js (dans un autre terminal)
cd frontend
npm run dev

# 4. Tests
pytest -q                          # Tests Python
cd frontend && npm run type-check  # Vérification TypeScript
```

**URLs :**
- Interface utilisateur : `http://localhost:3000`
- API FastAPI : `http://localhost:8000`
- Documentation API Swagger : `http://localhost:8000/docs`
- Documentation API ReDoc : `http://localhost:8000/redoc`
