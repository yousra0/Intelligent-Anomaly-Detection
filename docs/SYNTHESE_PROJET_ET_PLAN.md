# Synthese du projet et plan de suite

## 1. Contexte du projet

**Projet** : Systeme intelligent hybride base sur Deep Learning et LLM pour la detection et l'explication automatique de fraudes financieres.

**Dataset** : donnees transactionnelles client PwC Tunisie - volumetrie complete d'origine.

**Repertoire** : `C:\Users\lenovo\Desktop\anomaly_detection_project`

**Stack** : Python 3.10, scikit-learn, imbalanced-learn, PyTorch (AutoEncoder), SHAP, LIME, FastAPI, Groq API (LLM cloud), Jupyter, VS Code.

## 2. Objectif

Construire un systeme capable de :

1. detecter les fraudes avec des modeles ML/DL ;
2. expliquer automatiquement pourquoi une transaction est suspecte via SHAP, LIME et LLM ;
3. exposer le tout dans une application web FastAPI.

## 3. Etat d'avancement

### NB01 - Data Understanding : termine

1. Sample stratifie : 200 000 lignes, 258 fraudes (0.129%).
2. Ratio de desequilibre : 1:774.
3. Fraudes principalement sur les types de transactions les plus sensibles du jeu client.
4. Feature importante : `balance_diff_orig` (corr ~= 0.37).
5. Heures a risque : `[0-9, 23]` (environ 10.6x plus de fraudes).

### NB02 - Data Preparation : termine

1. 14 features finales.
2. Split 70/15/15 stratifie.
3. `StandardScaler` sans fuite de donnees.
4. SMOTE applique uniquement sur le train, avec ratio 1:10.
5. Artefacts sauvegardes : `scaler.pkl`, `features.json`, `class_weights.json`, datasets `.npy` et `.csv`.

### NB03 - Baseline Models : termine

| Modele | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: |
| XGB_smote | 0.846 | 0.835 | 0.868 |
| RF_smote | 0.795 | 0.805 | 0.841 |
| RF_balanced | 0.769 | 0.800 | 0.779 |

Modele actuel le plus performant : `XGB_smote`.

### NB04 - AutoEncoder PyTorch : termine

1. Architecture : 14 -> 10 -> 7 -> 4 -> 7 -> 10 -> 14.
2. Entraine uniquement sur `X_train_normal` sans fraudes.
3. Recall environ 0.359, F1 environ 0.475, ROC-AUC environ 0.954.
4. GPU MX450 utilise.
5. Ratio erreur de reconstruction fraude / legitime environ 157x.

### NB05 - LLM Integration : partiellement casse

Probleme principal : `src/llm_integration/llm_helper.py` utilise la variable `cfg` au niveau module, ce qui provoque un `NameError` a l'import.

Consequence : les appels Groq echouent et le notebook bascule sur un fallback rule-based.

### NB06 - SHAP + LIME : presque termine

1. SHAP RF_smote : `balance_diff_orig` est la feature numero 1.
2. LIME confirme localement l'importance de `balance_diff_orig` et `dest_zero_balance`.
3. Quelques corrections mineures restent a faire : references `RF_smote` obsoletes, seuils commentes, notes explicatives.

## 4. Corrections a faire en priorite

### 4.1 Corriger `src/llm_integration/llm_helper.py`

1. Identifier toute utilisation de `cfg` en dehors d'une fonction ou classe.
2. Deplacer cette logique dans `LLMHelper.__init__()` ou dans une methode dediee.
3. Verifier que l'import fonctionne sans erreur.

### 4.2 Mettre a jour le modele Groq

Dans `config/llm_config.yaml`, remplacer `llama-3.3-70b-versatile` par `llama3-8b-8192`.

### 4.3 Corriger NB06

1. Decommenter le chargement de `optimal_thresholds.json`.
2. Remplacer les references hardcodees a `RF_smote` par une lecture dynamique du meilleur modele depuis `baseline_report.json`.
3. Ajouter un commentaire indiquant que les valeurs SHAP du modele lineaire sont en log-odds, donc a une echelle differente.

## 5. Prochaine etape : application FastAPI

Construire un dossier `app/` avec les fichiers suivants :

| Fichier | Role |
| --- | --- |
| `app/main.py` | Entree FastAPI, chargement des modeles, routes, CORS |
| `app/services/predictor.py` | Pretraitement + prediction XGBoost + AutoEncoder |
| `app/routes/predict.py` | `POST /api/predict` (upload CSV -> resultats) |
| `app/routes/explain.py` | `GET /api/explain/{tx_id}` (SHAP + LIME + LLM) |
| `app/routes/report.py` | Generation du rapport PDF |
| `app/routes/models_compare.py` | Comparaison des modeles |
| `app/static/index.html` | Interface utilisateur simple |

### Workflow attendu

1. Upload CSV.
2. Prediction des fraudes.
3. Affichage des scores XGB et AutoEncoder.
4. Explication SHAP / LIME.
5. Resume en langage naturel via Groq.
6. Export PDF pour audit.

## 6. Structure du projet deja disponible

Les modeles et artefacts deja generes se trouvent principalement dans `outputs/models/` et `outputs/reports/`.

Le coeur data science est deja solide et couvre :

- l'EDA ;
- la preparation des donnees ;
- les modeles baseline ;
- l'AutoEncoder ;
- SHAP / LIME ;
- les premiers rapports d'explication.

## 7. Resume court pour la suite

Le projet est deja avance sur toute la partie detection et explicabilite. Il reste principalement :

1. corriger le helper LLM ;
2. corriger et relancer NB06 ;
3. construire l'application FastAPI finale.

## 8. Plateforme PwC - Application FastAPI

Backend FastAPI + Interface HTML + Rapport PDF. Livrable final estime a environ 2 jours.

### 8.1 Prerequis - avant de commencer

1. Verifier que NB05 et NB06 tournent sans erreur.
2. Installer les dependances FastAPI :

```bash
pip install fastapi uvicorn python-multipart fpdf2 plotly aiofiles
```

3. Creer la structure de dossiers :

```text
anomaly_detection_project/
└── app/
		├── main.py
		├── routes/
		│   ├── predict.py
		│   ├── explain.py
		│   ├── report.py
		│   └── models_compare.py
		├── services/
		│   ├── predictor.py
		│   ├── explainer.py
		│   ├── report_gen.py
		│   └── llm_service.py
		└── static/
				├── index.html
				└── app.js
```

### 8.2 app/services/predictor.py - charge les modeles et predit

Ce fichier charge tous les modeles une seule fois au demarrage, pas a chaque requete.

Fonctions attendues :

1. load_models() :
	 - charge scaler.pkl, features.json, optimal_thresholds.json ;
	 - charge xgb_smote.pkl comme modele principal ;
	 - charge FraudAutoEncoder depuis outputs/models/autoencoder/ ;
	 - charge ae_scores_test.npy si besoin pour les statistiques ;
	 - retourne un dictionnaire avec scaler, xgb, ae, features, thresholds.

2. predict_dataframe(df, models) :
	 - applique le preprocessing base sur les features et le scaler ;
	 - predit avec xgb_smote via predict_proba ;
	 - calcule le score AutoEncoder via ae.predict_score ;
	 - retourne les colonnes tx_id, xgb_score, ae_score, risk_level, is_fraud_pred.

Regle de calcul du risk_level :

- xgb_score >= 0.80 - CRITIQUE ;
- 0.50 <= xgb_score < 0.80 - ELEVE ;
- xgb_score < 0.50 - FAIBLE.

### 8.3 app/routes/predict.py - POST /api/predict

Le endpoint recoit un fichier CSV avec les colonnes du client et retourne un JSON avec les fraudes detectees.

Format attendu :

```json
{
	"n_transactions": 50000,
	"n_fraud": 847,
	"fraud_rate_pct": 1.69,
	"amount_at_risk": 12400000,
	"transactions": [
		{
			"tx_id": "T_001234",
			"type": "TRANSFER",
			"amount": 458320,
			"xgb_score": 0.94,
			"ae_score": 45.2,
			"risk_level": "CRITIQUE"
		}
	]
}
```

### 8.4 app/routes/explain.py - GET /api/explain/{tx_id}

Pour une transaction donnee, calculer SHAP, LIME et appeler Groq.

Format attendu :

```json
{
	"tx_id": "T_001234",
	"xgb_score": 0.94,
	"ae_score": 45.2,
	"shap_values": {
		"balance_diff_orig": 0.42,
		"type_TRANSFER": 0.31
	},
	"lime_rules": [
		"balance_diff_orig > 0.20 : +0.172"
	],
	"llm_explanation": {
		"risk_level": "CRITIQUE",
		"resume": "Transaction suspecte...",
		"raisons": ["...", "...", "..."],
		"actions_recommandees": ["...", "..."]
	}
}
```

### 8.5 app/static/index.html - interface 3 pages

Page 1 - Detection :

- bouton upload CSV et appel POST /api/predict ;
- affichage du nombre de fraudes detectees, du montant total a risque et du tableau trie par score avec couleurs ;
- clic sur une ligne pour aller a la page 2.

Page 2 - Detail transaction :

- score gauge de 0 a 1 ;
- graphe SHAP waterfall interactif avec Plotly ;
- regles LIME en texte ;
- bulle d'explication LLM avec resume Groq ;
- bouton de generation PDF.

Page 3 - Comparaison modeles :

- tableau des 7 modeles avec Recall, F1 et PR-AUC ;
- courbe ROC interactive Plotly ;
- courbe Precision-Recall ;
- histogramme des scores.

### 8.6 app/services/report_gen.py - rapport PDF PwC

Le rapport doit etre genere avec fpdf2 et suivre cette structure :

- Page 1 : page de garde PwC ;
- Page 2 : synthese globale ;
- Page 3 : top 10 transactions prioritaires ;
- Pages suivantes : fiches detaillees pour les fraudes critiques ;
- derniere page : graphiques, note methodologique et pied de page confidentiel.

### 8.7 app/main.py - point d'entree FastAPI

Le point d'entree doit :

1. appeler load_models() au demarrage via lifespan ;
2. stocker les modeles dans app.state ;
3. monter les routes predict, explain, report et models_compare ;
4. servir les fichiers statiques depuis app/static ;
5. permettre le lancement avec uvicorn app.main:app --reload --port 8000.

### 8.8 Test final attendu

- http://localhost:8000 ouvre l'interface HTML ;
- l'upload CSV affiche le tableau de fraudes en moins de 5 secondes ;
- le clic sur une transaction affiche SHAP waterfall et explication Groq ;
- le bouton PDF telecharge le rapport PwC.

## 9. Note operationnelle pour la passation

Le document sert de base pour reprendre le projet rapidement sans reexplorer tout le contexte. Il resume l'etat reel, les corrections bloqueantes et la cible de livrable final.