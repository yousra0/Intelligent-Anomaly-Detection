# Handoff projet - Anomaly Detection

## 1. But du projet

Le projet met en place une chaine complete de detection d'anomalies / fraude sur des transactions d'un client PwC Tunisie.

L'objectif final est double :

1. produire un pipeline data science reproductible pour comparer plusieurs approches de detection ;
2. transformer ce pipeline en livrable metier exploitable par une interface web et un rapport PDF.

## 2. Ce que contient le projet

### Donnees

- `data/raw/dataset_orig.csv` : jeu brut original.
- `data/processed/` : splits train / validation / test, versions normales, SMOTE et fichiers `.npy` / `.csv` deja prepares.
- `data/anomalies/` et `data/labels/` : dossiers de travail pour les sorties metier.

### Code source

- `src/preprocessing/` : nettoyage, suppression des colonnes a risque, encodage des variables, standardisation, split et SMOTE.
- `src/feature_engineering/` : creation des features temporelles et comportementales.
- `src/models/` : baselines supervisees et autoencoder PyTorch.
- `src/explainability/` : SHAP et LIME.
- `src/llm_integration/` : generation d'explications en langage naturel via API cloud.
- `src/utils/` : utilitaires de metrics, anomalies et chargement de rapports.
- `src/visualization/` : graphes de suivi et de comparaison.
- `src/pipeline/` et `src/main.py` : actuellement des stubs / placeholders.

### Notebooks

- `notebooks/01_data_understanding.ipynb` : exploration et diagnostic du jeu de donnees.
- `notebooks/02_data_preparation.ipynb` : feature engineering, split, scaling, SMOTE.
- `notebooks/03_baseline_models.ipynb` : logistic regression, random forest, XGB et seuils optimaux.
- `notebooks/04_autoencoder.ipynb` : autoencoder PyTorch et score d'anomalie.
- `notebooks/05_llm_integration.ipynb` : explications LLM.
- `notebooks/06_shap_lime.ipynb` : SHAP, LIME et dashboard d'explicabilite.

### Sorties deja produites

- `outputs/reports/` : rapports EDA, preparation, baseline, autoencoder, SHAP, LIME et LLM.
- `outputs/models/` : modeles sauvegardes, seuils optimaux, features, scores AE et valeurs SHAP.
- `outputs/figures/` : figures d'analyse et d'explicabilite.

## 3. Ce qui est deja realise

### Pipeline data

- Les features finales sont stables et documentees dans `config/config.yaml`.
- Le pipeline de preparation produit les splits train / val / test et la version normale pour l'autoencoder.
- Le dataset est deja structure pour les modeles supervisees et non supervisees.

### Modeles

- Baselines supervisees : logistic regression, random forest, XGB.
- Modele non supervise : autoencoder PyTorch.
- Les artefacts sont deja sauvegardes dans `outputs/models/`.

### Explicabilite

- SHAP est deja calcule pour RF, LR et AutoEncoder.
- LIME est deja calcule pour RF et AutoEncoder.
- Les rapports `shap_report.json` et `lime_report.json` existent deja.

### LLM

- L'architecture d'integration LLM est en place via `src/llm_integration/llm_helper.py`.
- Le notebook NB05 genere des explications structurees lorsque la configuration et la cle API sont correctes.

## 4. Etat reel du projet

Le projet est deja tres avance sur la partie data science. Le coeur du travail est termine pour :

- la preparation des donnees,
- les modeles baseline,
- l'autoencoder,
- SHAP / LIME,
- les sorties de rapport.

Les vraies zones encore a finaliser sont surtout :

- la cohesion entre NB05 et NB06,
- le nettoyage de la couche LLM,
- la transformation du pipeline en application FastAPI.

## 5. Corrections a faire maintenant dans NB06

### Priorite 1 - Cell C08

La cellule seuils optimaux est encore entierement commentees. Il faut decommenter le bloc pour afficher les seuils de tous les modeles.

Attendu : affichage de `LR_balanced`, `LR_smote`, `RF_balanced`, `RF_smote`, `XGB_smote`, `IsoForest` et `AutoEncoder`.

### Priorite 2 - Cell C07

La cellule doit utiliser le meilleur modele issu de `baseline_report.json` au lieu de supposer que RF est le modele de reference.

Il faut charger dynamiquement le meilleur modele via `best_model_recall` et afficher son recall / F1.

### Priorite 3 - Cell C14

Ajouter une note explicite sur l'echelle SHAP :

- `LinearExplainer` retourne des valeurs en espace log-odds.
- `TreeExplainer` retourne des valeurs en espace probabilite.
- Les magnitudes ne sont donc pas directement comparables.

### Priorite 4 - Reexecuter NB06

Une fois les corrections appliquees, relancer le notebook de bout en bout.

Objectif final attendu :

- 20 / 20 outputs,
- aucune erreur,
- figures 26 a 35 regenerees,
- SHAP RF_smote avec `balance_diff_orig` en premiere position,
- figure finale avec LLM reel si NB05 est corrige, sinon fallback rule-based acceptable.

## 6. Corrections a faire dans NB05 et dans la config LLM

### Bug bloquant 1 - `cfg is not defined`

Dans `src/llm_integration/llm_helper.py`, il ne faut aucun code qui s'exécute au niveau module et qui depend d'une variable `cfg` non definie.

Le fichier doit garder seulement :

- des imports,
- des constantes,
- des fonctions,
- des classes.

La lecture de la config doit rester dans l'initialisation de `LLMHelper`.

### Bug bloquant 2 - `ae_scores_test is not defined`

Ce bug est une cascade du bug precedent. Il disparaît une fois NB05 relance depuis le debut apres correction du helper LLM.

### Configuration a corriger

Dans `config/llm_config.yaml`, remplacer le modele Groq trop lourd par :

- `llama3-8b-8192`

### Nettoyage des impressions

Dans NB05, remplacer les anciens messages "Ollama" par des messages neutres sur le provider LLM.

### Reexecution attendue

Apres correction :

- NB05 tourne sans AssertionError,
- les appels Groq passent,
- les explications sont de vraies sorties LLM,
- NB06 peut afficher l'explication LLM reelle au lieu du fallback.

## 7. Ordre de travail recommande pour Claude Code

1. Corriger `src/llm_integration/llm_helper.py`.
2. Mettre a jour `config/llm_config.yaml`.
3. Nettoyer les cellules NB05 qui affichent encore "Ollama".
4. Reexecuter NB05 de bout en bout.
5. Appliquer les 3 corrections de NB06.
6. Reexecuter NB06 de bout en bout.
7. Passer ensuite a la plateforme FastAPI.

## 8. Ce qu'il faut construire ensuite pour la plateforme PwC

Le livrable final cible est une application FastAPI avec interface web et generation PDF.

### Structure attendue

```
app/
    main.py
    routes/
        predict.py
        explain.py
        report.py
        models_compare.py
    services/
        predictor.py
        explainer.py
        report_gen.py
        llm_service.py
    static/
        index.html
        app.js
```

### Comportement cible

- `POST /api/predict` : upload CSV et renvoie les transactions suspectes.
- `GET /api/explain/{tx_id}` : renvoie SHAP, LIME et explication LLM.
- `GET /api/models` : comparaison des modeles.
- `GET /api/report` : generation du PDF final.
- `GET /` : interface HTML.

### Recommandation d'implementation

- charger les modeles une seule fois au demarrage,
- conserver les artefacts deja produits dans `outputs/models/` et `outputs/reports/`,
- reutiliser les fonctions existantes de `src/` au lieu de re-implementer la logique,
- garder les scores et seuils centralises.

## 9. Points de vigilance importants

- Les fichiers `.npy` de `data/processed/` peuvent demander `allow_pickle=True` au chargement.
- Les notebooks doivent recharger les modules modifies avec `importlib.reload()` ou un restart kernel.
- `run_all.py` contient actuellement un ordre de notebooks a verifier, avec NB05 / NB06 inverses dans la liste. Il faut y faire attention avant toute exécution globale.
- Sur Windows, garder les prints console ASCII si possible pour eviter les problemes d'encodage.
- La configuration de `config/llm_config.yaml` doit rester propre et ne jamais contenir de cle API en dur.

## 10. Resume ultra-court

Le projet est deja construit sur toute la partie ML et explicabilite. Ce qu'il reste vraiment a faire, c'est :

1. corriger NB05 / le helper LLM ;
2. corriger NB06 et le relancer ;
3. convertir le pipeline en application FastAPI de production.
