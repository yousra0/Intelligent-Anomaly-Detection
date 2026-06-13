# Etat du projet et plan de suite

Ce fichier sert de repere rapide pour savoir ce qui est deja termine dans le projet et ce qu'il reste a faire avant le livrable final.

## 1. Ce qui est deja fait

### Donnees et preparation

- Le jeu de donnees client est deja charge, nettoye et structure.
- Les splits `train`, `validation` et `test` existent deja en version CSV et NPY dans `data/processed/`.
- Les features finales sont deja produites et sauvegardees dans les artefacts du projet.

### Modeles

- Les modeles baseline sont deja entraines et sauvegardes.
- Le modele AutoEncoder est deja disponible et ses scores sont exportes.
- Les seuils optimaux et les metriques de reference sont deja presentes dans `outputs/models/` et `outputs/reports/`.

### Explicabilite

- SHAP est deja calcule pour RF, LR et AutoEncoder.
- LIME est deja calcule pour RF et AutoEncoder.
- Les rapports `shap_report.json` et `lime_report.json` existent deja.
- Les figures d'explicabilite 26 a 35 sont deja produites ou pretes a etre regenerees apres correction.

### Integration LLM

- La couche d'integration LLM existe deja dans `src/llm_integration/`.
- Le notebook NB05 produit deja des explications structurees lorsque la configuration est valide.

## 2. Ce qui reste a corriger maintenant

### NB06 - SHAP + LIME

1. Decommenter la cellule des seuils optimaux pour afficher toutes les valeurs.
2. Corriger la reference du modele de base pour utiliser dynamiquement le meilleur modele issu de `baseline_report.json`.
3. Ajouter une note explicite sur l'echelle SHAP de `LinearExplainer` versus `TreeExplainer`.
4. Relancer NB06 de bout en bout pour verifier qu'il n'y a plus d'erreur.

### NB05 - Integration LLM

1. Corriger `src/llm_integration/llm_helper.py` pour supprimer toute execution au niveau module qui depend de `cfg`.
2. Remplacer le modele Groq trop lourd dans `config/llm_config.yaml` par `llama3-8b-8192`.
3. Nettoyer les anciens messages "Ollama" dans les impressions du notebook.
4. Relancer NB05 de bout en bout pour verifier que les appels Groq aboutissent correctement.

## 3. Ordre de travail recommande

1. Stabiliser NB05 et la configuration LLM.
2. Reexecuter NB05 jusqu'a obtenir de vraies explications LLM.
3. Appliquer les trois corrections NB06.
4. Reexecuter NB06 de bout en bout.
5. Passer ensuite a la plateforme FastAPI.

## 4. Cible finale a construire

Le livrable final vise une application FastAPI avec interface HTML, route de prediction, route d'explication, comparaison des modeles et generation d'un rapport PDF.

### Structure attendue

```text
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

### Fonctionnalites ciblees

- `POST /api/predict` pour charger un CSV et retourner les transactions suspectes.
- `GET /api/explain/{tx_id}` pour renvoyer SHAP, LIME et l'explication LLM.
- `GET /api/models` pour comparer les modeles.
- `GET /api/report` pour generer le PDF final.
- `GET /` pour servir l'interface web.

## 5. Points de vigilance

- Certains fichiers NPY de `data/processed/` peuvent necessiter `allow_pickle=True` au chargement.
- Apres modification d'un module Python, il faut recharger les imports ou relancer le kernel.
- Sur Windows, garder les prints console en ASCII limite les problemes d'encodage.
- L'ordre de `run_all.py` doit etre verifie avant une execution globale, surtout pour NB05 et NB06.

## 6. Resume court

Le coeur data science est deja en place. Les deux blocs encore a finaliser sont la stabilisation de NB05/NB06 et la transformation du pipeline en application FastAPI.