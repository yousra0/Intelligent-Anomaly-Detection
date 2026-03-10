# Intelligent Anomaly Detection

Ce projet implemente un pipeline de detection d'anomalies (fraude) autour des etapes suivantes:
exploration des donnees, preparation des features, entrainement/evaluation des modeles,
inference et production d'artefacts (rapports, figures, modeles, fichiers preprocesses).

## Structure du projet

```text
anomaly_detection_project/
|- README.md
|- requirements.txt
|- config/
|  |- config.yaml
|  |- ollama_config.yaml
|- data/
|  |- raw/                # donnees sources brutes (ex: dataset_orig.csv)
|  |- processed/          # jeux préparés pour l'entrainement/inference
|  |- labels/             # étiquettes / metadonnées de labels
|  |- anomalies/          # exports lies aux anomalies detectees
|- notebooks/
|  |- 01_data_understanding.ipynb
|  |- 02_data_preparation.ipynb
|- src/
|  |- main.py             # point d'entree principal
|  |- preprocessing/      # chargement, nettoyage, preparation des donnees
|  |- feature_engineering/# features temporelles et comportementales
|  |- models/             # modeles ML / autoencoder
|  |- pipeline/           # orchestration entrainement / inference
|  |- utils/              # fonctions utilitaires (evaluation, helpers)
|  |- visualization/      # fonctions de visualisation
|  |- ollama_integration/ # integration LLM/ollama pour explications
|- outputs/
|  |- models/             # modeles entraines et artefacts (scaler, metadonnées)
|  |- reports/            # rapports de preparation / EDA / metriques
|  |- figures/            # graphiques exportes
|  |- logs/               # logs d'execution
|  |- anomalies_report.csv
|  |- explanations.json
|- tests/                 # tests unitaires et d'integration
|- .venv/                 # environnement virtuel Python (local)
|- env/                   # ancien environnement local (a ignorer pour le code)
```

## Role de chaque couche

- `notebooks/`: analyse interactive, verification visuelle et experimentation.
- `src/`: logique metier reutilisable et testable.
- `tests/`: validation automatique des modules critiques.
- `data/`: separation claire entre donnees brutes et donnees transformees.
- `outputs/`: tous les resultats produits par les pipelines.
- `config/`: centralisation des parametres de run.
