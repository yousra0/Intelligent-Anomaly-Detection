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
|- .gitignore
|- README.md
|- requirements.txt
|- structure projet.docx
|- config/
|  |- config.yaml
|  |- ollama_config.yaml
|- data/
|  |- anomalies/
|  |- labels/
|  |- processed/
|  |  |- X_test.csv
|  |  |- X_test.npy
|  |  |- X_train.csv
|  |  |- X_train.npy
|  |  |- X_train_normal.csv
|  |  |- X_train_normal.npy
|  |  |- X_train_smote.csv
|  |  |- X_train_smote.npy
|  |  |- X_val.csv
|  |  |- X_val.npy
|  |  |- y_test.csv
|  |  |- y_test.npy
|  |  |- y_train.csv
|  |  |- y_train.npy
|  |  |- y_train_normal.csv
|  |  |- y_train_normal.npy
|  |  |- y_train_smote.csv
|  |  |- y_train_smote.npy
|  |  |- y_val.csv
|  |  |- y_val.npy
|  |- raw/
|  |  |- dataset_orig.csv
|- notebooks/
|  |- 01_data_understanding.ipynb
|  |- 02_data_preparation.ipynb
|- outputs/
|  |- anomalies_report.csv
|  |- explanations.json
|  |- figures/
|  |  |- 01_class_imbalance.png
|  |  |- 02_transaction_types.png
|  |  |- 03_amount_distribution.png
|  |  |- 04_fraud_by_amount_quantile.png
|  |  |- 05_temporal_analysis.png
|  |  |- 06_feature_correlations.png
|  |  |- 07_leakage_heatmap.png
|  |  |- 08_log_transform.png
|  |  |- 09_split.png
|  |  |- 10_scaling.png
|  |  |- 11_smote.png
|  |- logs/
|  |- models/
|  |  |- class_weights.json
|  |  |- features.json
|  |  |- scaler.pkl
|  |- reports/
|  |  |- eda_report.json
|  |  |- eda_summary.csv
|  |  |- eda_summary_stats.csv
|  |  |- feature_correlations.csv
|  |  |- prep_report.json
|- src/
|  |- main.py
|  |- feature_engineering/
|  |  |- behavioral_features.py
|  |  |- feature_builder.py
|  |  |- temporal_features.py
|  |  |- __init__.py
|  |- models/
|  |  |- autoencoder.py
|  |  |- ml_models.py
|  |  |- __init__.py
|  |- ollama_integration/
|  |  |- ollama_helper.py
|  |  |- __init__.py
|  |- pipeline/
|  |  |- inference_pipeline.py
|  |  |- training_pipeline.py
|  |  |- __init__.py
|  |- preprocessing/
|  |  |- data_loader.py
|  |  |- preprocessing.py
|  |  |- __init__.py
|  |- utils/
|  |  |- anomaly_utils.py
|  |  |- evaluator.py
|  |  |- __init__.py
|  |- visualization/
|  |  |- prep_plots.py
|  |  |- visualization.py
|  |  |- __init__.py
|- tests/
|  |- test_models.py
|  |- test_ollama.py
|  |- test_preprocessing.py
|  |- test_utils.py
|  |- test_visualization.py
|- .venv/   # environnement virtuel local (exclu du versioning)
|- env/     # ancien environnement local
```

## Role de chaque couche

- `notebooks/`: analyse interactive, verification visuelle et experimentation.
- `src/`: logique metier reutilisable et testable.
- `tests/`: validation automatique des modules critiques.
- `data/`: separation claire entre donnees brutes et donnees transformees.
- `outputs/`: tous les resultats produits par les pipelines.
- `config/`: centralisation des parametres de run.
