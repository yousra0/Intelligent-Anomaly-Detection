# Intelligent Anomaly Detection

Ce projet implemente un pipeline de detection de fraude de bout en bout :
exploration, preparation des donnees, entrainement/evaluation de modeles,
inference, puis generation d'artefacts (rapports, figures, modeles, exports).

## Structure du projet

```text
anomaly_detection_project/
|- README.md                               # documentation du projet
|- requirements.txt                        # dependances Python
|- structure projet.docx                   # document de cadrage initial
|- config/
|  |- config.yaml                          # configuration pipeline ML
|  |- ollama_config.yaml                   # configuration integration Ollama
|- data/
|  |- raw/
|  |  |- dataset_orig.csv                  # dataset source brut (PaySim)
|  |- processed/                           # jeux prets pour modelisation
|  |  |- X_train.npy / y_train.npy         # split train principal (numpy)
|  |  |- X_val.npy / y_val.npy             # split validation (numpy)
|  |  |- X_test.npy / y_test.npy           # split test final (numpy)
|  |  |- X_train_normal.npy                # train normal seulement (autoencoder)
|  |  |- y_train_normal.npy                # labels du train normal
|  |  |- X_train_smote.npy / y_train_smote.npy # train reequilibre SMOTE
|  |  |- *.csv                             # versions CSV des memes jeux
|  |- labels/                              # etiquettes/metadonnees annexes
|  |- anomalies/                           # exports d'anomalies detectees
|- notebooks/
|  |- 01_data_understanding.ipynb          # EDA et analyse du desequilibre
|  |- 02_data_preparation.ipynb            # preparation des features/splits
|  |- 03_baseline_models.ipynb             # baselines LR/RF + comparaison
|- outputs/
|  |- anomalies_report.csv                 # resultat d'inference (anomalies)
|  |- explanations.json                    # explications generees (LLM/rules)
|  |- figures/                             # figures exportees (EDA + modeles)
|  |- logs/                                # traces d'execution
|  |- models/
|  |  |- scaler.pkl                        # scaler fite sur train
|  |  |- features.json                     # schema features/target
|  |  |- class_weights.json                # poids de classes pour imbalance
|  |  |- *.pkl                             # modeles entraines sauvegardes
|  |- reports/
|  |  |- eda_report.json                   # synthese EDA
|  |  |- prep_report.json                  # rapport preparation donnees
|  |  |- baseline_report.json              # comparaison des modeles baseline
|  |  |- *.csv                             # tableaux de stats/correlations
|- src/
|  |- main.py                              # point d'entree applicatif
|  |- feature_engineering/
|  |  |- feature_builder.py                # orchestration feature engineering
|  |  |- temporal_features.py              # variables temporelles
|  |  |- behavioral_features.py            # variables comportementales
|  |  |- __init__.py                       # export du package
|  |- preprocessing/
|  |  |- data_loader.py                    # chargement robuste CSV/Excel
|  |  |- preprocessing.py                  # pipeline de preparation complet
|  |  |- __init__.py                       # export du package
|  |- models/
|  |  |- ml_models.py                      # wrappers LogisticRegression/RF
|  |  |- autoencoder.py                    # modele autoencoder pour anomalies
|  |  |- __init__.py                       # export du package
|  |- pipeline/
|  |  |- training_pipeline.py              # orchestration entrainement
|  |  |- inference_pipeline.py             # orchestration prediction/inference
|  |  |- __init__.py                       # export du package
|  |- visualization/
|  |  |- prep_plots.py                     # visualisations de preparation
|  |  |- model_plots.py                    # visualisations baseline modeles
|  |  |- visualization.py                  # fonctions de visualisation simples
|  |  |- __init__.py                       # exports centralises du package
|  |- utils/
|  |  |- evaluator.py                      # metriques, seuils, comparaisons
|  |  |- anomaly_utils.py                  # utilitaires anomalies
|  |  |- __init__.py                       # export du package
|  |- ollama_integration/
|  |  |- ollama_helper.py                  # integration Ollama (explications)
|  |  |- __init__.py                       # export du package
|- tests/
|  |- test_preprocessing.py                # tests pipeline de preparation
|  |- test_models.py                       # tests modeles ML
|  |- test_visualization.py                # tests fonctions de visualisation
|  |- test_utils.py                        # tests utilitaires/metriques
|  |- test_ollama.py                       # tests integration Ollama
|- .venv/                                  # environnement virtuel local actif
|- env/                                    # ancien environnement local
```

## Role des couches

- `config/` : centralise les parametres (pipeline et integration Ollama).
- `data/` : separe les donnees brutes des donnees preparees pour l'entrainement.
- `notebooks/` : experimentation, analyse visuelle, validation incremental.
- `src/` : logique metier reutilisable, versionnable et testable.
- `outputs/` : artefacts produits par les runs (modeles, rapports, figures, logs).
- `tests/` : filet de securite pour prevenir les regressions.
