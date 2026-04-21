# Intelligent Anomaly Detection

Pipeline complet de detection d'anomalies/fraude base sur des donnees transactionnelles.
Le projet couvre toute la chaine: EDA, preparation des donnees, modeles baseline,
autoencoder, evaluation et export des artefacts (rapports, modeles, figures).

## Objectifs

- Construire un workflow reproductible de detection d'anomalies.
- Comparer des modeles supervises baseline et un autoencoder.
- Produire des sorties exploitables pour l'analyse metier.
- Garder une base de code modulaire et testable.

## Demarrage rapide (Windows)

### 1. Creer et activer un environnement virtuel

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Installer les dependances

```powershell
pip install -r requirements.txt
```

### 3. Lancer tout le pipeline notebooks

```powershell
python run_all.py
```

## Execution guidee avec run_all.py

Le script [run_all.py](run_all.py) verifie d'abord les modules de [src](src), puis execute
les notebooks dans l'ordre:

1. [notebooks/01_data_understanding.ipynb](notebooks/01_data_understanding.ipynb)
2. [notebooks/02_data_preparation.ipynb](notebooks/02_data_preparation.ipynb)
3. [notebooks/03_baseline_models.ipynb](notebooks/03_baseline_models.ipynb)
4. [notebooks/04_autoencoder.ipynb](notebooks/04_autoencoder.ipynb)

Commandes utiles:

```powershell
python run_all.py --check-only      # verifie uniquement les imports src/
python run_all.py --from 03          # reprend a partir du notebook 03
python run_all.py --only 04          # execute uniquement le notebook 04
python run_all.py --timeout 3600     # timeout par notebook en secondes
```

## Structure du projet

```text
anomaly_detection_project/
|- README.md
|- requirements.txt
|- run_all.py
|- config/
|  |- config.yaml
|  |- ollama_config.yaml
|- data/
|  |- raw/
|  |  |- dataset_orig.csv
|  |- processed/
|  |  |- X_train*.npy, X_val.npy, X_test.npy
|  |  |- y_train*.npy, y_val.npy, y_test.npy
|  |  |- versions CSV equivalentes
|  |- anomalies/
|  |- labels/
|- notebooks/
|  |- 01_data_understanding.ipynb
|  |- 02_data_preparation.ipynb
|  |- 03_baseline_models.ipynb
|  |- 04_autoencoder.ipynb
|- outputs/
|  |- anomalies_report.csv
|  |- explanations.json
|  |- figures/
|  |- logs/
|  |- models/
|  |  |- class_weights.json
|  |  |- features.json
|  |  |- optimal_thresholds.json
|  |  |- ae_scores_val.npy
|  |  |- ae_scores_test.npy
|  |  |- autoencoder/autoencoder_weights.keras
|  |- reports/
|     |- eda_report.json
|     |- prep_report.json
|     |- baseline_report.json
|     |- autoencoder_report.json
|- src/
|  |- main.py
|  |- feature_engineering/
|  |- preprocessing/
|  |- models/
|  |- pipeline/
|  |- utils/
|  |- visualization/
|  |- ollama_integration/
|- tests/
```

## Modules principaux

- [src/preprocessing](src/preprocessing): chargement, nettoyage et preparation des jeux de donnees.
- [src/feature_engineering](src/feature_engineering): creation de variables temporelles et comportementales.
- [src/models](src/models): modeles baseline et autoencoder.
- [src/pipeline](src/pipeline): orchestration entrainement et inference.
- [src/utils](src/utils): metriques, evaluation et utilitaires anomalies.
- [src/visualization](src/visualization): generation des figures et graphiques de suivi.
- [src/ollama_integration](src/ollama_integration): aide a la generation d'explications.

## Principales sorties generees

- Rapports JSON/CSV dans [outputs/reports](outputs/reports).
- Scores et poids de modeles dans [outputs/models](outputs/models).
- Exports d'anomalies dans [outputs/anomalies_report.csv](outputs/anomalies_report.csv).
- Visualisations dans [outputs/figures](outputs/figures).

## Tests

Executer les tests unitaires:

```powershell
pytest -q
```

Fichiers de tests:

- [tests/test_preprocessing.py](tests/test_preprocessing.py)
- [tests/test_models.py](tests/test_models.py)
- [tests/test_visualization.py](tests/test_visualization.py)
- [tests/test_utils.py](tests/test_utils.py)
- [tests/test_ollama.py](tests/test_ollama.py)

## Notes importantes

- Les fichiers [config/config.yaml](config/config.yaml) et [config/ollama_config.yaml](config/ollama_config.yaml)
	sont presents mais actuellement vides. Si vous souhaitez piloter davantage le pipeline par configuration,
	vous pouvez y centraliser vos parametres (chemins, hyperparametres, seuils, options d'inference).
- [src/main.py](src/main.py) est present mais vide. Le point d'entree principal recommande pour le moment
	est [run_all.py](run_all.py) ou l'execution notebook par notebook.
