# JSON Schema Standards — Anomaly Detection Project

Ce document définit les schémas standards pour les rapports JSON générés par les notebooks, assurant la cohérence et la réutilisabilité des données entre les modules.

---

## 1. `baseline_report.json`

**Localisation** : `outputs/reports/baseline_report.json`

**Objectif** : Documenter les performances des modèles baseline (NB03), hyperparamètres, et seuils optimaux.

### Schéma Complet

```json
{
  "timestamp": "ISO 8601 datetime string",
  "source": "NB03_baseline_models.ipynb",
  "data_info": {
    "n_train": "integer - nombre d'échantillons train original",
    "n_fraud_train": "integer - nombre de fraudes en train",
    "n_smote": "integer - nombre d'échantillons après SMOTE",
    "n_fraud_smote": "integer - nombre de fraudes après SMOTE",
    "n_val": "integer",
    "n_fraud_val": "integer",
    "n_test": "integer",
    "n_fraud_test": "integer",
    "n_features": "integer - nombre de features"
  },
  "baseline_metier": {
    "model": "string - nom du modèle métier (ex: isFlaggedFraud)",
    "recall": "float [0,1]",
    "precision": "float [0,1]",
    "f1": "float [0,1]"
  },
  "models": [
    {
      "name": "string - nom unique du modèle (LR_balanced, RF_smote, XGB_smote, etc)",
      "optimal_threshold": "float [0,1] - seuil déterminé sur validation",
      "train_time_s": "float - temps d'entraînement en secondes",
      "hyperparameters": {
        "key": "value pairs - hyperparamètres utilisés"
      },
      "test_metrics": {
        "model": "string",
        "threshold": "float",
        "recall": "float [0,1]",
        "precision": "float [0,1]",
        "f1": "float [0,1]",
        "accuracy": "float [0,1]",
        "confusion_matrix": [[int, int], [int, int]],
        "tp": "int - true positives",
        "fp": "int - false positives",
        "fn": "int - false negatives",
        "tn": "int - true negatives",
        "roc_auc": "float [0,1]",
        "pr_auc": "float [0,1]"
      }
    }
  ],
  "best_model_recall": "string - nom du modèle avec meilleur recall",
  "best_model_f1": "string - nom du modèle avec meilleur F1",
  "feature_importances_rf_balanced": [
    {
      "feature": "string",
      "importance": "float"
    }
  ]
}
```

### Notes d'Utilisation

- **Timestamp** : Permet de tracer quand le rapport a été généré
- **Source** : Identifie le notebook d'origine (important pour reproduire les résultats)
- **data_info** : Mémoriser les dimensions des données utilisées
- **hyperparameters** : Permet de recréer exactement les modèles
- **optimal_threshold** : Seuil maximisant F-beta (beta=2) sur validation, appliqué au test

### Validation en Python

```python
import json
from pathlib import Path
from src.utils.baseline_config import load_baseline_metrics

# Charger les métriques
metrics = load_baseline_metrics(Path('outputs/reports/baseline_report.json'))
# Retourne dict avec recall, precision, f1, model (toujours un fallback sûr)

# Accès aux modèles complets
with open('outputs/reports/baseline_report.json') as f:
    report = json.load(f)
    
for model in report['models']:
    print(f"Modèle: {model['name']}")
    print(f"  Seuil optimal: {model['optimal_threshold']:.4f}")
    print(f"  F1 (test): {model['test_metrics']['f1']:.4f}")
```

---

## 2. `optimal_thresholds.json`

**Localisation** : `outputs/models/optimal_thresholds.json`

**Objectif** : Stocker les seuils optimaux par modèle (déterminés sur validation), avec métadonnées de source.

### Schéma Complet

```json
{
  "timestamp": "ISO 8601 datetime string",
  "source": "NB03_baseline_models.ipynb",
  "val_metrics": {
    "n_val": "integer - taille de la validation set",
    "n_fraud_val": "integer - nombre de fraudes en validation"
  },
  "thresholds": {
    "LR_balanced": "float [0,1]",
    "LR_smote": "float [0,1]",
    "RF_balanced": "float [0,1]",
    "RF_smote": "float [0,1]",
    "XGB_smote": "float [0,1]",
    "IsoForest": "float [0,1]"
  },
  "description": "Seuils optimaux par modèle, maximisant F-beta (beta=2) sur validation set"
}
```

### Notes d'Utilisation

- **Validation Set Immuable** : Ces seuils ne changent jamais — sélectionnés **une seule fois** sur `X_val` (jamais sur test)
- **Source Traçable** : Le timestamp permet de corréler avec baseline_report.json
- **Métriques Val** : Mémoriser les dimensions de la validation set qui a produit ces seuils

### Accès en Python

```python
import json

with open('outputs/models/optimal_thresholds.json') as f:
    thresholds_data = json.load(f)

# Accès simple aux seuils
thresholds = thresholds_data['thresholds']
print(f"RF_smote optimal threshold: {thresholds['RF_smote']:.4f}")

# Retrouver quand ces seuils ont été calculés
print(f"Timestamp: {thresholds_data['timestamp']}")
print(f"Validation set size: {thresholds_data['val_metrics']['n_val']}")
```

---

## 3. Règles de Cohérence

### 3.1 Nommage des Modèles

Tous les modèles doivent utiliser ces noms standardisés (sans espaces trailing) :
- `LR_balanced`
- `LR_smote`
- `RF_balanced`
- `RF_smote`
- `XGB_smote`
- `IsoForest`

**Vérification** : Aucun espace avant la fin du nom (ex : **`XGB_smote ` ❌** → **`XGB_smote` ✅**)

### 3.2 Métriques

Les métriques doivent **toujours** inclure :
- `recall` (priorité 1 — détection de fraude)
- `precision` (priorité 2 — taux faux positif)
- `f1` (équilibre global)
- `pr_auc` (PR-AUC robuste sous déséquilibre)
- `roc_auc` (informatif mais moins pertinent ici)
- `confusion_matrix` ou `tp, fp, fn, tn`

Arrondir à 4 décimales (représentation lisible standard).

### 3.3 Seuils

- **Seuil Optimal** : Déterminé sur `X_val` uniquement (jamais `X_test`)
- **Métrique de Sélection** : F-beta avec `beta=2` (rappel-prioritaire pour la fraude)
- **Format** : Float [0,1] avec 4 décimales (ex : `0.6291`)

### 3.4 Encodage

Tous les fichiers JSON doivent utiliser :
- **Encodage** : UTF-8
- **Indentation** : 2 espaces
- **ensure_ascii** : False (permettre les accents français)

```python
json.dump(data, f, indent=2, ensure_ascii=False, encoding='utf-8')
```

---

## 4. Versionning et Évolution

### 4.1 Changements Rétro-compatibles (✅ autorisés)

- Ajouter un champ optionnel à `models[]`
- Ajouter un sous-objet à `data_info`

### 4.2 Changements Rompants (⚠️ nécessitent migration)

- Renommer un modèle existant
- Changer le format d'une métrique
- Supprimer un champ obligatoire

→ **Documenter la migration dans `CHANGELOG.md`**

---

## 5. Checklist de Validation

Avant de commiter un rapport JSON, vérifier :

```python
import json
from pathlib import Path

def validate_baseline_report(path):
    """Valide le schéma de baseline_report.json"""
    with open(path) as f:
        report = json.load(f)
    
    assert 'timestamp' in report, "FAIL: timestamp absent"
    assert 'models' in report, "FAIL: models absent"
    assert len(report['models']) >= 4, f"FAIL: {len(report['models'])} modèles (attendu ≥4)"
    
    for model in report['models']:
        assert model['name'] in [
            'LR_balanced', 'LR_smote', 'RF_balanced', 'RF_smote', 
            'XGB_smote', 'IsoForest'
        ], f"FAIL: nom modèle invalide '{model['name']}'"
        
        metrics = model['test_metrics']
        for key in ['recall', 'precision', 'f1', 'threshold']:
            assert key in metrics, f"FAIL: {key} absent dans metrics de {model['name']}"
    
    print("✅ Rapport valide")

# Utilisation
validate_baseline_report('outputs/reports/baseline_report.json')
```

---

## 6. Exemples Pratiques

### Exemple 1 : Charger et comparer deux rapports

```python
import json
import pandas as pd

def compare_reports(old_path, new_path):
    """Comparer deux baseline_report.json"""
    with open(old_path) as f:
        old = json.load(f)
    with open(new_path) as f:
        new = json.load(f)
    
    # Extraire les métriques test
    old_df = pd.DataFrame([m['test_metrics'] for m in old['models']])
    new_df = pd.DataFrame([m['test_metrics'] for m in new['models']])
    
    print("ANCIEN rapport :")
    print(old_df[['model', 'recall', 'f1', 'pr_auc']].to_string())
    print("\nNOUVEAU rapport :")
    print(new_df[['model', 'recall', 'f1', 'pr_auc']].to_string())
    
    return old_df, new_df
```

### Exemple 2 : Inférence avec seuils versionnés

```python
import json
import joblib

# Charger le modèle et les seuils
model = joblib.load('outputs/models/rf_smote.pkl')
with open('outputs/models/optimal_thresholds.json') as f:
    thresholds_data = json.load(f)

threshold = thresholds_data['thresholds']['RF_smote']
scores = model.predict_proba(X_test)[:, 1]
predictions = (scores >= threshold).astype(int)

print(f"Seuil utilisé: {threshold:.4f}")
print(f"Prédictions: {predictions[:10]}")
```

---

## 7. Référence Rapide

| Fichier | Clé d'Accès | Type | Exemple |
|---------|-------------|------|---------|
| `baseline_report.json` | `['models'][0]['name']` | str | `"RF_smote"` |
| `baseline_report.json` | `['models'][0]['test_metrics']['f1']` | float | `0.8052` |
| `baseline_report.json` | `['timestamp']` | str (ISO 8601) | `"2026-05-13T14:30:45.123456"` |
| `optimal_thresholds.json` | `['thresholds']['RF_smote']` | float | `0.6291` |
| `optimal_thresholds.json` | `['val_metrics']['n_fraud_val']` | int | `38` |

---

**Dernière mise à jour** : 2026-05-13  
**Responsable** : Notebook 03_baseline_models.ipynb
