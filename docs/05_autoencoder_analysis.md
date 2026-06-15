# Notebook 05 — AutoEncoder : Analyse détaillée

**Notebook :** `notebooks/05_autoencoder.ipynb`
**Objectif :** Construire, entraîner et évaluer un AutoEncoder PyTorch pour la détection non-supervisée de fraudes. L'AutoEncoder apprend la représentation des transactions normales uniquement et détecte les fraudes par leur erreur de reconstruction anormalement élevée.

---

## Table des matières

1. [Configuration & Imports](#1-configuration--imports)
2. [Chargement des données](#2-chargement-des-données)
3. [Construction de l'AutoEncoder](#3-construction-de-lautoencoder)
4. [Entraînement](#4-entraînement)
5. [Analyse de l'erreur de reconstruction](#5-analyse-de-lerreur-de-reconstruction)
6. [Recherche du seuil optimal](#6-recherche-du-seuil-optimal)
7. [Évaluation finale — Test Set](#7-évaluation-finale--test-set)
8. [Visualisation de l'espace latent](#8-visualisation-de-lespace-latent)
9. [Comparaison AutoEncoder vs Baselines ML](#9-comparaison-autoencoder-vs-baselines-ml)
10. [Sauvegarde des artefacts](#10-sauvegarde-des-artefacts)
11. [Synthèse finale](#11-synthèse-finale)

---

## Principe fondamental de l'AutoEncoder pour la détection de fraude

Un AutoEncoder est un réseau de neurones entraîné à **compresser** puis **reconstruire** ses entrées. Dans ce contexte :

1. **Entraînement** : uniquement sur des transactions **normales** (0 fraude). Le réseau apprend à compresser les patterns normaux dans un espace latent de dimension 4, puis à les reconstruire fidèlement.
2. **Inférence** : pour une transaction inconnue, on mesure l'**erreur de reconstruction** (MSE entre l'entrée et la sortie reconstruite).
   - Transaction **normale** → le réseau l'a déjà vue → erreur **faible**
   - Transaction **frauduleuse** → pattern inconnu → le réseau échoue à la reconstruire → erreur **élevée**
3. **Seuillage** : si l'erreur dépasse un seuil θ → transaction classifiée comme fraude.

**Avantage clé vs modèles supervisés :** aucun label de fraude n'est nécessaire à l'entraînement — le modèle peut détecter des schémas de fraude entièrement nouveaux (zero-day fraud).

---

## 1. Configuration & Imports

### Ce qui est fait
Configuration PyTorch avec détection automatique du GPU, initialisation des graines, style visuel, résolution du répertoire racine.

### Environnement d'exécution

| Paramètre | Valeur |
|---|---|
| Framework | **PyTorch 2.6.0+cu124** |
| Device | **CUDA** (GPU NVIDIA GeForce MX450) |
| CUDA Compute Capability | (7, 5) — Turing architecture |
| Graine | 42 (`torch.manual_seed` + `np.random.seed`) |

### Choix : PyTorch sur GPU
PyTorch est préféré à Keras/TensorFlow ici pour :
- Contrôle fin de l'architecture (couches personnalisées, boucle d'entraînement explicite)
- Déploiement plus léger en production (pas de TF runtime)
- GPU MX450 disponible : accélération 3–10× sur les matrices de la taille du bottleneck

### Modules internes chargés

| Module | Rôle |
|---|---|
| `src.models.autoencoder.FraudAutoEncoder` | Wrapper PyTorch de l'AutoEncoder |
| `src.utils.evaluator` | `compute_fraud_metrics`, `print_metrics_report` |
| `src.utils.baseline_config` | Chargement des métriques de référence NB04 |
| `src.visualization.autoencoder_plots` | 6 fonctions de visualisation spécialisées |

---

## 2. Chargement des données

### Ce qui est fait
Chargement uniquement des trois datasets nécessaires à l'AutoEncoder depuis `data/processed/`. Le dataset SMOTE n'est pas utilisé — l'AutoEncoder est non-supervisé.

### Datasets utilisés

| Dataset | Shape | Fraudes | Rôle |
|---|---|---|---|
| `X_train_normal` | (139 818, 14) | **0** | Entraînement de l'AutoEncoder — UNIQUEMENT des transactions normales |
| `X_val` | (30 000, 14) | 38 | Sélection du seuil optimal (jamais vu pendant l'entraînement) |
| `X_test` | (30 001, 14) | 39 | Évaluation finale uniquement |

### Validation critique : assertion 0 fraude en train
```python
assert int(y_normal.sum()) == 0, 'X_train_normal contient des fraudes !'
✅ X_train_normal : 0 fraudes confirmé
```
Si des fraudes étaient présentes dans le train, l'AutoEncoder les apprendrait comme "normales" et ne les détecterait plus à l'inférence — ce qui invaliderait le principe même du modèle.

### Références baseline chargées depuis `baseline_report.json`

| Référence | Valeur |
|---|---|
| Baseline métier (isFlaggedFraud) Recall | 0.0039 |
| Baseline métier F1 | 0.0077 |
| XGB_smote Recall | **0.8462** |
| XGB_smote F1 | **0.8354** |
| XGB_smote PR-AUC | **0.8677** |

---

## 3. Construction de l'AutoEncoder

### Architecture choisie : `14 → 10 → 7 → [4] → 7 → 10 → 14`

```
Entrée      : 14 features
Encodeur    : 14 → 10 → 7
Bottleneck  : [4]  ← espace latent
Décodeur    : 4  → 7 → 10
Sortie      : 14 features reconstruites
```

### Paramètres du modèle et justifications

| Paramètre | Valeur | Justification |
|---|---|---|
| `encoder_dims` | `[10, 7]` | Compression progressive : 14→10→7. Évite une compression trop brutale qui ferait perdre de l'information |
| `bottleneck_dim` | **4** | Espace latent très compressé (ratio 14:4 = 3.5:1). Force le modèle à apprendre la structure essentielle des transactions normales. Trop petit → perte d'information ; trop grand → fraudes facilement reconstruites |
| `decoder_dims` | `[7, 10]` | Symétrique à l'encodeur — reconstruction progressive et symétrique |
| `activation` | `relu` | Activation standard pour les couches cachées. Évite le problème du gradient qui disparaît (vanishing gradient) comparé à sigmoid/tanh |
| `output_activation` | `linear` | Indispensable : après `StandardScaler`, les features ont des valeurs positives ET négatives. Une activation sigmoid/relu en sortie bloquerait les valeurs négatives |
| `dropout_rate` | `0.2` | Régularisation : désactive 20 % des neurones aléatoirement à chaque forward pass. Empêche la mémorisation avec seulement 181 exemples de fraude (même non vus) |
| `use_batch_norm` | `True` | Normalise les activations à chaque couche → stabilise l'entraînement et accélère la convergence |
| `l2_reg` | `1e-5` | Pénalisation L2 des poids (weight decay) → régularisation supplémentaire, évite l'overfitting sur les patterns normaux très similaires |
| `epochs` | 100 | Maximum — en pratique stoppé par EarlyStopping |
| `batch_size` | **256** | Adapté à la RAM GPU du MX450. Assez grand pour un gradient stable sur 139 818 transactions |
| `learning_rate` | `1e-3` | Taux d'apprentissage standard Adam. Sera réduit automatiquement par ReduceLROnPlateau |
| `patience` | 10 | EarlyStopping : arrête après 10 epochs sans amélioration de val_loss |
| `val_split` | 0.1 | 10 % de X_train_normal réservé pour surveiller val_loss → 13 982 lignes de validation interne |

### Statistiques du modèle

| Indicateur | Valeur |
|---|---|
| **Total paramètres** | **596** |
| Device | cuda (GPU) |
| BatchNorm | Oui |
| Dropout | 0.2 |

**Note sur le nombre de paramètres :** 596 paramètres seulement pour 14 features d'entrée. Ce modèle volontairement compact force la généralisation et évite l'overfitting. À titre de comparaison, RF_smote occupait 6 023 Ko (env. 1 M de paramètres).

### Choix : architecture symétrique encodeur/décodeur
La symétrie `[10, 7] → [4] → [7, 10]` est une pratique standard pour les AutoEncoders de détection d'anomalies. Elle garantit que la capacité de reconstruction est équilibrée entre compression et décompression.

---

## 4. Entraînement

### Ce qui est fait
Entraînement sur `X_train_normal` (139 818 transactions, 0 fraude) avec deux mécanismes de régulation automatique : EarlyStopping et ReduceLROnPlateau.

### Règle fondamentale
```
Entraînement → UNIQUEMENT sur X_train_normal
Val_split    → 10 % de X_train_normal (13 982 lignes) pour surveiller val_loss
X_val / X_test → jamais consultés pendant le fit
```

### Mécanismes de régulation automatique

**EarlyStopping (patience=10) :**
Arrête l'entraînement si `val_loss` n'améliore pas pendant 10 epochs consécutives. Évite l'overfitting sur les patterns normaux et économise du temps de calcul.

**ReduceLROnPlateau :**
Divise le learning rate par 2 si `val_loss` ne s'améliore pas pendant 5 epochs. Permet un affinement progressif en fin de convergence.

### Déroulement de l'entraînement

```
Epoch  10/100  loss=0.2352  val_loss=0.1809
Epoch  20/100  loss=0.2330  val_loss=0.1795
Epoch  30/100  loss=0.2320  val_loss=0.1755
  → ReduceLROnPlateau: LR → 5.00e-04  (÷2)
  → EarlyStopping: 5/10
  → ReduceLROnPlateau: LR → 2.50e-04  (÷2 à nouveau)
  → EarlyStopping: 10/10
Early stopping at epoch 36
```

### Résultats de l'entraînement

| Indicateur | Valeur |
|---|---|
| Epochs effectifs | **36** / 100 (arrêt anticipé) |
| Best val_loss | **0.175100** |
| MSE moyen (train normal) | 0.174414 |
| MSE p95 (train normal) | 0.462719 |
| MSE p99 (train normal) | 1.004359 |
| **Temps d'entraînement** | **191.3 secondes** (GPU MX450) |

**Interprétation :** Le modèle converge rapidement (36 epochs vs 100 max). La val_loss finale de 0.175 indique que le modèle reconstruit les transactions normales avec une erreur MSE moyenne de ~0.175 — faible et stable.

### Visualisation produite
`18_training_history.png` : courbes loss et val_loss sur les 36 epochs, avec les points de déclenchement de ReduceLROnPlateau.

---

## 5. Analyse de l'erreur de reconstruction

### Ce qui est fait
Calcul des erreurs de reconstruction (MSE) pour chaque transaction dans les trois splits, et analyse de la séparabilité entre transactions normales et frauduleuses.

### Principe de la détection
```
score_anomalie(x) = MSE(x, AutoEncoder(x)) = mean((x - x̂)²)
```
Plus cette erreur est élevée, plus la transaction est anormale.

### Statistiques des erreurs de reconstruction

| Dataset | Moyenne | Std | p95 | Maximum |
|---|---|---|---|---|
| Train normal | 0.1745 | 0.1893 | 0.4631 | 24.6406 |
| Val — légitimes | 0.1770 | 0.1929 | 0.4680 | 11.3761 |
| **Val — fraudes** | **21.7664** | **59.2405** | **129.7503** | **314.1599** |
| Test — légitimes | 0.1742 | 0.1744 | 0.4668 | 3.0860 |
| **Test — fraudes** | **19.6136** | **63.9157** | **86.7439** | **313.6496** |

### Séparabilité — indicateur clé

```
Ratio MSE fraudes / légitimes (val) : 122.95×
```

Les fraudes ont une erreur de reconstruction **123× plus élevée** que les transactions normales. Cette séparabilité très forte confirme que l'AutoEncoder a bien appris à distinguer les deux patterns, même sans avoir vu aucune fraude pendant l'entraînement.

### Seuil initial : percentile p95 du train normal

```python
threshold_p95 = np.percentile(errors_normal, 95) = 0.463145
```

**Justification :** Le p95 signifie que 5 % des transactions normales dépassent ce seuil — taux de faux positifs acceptable comme point de départ. Ce seuil est ensuite affiné via l'optimisation sur val (section 6).

### Visualisation produite
`19_reconstruction_error.png` : distribution des erreurs (log-scale) pour transactions normales vs fraudes, avec le seuil p95 marqué. La séparation visuelle entre les deux distributions est claire.

---

## 6. Recherche du seuil optimal

### Ce qui est fait
Balayage de 200 seuils candidats entre le percentile 50 et 99.9 de l'erreur sur val, optimisation sur le **F1-score**.

### Règle : seuil sélectionné sur `X_val` uniquement
```
X_val → sélection du seuil optimal
X_test → évaluation finale avec ce seuil figé (jamais consulté avant)
```
Utiliser X_test pour choisir le seuil serait du data snooping — les métriques finales seraient gonflées artificiellement.

### Choix de la métrique d'optimisation : F1 (et non F-beta2)
Contrairement aux modèles supervisés (qui utilisaient F-beta(β=2) pour favoriser le Recall), l'AutoEncoder utilise **F1** pour la sélection du seuil.

**Raison :** L'AutoEncoder génère naturellement moins de faux positifs que les modèles supervisés (pas de biais d'apprentissage vers la classe fraude) — un seuil F1 suffit pour équilibrer Recall et Precision dans ce contexte non-supervisé.

### Résultat de l'optimisation

```
Seuil optimal (f1) : 1.753022
F1 (val) = 0.4412
```

Le seuil optimal de **1.75** est nettement supérieur au seuil p95 initial de 0.46. Cela signifie qu'il faut une erreur de reconstruction 4× plus élevée que le p95 pour classifier une transaction comme fraude — cohérent avec la grande variance des erreurs sur les transactions normales (std=0.19, max=24.6).

### Résultats sur Validation (seuil=1.75)

| Métrique | Valeur |
|---|---|
| Recall | 0.3947 |
| Precision | 0.5000 |
| F1-Score | 0.4412 |
| Accuracy | 0.9987 |
| PR-AUC | 0.3851 |
| ROC-AUC | 0.9431 |
| TP | 15 |
| FN | 23 |
| FP | 15 |
| TN | 29 947 |

---

## 7. Évaluation finale — Test Set

### Ce qui est fait
Application du seuil optimal (1.753022, sélectionné sur val) sur `X_test`. C'est la **seule et unique fois** que X_test est utilisé dans ce notebook.

### Résultats sur Test (seuil=1.75)

| Métrique | AutoEncoder | vs XGB_smote | Δ |
|---|---|---|---|
| **Recall** | **0.3590** | 0.8462 | −0.4872 |
| Precision | 0.6087 | 0.8250 | −0.2163 |
| **F1-Score** | **0.4516** | 0.8354 | −0.3838 |
| Accuracy | 0.9989 | — | — |
| **PR-AUC** | **0.3820** | 0.8677 | −0.4857 |
| ROC-AUC | 0.9358 | 0.9975 | −0.0617 |
| TP | 14 | 33 | −19 |
| FN | 25 | 6 | +19 |
| FP | **9** | 6 | +3 |
| TN | 29 953 | 29 956 | −3 |

### Visualisations produites

| Fichier | Contenu |
|---|---|
| `20_threshold_roc_pr.png` | Courbes ROC et PR-AUC sur test, avec seuil optimal marqué |
| `21_ae_confusion_matrix.png` | Matrice de confusion (TP=14, FN=25, FP=9, TN=29 953) |

### Analyse des résultats

**Points positifs :**
- Recall = 0.359 : **92× la baseline métier** (isFlaggedFraud Recall=0.0039)
- Precision = 0.609 : très bonne pour un modèle non-supervisé — seulement **9 faux positifs** sur 30 001 transactions
- ROC-AUC = 0.936 : excellente discrimination globale
- **0 fraude vue pendant l'entraînement** — c'est l'avantage fondamental

**Points d'amélioration :**
- Recall de 0.359 vs 0.846 pour XGB_smote : 25 fraudes manquées sur 39
- PR-AUC de 0.382 vs 0.868 : l'espace de précision-rappel est nettement inférieur aux supervisés
- **Explication :** les fraudes CASH_OUT dans ce dataset partagent beaucoup de features avec des transactions légitimes de grande amplitude — l'AutoEncoder a du mal à les distinguer uniquement par l'erreur de reconstruction

---

## 8. Visualisation de l'espace latent

### Ce qui est fait
Encodage de toutes les transactions dans l'espace latent 4D (`ae.encode()`), puis projection PCA 2D pour visualiser la séparation entre normaux et fraudes.

### Résultats

```
Espace latent — forme : (139818, 4)  (dim=4)
Fraudes dans val      : 38 sur 30 000
```

### Choix : PCA 2D du bottleneck
La PCA est préférée à t-SNE ici car :
- Plus rapide sur 139 818 points
- Préserve la structure globale (variance maximale) plutôt que locale
- Interprétable : les axes PCA1/PCA2 correspondent aux directions de variance maximale dans l'espace latent

### Visualisation produite
`23_latent_space.png` : scatter plot 2D (PCA du bottleneck) — points normaux en bleu, fraudes en rouge. Si l'AutoEncoder est efficace, les fraudes doivent apparaître dans des régions distinctes de l'espace latent.

**Interprétation attendue :** Les fraudes devraient se positionner en périphérie du nuage de points normaux — leur représentation dans l'espace latent est différente car le modèle n'a jamais appris à les compresser efficacement.

---

## 9. Comparaison AutoEncoder vs Baselines ML

### Ce qui est fait
Comparaison directe des performances de l'AutoEncoder contre les 6 modèles du notebook 04, sur le test set avec seuil optimal.

### Classement final — test set

| Rang | Modèle | Type | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| 🥇 1 | XGB_smote (t=0.35) | Supervisé | 0.846 | 0.835 | 0.868 |
| 🥈 2 | RF_smote (t=0.63) | Supervisé | 0.795 | 0.805 | 0.841 |
| 🥉 3 | RF_balanced (t=0.62) | Supervisé | 0.769 | 0.800 | 0.779 |
| 4 | LR_smote (t=0.96) | Supervisé | 0.692 | 0.711 | 0.739 |
| 5 | LR_balanced (t=1.00) | Supervisé | 0.641 | 0.649 | 0.704 |
| **6** | **AutoEncoder (t=1.75)** | **Non-supervisé** | **0.359** | **0.452** | **0.382** |
| 7 | IsoForest (t=0.78) | Non-supervisé | 0.333 | 0.347 | 0.341 |

### Visualisation produite
`22_ae_vs_baselines.png` : graphique comparatif Recall / F1 / PR-AUC pour tous les modèles.

### Analyse comparative

**AutoEncoder vs IsoForest (comparaison non-supervisée) :**
| Métrique | AutoEncoder | IsoForest | Δ |
|---|---|---|---|
| Recall | 0.359 | 0.333 | +0.026 |
| F1 | 0.452 | 0.347 | +0.105 |
| PR-AUC | 0.382 | 0.341 | +0.041 |

L'AutoEncoder surpasse IsoForest sur toutes les métriques dans le contexte non-supervisé, validant le choix de cette architecture pour la détection d'anomalies.

**Pourquoi l'AutoEncoder est inférieur aux supervisés :**
Les modèles supervisés (XGB_smote, RF_smote) ont un avantage structurel : ils ont vu les 181 vraies fraudes pendant l'entraînement et ont appris leur frontière de décision précise. L'AutoEncoder doit inférer cette frontière uniquement depuis les patterns normaux — tâche plus difficile lorsque les fraudes partagent des features communes avec les légitimes.

**Valeur ajoutée de l'AutoEncoder en production :**
1. **Zero-day fraud** : nouveaux schémas de fraude inconnus (non présents dans les 181 exemples d'entraînement) seront détectés car leur erreur de reconstruction sera élevée
2. **Pas de dépendance aux labels** : applicable dans les premiers mois d'une nouvelle relation client (pas encore de fraudes historiques)
3. **Complémentarité** : combiner AutoEncoder + XGB_smote en ensemble permet de capturer à la fois les fraudes connues (XGB) et inconnues (AE)

---

## 10. Sauvegarde des artefacts

### Ce qui est fait
Sauvegarde complète du modèle, des scores d'anomalie, et du rapport JSON avec métadonnées.

### Artefacts produits

**Modèle (`outputs/models/autoencoder/`) :**

| Fichier | Contenu |
|---|---|
| `autoencoder_weights.pt` | Poids PyTorch du modèle (format .pt natif) |
| `autoencoder_config.json` | Configuration complète (architecture, hyperparamètres, seuil) |
| `autoencoder_metadata.json` | Métadonnées d'entraînement (epochs, val_loss, temps, SHA-256 des poids) |

**Scores d'anomalie (`outputs/models/`) :**

| Fichier | Shape | Échelle | Usage |
|---|---|---|---|
| `ae_scores_test.npy` | (30 001,) | MSE brute (0–314) | Pipeline inférence, NB06 SHAP |
| `ae_scores_val.npy` | (30 000,) | MSE brute (0–314) | Seuil dans baseline_report |

**Rapport JSON (`outputs/reports/autoencoder_report.json`) :**
```json
{
  "architecture": {
    "model_class": "FraudAutoEncoder",
    "encoder_dims": [10, 7],
    "bottleneck_dim": 4,
    "decoder_dims": [7, 10],
    "n_features": 14,
    "total_params": 596,
    "weights_sha256": "..."
  },
  "training": {
    "n_samples": 139818,
    "n_epochs": 36,
    "best_val_loss": 0.175100,
    "train_time_s": 191.3,
    "training_seed": 42
  }
}
```

### Important : deux fonctions de score coexistent

| Méthode | Échelle | Usage |
|---|---|---|
| `ae.reconstruction_error(X)` | MSE brute (~0–400) | Seuillage, comparaison avec threshold_opt, sauvegarde dans .npy |
| `ae.predict_score(X)` | [0, 1] normalisé (÷ p99 train) | Affichage API FastAPI, rapport auditeur |

Ces deux fonctions ne sont **pas interchangeables** pour le seuillage : `threshold_opt = 1.753` est sur l'échelle MSE brute, pas sur [0,1].

---

## 11. Synthèse finale

### Récapitulatif de l'architecture et des résultats

| Aspect | Valeur |
|---|---|
| Architecture | 14 → 10 → 7 → **[4]** → 7 → 10 → 14 |
| Total paramètres | **596** |
| Données d'entraînement | 139 818 transactions normales (0 fraude) |
| Epochs effectifs | 36 / 100 (EarlyStopping) |
| Temps d'entraînement | 191.3 s (GPU MX450) |
| Best val_loss | 0.1751 |
| Ratio séparabilité MSE | **122.95×** (fraudes / légitimes) |
| Seuil optimal | 1.753 (MSE brute, optimisé F1 sur val) |
| **Recall test** | **0.3590** (vs 0.0039 baseline = ×92) |
| **F1 test** | **0.4516** |
| **PR-AUC test** | **0.3820** |
| Faux positifs test | **9 / 30 001** transactions |

### Comparaison avec les deux autres modèles non-supervisés

| Modèle | Labels requis | Recall | F1 | PR-AUC | FP |
|---|---|---|---|---|---|
| IsoForest | Non | 0.333 | 0.347 | 0.341 | 23 |
| **AutoEncoder** | **Non** | **0.359** | **0.452** | **0.382** | **9** |
| XGB_smote | **Oui** | 0.846 | 0.835 | 0.868 | 6 |

L'AutoEncoder dépasse IsoForest sur toutes les métriques tout en utilisant uniquement 9 FP (vs 23 pour IF). Il reste inférieur aux supervisés en Recall et F1, mais offre un avantage unique : la détection de fraudes inconnues.

### Ce que l'AutoEncoder apporte vs les baselines ML

| Critère | Baselines ML (supervisé) | AutoEncoder (non-supervisé) |
|---|---|---|
| Labels requis à l'entraînement | ✅ Oui (181 fraudes) | ❌ Non — 0 fraudes |
| Détecte fraudes **inconnues** | ❌ Limité aux patterns connus | ✅ Zero-day fraud |
| Explicabilité | Feature importances Gini/SHAP | Score d'anomalie (erreur MSE) |
| Complémentarité | — | ✅ Combinable avec XGB_smote en ensemble |
| Performances (Recall) | 0.846 (XGB) | 0.359 |
| Faux positifs | 6 (XGB) | 9 |

### Décisions impactant les notebooks suivants

1. **`ae_scores_test.npy`** → importé dans NB06 pour les explications SHAP et LIME des prédictions de l'AutoEncoder
2. **`autoencoder_weights.pt`** → chargé dans l'API FastAPI (`app/`) pour l'inférence en production
3. **Seuil 1.753** → enregistré dans `autoencoder_report.json`, utilisé par l'API
4. **Complémentarité AutoEncoder + XGB** → le NB07 (LLM) peut combiner les scores des deux modèles pour générer un rapport d'explication plus riche
5. **`predict_score()` → [0,1]** → utilisé dans l'API pour afficher un "score de risque" entre 0 et 100 % aux auditeurs PwC
