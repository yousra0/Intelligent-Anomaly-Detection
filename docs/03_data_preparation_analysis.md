# Notebook 03 — Data Preparation : Analyse détaillée

**Notebook :** `notebooks/03_data_preparation.ipynb`
**Objectif :** Transformer le jeu de données brut en datasets prêts à l'entraînement : nettoyage, feature engineering, encodage, split, normalisation, et gestion du déséquilibre de classes.

---

## Table des matières

1. [Configuration & Imports](#1-configuration--imports)
2. [Rechargement du sample](#2-rechargement-du-sample)
3. [Feature Engineering](#3-feature-engineering)
4. [Suppression du Data Leakage](#4-suppression-du-data-leakage)
5. [One-Hot Encoding de `type`](#5-one-hot-encoding-de-type)
6. [Transformation log1p de `amount`](#6-transformation-log1p-de-amount)
7. [Split stratifié 70 / 15 / 15](#7-split-stratifié-70--15--15)
8. [Normalisation — StandardScaler](#8-normalisation--standardscaler)
9. [Gestion du déséquilibre de classes](#9-gestion-du-déséquilibre-de-classes)
10. [Vérification finale des datasets](#10-vérification-finale-des-datasets)
11. [Sauvegarde des artefacts](#11-sauvegarde-des-artefacts)
12. [Récapitulatif du pipeline](#12-récapitulatif-du-pipeline)

---

## 1. Configuration & Imports

### Ce qui est fait
Initialisation de toutes les constantes du pipeline en un seul bloc central, résolution du répertoire racine, et création des dossiers de sortie nécessaires.

### Bibliothèques importées
`pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `sklearn` (`train_test_split`, `StandardScaler`, `compute_class_weight`), `imblearn.over_sampling.SMOTE`, et les modules internes `src.preprocessing.data_loader` et `src.preprocessing.feature_engineering`.

### Constantes centrales définies

| Constante | Valeur | Justification |
|---|---|---|
| `RANDOM_STATE` | 42 | Graine unique pour reproductibilité totale (split, SMOTE, numpy, random) |
| `SAMPLE_SIZE` | 200 000 | Identique au notebook 02 — reproductibilité du sample |
| `TRAIN_RATIO` | 0.70 | ~180 fraudes en train — minimum viable pour l'apprentissage |
| `VAL_RATIO` | 0.15 | ~38 fraudes — suffisant pour calculer une PR-AUC fiable |
| `TEST_RATIO` | 0.15 | ~39 fraudes — réservé à l'évaluation finale uniquement |
| `HIGH_RISK_HOURS` | `[0,1,2,3,4,5,6,7,8,9,23]` | Heures identifiées dans l'EDA (ratio 10.63×) |
| `SCALE_COLS` | 6 colonnes continues | `step`, `hour`, `day`, `week`, `log_amount`, `balance_diff_orig` |
| `COLS_TO_DROP` | 7 colonnes | Leakage métier + identifiants non-encodables |

### Choix : centraliser toutes les constantes dès le début
Regrouper les hyperparamètres en début de notebook (plutôt que de les disperser dans les cellules) permet de modifier le pipeline entier en un seul endroit et rend le code reproductible et auditable — pratique essentielle dans un contexte PwC.

---

## 2. Rechargement du sample

### Ce qui est fait
Rechargement du dataset complet (6,3 M lignes) puis extraction d'un échantillon stratifié **identique à celui du notebook 02**, avec assertions de reproductibilité pour garantir la cohérence inter-notebooks.

### Choix : reproduire exactement le même sample que l'EDA
Les mêmes paramètres (`train_size=200_000`, `stratify=isFraud`, `random_state=42`) sont utilisés pour garantir que les décisions prises dans le notebook 02 (distributions observées, corrélations, baseline) s'appliquent exactement aux données préparées ici. Deux assertions vérifient le résultat attendu.

### Résultats

| Indicateur | Valeur | Statut |
|---|---|---|
| Dataset complet | 6 362 620 lignes | — |
| Fraudes dans le complet | 8 213 (0.1291 %) | — |
| Sample extrait | 200 000 lignes | OK |
| Fraudes dans le sample | 258 (0.1290 %) | OK |
| Ratio d'imbalance | 1 : 774 | OK |
| Plage de `step` | [1, 741] | OK |
| Verdict assertion | Sample identique au notebook 02 | ✅ |

---

## 3. Feature Engineering

### Ce qui est fait
Reproduction fidèle des 7 features dérivées identifiées dans l'EDA (notebook 02), avec validation systématique de chaque feature contre les valeurs de référence mesurées dans l'EDA.

### Features créées et leur justification

| Feature | Calcul | Corrélation `isFraud` | Source EDA |
|---|---|---|---|
| `hour` | `step % 24` | 0.0372 | Cell 53 — patterns journaliers |
| `day` | `step // 24` | 0.0338 | Cell 53 — patterns mensuels |
| `week` | `step // 168` | 0.0312 | Cell 53 — patterns hebdomadaires |
| `high_risk_hour` | 1 si `hour` ∈ [0–9, 23] | 0.0531 | Cell 57 — ratio 10.63× détecté |
| `is_transfer_or_cashout` | 1 si `type` ∈ {TRANSFER, CASH_OUT} | 0.0409 | Cell 71 — seuls types avec fraudes |
| `balance_diff_orig` | `oldbalanceOrg − newbalanceOrig` | **0.3662** | Cell 70 — signal dominant |
| `dest_zero_balance` | 1 si dest client et balance 0/0 | 0.1088 | Cell 72 — signe de compte mule |

### Validation des features contre l'EDA

| Feature | Valeur observée | Valeur attendue (EDA) | Statut |
|---|---|---|---|
| `hour` range | [0, 23] | [0, 23] | OK |
| `day` range | [0, 30] | [0, 30] | OK |
| `week` range | [0, 4] | [0, 4] | OK |
| `high_risk_hour` ratio | 10.63× | ~10.63× | OK |
| `dest_zero_balance` taux fraude | 2.5212 % | ~2.52 % | OK |
| `balance_diff_orig` corrélation | 0.3662 | ~0.3662 | OK |

---

## 4. Suppression du Data Leakage

### Ce qui est fait
Suppression de 7 colonnes identifiées dans l'EDA comme sources de leakage métier ou d'information inutilisable en production.

### Colonnes supprimées et justifications

| Colonne | Raison d'exclusion |
|---|---|
| `nameOrig` | Identifiant textuel, cardinalité 200 000 — non-encodable en production, ne généralise pas |
| `nameDest` | Idem, cardinalité 174 532 |
| `oldbalanceOrg` | Solde brut source avant transaction — `balance_diff_orig` en extrait déjà le signal pertinent |
| `newbalanceOrig` | Solde brut source après transaction — idem |
| `oldbalanceDest` | Solde brut destination — `dest_zero_balance` couvre le signal pertinent |
| `newbalanceDest` | Idem |
| `isFlaggedFraud` | Flag administratif existant — couvre moins de 3 % des fraudes réelles (leakage direct) |

### Résultat

```
Colonnes avant suppression : 18
Colonnes après suppression : 11

Colonnes restantes : step, type, amount, isFraud,
                     hour, day, week, high_risk_hour,
                     is_transfer_or_cashout, balance_diff_orig, dest_zero_balance
```

---

## 5. One-Hot Encoding de `type`

### Ce qui est fait
Encodage de la variable catégorielle `type` en 5 colonnes binaires (`type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`), avec validation des effectifs contre les valeurs mesurées dans l'EDA.

### Choix : `drop_first=False` (conserver les 5 colonnes)
Pour la régression linéaire, supprimer une colonne (`drop_first=True`) évite la multicolinéarité parfaite. Ici, `drop_first=False` est retenu car :
- L'AutoEncoder et les modèles à arbres (XGBoost, Isolation Forest) n'ont pas ce problème de multicolinéarité.
- Conserver les 5 colonnes donne au modèle une représentation explicite de chaque type — utile notamment pour le débogage et l'interprétabilité SHAP.
- La feature `is_transfer_or_cashout` (déjà présente) redonde partiellement, mais son signal discret est plus facile à interpréter pour un auditeur PwC.

### Validation des effectifs

| Colonne | Lignes | Attendu (EDA) | Statut |
|---|---|---|---|
| `type_CASH_IN` | 43 677 | 43 677 | ✅ |
| `type_CASH_OUT` | 70 282 | 70 282 | ✅ |
| `type_DEBIT` | 1 305 | 1 305 | ✅ |
| `type_PAYMENT` | 67 821 | 67 821 | ✅ |
| `type_TRANSFER` | 16 915 | 16 915 | ✅ |

Une assertion vérifie également que chaque ligne a exactement 1 type actif (somme des 5 colonnes = 1).

---

## 6. Transformation log1p de `amount`

### Ce qui est fait
Remplacement de la colonne `amount` par `log_amount = log1p(amount)`, avec mesure de la réduction de skewness et visualisation comparative.

### Justification mesurée dans l'EDA (Cell 47-48)
- Skewness brute = **30.80** — distribution extrêmement asymétrique à droite
- Ratio max/médiane ≈ **931×** (max 69.9 M, médiane 75 033)
- Sans transformation, les montants extrêmes dominent les visualisations et perturbent les algorithmes sensibles aux échelles (AutoEncoder, LSTM, StandardScaler)

### Choix : `log1p` plutôt que `log`, Box-Cox, ou Yeo-Johnson
- `log1p(x) = log(1 + x)` : définie en 0 (certains montants du dataset sont nuls), contrairement à `log(x)` qui diverge
- Plus simple et plus interprétable que Box-Cox ou Yeo-Johnson pour un rapport d'audit
- Transformée inverse triviale si besoin de présenter des montants réels : `expm1(y)`

### Résultats

| Indicateur | Avant | Après |
|---|---|---|
| Skewness | 30.80 | **−0.55** |
| Réduction de skewness | — | **98 %** |
| Plage | [0, 69 886 731] | [0.34, 18.06] |
| Médiane | 75 033 | 11.23 |

La skewness passe de 30.80 à −0.55 (légèrement asymétrique à gauche), ce qui est acceptable pour tous les modèles du pipeline. La colonne `amount` brute est supprimée après transformation.

---

## 7. Split stratifié 70 / 15 / 15

### Ce qui est fait
Division du dataset en trois ensembles (train, validation, test) avec stratification sur `isFraud`, puis création d'un quatrième sous-ensemble `train_normal` (train filtré sans aucune fraude) destiné à l'entraînement non-supervisé de l'AutoEncoder.

### Choix des ratios 70 / 15 / 15
- **70 % train** → ~180 fraudes : minimum viable pour qu'un modèle supervisé apprenne la frontière de décision
- **15 % val** → ~38 fraudes : suffisant pour calculer une PR-AUC fiable et suivre la convergence
- **15 % test** → ~39 fraudes : réservé strictement à l'évaluation finale (jamais utilisé pendant l'entraînement)

### Procédure de split en deux étapes
```
Étape 1 : train (70%) | temp (30%)   → stratify=isFraud
Étape 2 : val (50% temp) | test (50% temp)   → stratify=isFraud
```
Cette procédure garantit une stratification correcte à chaque étape, évitant les biais liés au tirage aléatoire sur une classe si rare (0.13 %).

### Résultats

| Dataset | Lignes | Fraudes | Taux fraude | Usage |
|---|---|---|---|---|
| `X_train_sc` | 139 999 | 181 | 0.1293 % | Modèles supervisés |
| `X_norm_sc` (train normal) | 139 818 | **0** | 0.0000 % | AutoEncoder non-supervisé |
| `X_val_sc` | 30 000 | 38 | 0.1267 % | Validation en cours d'entraînement |
| `X_test_sc` | 30 001 | 39 | 0.1300 % | Évaluation finale uniquement |

**Nombre total de features :** 14

```
step, hour, day, week, high_risk_hour, is_transfer_or_cashout,
balance_diff_orig, dest_zero_balance,
type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER,
log_amount
```

### Importance du train_normal
L'AutoEncoder apprend la distribution des transactions **normales** uniquement. Si des fraudes étaient présentes à l'entraînement, le modèle les apprendrait comme "normales" et ne les détecterait plus à l'inférence. `X_norm_sc` = `X_train_sc` filtré sur `y_train == 0`.

---

## 8. Normalisation — StandardScaler

### Ce qui est fait
Normalisation des 6 colonnes continues (µ=0, σ=1) en fittant le scaler **uniquement sur `X_train`**, puis en appliquant `.transform()` sur tous les splits.

### Colonnes scalées vs non-scalées

| Type | Colonnes | Raison |
|---|---|---|
| **Scalées** | `step`, `hour`, `day`, `week`, `log_amount`, `balance_diff_orig` | Variables continues avec des plages très différentes |
| **Non scalées** | `high_risk_hour`, `is_transfer_or_cashout`, `dest_zero_balance`, `type_*` | Déjà en 0/1 — scaler n'apporterait rien |

### Choix : StandardScaler plutôt que MinMaxScaler ou RobustScaler
- `balance_diff_orig` est fortement négatif pour les non-fraudes (moyenne = −23 074 mesurée dans l'EDA) et très positif pour les fraudes (1 491 319) → l'écart-type est le bon dénominateur
- `log_amount` a encore des valeurs extrêmes modérées (max ~17.06 pour un montant de 69.9 M)
- `MinMaxScaler` est plus vulnérable aux outliers extrêmes (les 0.13 % de fraudes sont des outliers)
- `RobustScaler` aurait été une alternative valide, mais `StandardScaler` est plus standard dans la littérature sur la détection de fraudes

### Règle anti-leakage critique
```
scaler.fit()    → UNIQUEMENT sur X_train
scaler.transform() → X_train, X_val, X_test, X_norm, X_smote
```
Fitter le scaler sur val ou test reviendrait à utiliser des informations futures dans la normalisation, gonflant artificiellement les métriques.

### Stats pré-scaling (X_train)

| Feature | Moyenne | Std | Min | Max |
|---|---|---|---|---|
| `step` | 243.64 | 142.45 | 1.00 | 741.00 |
| `hour` | 15.32 | 4.32 | 0.00 | 23.00 |
| `day` | 9.51 | 5.93 | 0.00 | 30.00 |
| `week` | 1.01 | 0.86 | 0.00 | 4.00 |
| `log_amount` | 10.84 | 1.81 | 0.46 | 17.57 |
| `balance_diff_orig` | (large négatif) | (très grande) | — | — |

### Validation post-scaling

| Feature | Moyenne post-scaling | Std post-scaling | Statut |
|---|---|---|---|
| `step` | ≈ 0.0000 | 1.0000 | ✅ |
| `hour` | ≈ 0.0000 | 1.0000 | ✅ |
| `day` | ≈ 0.0000 | 1.0000 | ✅ |
| `week` | ≈ 0.0000 | 1.0000 | ✅ |
| `log_amount` | ≈ 0.0000 | 1.0000 | ✅ |
| `balance_diff_orig` | ≈ 0.0000 | 1.0000 | ✅ |

La moyenne sur `X_val[log_amount]` = −0.0026 (≠ 0), ce qui confirme que le scaler a été fitté sur le train uniquement — comportement attendu et correct.

---

## 9. Gestion du déséquilibre de classes

### Contexte
Ratio mesuré dans l'EDA : **1 : 774** (181 fraudes / 139 818 non-fraudes dans le train).

Deux stratégies complémentaires sont appliquées **uniquement sur le train** — val et test conservent la distribution naturelle pour que les métriques reflètent les conditions réelles de production.

---

### Stratégie 1 : `class_weight='balanced'`

**Principe :** Pénalise plus fortement les erreurs sur la classe minoritaire lors de l'optimisation, sans modifier les données d'entraînement.

**Formule sklearn :**
```
w_i = n_samples / (n_classes × count_i)
```

**Résultats :**

| Classe | Poids calculé | Interprétation |
|---|---|---|
| 0 (non-fraude) | 0.500647 | Pénalisation minimale |
| 1 (fraude) | **386.74** | Chaque erreur sur fraude compte 387× plus |
| Ratio poids 1/0 | 772× | Cohérent avec le ratio d'imbalance 1:774 |

**Usage :** Paramètre direct des modèles sklearn et Keras (`class_weight` dans `.fit()`). Aucune modification des données — applicable à tous les modèles du pipeline.

---

### Stratégie 2 : SMOTE (Synthetic Minority Over-Sampling Technique)

**Principe :** Génère des exemples synthétiques de fraude par interpolation linéaire entre des exemples réels et leurs k plus proches voisins dans l'espace des features.

**Paramètres choisis :**
- `sampling_strategy=0.1` : ratio cible fraudes/non-fraudes = 1:10 (et non 1:1)
- `k_neighbors=5` : nombre de voisins utilisés pour l'interpolation

**Choix : ratio cible 1:10 et non 1:1**
Un ratio 1:1 génèrerait des dizaines de milliers d'exemples synthétiques très éloignés des fraudes réelles, conduisant à de l'overfitting sur des patterns artificiels. Le ratio 1:10 est un compromis : assez de diversité pour améliorer la frontière de décision, sans sur-représentation excessive.

**Résultats :**

| Indicateur | Avant SMOTE | Après SMOTE |
|---|---|---|
| Lignes totales | 139 999 | **153 799** |
| Fraudes réelles | 181 (0.1293 %) | 181 |
| Fraudes synthétiques | 0 | **13 800** |
| Fraudes totales | 181 | **13 981 (9.09 %)** |
| Ratio final | 1 : 774 | **1 : 10** |

**Validation de la qualité des exemples synthétiques :**

| Indicateur | Valeur |
|---|---|
| Fraudes réelles — `log_amount` moyenne | 1.2140 |
| Fraudes synthétiques — `log_amount` moyenne | 1.2347 |
| Delta | 0.0207 ✅ |

Le delta très faible entre la distribution réelle et synthétique confirme que SMOTE génère des exemples cohérents avec les fraudes réelles.

**Usage :** `X_smote_df` est utilisé uniquement pour les modèles supervisés classiques (XGBoost, Random Forest). L'AutoEncoder utilise `X_norm_sc` (pas de SMOTE). Les métriques finales sont toujours calculées sur `X_test_sc` à distribution naturelle 1:774.

---

## 10. Vérification finale des datasets

### Ce qui est fait
Vérification croisée de tous les datasets produits (dimensions, taux de fraude, usage) et drift check de distribution entre train et test.

### Inventaire des datasets produits

| Dataset | Dimensions | Fraudes | Taux | Usage |
|---|---|---|---|---|
| `X_train_sc` | (139 999, 14) | 181 | 0.1293 % | Tous modèles supervisés |
| `X_norm_sc` | (139 818, 14) | 0 | 0.0000 % | AutoEncoder uniquement |
| `X_smote_df` | (153 799, 14) | 13 981 | 9.0904 % | Modèles supervisés classiques |
| `X_val_sc` | (30 000, 14) | 38 | 0.1267 % | Suivi métriques entraînement |
| `X_test_sc` | (30 001, 14) | 39 | 0.1300 % | Évaluation finale |

### Drift check train vs test

Vérifie que les distributions des colonnes scalées sont similaires entre train et test (|Δ moyennes| < 0.15 après scaling).

| Feature | Moyenne train | Moyenne test | |Δ| | Statut |
|---|---|---|---|---|
| `step` | −0.0000 | −0.0006 | 0.0006 | ✅ |
| `hour` | −0.0000 | +0.0045 | 0.0045 | ✅ |
| `day` | −0.0000 | −0.0008 | 0.0008 | ✅ |
| `week` | +0.0000 | −0.0007 | 0.0007 | ✅ |
| `log_amount` | +0.0000 | −0.0015 | 0.0015 | ✅ |
| `balance_diff_orig` | −0.0000 | −0.0061 | 0.0061 | ✅ |

Tous les deltas sont bien en dessous du seuil de 0.15 — pas de drift de distribution entre train et test.

---

## 11. Sauvegarde des artefacts

### Ce qui est fait
Export de tous les artefacts du pipeline dans les répertoires `data/processed/` et `outputs/models/`, avec rapport de synthèse JSON.

### Artefacts produits

**Métadonnées et modèles (`outputs/models/`) :**

| Fichier | Contenu |
|---|---|
| `features.json` | Liste des 14 features, colonnes scalées, colonnes binaires, colonnes type, target, heures à risque, ratios de split, comptages de fraudes |
| `class_weights.json` | `{0: 0.5006, 1: 386.74}` — poids pour tous les modèles supervisés |
| `scaler.pkl` | `StandardScaler` fitté sur les 139 999 lignes de `X_train` — réutilisé pour l'inférence en production |

**Datasets (`data/processed/`) :**

| Fichier | Format | Contenu |
|---|---|---|
| `train.csv` / `train.npy` | CSV + NPY | X_train_sc (139 999 lignes) |
| `val.csv` / `val.npy` | CSV + NPY | X_val_sc (30 000 lignes) |
| `test.csv` / `test.npy` | CSV + NPY | X_test_sc (30 001 lignes) |
| `train_normal.csv` / `.npy` | CSV + NPY | X_norm_sc (139 818 lignes) |
| `train_smote.csv` / `.npy` | CSV + NPY | X_smote_df (153 799 lignes) |

### Choix : double format CSV + NPY
- **CSV** : lisible humainement, utile pour le débogage et l'audit
- **NPY** (NumPy binary) : chargement rapide dans les notebooks de modélisation, sans parsing de chaînes

### Rapport de synthèse (`prep_report.json`)
Récapitulatif de toutes les étapes du pipeline (action, détail, résultat) pour la traçabilité de l'audit.

---

## 12. Récapitulatif du pipeline

| # | Étape | Entrée | Sortie | Résultat |
|---|---|---|---|---|
| 1 | Sampling stratifié | 6 362 620 lignes | 200 000 lignes | 258 fraudes · 0.1290 % · ratio 1:774 |
| 2 | Feature Engineering | 11 colonnes | +7 features dérivées | `hour`, `day`, `week`, `high_risk_hour`, `is_transfer_or_cashout`, `balance_diff_orig`, `dest_zero_balance` |
| 3 | Suppression leakage | 18 colonnes | 11 colonnes | −7 colonnes (`nameOrig`, `nameDest`, balances brutes, `isFlaggedFraud`) |
| 4 | One-Hot Encoding | `type` (5 modalités) | 5 colonnes `type_*` | `drop_first=False`, validé contre l'EDA |
| 5 | Transformation log1p | `amount` (skew 30.80) | `log_amount` (skew −0.55) | Réduction de skewness de 98 % |
| 6 | Split 70/15/15 | 200 000 lignes | 4 datasets | Train 140 k · Val 30 k · Test 30 k · Train normal 140 k |
| 7 | StandardScaler | 6 colonnes continues | Normalisées µ=0, σ=1 | Fit uniquement sur train (anti-leakage) |
| 8 | class_weight | y_train | Poids {0: 0.50, 1: 386.74} | Compensation déséquilibre sans modification des données |
| 9 | SMOTE | Train 140 k (181 fraudes) | Train SMOTE 154 k (13 981 fraudes) | Ratio 1:774 → 1:10, delta synthétique 0.0207 ✅ |
| 10 | Sauvegarde | Tous datasets | CSV + NPY + JSON + PKL | 5 datasets · scaler · features.json · class_weights.json |

### Décisions impactant les notebooks suivants

1. **14 features finales** → architecture fixe pour tous les modèles (AutoEncoder, XGBoost, LSTM)
2. **`X_norm_sc` (0 fraude)** → dataset dédié pour l'entraînement non-supervisé de l'AutoEncoder
3. **`X_smote_df` (ratio 1:10)** → utilisé pour XGBoost et Random Forest supervisés
4. **`class_weights.json`** → importé directement dans les notebooks de modélisation via `json.load()`
5. **`scaler.pkl`** → importé pour normaliser les nouvelles transactions en production (API FastAPI)
6. **Baseline à dépasser** : Recall ≈ 0.4 %, F1 ≈ 0.8 % (règle `isFlaggedFraud` du notebook 02)
