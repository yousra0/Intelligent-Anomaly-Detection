# Notebook 04 — Baseline Models : Analyse détaillée

**Notebook :** `notebooks/04_baseline_models.ipynb`
**Objectif :** Établir des références supervisées et non-supervisées solides avant les approches Deep Learning. Ces modèles fixent les seuils de performance que l'AutoEncoder et le LSTM devront dépasser.

---

## Table des matières

1. [Configuration & Imports](#1-configuration--imports)
2. [Chargement des données préparées](#2-chargement-des-données-préparées)
3. [Modèle 1 — Logistic Regression `LR_balanced`](#3-modèle-1--logistic-regression-lr_balanced)
4. [Modèle 2 — Logistic Regression `LR_smote`](#4-modèle-2--logistic-regression-lr_smote)
5. [Modèle 3 — Random Forest `RF_balanced`](#5-modèle-3--random-forest-rf_balanced)
6. [Modèle 4 — Random Forest `RF_smote`](#6-modèle-4--random-forest-rf_smote)
7. [Modèle 5 — XGBoost `XGB_smote`](#7-modèle-5--xgboost-xgb_smote)
8. [Modèle 6 — Isolation Forest `IsoForest`](#8-modèle-6--isolation-forest-isoforest)
9. [Analyse du seuil de décision optimal](#9-analyse-du-seuil-de-décision-optimal)
10. [Comparaison des 6 modèles](#10-comparaison-des-6-modèles)
11. [Feature Importances — Random Forest](#11-feature-importances--random-forest)
12. [Sauvegarde des modèles et rapports](#12-sauvegarde-des-modèles-et-rapports)
13. [Synthèse finale](#13-synthèse-finale)

---

## Contexte de départ

| Donnée | Valeur |
|---|---|
| Train | 139 999 lignes — **181 fraudes** (0.1293 %) |
| Val | 30 000 lignes — **38 fraudes** |
| Test | 30 001 lignes — **39 fraudes** |
| Ratio d'imbalance | **1 : 774** |
| class_weight[1] | **386.74** |
| **Baseline métier à dépasser** | Recall=0.0039 · Precision=1.0 · F1=0.0077 |

---

## 1. Configuration & Imports

### Ce qui est fait
Chargement des bibliothèques, résolution du répertoire racine, création des dossiers de sortie, et chargement de la baseline métier depuis `baseline_report.json`.

### Modules internes utilisés
| Module | Rôle |
|---|---|
| `src.models.ml_models` | Wrappers `FraudLogisticRegression`, `FraudRandomForest` |
| `src.utils.evaluator` | `compute_fraud_metrics`, `find_optimal_threshold`, `compare_models` |
| `src.visualization.model_plots` | Courbes PR/ROC, matrices de confusion, feature importances, comparaison |
| `src.utils.baseline_config` | Chargement de la baseline métier (`isFlaggedFraud`) |

### Choix : wrappers custom pour LR et RF
Les classes `FraudLogisticRegression` et `FraudRandomForest` encapsulent les modèles sklearn avec une interface unifiée (`.fit()`, `.predict()`, `.predict_proba()`, `.summary()`, `.save()`, `.get_feature_importances()`). Cela garantit la cohérence du code entre notebooks et facilite la sauvegarde avec métadonnées.

### Baseline de référence chargée

| Métrique | Valeur (isFlaggedFraud) |
|---|---|
| Recall | 0.0039 |
| Precision | 1.0000 |
| F1-Score | 0.0077 |
| **Objectif minimal** | **Recall > 0.0039 et F1 > 0.0077** |

---

## 2. Chargement des données préparées

### Ce qui est fait
Chargement des artefacts produits par le notebook 03 : fichiers NPY (rapides) pour les datasets et JSON pour les métadonnées.

### Datasets chargés

| Dataset | Shape | Fraudes | Taux | Usage |
|---|---|---|---|---|
| `X_train` | (139 999, 14) | 181 | 0.1293 % | LR_balanced, RF_balanced |
| `X_smote` | (153 799, 14) | 13 981 | 9.0904 % | LR_smote, RF_smote, XGB_smote |
| `X_val` | (30 000, 14) | 38 | 0.1267 % | Sélection des seuils optimaux |
| `X_test` | (30 001, 14) | 39 | 0.1300 % | Évaluation finale |
| `X_normal` | (139 818, 14) | 0 | 0.0000 % | AutoEncoder (notebooks suivants) |

### Compatibilité des artefacts NPY
Un cast explicite `float32` / `int64` est appliqué si le dtype des fichiers NPY est `object` (artefacts historiques). Une validation vérifie que le nombre de colonnes correspond à `features.json` (14 colonnes attendues).

---

## 3. Modèle 1 — Logistic Regression `LR_balanced`

### Choix du modèle
La régression logistique est le modèle linéaire de référence en classification binaire. Sa simplicité et son interprétabilité en font un premier baseline indispensable avant les approches non-linéaires. Elle fournit aussi des probabilités bien calibrées pour l'analyse de seuil.

### Stratégie de gestion du déséquilibre
`class_weight='balanced'` pénalise les erreurs sur la classe fraude d'un facteur **386.74×** sans modifier les données d'entraînement. Le modèle voit les données dans leur distribution naturelle (1:774).

### Paramètres choisis et justifications

| Paramètre | Valeur | Justification |
|---|---|---|
| `C` | 0.1 | Régularisation forte — adaptée au petit nombre de fraudes (181). Évite l'overfitting sur la classe positive très peu représentée |
| `max_iter` | 1000 | Convergence garantie avec lbfgs sur des données scalées |
| `class_weight` | `'balanced'` | Compense le ratio 1:774 par pondération des erreurs |
| `solver` | `lbfgs` (défaut) | Efficace pour des datasets de taille moyenne avec régularisation L2 |
| `random_state` | 42 | Reproductibilité |
| **Temps d'entraînement** | **0.67 s** | |

### Résultats — seuil 0.5

| Split | Recall | Precision | F1 | PR-AUC | ROC-AUC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Validation | 0.9737 | 0.0391 | 0.0752 | 0.6379 | 0.9950 | 37 | 1 | 909 |
| Test | 0.9487 | 0.0420 | ~0.08 | — | — | — | — | — |

**Lecture :** Recall excellent (97 %) mais Precision très faible (4 %) → le modèle détecte presque toutes les fraudes mais génère un nombre massif de faux positifs (909 sur val). Utile pour le Recall brut, pas pour la Precision.

---

## 4. Modèle 2 — Logistic Regression `LR_smote`

### Choix du modèle
Même architecture que `LR_balanced` mais entraîné sur les données rééchantillonnées par SMOTE. Permet de comparer les deux stratégies de gestion du déséquilibre (class_weight vs SMOTE) à architecture constante.

### Stratégie de gestion du déséquilibre
Entraînement sur `X_smote` (ratio 1:10, 13 981 fraudes dont 13 800 synthétiques). `class_weight=None` car SMOTE gère déjà le déséquilibre. L'évaluation reste sur Val/Test à distribution naturelle (1:774).

### Paramètres choisis

| Paramètre | Valeur | Justification |
|---|---|---|
| `C` | 0.1 | Même régularisation que LR_balanced — comparaison à égalité |
| `max_iter` | 1000 | Convergence |
| `class_weight` | `None` | SMOTE gère le déséquilibre — double pondération serait redondant |
| **Temps d'entraînement** | **0.63 s** | |

### Résultats — seuil 0.5

| Split | Recall | Precision | F1 | PR-AUC | ROC-AUC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Validation | 0.8158 | 0.1890 | 0.3069 | 0.6565 | 0.9924 | 31 | 7 | 133 |
| Test | 0.7692 | 0.2113 | ~0.33 | — | — | — | — | — |

**Lecture :** SMOTE améliore nettement la Precision (19 % vs 4 % pour LR_balanced) au prix d'un léger Recall inférieur. Le F1 est 4× meilleur. Les faux positifs passent de 909 à 133.

---

## 5. Modèle 3 — Random Forest `RF_balanced`

### Choix du modèle
Le Random Forest capture des interactions **non-linéaires** entre les features que la régression logistique ne peut pas apprendre. Notamment, la combinaison `balance_diff_orig` + `is_transfer_or_cashout` + `dest_zero_balance` est une règle de décision non-linéaire typique d'une fraude TRANSFER. Le RF est aussi robuste aux features redondantes.

### Stratégie de gestion du déséquilibre
`class_weight='balanced'` — le RF assigne un poids de 386.74 à chaque nœud de décision pour les exemples de fraude.

### Paramètres choisis et justifications

| Paramètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 300 | Plus d'arbres = variance réduite. Au-delà de 300, le gain est marginal sur ce dataset |
| `max_depth` | 10 | Limite l'overfitting. Profondeur illimitée sur 181 fraudes = mémorisation |
| `min_samples_leaf` | 5 | Au moins 5 exemples par feuille — évite les splits sur bruit statistique |
| `class_weight` | `'balanced'` | Compensation du déséquilibre 1:774 |
| `random_state` | 42 | Reproductibilité |
| **Temps d'entraînement** | **11.32 s** | 300 arbres × profondeur 10 |

### Résultats — seuil 0.5

| Split | Recall | Precision | F1 | PR-AUC | ROC-AUC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Validation | 0.8158 | 0.3069 | 0.4460 | 0.7570 | 0.9962 | 31 | 7 | 70 |
| Test | 0.7949 | 0.3333 | ~0.47 | — | — | — | — | — |

**Lecture :** La PR-AUC de 0.757 confirme que le RF exploite bien les interactions non-linéaires. Precision meilleure que LR (31 % vs 4 %), faux positifs réduits à 70 (vs 909 pour LR_balanced).

---

## 6. Modèle 4 — Random Forest `RF_smote`

### Choix du modèle
RF + SMOTE est souvent la combinaison la plus performante pour les datasets très déséquilibrés : le rééchantillonnage enrichit les exemples de fraude à la frontière de décision, et le RF exploite ces exemples synthétiques pour affiner ses règles.

### Stratégie de gestion du déséquilibre
Entraînement sur `X_smote`. `class_weight=None` — le SMOTE suffit.

### Paramètres choisis

| Paramètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 300 | Cohérence avec RF_balanced |
| `max_depth` | 10 | Même régularisation — comparaison équitable |
| `min_samples_leaf` | 5 | Idem |
| `class_weight` | `None` | SMOTE gère le déséquilibre |
| **Temps d'entraînement** | **15.28 s** | Dataset SMOTE plus grand (153 799 vs 139 999 lignes) |

### Résultats — seuil 0.5

| Split | Recall | Precision | F1 | PR-AUC | ROC-AUC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Validation | 0.8684 | 0.5000 | 0.6346 | 0.8427 | 0.9980 | 33 | 5 | 33 |
| Test | 0.8462 | 0.5593 | ~0.67 | — | — | — | — | — |

**Lecture :** Meilleur F1 parmi les Random Forests (0.635 sur val). La Precision atteint 50 % (33 FP seulement) tout en maintenant un Recall de 87 %. PR-AUC de 0.843 — proche de XGBoost.

---

## 7. Modèle 5 — XGBoost `XGB_smote`

### Choix du modèle
XGBoost est l'algorithme de référence pour les données tabulaires déséquilibrées. Il combine boosting par gradient (amélioration itérative des erreurs), régularisation intégrée (L1/L2), et entraînement efficace sur GPU/CPU. Sa capacité à apprendre des interactions complexes entre features le rend supérieur au RF dans la plupart des benchmarks sur ce type de données.

### Stratégie de gestion du déséquilibre
Entraîné sur `X_smote`. `scale_pos_weight=1` car SMOTE a déjà équilibré les classes à 1:10 — doubler la pondération serait contre-productif.

### Paramètres choisis et justifications

| Paramètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 300 | Cohérence avec les autres modèles |
| `max_depth` | 6 | Plus faible que RF (10) car XGBoost utilise le boosting — des arbres plus profonds + boosting = sur-apprentissage plus rapide |
| `learning_rate` | 0.1 | Pas de gradient modéré — compromis vitesse/généralisation standard |
| `subsample` | 0.8 | 80 % des lignes par arbre — régularisation par sous-échantillonnage des lignes |
| `colsample_bytree` | 0.8 | 80 % des features par arbre — régularisation par sous-échantillonnage des colonnes |
| `scale_pos_weight` | 1 | SMOTE gère le déséquilibre — pas de pondération supplémentaire |
| `eval_metric` | `'aucpr'` | PR-AUC comme métrique d'évaluation — plus pertinente que logloss sur données déséquilibrées |
| `eval_set` | `[(X_val, y_val)]` | Suivi de la convergence sur val à chaque itération |
| `n_jobs` | -1 | Parallélisation sur tous les cœurs CPU |
| **Temps d'entraînement** | **10.34 s** | |

### Résultats — seuil 0.5

| Split | Recall | Precision | F1 | PR-AUC | ROC-AUC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Validation | 0.7895 | 0.8824 | 0.8333 | 0.9078 | 0.9992 | 30 | 8 | **4** |
| Test | 0.8205 | 0.8205 | ~0.82 | — | — | — | — | — |

**Lecture :** XGBoost est le seul modèle à atteindre une Precision > 80 % avec seuil=0.5. Seulement **4 faux positifs** sur 30 000 transactions de validation — résultat exceptionnel pour un modèle supervisé. PR-AUC de 0.908, le plus élevé de tous les modèles.

---

## 8. Modèle 6 — Isolation Forest `IsoForest`

### Choix du modèle
L'Isolation Forest est un modèle **non-supervisé** : il n'utilise pas les labels `isFraud` pendant l'entraînement. Il détecte les anomalies en isolant les points dans un espace de features via des arbres aléatoires — les points qui s'isolent rapidement (en peu de splits) sont considérés comme anormaux.

### Intérêt dans ce pipeline
- **Comparaison directe avec l'AutoEncoder** (notebook 05) : les deux modèles fonctionnent sans labels, ce qui est un argument fort pour la production (détection de nouveaux schémas de fraude inconnus)
- **Benchmarking non-supervisé** : fixe la cible minimale que l'AutoEncoder doit dépasser

### Paramètres choisis et justifications

| Paramètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 200 | 200 arbres d'isolation — bon compromis stabilité/vitesse pour ce dataset |
| `contamination` | 0.001293 (0.1293 %) | Taux de fraude réel mesuré dans le train — informe l'IF du pourcentage attendu d'anomalies |
| `random_state` | 42 | Reproductibilité |
| `n_jobs` | -1 | Parallélisation |
| **Temps d'entraînement** | **3.67 s** | Très rapide — pas de labels à traiter |

### Particularité de l'évaluation
`score_samples()` retourne des valeurs négatives (plus négatif = plus anomal). Une normalisation `MinMaxScaler` est appliquée pour produire des scores entre 0 et 1 comparables aux probabilités des autres modèles.

```
Scores val  : min=0.0005  max=0.9538
Scores test : min=0.0005  max=1.0145
```

### Résultats — seuil 0.5

| Split | Recall | Precision | F1 | PR-AUC | ROC-AUC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Validation | 0.8158 | 0.0155 | 0.0304 | 0.3484 | 0.9580 | 31 | 7 | **1 973** |
| Test | ~0.77 | ~0.015 | ~0.03 | — | — | — | — | — |

**Lecture :** Recall acceptable (82 %) mais Precision catastrophique (1.5 %) avec 1 973 faux positifs sur val. La PR-AUC de 0.348 est nettement inférieure aux modèles supervisés — attendu pour un modèle non-supervisé sur un dataset avec des features aussi discriminantes.

---

## 9. Analyse du seuil de décision optimal

### Ce qui est fait
Le seuil par défaut de 0.5 n'est pas optimal pour un dataset aussi déséquilibré. Pour chaque modèle, on cherche le seuil qui maximise le **F-beta (β=2)** sur la validation, puis on applique ce seuil au test.

### Choix de la métrique F-beta (β=2) pour la sélection du seuil
F-beta avec β=2 pondère le Recall **2× plus** que la Precision :

```
F_beta = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

**Justification métier :** Dans la détection de fraude financière pour PwC Tunisie, manquer une fraude réelle (Faux Négatif) est beaucoup plus coûteux qu'alerter sur une transaction légitime (Faux Positif). Un auditeur peut vérifier manuellement un FP en quelques minutes ; une fraude non détectée représente une perte financière réelle pour le client.

### Règle d'or : seuil sélectionné sur `X_val`, jamais sur `X_test`
Sélectionner le seuil sur le test set reviendrait à du **data snooping** — les métriques finales seraient artificiellement gonflées. Le seuil optimal est toujours cherché sur val, puis figé avant d'évaluer sur test.

### Seuils optimaux trouvés (sur validation)

| Modèle | Seuil optimal | F-beta2 (val) | Interprétation |
|---|---|---|---|
| `LR_balanced` | 0.9986 | 0.6436 | Seuil très élevé : le modèle génère des scores très étalés |
| `LR_smote` | 0.9621 | 0.6599 | Idem — scores concentrés près de 0 ou 1 |
| `RF_balanced` | 0.6235 | 0.7104 | Seuil proche de 0.5 — scores bien calibrés |
| `RF_smote` | 0.6291 | 0.8115 | Idem |
| `XGB_smote` | **0.3547** | **0.8505** | Seuil plus bas — XGBoost produit des scores conservateurs mais très discriminants |
| `IsoForest` | 0.7842 | 0.4427 | Seuil élevé pour réduire les 1 973 FP — résultat toujours médiocre |

---

## 10. Comparaison des 6 modèles

### Classement final — test set avec seuil optimal (F-beta2 sur val)

| Rang | Modèle | Seuil | Recall | Precision | F1 | PR-AUC | ROC-AUC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 1 | `XGB_smote` | 0.35 | **0.8462** | 0.8250 | **0.8354** | **0.8677** | **0.9975** | 33 | 6 | 6 |
| 🥈 2 | `RF_smote` | 0.63 | 0.7949 | 0.8158 | 0.8052 | 0.8405 | 0.9940 | 31 | 8 | 7 |
| 🥉 3 | `RF_balanced` | 0.62 | 0.7692 | **0.8333** | 0.8000 | 0.7794 | 0.9922 | 30 | 9 | 6 |
| 4 | `LR_smote` | 0.96 | 0.6923 | 0.7297 | 0.7105 | 0.7393 | 0.9949 | 27 | 12 | 10 |
| 5 | `LR_balanced` | 1.00 | 0.6410 | 0.6579 | 0.6494 | 0.7044 | 0.9961 | 25 | 14 | 13 |
| 6 | `IsoForest` | 0.78 | 0.3333 | 0.3611 | 0.3467 | 0.3408 | 0.9577 | 13 | 26 | 23 |

### Visualisations produites

| Fichier | Contenu |
|---|---|
| `12_pr_curves.png` | Courbes Precision-Recall des 6 modèles sur test |
| `13_roc_curves.png` | Courbes ROC des 6 modèles sur test |
| `14_confusion_matrices.png` | 6 matrices de confusion (2 lignes × 3 colonnes) |
| `15_feature_importances.png` | Importances Gini — RF_balanced |
| `16_threshold_analysis.png` | Recall/Precision/F1 vs seuil pour chaque modèle (6 graphiques) |
| `17_model_comparison.png` | Graphique comparatif récapitulatif — Recall, F1, PR-AUC |

### Enseignements clés de la comparaison

1. **SMOTE > class_weight** à architecture égale : RF_smote (F1=0.805) > RF_balanced (F1=0.800), LR_smote (F1=0.711) > LR_balanced (F1=0.649)
2. **XGBoost domine** tous les modèles supervisés : PR-AUC=0.868, F1=0.835, seulement 6 faux positifs sur test
3. **L'Isolation Forest est limité** avec des features métier explicites : les features dérivées (balance_diff_orig, is_transfer_or_cashout) contiennent déjà la logique de détection — un modèle non-supervisé ne tire pas parti de leur structure comme le font les modèles supervisés
4. **ROC-AUC élevé pour tous** (> 0.95) : le ROC-AUC est trompeur sur données déséquilibrées — il ne mesure pas la PR-AUC qui est la métrique pertinente ici

---

## 11. Feature Importances — Random Forest

### Ce qui est fait
Calcul des importances de features basées sur la **réduction d'impureté Gini** pour `RF_balanced` et `RF_smote`. Ces importances confirment que les features identifiées dans l'EDA sont bien exploitées par les modèles.

### Feature Importances — RF_balanced

| Rang | Feature | Importance Gini |
|---|---|---|
| 1 | `balance_diff_orig` | **0.3699** |
| 2 | `is_transfer_or_cashout` | 0.1171 |
| 3 | `dest_zero_balance` | 0.1152 |
| 4 | `log_amount` | 0.0965 |
| 5 | `hour` | 0.0737 |
| 6 | `step` | 0.0552 |
| 7 | `type_TRANSFER` | 0.0450 |
| 8 | `day` | 0.0389 |
| 9 | `high_risk_hour` | 0.0241 |
| 10 | `type_PAYMENT` | 0.0236 |
| 11 | `type_CASH_OUT` | 0.0235 |
| 12 | `week` | 0.0098 |
| 13 | `type_CASH_IN` | 0.0075 |
| 14 | `type_DEBIT` | 0.0001 |

### Feature Importances — RF_smote

| Rang | Feature | Importance Gini |
|---|---|---|
| 1 | `balance_diff_orig` | **0.3850** |
| 2 | `hour` | 0.1176 |
| 3 | `dest_zero_balance` | 0.1153 |
| 4 | `type_TRANSFER` | 0.0892 |
| 5 | `log_amount` | 0.0808 |
| 6 | `day` | 0.0534 |
| 7 | `type_CASH_OUT` | 0.0457 |
| 8 | `step` | 0.0363 |
| 9 | `is_transfer_or_cashout` | 0.0313 |
| 10 | `week` | 0.0175 |
| 11 | `type_CASH_IN` | 0.0112 |
| 12 | `high_risk_hour` | 0.0096 |
| 13 | `type_PAYMENT` | 0.0070 |
| 14 | `type_DEBIT` | 0.0000 |

### Analyse des importances

**Cohérence avec l'EDA :** `balance_diff_orig` domine avec ~38 % d'importance dans les deux RF, ce qui est cohérent avec sa corrélation EDA de 0.3662 — la plus forte de toutes les features.

**Différences RF_balanced vs RF_smote :**
- `hour` : 2e en RF_smote (0.118) vs 5e en RF_balanced (0.074). SMOTE amplifie les patterns temporels en générant des exemples synthétiques qui couvrent mieux les heures à risque.
- `is_transfer_or_cashout` : 2e en RF_balanced (0.117) vs 9e en RF_smote (0.031). Avec SMOTE, le type de transaction est capturé directement via `type_TRANSFER` (4e, 0.089).
- `type_DEBIT` : importance quasi nulle (0.000–0.0001) dans les deux — confirmé : aucune fraude de type DEBIT.

**Features candidates à supprimer en modélisation avancée :** `type_DEBIT`, `week` et `type_CASH_IN` ont une importance très faible et pourraient être retirées sans perte de performance. Elles sont conservées pour la cohérence du pipeline.

---

## 12. Sauvegarde des modèles et rapports

### Modèles sauvegardés (`outputs/models/`)

| Fichier | Taille | Format |
|---|---|---|
| `lr_balanced.pkl` | 2 Ko | joblib (wrapper + métadonnées) |
| `lr_smote.pkl` | 2 Ko | joblib (wrapper + métadonnées) |
| `rf_balanced.pkl` | 3 216 Ko | joblib (300 arbres) |
| `rf_smote.pkl` | 6 023 Ko | joblib (300 arbres sur SMOTE) |
| `xgb_smote.pkl` | 636 Ko | joblib (modèle XGBoost compilé) |
| `iso_forest.pkl` | 2 676 Ko | joblib (200 arbres d'isolation) |
| `iso_forest_scaler.pkl` | 1 Ko | MinMaxScaler pour les scores IF |

**Note :** RF_smote est 2× plus lourd que RF_balanced (6 023 Ko vs 3 216 Ko) car il a été entraîné sur 153 799 lignes (SMOTE) vs 139 999 — les arbres sont plus denses.

### Rapports JSON sauvegardés

| Fichier | Contenu |
|---|---|
| `baseline_report.json` | Rapport complet : hyperparamètres, métriques val/test, informations dataset, timestamp (2026-06-12T11:51:56) |
| `optimal_thresholds.json` | Seuils optimaux par modèle (F-beta2 sur val), versionné avec timestamp et métadonnées |

### Structure de `baseline_report.json`
```json
{
  "timestamp": "2026-06-12T11:51:56.864712",
  "source": "NB03_baseline_models.ipynb",
  "data_info": { "n_train": 139999, "n_fraud_train": 181, ... },
  "baseline_metier": { "recall": 0.0039, "precision": 1.0, "f1": 0.0077 },
  "models": [ { "name": "XGB_smote", "recall": 0.8462, "f1": 0.8354, ... }, ... ]
}
```

---

## 13. Synthèse finale

### Classement des modèles par catégorie

**Meilleur modèle supervisé : `XGB_smote`**
- Recall = 0.846, F1 = 0.835, PR-AUC = 0.868
- Seulement **6 faux positifs** sur 30 001 transactions de test
- Devient la **référence absolue** pour les modèles Deep Learning (AutoEncoder, LSTM)

**2e meilleur supervisé : `RF_smote`**
- Recall = 0.795, F1 = 0.805, PR-AUC = 0.841
- Performances très proches de XGBoost — utile comme fallback si XGBoost n'est pas disponible en production

**Meilleur modèle non-supervisé : `IsoForest`**
- Recall = 0.333, F1 = 0.347, PR-AUC = 0.341
- Fixe la **cible minimale** pour l'AutoEncoder du notebook 05

### Récapitulatif des performances vs baseline

| Modèle | Recall | F1 | vs Baseline Recall | vs Baseline F1 |
|---|---|---|---|---|
| Baseline isFlaggedFraud | 0.0039 | 0.0077 | — | — |
| IsoForest (t=0.78) | 0.3333 | 0.3467 | **×85** | **×45** |
| LR_balanced (t=1.00) | 0.6410 | 0.6494 | **×164** | **×84** |
| LR_smote (t=0.96) | 0.6923 | 0.7105 | **×177** | **×92** |
| RF_balanced (t=0.62) | 0.7692 | 0.8000 | **×197** | **×104** |
| RF_smote (t=0.63) | 0.7949 | 0.8052 | **×204** | **×105** |
| **XGB_smote (t=0.35)** | **0.8462** | **0.8354** | **×217** | **×108** |

### Cibles pour les notebooks suivants

| Notebook | Modèle | Cible minimale (Recall) | Cible ambitieuse |
|---|---|---|---|
| NB05 — AutoEncoder | Détection non-supervisée | > 0.333 (IsoForest) | > 0.795 (RF_smote) |
| NB05 — LSTM | Détection supervisée séquentielle | > 0.846 (XGB_smote) | > 0.90 |

### Points forts et limites de l'approche baseline

**Points forts :**
- XGBoost atteint un F1=0.835 avec seulement 6 FP sur 30 001 transactions — excellent pour un contexte d'audit où les faux positifs ont un coût d'investigation
- Les modèles supervisés exploitent efficacement les features métier (balance_diff_orig dominant à 37-38 %)
- La PR-AUC > 0.84 pour RF_smote et XGB_smote confirme un pouvoir discriminant très fort

**Limites :**
- Tous les modèles ont été entraînés sur les mêmes 181 fraudes — le faible nombre de positifs réels rend le pipeline sensible aux biais d'échantillonnage
- L'Isolation Forest performe mal avec des features métier explicites : ces features contiennent déjà la logique de détection que le modèle non-supervisé devrait découvrir seul
- Les schémas de fraude **inconnus** (nouveaux patterns non présents en entraînement) ne seront pas détectés par les modèles supervisés — justifie le développement de l'AutoEncoder
