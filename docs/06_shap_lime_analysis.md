# Notebook 06 — Explicabilité SHAP + LIME : Analyse détaillée

**Notebook :** `notebooks/06_shap_lime.ipynb`
**Objectif :** Expliquer les décisions de tous les modèles (LR, RF, XGBoost, AutoEncoder) en combinant SHAP (global + local) et LIME (local), puis construire les prompts enrichis qui alimenteront le LLM du notebook 07.

---

## Table des matières

1. [Pourquoi l'explicabilité est indispensable](#1-pourquoi-lexplicabilité-est-indispensable)
2. [SHAP — Principe et choix](#2-shap--principe-et-choix)
3. [LIME — Principe et choix](#3-lime--principe-et-choix)
4. [Relation SHAP + LIME → LLM](#4-relation-shap--lime--llm)
5. [Configuration & Imports](#5-configuration--imports)
6. [Chargement des données et modèles](#6-chargement-des-données-et-modèles)
7. [SHAP — Random Forest (TreeExplainer)](#7-shap--random-forest-treeexplainer)
8. [SHAP — Logistic Regression (LinearExplainer)](#8-shap--logistic-regression-linearexplainer)
9. [SHAP — AutoEncoder (KernelExplainer)](#9-shap--autoencoder-kernelexplainer)
10. [SHAP — XGBoost (TreeExplainer)](#10-shap--xgboost-treeexplainer)
11. [Comparaison SHAP tous modèles](#11-comparaison-shap-tous-modèles)
12. [Sélection de la transaction à expliquer](#12-sélection-de-la-transaction-à-expliquer)
13. [SHAP Waterfall — transaction spécifique](#13-shap-waterfall--transaction-spécifique)
14. [LIME — RF_smote](#14-lime--rf_smote)
15. [LIME — AutoEncoder](#15-lime--autoencoder)
16. [LIME — XGBoost](#16-lime--xgboost)
17. [SHAP vs LIME côte-à-côte](#17-shap-vs-lime-côte-à-côte)
18. [Dashboard complet + LLM](#18-dashboard-complet--llm)
19. [Sauvegarde des artefacts](#19-sauvegarde-des-artefacts)
20. [Synthèse finale](#20-synthèse-finale)

---

## 1. Pourquoi l'explicabilité est indispensable

Dans le contexte PwC Tunisie, les auditeurs ne peuvent pas accepter une décision de modèle comme une boîte noire. Trois raisons imposent l'explicabilité :

1. **Exigence réglementaire :** Les réglementations financières (IFRS, Bâle III, directives AMF) imposent que toute décision automatisée sur une transaction puisse être justifiée et auditée.
2. **Confiance de l'auditeur :** Un modèle qui dit "cette transaction est une fraude avec 88 % de probabilité" sans expliquer *pourquoi* ne sera pas utilisé en pratique. L'auditeur a besoin de comprendre quels signaux ont déclenché l'alerte.
3. **Détection des biais :** L'explicabilité permet de vérifier que le modèle ne fraude pas lui-même en s'appuyant sur des features incorrectes ou des artefacts du dataset.

---

## 2. SHAP — Principe et choix

### Qu'est-ce que SHAP ?

SHAP (SHapley Additive exPlanations) est basé sur la théorie des jeux coopératifs (valeurs de Shapley, 1953). Pour chaque prédiction, SHAP calcule la **contribution marginale** de chaque feature en mesurant comment la prédiction change lorsqu'on retire cette feature de toutes les combinaisons possibles de features.

**Propriété clé :** La somme des valeurs SHAP de toutes les features égale exactement la prédiction du modèle moins la prédiction moyenne (base value).

```
prédiction(x) = base_value + Σ shap_i(x)
```

### Pourquoi SHAP plutôt que d'autres méthodes ?

| Méthode | Cohérence théorique | Fidélité locale | Fidélité globale | Vitesse |
|---|---|---|---|---|
| **SHAP** | ✅ Axiomes de Shapley | ✅ Exacte | ✅ Agrégeable | Dépend du type |
| LIME | ❌ Approximation locale | ✅ Bonne localement | ❌ Inconsistante globalement | ✅ Rapide |
| Feature Importance Gini | ❌ Biaisée (features continues) | ❌ Globale seulement | ✅ | ✅ Très rapide |
| Permutation Importance | ✅ | ❌ | ✅ | ❌ Lente |

**SHAP est choisi ici pour :**
- Sa cohérence mathématique (additivité, efficacité, symétrie)
- Sa capacité à produire des explications **globales** (importance moyenne sur le dataset) ET **locales** (explication d'une transaction individuelle)
- Les explications globales SHAP sont directement réutilisables par le LLM pour contextualiser sa réponse

### Trois explainers SHAP selon le type de modèle

| Explainer | Modèle cible | Complexité | Précision |
|---|---|---|---|
| `TreeExplainer` | Random Forest, XGBoost, arbres | O(T × D × L) — très rapide | Exacte (pas d'approximation) |
| `LinearExplainer` | Logistic Regression, modèles linéaires | O(N × F) — très rapide | Exacte sur modèle linéaire |
| `KernelExplainer` | Tout modèle (boîte noire) | O(N × M²) — très lent | Approximation Monte-Carlo |

---

## 3. LIME — Principe et choix

### Qu'est-ce que LIME ?

LIME (Local Interpretable Model-agnostic Explanations) génère des explications **locales** : pour une transaction donnée, LIME :
1. Crée des perturbations artificielles autour de la transaction (en activant/désactivant des features)
2. Fait prédire le modèle cible sur ces perturbations
3. Entraîne un **modèle linéaire simple** (régression linéaire ou Ridge) sur ces perturbations, pondérées par leur proximité à la transaction originale
4. Les coefficients de ce modèle linéaire constituent l'explication

**Sortie :** Des règles de la forme `feature > seuil → poids` qui expliquent localement la décision du modèle.

### Pourquoi LIME en complément de SHAP ?

| Dimension | SHAP | LIME |
|---|---|---|
| **Granularité** | Valeur numérique de contribution | Règle interprétable en langage naturel |
| **Portée** | Locale ET globale | Locale uniquement |
| **Format pour LLM** | Score numérique → `"balance_diff_orig : +0.32"` | Règle → `"balance_diff_orig > 0.20"` |
| **Robustesse** | Stable (déterministe pour TreeExplainer) | Variable selon l'échantillonnage |
| **Interprétabilité auditeur** | Nécessite une explication | Directement lisible |

**La complémentarité SHAP + LIME est essentielle :** SHAP dit *combien* chaque feature contribue, LIME dit *dans quelles conditions* (règles à seuil). Le LLM reçoit les deux types d'information pour générer une explication plus riche et vérifiable.

---

## 4. Relation SHAP + LIME → LLM

C'est le cœur de l'architecture du notebook 06. Les explications SHAP et LIME ne sont pas produites pour être lues directement dans le notebook — elles sont **structurées en JSON** et injectées dans le prompt LLM du notebook 07.

### Pipeline d'explicabilité → LLM

```
Transaction suspecte
        │
        ├─── SHAP (RF, XGB)   → top 3 features avec valeur numérique et direction
        │                          {"feature": "balance_diff_orig", "shap_value": +0.32, "direction": "↑ risque fraude"}
        │
        ├─── LIME (RF, XGB)   → règles en langage naturel
        │                          "balance_diff_orig > 0.20 → +0.17"
        │
        ├─── AutoEncoder       → score d'anomalie brut + erreur par feature
        │                          {"ae_score": 313.65, "threshold": 1.75, "feat_errors": [...]}
        │
        └─── LLM (Groq/Ollama) → Explication en français pour l'auditeur PwC
                                   "Cette transaction présente un risque ÉLEVÉ..."
```

### Pourquoi le LLM a besoin de SHAP + LIME

Un LLM seul (sans SHAP/LIME) ne peut pas expliquer les décisions des modèles ML parce qu'il n'a pas accès aux poids appris. En lui fournissant les valeurs SHAP et les règles LIME :
- Le LLM dispose d'une **base factuelle vérifiable** pour construire son explication
- Il peut formuler une explication **causale** : "le compte émetteur a été vidé (balance_diff_orig élevé), ce qui est le principal signal de fraude selon le modèle"
- L'explication est **traçable** : l'auditeur peut vérifier les valeurs SHAP dans le JSON et les croiser avec l'explication textuelle

Les fichiers `shap_report.json`, `lime_report.json` et `shap_top_features.json` produits par ce notebook sont **directement importés par le notebook 07** pour alimenter les prompts LLM.

---

## 5. Configuration & Imports

### Ce qui est fait
Initialisation des bibliothèques, chemins, palette de couleurs, et dictionnaire de labels lisibles pour les 14 features.

### Versions des bibliothèques

| Bibliothèque | Version |
|---|---|
| SHAP | **0.51.0** |
| LIME | installée (version unknown) |

### Palette de couleurs standardisée

| Couleur | Code | Usage |
|---|---|---|
| Rouge | `#E74C3C` (POS) | Features qui augmentent le risque fraude |
| Bleu | `#2980B9` (NEG) | Features qui diminuent le risque fraude |
| Violet | `#8E44AD` (PURP) | AutoEncoder |

### Dictionnaire FEAT_LABELS
Mapping des noms techniques vers des labels lisibles en français pour les auditeurs :
```python
'balance_diff_orig'      → 'Diff. solde emetteur'
'is_transfer_or_cashout' → 'TRANSFER/CASH_OUT'
'dest_zero_balance'      → 'Dest. solde nul'
'high_risk_hour'         → 'Heure a risque'
'log_amount'             → 'Montant (log)'
```

### Features catégorielles chargées dynamiquement
Les 8 features binaires/one-hot sont chargées depuis `features.json` (pas codées en dur) :
```
['dest_zero_balance', 'high_risk_hour', 'is_transfer_or_cashout',
 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
```

---

## 6. Chargement des données et modèles

### Ce qui est fait
Chargement de tous les modèles entraînés (NB04 + NB05), extraction des estimateurs sklearn bruts depuis les wrappers custom, et chargement des scores AutoEncoder.

### Datasets chargés

| Dataset | Shape | Fraudes | Usage |
|---|---|---|---|
| `X_train` | (139 999, 14) | 181 | Background SHAP (300 transactions normales aléatoires) |
| `X_test` | (30 001, 14) | 39 | Calcul SHAP (500 premières), explication LIME (1 transaction) |

### Modèles chargés et types

| Variable | Type sklearn | Seuil |
|---|---|---|
| `rf_smote_raw` | `RandomForestClassifier` | 0.6291 |
| `lr_smote_raw` | `LogisticRegression` | 0.9621 |
| `xgb_smote` | `XGBClassifier` | 0.3547 |
| `ae` | `FraudAutoEncoder` (PyTorch) | **1.7530** |

**Extraction des estimateurs bruts depuis les wrappers :** Les classes `FraudRandomForest` et `FraudLogisticRegression` encapsulent les modèles sklearn. SHAP a besoin du modèle sklearn natif. Une fonction `get_raw()` extrait l'attribut `model` ou `estimator_` du wrapper.

---

## 7. SHAP — Random Forest (TreeExplainer)

### Choix de l'explainer : TreeExplainer

Le `TreeExplainer` est l'explainer SHAP optimal pour les modèles basés sur des arbres (Random Forest, XGBoost). Il exploite la structure arborescente pour calculer des valeurs SHAP **exactes** (pas d'approximation) en temps polynomial — O(T × D × L) où T = nombre d'arbres, D = profondeur, L = nombre de feuilles.

Pour RF_smote (300 arbres, profondeur 10) : calcul en **7.1 secondes** sur 500 transactions.

### Paramètres d'exécution

| Paramètre | Valeur | Justification |
|---|---|---|
| Background | 300 transactions normales aléatoires | Référence pour calculer l'impact marginal de chaque feature |
| Transactions expliquées | 500 premières de X_test | Compromis représentativité / temps de calcul |
| `check_additivity` | False | Évite les erreurs numériques mineures sur les arbres non-pondérés |
| Classe retenue | Classe 1 (fraude) | Pour classification binaire, SHAP retourne [classe0, classe1] — on prend classe1 |

### Résultats SHAP RF_smote

**Top 5 features par importance SHAP moyenne (|valeur| sur 500 transactions) :**

| Rang | Feature | |SHAP| moyen | Interprétation |
|---|---|---|---|---|
| 1 | `balance_diff_orig` | **0.0440** | Le vide du compte source est le signal dominant |
| 2 | `type_TRANSFER` | 0.0236 | Le type TRANSFER est fortement associé aux fraudes |
| 3 | `type_CASH_OUT` | 0.0194 | Le type CASH_OUT contribue aussi significativement |
| 4 | `dest_zero_balance` | 0.0159 | Compte destination vide = signal de compte mule |
| 5 | `hour` | 0.0140 | L'heure de la transaction a un impact mesurable |

**Valeur moyenne globale :** `|SHAP| moyen = 0.0115`

### Graphiques produits

**Figure 24 — Bar importances SHAP RF_smote :** Graphique barres horizontales, top 10 features triées par importance SHAP décroissante. `balance_diff_orig` domine avec 0.0440, presque 2× le suivant.

**Figure 25 — Beeswarm SHAP RF_smote :** Chaque point représente une transaction. La couleur indique la valeur de la feature (rouge = valeur élevée, bleu = faible). La position horizontale indique la contribution SHAP (droite = augmente le risque fraude). Ce graphique révèle la **direction** de l'impact : `balance_diff_orig` élevé → forte probabilité fraude (points rouges à droite).

---

## 8. SHAP — Logistic Regression (LinearExplainer)

### Choix de l'explainer : LinearExplainer

Le `LinearExplainer` calcule des valeurs SHAP analytiquement pour les modèles linéaires. Pour une régression logistique, les valeurs SHAP sont proportionnelles aux coefficients × (valeur_feature − moyenne_background). Calcul en **0.1 seconde** (instantané).

### Résultats SHAP LR_smote

**|SHAP| moyen LR = 0.5975** — nettement plus élevé que RF (0.0115). Cela reflète l'échelle différente des probabilités logistiques (espace log-odds) vs probabilités RF (espace [0,1] directement).

**Top 5 features :**

| Rang | Feature | |SHAP| moyen | Interprétation |
|---|---|---|---|---|
| 1 | `type_CASH_OUT` | **2.8956** | La régression logistique pondère fortement le type CASH_OUT |
| 2 | `log_amount` | 1.0191 | Le montant (log) a un poids élevé dans le modèle linéaire |
| 3 | `balance_diff_orig` | 0.9964 | Cohérent avec RF, signal fort |
| 4 | `type_TRANSFER` | 0.8405 | Important également |
| 5 | `type_CASH_IN` | 0.4883 | Négativement corrélé (CASH_IN = pas de fraude) |

**Différence clé RF vs LR :** La LR place `type_CASH_OUT` en #1 (impact linéaire direct du coefficient), tandis que RF place `balance_diff_orig` en #1 (interaction non-linéaire avec d'autres features). Cela illustre que la LR capture des relations linéaires plus simples, RF des interactions plus complexes.

**Figure 26 :** Bar importances SHAP LR_smote (bleu), top 10 features.

---

## 9. SHAP — AutoEncoder (KernelExplainer)

### Pourquoi KernelExplainer et non DeepExplainer

Le `DeepExplainer` SHAP est conçu pour Keras/TensorFlow uniquement. L'AutoEncoder de ce projet est en **PyTorch** — `DeepExplainer` n'est pas compatible. Le `KernelExplainer` est l'alternative universelle (model-agnostic).

### Complexité et limitation du KernelExplainer

```
Complexité : O(N × M²) où N = transactions expliquées, M = taille du background
Ici : N=50, M=50, nsamples=200 → ~2-5 minutes acceptable en notebook
⚠️ INUTILISABLE en production temps réel
```

**Choix pour la production :** En production (API FastAPI), les erreurs de reconstruction par feature sont utilisées directement comme proxy d'importance sans SHAP. Ce proxy est implémenté dans `app/services/explainer.py` comme fonction dédiée :

```python
# app/services/explainer.py
def compute_ae_feature_errors(tx_arr, ae, feature_cols) -> dict[str, float]:
    """Proxy AE : |x - AE(x)| par feature. Remplace KernelExplainer."""
    recon = ae.model(tx_tensor).cpu().numpy()[0]
    errors = np.abs(tx_arr - recon)
    return {feat: round(float(e), 6) for feat, e in zip(feature_cols, errors)}
```

La réponse de `GET /api/explain/{tx_id}` expose ce proxy sous deux champs distincts :
- `ae_feature_errors` : dict complet des 14 erreurs de reconstruction
- `ae_top_features` : top-3 features triées par erreur décroissante (directement lisible par l'auditeur)

Ces valeurs sont séparées des `shap_values_xgb` (voir section 10) pour éviter toute confusion : les SHAP expliquent XGBoost, les erreurs AE expliquent l'AutoEncoder.

### Paramètres KernelExplainer

| Paramètre | Valeur | Justification |
|---|---|---|
| Background | 50 transactions aléatoires du train | M petit pour réduire la complexité O(M²) |
| Transactions expliquées | 50 max | N limité pour le même motif |
| Fonction cible | `ae.predict_score()` [0,1] | Score normalisé cohérent pour la comparaison |
| `nsamples` | 200 | Nombre d'évaluations Monte-Carlo par transaction |

### Résultats SHAP AutoEncoder

**Top 5 features :**

| Rang | Feature | |SHAP| moyen |
|---|---|---|
| 1 | `hour` | **0.0556** |
| 2 | `balance_diff_orig` | 0.0266 |
| 3 | `log_amount` | 0.0231 |
| 4 | `week` | 0.0213 |
| 5 | `step` | 0.0090 |

**Analyse :** L'AutoEncoder place `hour` (heure de la transaction) en #1, contrairement aux modèles supervisés qui placent `balance_diff_orig`. Cela révèle que l'AutoEncoder a appris une **structure temporelle forte** dans les transactions normales : certaines heures ont des patterns très distinctifs, et les transactions frauduleuses surviennent souvent à des heures inhabituelles pour les patterns normaux.

**Figure 27 :** Bar importances SHAP AutoEncoder (violet), top 10 features.

---

## 10. SHAP — XGBoost (TreeExplainer)

XGBoost utilise également le `TreeExplainer` (même classe que RF mais algorithme différent). Calcul sur 500 transactions, shape résultante : `(500, 14)`.

Les valeurs SHAP XGBoost sont utilisées dans la section de cohérence SHAP vs LIME pour la transaction frauduleuse.

---

## 11. Comparaison SHAP tous modèles

### Ce qui est fait
Production d'une figure comparative (3 graphiques côte-à-côte) montrant les top 8 features SHAP pour RF_smote, LR_smote et AutoEncoder.

### Top 5 par modèle — synthèse

| Rang | RF_smote | LR_smote | AutoEncoder |
|---|---|---|---|
| 1 | `balance_diff_orig` (0.0440) | `type_CASH_OUT` (2.8956) | `hour` (0.0556) |
| 2 | `type_TRANSFER` (0.0236) | `log_amount` (1.0191) | `balance_diff_orig` (0.0266) |
| 3 | `type_CASH_OUT` (0.0194) | `balance_diff_orig` (0.9964) | `log_amount` (0.0231) |
| 4 | `dest_zero_balance` (0.0159) | `type_TRANSFER` (0.8405) | `week` (0.0213) |
| 5 | `hour` (0.0140) | `type_CASH_IN` (0.4883) | `step` (0.0090) |

### Enseignements de la comparaison

**Consensus entre modèles :** `balance_diff_orig` apparaît dans le top 3 de tous les modèles — c'est la feature la plus universellement informative, cohérente avec la corrélation EDA de 0.3662.

**Divergences révélatrices :**
- **LR_smote** surpondère `type_CASH_OUT` (2.89) car la régression logistique ne peut pas apprendre que l'effet de CASH_OUT est conditionnel à la valeur de `balance_diff_orig` — elle capture une corrélation marginale.
- **AutoEncoder** place `hour` en #1 : le modèle non-supervisé a appris que les patterns temporels des transactions normales sont réguliers et que les fraudes brisent cette régularité.
- **RF_smote** trouve un meilleur équilibre : il capture les interactions `balance_diff_orig × type_TRANSFER × dest_zero_balance` que la LR ne peut pas voir.

**Figure 28 :** 3 graphiques horizontaux côte-à-côte, un par modèle.

---

## 12. Sélection de la transaction à expliquer

### Critère de sélection : TP avec score AutoEncoder le plus élevé
La transaction retenue pour l'explication locale SHAP + LIME est le **vrai positif (TP)** avec le score AutoEncoder maximal parmi toutes les transactions du test set correctement détectées par l'AutoEncoder.

```python
tp_mask = (ae_scores_test >= ae.threshold) & (y_test_arr == 1)
tx_idx  = tp_idx[np.argmax(ae_scores_test[tp_idx])]
```

**Justification du critère :** Choisir le TP avec le score AE le plus élevé garantit une transaction que les deux types de modèles (supervisé ET non-supervisé) détectent avec confiance maximale — la meilleure candidate pour une explication riche et cohérente entre méthodes.

### Transaction sélectionnée (idx=29340)

| Attribut | Valeur |
|---|---|
| Index | **29 340** |
| Label réel | **1 (fraude confirmée)** |
| Score AutoEncoder (MSE brute) | **313.65** (max ≈ 314 — anomalie extrême) |
| Probabilité RF_smote | **0.9880** (99 % de confiance fraude) |

### Features de la transaction (valeurs standardisées)

| Feature | Valeur standardisée | Interprétation |
|---|---|---|
| `is_transfer_or_cashout` | 1.0000 | Transaction de type TRANSFER ou CASH_OUT |
| `balance_diff_orig` | **66.0686** | +66 écarts-types au-dessus de la moyenne — compte émetteur entièrement vidé |
| `dest_zero_balance` | 1.0000 | Compte destination avec balance 0 avant et après |
| `type_TRANSFER` | 1.0000 | C'est un TRANSFER |
| `type_CASH_OUT` | 0.0000 | Pas un CASH_OUT |
| `log_amount` | 2.9061 | Montant relativement modeste (log-scale) |
| `high_risk_hour` | 0.0000 | Pas une heure à risque élevé (contre-intuitif) |

**Profil de la fraude :** TRANSFER à une heure normale mais avec un vide total du compte source (`balance_diff_orig = 66.07`) vers un compte destination "mule" (balance 0/0). L'AE score de 313.65 (sur un max de ~314) indique que cette transaction est la plus anormale du dataset de test.

---

## 13. SHAP Waterfall — transaction spécifique

### Ce qui est fait
Calcul des valeurs SHAP individuelles pour la transaction idx=29340 avec RF_smote, affichage sous forme de graphique waterfall.

### Valeurs SHAP pour cette transaction (RF_smote)

| Rang | Feature | Valeur SHAP | Valeur feature | Interprétation |
|---|---|---|---|---|
| 1 | Diff. solde emetteur | **+0.3205** | 66.07 | Vide total du compte → signal fraude fort |
| 2 | Dest. solde nul | **+0.2350** | 1.0 | Compte mule détecté → signal fraude fort |
| 3 | TRANSFER | **+0.1143** | 1.0 | Type TRANSFER → associé aux fraudes |
| 4 | Montant (log) | **+0.0974** | 2.91 | Montant contribue positivement |
| 5 | Jour | **+0.0736** | 3.46 | Jour particulier → signal mineur |
| 6 | CASH_OUT | **−0.0519** | 0.0 | Absence de CASH_OUT → réduit le risque |
| 7 | TRANSFER/CASH_OUT | **+0.0480** | 1.0 | Feature binaire confirme |
| 8 | Etape temporelle | **+0.0386** | 3.41 | Step contribue positivement |
| 9 | CASH_IN | **+0.0154** | 0.0 | Absence de CASH_IN → signal mineur |
| 10 | Heure | **−0.0133** | −1.23 | Heure dans une plage normale → réduit le risque |

**Lecture du waterfall :** Les 8 premières features poussent la probabilité de fraude vers le haut (rouge, valeurs positives). Seules `CASH_OUT` (absent) et `Heure` (normale) réduisent légèrement le score. La contribution cumulée est dominée par `balance_diff_orig` (+0.32) et `dest_zero_balance` (+0.23).

**Figure 31 :** Waterfall horizontal, rouge pour les features qui augmentent le risque, bleu pour celles qui le réduisent.

---

## 14. LIME — RF_smote

### Ce qui est fait
Explication locale de la transaction idx=29340 avec RF_smote via le module `LIMEExplainer` custom (`src/explainability/lime_explainer.py`).

### Paramètres LIME RF_smote

| Paramètre | Valeur | Justification |
|---|---|---|
| `mode` | `'classification'` | RF prédit P(fraude) ∈ [0,1] |
| `num_features` | 10 | Top 10 features dans l'explication |
| `num_samples` | 1000 | 1000 perturbations — 3× plus rapide que 3000 avec qualité similaire pour 14 features |
| `label` | 1 | Explication pour la classe fraude |
| Temps d'exécution | **0.1 s** | LIME très rapide en classification |

### Résultats LIME RF_smote

**Prédiction locale = 0.7172** (le modèle linéaire local prédit 72 % de probabilité fraude pour cette transaction)

**Règles LIME et poids (top 10) :**

| Règle | Poids | Direction |
|---|---|---|
| `balance_diff_orig > 0.20` | **+0.1722** | ↑ risque fraude |
| `dest_zero_balance = 1` | **+0.1437** | ↑ risque fraude |
| `type_TRANSFER = 1` | **+0.1068** | ↑ risque fraude |
| `day > 0.76` | **+0.1001** | ↑ risque fraude |
| `hour <= −0.77` | **+0.0895** | ↑ risque fraude |
| `type_CASH_OUT = 0` | **−0.0884** | ↓ risque fraude (absence) |
| `is_transfer_or_cashout = 1` | **+0.0621** | ↑ risque fraude |
| `week > 1.15` | **+0.0476** | ↑ risque fraude |
| `high_risk_hour = 0` | **+0.0361** | ↑ risque fraude |
| `type_PAYMENT = 0` | **+0.0361** | ↑ risque fraude |

**Avantage des règles LIME :** Contrairement aux valeurs SHAP numériques, les règles LIME sont directement lisibles par un auditeur non-technique. "Le compte émetteur a un solde différentiel > 0.20 et le compte destination a une balance nulle" est une phrase que l'auditeur peut vérifier dans les données brutes.

**Figure 29 :** Waterfall LIME RF_smote, rouge pour les règles qui augmentent le risque, bleu pour celles qui le réduisent.

---

## 15. LIME — AutoEncoder

### Ce qui est fait
Explication locale de la même transaction avec l'AutoEncoder via LIME en mode régression.

### Paramètres LIME AutoEncoder

| Paramètre | Valeur | Justification |
|---|---|---|
| `mode` | `'regression'` | L'AE prédit un score continu (anomalie [0,1]) et non une classe |
| `predict_fn` | `ae.predict_score()` [0,1] | Score normalisé, cohérent avec le mode régression |
| `num_samples` | 500 | Réduit à 500 (vs 1000 pour RF) car l'AE est plus lent |
| Temps d'exécution | **0.0 s** | Instantané |

### Résultats LIME AutoEncoder

**Prédiction locale = 0.1962** (le modèle linéaire local prédit un score d'anomalie de 0.196 sur [0,1])

**Figure 30 :** Waterfall LIME AutoEncoder (violet).

**Analyse comparative LIME RF vs LIME AE :**
- RF prédit localement 0.717 (très confiant en fraude)
- AE prédit localement 0.196 (score modéré)
- La différence s'explique : RF_smote a été entraîné directement sur des fraudes, il reconnaît ce pattern avec haute confiance. L'AE prédit via l'erreur de reconstruction — 0.196 correspond à un score normalisé, mais le score brut AE de 313.65 est en réalité extrêmement élevé (cf. section 12).

---

## 16. LIME — XGBoost

### Résultats LIME XGBoost

**Prédiction locale = 0.8861** (88.6 % de probabilité fraude)

LIME XGBoost est plus confiant que RF (0.717) — cohérent avec les métriques de NB04 où XGBoost avait le meilleur PR-AUC.

**Figure 32 :** Waterfall LIME XGBoost.

---

## 17. SHAP vs LIME côte-à-côte

### Ce qui est fait
Visualisation comparative SHAP (RF_smote) vs LIME (RF_smote) sur la même transaction, côte-à-côte.

### Analyse de cohérence SHAP vs LIME (XGBoost, transaction frauduleuse)

**Top 3 features SHAP (XGBoost) :**

| Feature | Valeur SHAP | Direction |
|---|---|---|
| `dest_zero_balance` | **+6.4968** | ↑ risque fraude |
| `type_TRANSFER` | **+4.9656** | ↑ risque fraude |
| `balance_diff_orig` | **−3.5381** | ↓ risque fraude |

**Top 3 règles LIME (XGBoost) :**

| Règle | Poids | Direction |
|---|---|---|
| `balance_diff_orig > 0.20` | **+0.6215** | ↑ risque |
| `dest_zero_balance = 1` | **+0.1059** | ↑ risque |
| `−0.74 < log_amount <= 0.21` | **+0.0734** | ↑ risque |

**Chevauchement SHAP ∩ LIME :**
```
Features communes (top 3) : {'dest_zero_balance', 'balance_diff_orig'}
Chevauchement : 2/3 features communes ✅ → Cohérence bonne
```

**Interprétation :** SHAP et LIME identifient les mêmes features clés (`balance_diff_orig` et `dest_zero_balance`) mais avec des signes différents pour SHAP XGBoost. Cette apparente contradiction s'explique : SHAP calcule la contribution marginale dans le contexte global du modèle (effets d'interaction inclus), tandis que LIME approche localement avec un modèle linéaire. Pour XGBoost, `balance_diff_orig = 66.07` peut avoir un effet négatif sur le log-odds global (valeur extrême en dehors de la distribution d'entraînement) tout en étant positivement corrélé localement avec la fraude selon LIME.

**Figure 33 :** Deux graphiques côte-à-côte — gauche SHAP (rouge), droite LIME (bleu).

---

## 18. Dashboard complet + LLM

### Ce qui est fait
Construction d'un dashboard intégré combinant SHAP, LIME, erreurs de reconstruction AE par feature, et explication LLM, avec appel au LLM Groq.

### Erreurs de reconstruction par feature (AE)

Pour la transaction idx=29340 :
```python
tx_rec = AutoEncoder(tx)           # reconstruction
feat_errs = |tx - tx_rec|          # erreur absolue par feature (14 valeurs)
```

Ces erreurs sont passées au LLM comme information complémentaire : "quelles features l'AutoEncoder a-t-il le plus de mal à reconstruire ?" — ce sont les features les plus anormales selon le modèle non-supervisé.

### Appel LLM (Groq)

| Paramètre | Valeur |
|---|---|
| Provider | **Groq** |
| Timeout | 90 s |
| Entrées | `transaction`, `feature_errors` (AE), `ae_score`, `threshold` |
| Résultat | `status: ok`, `risk_level: ÉLEVÉ` |

**Figure 34 — Dashboard complet (5 panneaux) :**
1. **Top-left** : SHAP waterfall RF_smote (contributions par feature)
2. **Top-right** : LIME RF_smote (règles interprétables)
3. **Mid-left** : Erreurs AE par feature (proxy d'importance non-supervisé)
4. **Mid-right** : Scores comparatifs (RF=0.988, AE=313.65, seuil=1.75)
5. **Bottom** : Explication LLM en français (Risque ÉLEVÉ)

---

## 19. Sauvegarde des artefacts

### Ce qui est fait
Export de tous les résultats en JSON et NPY pour réutilisation par le notebook 07 (LLM).

### Fichiers produits

**Rapports JSON (`outputs/reports/`) :**

| Fichier | Contenu |
|---|---|
| `shap_report.json` | Importances SHAP moyennes par feature pour RF_smote, LR_smote, AutoEncoder (toutes triées par |SHAP|) |
| `lime_report.json` | Transaction idx=29340 : scores RF/AE, prédictions locales LIME, top 5 règles RF, explication LLM |
| `shap_top_features.json` | Top 3 features SHAP (XGBoost + RF) pour les 10 premières transactions frauduleuses détectées — utilisé par le LLM |

**Valeurs SHAP brutes (`outputs/models/`) :**

| Fichier | Shape | Usage |
|---|---|---|
| `shap_values_rfsmote.npy` | (500, 14) | Analyses complémentaires, audit |
| `shap_values_lrsmote.npy` | (500, 14) | Idem |
| `shap_values_autoencoder.npy` | (50, 14) | Idem (50 transactions seulement — KernelExplainer) |

### Figures produites

| Figure | Contenu |
|---|---|
| `24_shap_rf_smote_bar.png` | Bar importances SHAP RF_smote — top 10 |
| `25_shap_rf_smote_beeswarm.png` | Beeswarm SHAP RF_smote — distribution des contributions |
| `26_shap_lr_smote_bar.png` | Bar importances SHAP LR_smote |
| `27_shap_ae_bar.png` | Bar importances SHAP AutoEncoder (KernelExplainer) |
| `28_shap_comparison_all.png` | Comparaison côte-à-côte RF / LR / AE |
| `29_lime_rf.png` | Waterfall LIME RF_smote — transaction idx=29340 |
| `30_lime_ae.png` | Waterfall LIME AutoEncoder |
| `31_shap_waterfall_tx.png` | SHAP waterfall transaction spécifique (RF_smote) |
| `32_lime_xgb.png` | Waterfall LIME XGBoost |
| `33_shap_vs_lime.png` | SHAP vs LIME côte-à-côte (RF_smote) |
| `34_dashboard.png` | Dashboard complet 5 panneaux (SHAP + LIME + AE + LLM) |

---

## 20. Synthèse finale

### Résultats SHAP par modèle

| Modèle | Feature #1 | Feature #2 | Feature #3 | Explainer | Temps |
|---|---|---|---|---|---|
| **RF_smote** | `balance_diff_orig` (0.044) | `type_TRANSFER` (0.024) | `type_CASH_OUT` (0.019) | TreeExplainer | 7.1 s |
| **LR_smote** | `type_CASH_OUT` (2.896) | `log_amount` (1.019) | `balance_diff_orig` (0.996) | LinearExplainer | 0.1 s |
| **AutoEncoder** | `hour` (0.056) | `balance_diff_orig` (0.027) | `log_amount` (0.023) | KernelExplainer | ~5 min |
| **XGBoost** | (via top_features) | `dest_zero_balance` | `type_TRANSFER` | TreeExplainer | — |

### Résultats LIME — transaction idx=29340

| Modèle | Prédiction locale | Top règle | 2e règle |
|---|---|---|---|
| **RF_smote** | **0.7172** | `balance_diff_orig > 0.20` (+0.172) | `dest_zero_balance = 1` (+0.144) |
| **AutoEncoder** | 0.1962 | (score normalisé) | — |
| **XGBoost** | **0.8861** | `balance_diff_orig > 0.20` (+0.622) | `dest_zero_balance = 1` (+0.106) |

### Cohérence inter-méthodes

| Paire | Chevauchement top 3 | Verdict |
|---|---|---|
| SHAP XGB vs LIME XGB | 2/3 (`dest_zero_balance`, `balance_diff_orig`) | ✅ Cohérence bonne |
| SHAP RF vs Feature Importance Gini RF (NB04) | Feature #1 identique (`balance_diff_orig`) | ✅ Cohérence parfaite |
| RF vs LR (SHAP) | `balance_diff_orig` top 3 dans les deux | ✅ Consensus |
| AE vs RF (SHAP) | `balance_diff_orig` top 2 dans les deux | ✅ Consensus partiel |

### LLM — Décision finale

```
Transaction idx=29340 : LLM status=ok | Risque=ÉLEVÉ
```

Le LLM Groq a reçu les valeurs SHAP, les règles LIME et les erreurs AE, et a produit une explication en français classant la transaction à risque ÉLEVÉ — cohérent avec RF=0.988 et AE=313.65.

### Décisions impactant le notebook 07

1. **`shap_top_features.json`** → injecté dans les prompts LLM du NB07 pour chaque transaction suspecte
2. **`shap_report.json`** → contexte global du modèle fourni au LLM (quelles features comptent globalement)
3. **`lime_report.json`** → règles interprétables pour la transaction analysée, incluses dans le prompt
4. **`feat_errs` (AE)** → proxy d'importance non-supervisé pour alimenter le LLM sans KernelExplainer — encapsulé en production dans `compute_ae_feature_errors()` (`app/services/explainer.py`)
5. **Connexion LLM Groq testée** → confirmée fonctionnelle (status=ok), utilisée massivement en NB07

### Décisions impactant l'API FastAPI (backend)

| Décision NB06 | Déploiement backend |
|---------------|---------------------|
| KernelExplainer AE → trop lent | Remplacé par `compute_ae_feature_errors()` — résultat exposé dans `ae_feature_errors` + `ae_top_features` |
| SHAP XGB via TreeExplainer | Déployé dans `compute_shap()`, exposé sous `shap_values_xgb` (nom explicite — pas de confusion avec l'AE) |
| LinearExplainer LR non déployé | Documenté comme hors-scope API : `/api/explain` expose uniquement XGB et AE ; LR non exposé |
| LIME RF/XGB | Déployé dans `compute_lime()`, inclus dans `GET /api/explain/{tx_id}`, désactivé dans `POST /api/explain/batch` (performance) |

### Choix architecturaux clés

| Décision | Choix | Alternative rejetée | Raison |
|---|---|---|---|
| Explainer AE (notebook) | KernelExplainer | DeepExplainer | DeepExplainer = Keras uniquement |
| Explainer AE (production) | `compute_ae_feature_errors()` — `\|x-AE(x)\|` | KernelExplainer | KernelExplainer trop lent (O(N×M²)) ; proxy instantané et suffisant pour le LLM |
| Explainer RF | TreeExplainer | KernelExplainer | TreeExplainer exact et 50× plus rapide |
| Explainer LR | LinearExplainer | KernelExplainer | LinearExplainer analytique et instantané |
| SHAP en production | XGB uniquement (`shap_values_xgb`) | RF / LR / AE | Performance : TreeExplainer XGB < 1s ; exposer RF ou LR doublerait le temps de réponse sans apport pour l'auditeur |
| LIME num_samples | 1000 | 3000 | 3× plus rapide, qualité similaire sur 14 features |
| LIME en batch | Désactivé | Activé | LIME est la partie la plus lente — inacceptable sur 20+ transactions en production |
| Transaction cible | TP score AE max | Première fraude du test | Maximise la richesse de l'explication |
| Format sortie LLM | JSON structuré | Texte libre | Parseable, reproductible, auditeur peut vérifier |
