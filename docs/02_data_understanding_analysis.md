# Notebook 02 — Data Understanding : Analyse détaillée

**Notebook :** `notebooks/02_data_understanding.ipynb`
**Objectif :** Exploration complète du jeu de données transactionnel, qualification de sa qualité, identification des patterns liés aux fraudes, et construction des features dérivées à utiliser dans la modélisation.

---

## Table des matières

1. [Configuration & Imports](#1-configuration--imports)
2. [Chargement des données](#2-chargement-des-données)
3. [Vue générale des données brutes](#3-vue-générale-des-données-brutes)
4. [Qualité des données](#4-qualité-des-données)
5. [Analyse du déséquilibre de classes](#5-analyse-du-déséquilibre-de-classes)
6. [Baseline — règle métier isFlaggedFraud](#6-baseline--règle-métier-isflaggedfraud)
7. [Distribution des types de transaction](#7-distribution-des-types-de-transaction)
8. [Distribution des montants](#8-distribution-des-montants)
9. [Analyse temporelle](#9-analyse-temporelle)
10. [Analyse réseau nameOrig / nameDest](#10-analyse-réseau-nameorig--namedest)
11. [Analyse par client — nameOrig](#11-analyse-par-client--nameorig)
12. [Feature Engineering exploratoire](#12-feature-engineering-exploratoire)
13. [Vérification du Data Leakage](#13-vérification-du-data-leakage)
14. [Sauvegarde du rapport EDA](#14-sauvegarde-du-rapport-eda)
15. [Synthèse des décisions](#15-synthèse-des-décisions)

---

## 1. Configuration & Imports

### Ce qui est fait
Résolution dynamique du répertoire racine du projet (compatible exécution depuis `project_root/` ou depuis `notebooks/`), import des bibliothèques, initialisation des chemins de sortie, et paramétrage visuel global.

### Choix techniques
- **Résolution dynamique du `project_root`** : le notebook vérifie si `data/` existe au niveau courant ou un niveau au-dessus, rendant le code robuste quel que soit le répertoire de lancement.
- **Style visuel unifié** : `seaborn whitegrid`, DPI 120, taille de police 13 pour les titres et 11 pour les axes. Ce choix garantit une cohérence visuelle dans tous les graphiques du notebook et facilite leur intégration dans les rapports.
- **Graine aléatoire fixée à 42** : assure la reproductibilité totale de l'échantillonnage et des résultats entre exécutions.
- **Taille d'échantillon : 200 000 lignes** : compromis entre représentativité statistique et vitesse d'exécution. Le dataset complet contient ~6,3 M lignes, trop lourd pour une EDA interactive.

### Bibliothèques importées
`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn.model_selection.train_test_split`, modules internes `src.preprocessing.data_loader` et `src.preprocessing.feature_engineering`.

---

## 2. Chargement des données

### Ce qui est fait
Chargement du dataset complet en mémoire, puis extraction d'un échantillon stratifié de 200 000 lignes préservant la proportion exacte de fraudes. Un hash SHA-256 est calculé pour l'échantillon afin de garantir la traçabilité.

### Choix : échantillonnage stratifié sur `isFraud`
La stratification sur la variable cible `isFraud` est obligatoire ici. Un échantillonnage aléatoire simple risquerait d'exclure une fraction disproportionnée des 0,13 % de fraudes. La stratification garantit que le sample reflète fidèlement la distribution originale.

### Résultats obtenus
| Indicateur | Valeur |
|---|---|
| Fraudes dans le dataset complet | 0.1291 % |
| Fraudes dans l'échantillon | 0.1290 % |
| Delta | 0.000001 (OK) |
| Méthode | `stratified_split` |
| SHA-256 du split | `f6547a1c...` |

La différence de 0.000001 % confirme que la stratification fonctionne parfaitement.

---

## 3. Vue générale des données brutes

### Ce qui est fait
Inspection de la structure du DataFrame : dimensions, dictionnaire des 11 variables, types de données, statistiques descriptives, et vérification de la variance des features numériques brutes.

### Résultats : structure du dataset

**Dimensions :** 200 000 lignes × 11 colonnes

| Colonne | Type | Description |
|---|---|---|
| `step` | int64 | Unité de temps : 1 step = 1 heure, max 744 = 31 jours |
| `type` | str | Type de transaction : CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | float64 | Montant de la transaction en monnaie locale |
| `nameOrig` | str | Identifiant du compte source |
| `oldbalanceOrg` | float64 | Solde du compte source avant la transaction |
| `newbalanceOrig` | float64 | Solde du compte source après la transaction |
| `nameDest` | str | Identifiant du compte destination |
| `oldbalanceDest` | float64 | Solde du compte destination avant la transaction |
| `newbalanceDest` | float64 | Solde du compte destination après la transaction |
| `isFraud` | int64 | Cible : 1 = fraude, 0 = légitime |
| `isFlaggedFraud` | int64 | Règle métier existante (voir section 6) |

**Statistiques descriptives clés :**
- `step` : 1 à 743 → 31 jours complets couverts
- `amount` : médiane ≈ 75 033, moyenne ≈ 181 937, max ≈ 69 886 731 → forte asymétrie
- `oldbalanceDest` / `newbalanceDest` : 25e percentile = 0 → nombreux comptes destination vides

**Variance des features numériques brutes :**
Toutes les features ont une variance très supérieure à 1e-4, confirmant qu'aucune n'est constante. `StandardScaler` sera applicable sans risque d'instabilité numérique.

---

## 4. Qualité des données

### Ce qui est fait
Trois vérifications systématiques : valeurs manquantes, doublons, et cohérence des identifiants (préfixes).

### Choix : vérification avant tout feature engineering
Ces contrôles sont effectués sur les données brutes, avant toute transformation, afin de ne pas masquer d'éventuels problèmes de qualité par des opérations de nettoyage en aval.

### Résultats

**Valeurs manquantes :** 0 sur toutes les colonnes. Aucune imputation nécessaire.

**Doublons :** 0 lignes dupliquées. Les métriques d'évaluation ne seront pas biaisées par des observations répétées.

**Cohérence des identifiants :**
| Colonne | Préfixe C (clients) | Préfixe M (marchands) |
|---|---|---|
| `nameOrig` | 200 000 (100 %) | 0 |
| `nameDest` | 132 179 (66.1 %) | 67 821 (33.9 %) |

Note : Les marchands (préfixe M) n'ont pas de balance enregistrée, ce qui explique les valeurs 0/0 fréquentes sur `oldbalanceDest` / `newbalanceDest` pour les destinations marchandes. Ce n'est pas une anomalie de données.

---

## 5. Analyse du déséquilibre de classes

### Ce qui est fait
Calcul du taux et du ratio de déséquilibre entre les classes, suivi de deux visualisations : histogramme en échelle logarithmique et graphique proportionnel horizontal.

### Choix : visualisation en échelle log
Avec un ratio de 1:774, un graphique en échelle linéaire rendrait la classe minoritaire (fraudes) quasiment invisible. L'échelle logarithmique permet de voir les deux barres simultanément de manière lisible.

### Résultats

| Classe | Nombre | Proportion |
|---|---|---|
| Non-fraude (0) | 199 742 | 99.871 % |
| Fraude (1) | 258 | 0.129 % |
| **Ratio d'imbalance** | **1 : 774** | |

**Implications pour la modélisation :**
- L'accuracy seule ne vaut rien comme métrique : un modèle naïf qui prédit toujours 0 atteindrait 99.87 % d'accuracy.
- Les modèles devront utiliser `class_weight='balanced'` ou SMOTE pour compenser.
- Les métriques pertinentes seront Recall, Precision et F1-score sur la classe fraude.

---

## 6. Baseline — règle métier isFlaggedFraud

### Ce qui est fait
Évaluation des performances de la règle métier existante `isFlaggedFraud`, qui flag automatiquement les transactions de type TRANSFER dépassant 200 000 unités monétaires. Cette baseline représente ce que le système actuel (sans ML) est capable de détecter.

### Choix : calculer toutes les métriques de classification
La simple précision ne suffit pas. On calcule TP, FN, FP, Recall, Precision et F1 pour avoir une image complète des limites de la règle.

### Résultats

| Indicateur | Valeur |
|---|---|
| Fraudes réelles totales | 258 |
| Transactions flaggées par la règle | 1 |
| Vrais Positifs (TP) | 1 |
| Faux Négatifs (FN) | 257 |
| Recall | ≈ 0.4 % |
| Precision | 100 % |
| F1-score | ≈ 0.8 % |

**Analyse des montants des fraudes non flaggées :**
- 257 fraudes non flaggées (99.6 %) ont un montant moyen de 1 497 122, soit bien au-dessus du seuil de 200 000.
- Ce résultat paradoxal s'explique : la règle exige que le type soit TRANSFER **ET** que le montant dépasse 200 000. Sur le sample de 200 000 lignes, une seule transaction remplit les deux conditions.
- Conclusion : **la règle métier est quasi inutile** avec un Recall de 0.4 %. Les modèles ML devront impérativement la dépasser. C'est la cible minimale à surpasser.

---

## 7. Distribution des types de transaction

### Ce qui est fait
Comptage du volume par type de transaction et calcul du taux de fraude par type, avec deux graphiques : volume et taux de fraude.

### Résultats

| Type | Total | Fraudes | Taux fraude |
|---|---|---|---|
| TRANSFER | 16 915 | 131 | 0.7745 % |
| CASH_OUT | 70 282 | 127 | 0.1807 % |
| CASH_IN | 43 677 | 0 | 0.0000 % |
| DEBIT | 1 305 | 0 | 0.0000 % |
| PAYMENT | 67 821 | 0 | 0.0000 % |

**Observation clé :** TRANSFER et CASH_OUT sont les **seuls types impliqués dans les fraudes**. Cette observation justifie directement la création de la feature binaire `is_transfer_or_cashout` en section 12.

---

## 8. Distribution des montants

### Ce qui est fait
Calcul des indicateurs d'asymétrie (skewness, kurtosis), puis visualisation comparative entre la distribution brute et la distribution après transformation `log1p`. Analyse du taux de fraude par tranche de montant (déciles).

### Choix : transformation log1p
La transformation `log1p(x) = log(1 + x)` est préférée à `log(x)` car elle est définie en 0 (montant = 0 est possible dans le dataset). Elle est choisie plutôt que la normalisation Box-Cox ou Yeo-Johnson pour sa simplicité et son interprétabilité.

### Résultats

| Indicateur | Valeur |
|---|---|
| Skewness brute | 30.80 (très asymétrique à droite) |
| Kurtosis | 1814.32 (queue très lourde) |
| Médiane | 75 033 |
| Moyenne | 181 937 |
| Maximum | 69 886 731 |
| Ratio max/médiane | 931x |

Après `log1p`, la skewness descend à ~0.4, proche d'une distribution symétrique. Les modèles sensibles aux échelles (AutoEncoder, LSTM, XGBoost) bénéficieront directement de cette transformation.

---

## 9. Analyse temporelle

### Ce qui est fait
Dérivation de trois features temporelles depuis `step`, puis calcul du taux de fraude par heure, par jour et par semaine. Identification des "heures à risque" statistiquement au-dessus de la moyenne globale.

### Choix : dériver `hour`, `day`, `week` depuis `step`
La variable `step` est une variable continue qui encode le temps absolu en heures depuis le début de la simulation. La décomposer en composantes cycliques (heure, jour, semaine) permet aux modèles de capturer des patterns comportementaux récurrents que `step` brut ne permet pas d'identifier.

### Résultats

**Features créées :**
| Feature | Plage | Calcul |
|---|---|---|
| `hour` | 0–23 | `step % 24` |
| `day` | 0–30 | `step // 24` |
| `week` | 0–4 | `step // 168` |

**Heures à risque identifiées :** `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 23]` (nuit et début de matinée)

| Contexte | Taux de fraude |
|---|---|
| Heures normales (10h–22h) | 0.0688 % |
| Heures à risque (0h–9h, 23h) | 0.7311 % |
| **Ratio risque/normal** | **10.63x** |

Le facteur 10.63 est statistiquement très significatif. Les fraudes se concentrent massivement la nuit, période où les équipes de contrôle sont moins disponibles. La feature binaire `high_risk_hour` est créée directement depuis cette observation.

---

## 10. Analyse réseau nameOrig / nameDest

### Ce qui est fait
Identification du mécanisme de fraude en deux temps : TRANSFER (vider le compte source vers un compte mule) suivi de CASH_OUT (retrait en liquide par le compte mule). Analyse du chevauchement entre destinations frauduleuses et sources de CASH_OUT, et analyse des balances destination nulles.

### Résultats

**Chevauchement TRANSFER → CASH_OUT :**
| Indicateur | Valeur |
|---|---|
| Comptes destination dans fraudes | 258 |
| Comptes source de CASH_OUT | 70 282 |
| Chevauchement (comptes mules) | 0 (dans le sample) |

Le chevauchement nul sur le sample de 200 000 lignes s'explique par la taille de l'échantillon : les comptes mules sont rares et il est peu probable que les deux parties du schéma (TRANSFER + CASH_OUT) apparaissent dans le même échantillon de 200 000 transactions sur 6,3 M.

**Transactions avec balance destination 0/0 :**
| Indicateur | Valeur |
|---|---|
| Transactions avec dest balance 0/0 | 73 017 (36.5 %) |
| Fraudes parmi elles | 131 |
| Taux de fraude sur balance 0/0 | 0.1794 % |
| Taux de fraude global | 0.1290 % |

Le taux de fraude est 39 % plus élevé sur les transactions avec balance destination nulle avant et après. Cela justifie la feature `dest_zero_balance`.

---

## 11. Analyse par client — nameOrig

### Ce qui est fait
Profilage des clients frauduleux : nombre de fraudes par client, types de transactions utilisés, et comparaison avec les clients normaux.

### Résultats

| Indicateur | Valeur |
|---|---|
| Clients impliqués dans au moins 1 fraude | 258 |
| Clients avec exactement 1 fraude | 258 |
| Clients avec 2 fraudes ou plus | 0 |

**Types de transactions — clients frauduleux vs normaux :**

| Type | Clients frauduleux | Clients normaux |
|---|---|---|
| TRANSFER | 50.78 % | 8.47 % |
| CASH_OUT | 49.22 % | 35.12 % |
| CASH_IN | 0.00 % | 21.87 % |
| PAYMENT | 0.00 % | 33.89 % |
| DEBIT | 0.00 % | 0.65 % |

**Conclusion clé :** Les clients frauduleux utilisent **exclusivement TRANSFER et CASH_OUT**, aucun n'utilise PAYMENT ou DEBIT. Cela confirme que la feature `is_transfer_or_cashout` est une feature métier essentielle pour la modélisation.

---

## 12. Feature Engineering exploratoire

### Ce qui est fait
Construction et validation statistique des 3 features dérivées métier, puis classement de toutes les features candidates par corrélation absolue avec `isFraud`.

### Feature 1 : `balance_diff_orig`

**Définition :** `oldbalanceOrg - newbalanceOrig`

**Justification :** Un compte source dont le solde diminue fortement (voire passe à zéro) est le principal signal d'un TRANSFER ou CASH_OUT frauduleux.

**Résultats :**
| Classe | Moyenne | Médiane |
|---|---|---|
| Non-fraude | -23 074 | 0.0 |
| Fraude | 1 491 319 | 387 408 |

**Corrélation avec `isFraud` : 0.3662** — c'est la feature la plus corrélée au label.

**Note de disponibilité en production :** Cette feature suppose que les deux balances sont observables au moment de la prédiction. Si la détection doit intervenir avant la validation comptable, `newbalanceOrig` pourrait ne pas encore être disponible. Elle est conservée car le dataset simule une détection post-transaction.

### Feature 2 : `is_transfer_or_cashout`

**Définition :** Binaire, 1 si le type est TRANSFER ou CASH_OUT, 0 sinon.

**Justification :** 100 % des fraudes observées appartiennent à ces deux types.

**Résultats :**
| Valeur | Fraudes | Taux fraude |
|---|---|---|
| 0 (autres types) | 0 | 0.000 % |
| 1 (TRANSFER/CASH_OUT) | 258 | 0.296 % |

**Corrélation avec `isFraud` : 0.0409** — faible en valeur absolue, mais cette feature est **100 % discriminante** : aucune fraude en dehors de TRANSFER/CASH_OUT.

### Feature 3 : `dest_zero_balance`

**Définition :** Binaire, 1 si le compte destination est un client (préfixe C) avec `oldbalanceDest = 0` ET `newbalanceDest = 0`.

**Justification :** Les comptes mules utilisés dans les fraudes tendent à rester vides avant et après la réception (l'argent est immédiatement retiré par d'autres canaux).

**Corrélation avec `isFraud` : 0.1088**

### Classement final des features par corrélation absolue avec `isFraud`

| Feature | Corrélation absolue | Niveau |
|---|---|---|
| `balance_diff_orig` | 0.3662 | Fort |
| `dest_zero_balance` | 0.1088 | Fort |
| `amount` | 0.0760 | Modéré |
| `high_risk_hour` | 0.0531 | Modéré |
| `is_transfer_or_cashout` | 0.0409 | Faible* |
| `hour` | 0.0372 | Faible |
| `day` | ~0.02 | Faible |
| `week` | ~0.01 | Faible |

*`is_transfer_or_cashout` a une corrélation faible mais une importance métier maximale (condition nécessaire à la fraude).

---

## 13. Vérification du Data Leakage

### Ce qui est fait
Identification des variables qui ne doivent pas être utilisées en entrée des modèles, soit parce qu'elles sont causalement postérieures à la fraude (leakage temporel), soit parce qu'elles révèlent directement le label.

### Choix : approche de vérification systématique
Deux types de leakage sont vérifiés :
1. **Leakage métier** : colonnes dont la valeur post-fraude est non représentative du contexte réel de prédiction
2. **Leakage direct** : colonnes corrélées à 100 % avec le label

### Résultats

**Colonnes EXCLUES et justifications :**

| Colonne | Raison d'exclusion |
|---|---|
| `oldbalanceOrg` | Les transactions frauduleuses sont annulées dans le dataset → les balances ne reflètent pas ce qu'un modèle verrait en temps réel |
| `newbalanceOrig` | Idem |
| `oldbalanceDest` | Idem |
| `newbalanceDest` | Idem |
| `isFlaggedFraud` | Corrélation de 100 % avec `isFraud` quand = 1 → inclusion = triche pure |
| `nameOrig` | Cardinalité trop élevée (200 000 valeurs uniques), pas de généralisation possible |
| `nameDest` | Idem (174 532 valeurs uniques) |

**Preuve numérique sur `isFlaggedFraud` :**
- Corrélation avec `isFraud` : 0.0622 (apparemment faible)
- Mais : taux de fraude quand `isFlaggedFraud = 1` : **100 %**
- Inclusion = fuite d'information directe vers le label

---

## 14. Sauvegarde du rapport EDA

### Ce qui est fait
Export de l'ensemble des métriques clés de l'exploration dans trois fichiers de sortie, avec calcul du hash SHA-256 pour chaque artefact afin d'assurer la traçabilité.

### Fichiers produits

| Fichier | Contenu |
|---|---|
| `outputs/reports/eda_full_report.json` | Rapport EDA complet (toutes métriques) |
| `outputs/reports/eda_summary_stats.csv` | Statistiques descriptives en format tabulaire |
| `outputs/reports/feature_correlations.csv` | Corrélations absolues de chaque feature avec `isFraud` |

### Choix : hash SHA-256 par artefact
Le hash garantit qu'un notebook relancé sur un autre dataset ou un autre sample ne produira pas silencieusement des métriques différentes sans que l'utilisateur ne le remarque. C'est une pratique d'audit important dans le contexte PwC.

---

## 15. Synthèse des décisions

### Features retenues pour la modélisation

| Feature | Type | Justification |
|---|---|---|
| `log_amount` | Continue | Réduit la skewness de 30.8 à ~0.4 |
| `hour` | Continue | Patterns temporels journaliers liés aux fraudes |
| `day` | Continue | Patterns temporels mensuels |
| `week` | Continue | Patterns temporels hebdomadaires |
| `high_risk_hour` | Binaire | 10.63x plus de fraudes entre 0h–9h et 23h |
| `balance_diff_orig` | Continue | Corrélation la plus forte avec `isFraud` (0.366) |
| `is_transfer_or_cashout` | Binaire | Condition nécessaire à la fraude (100 % des cas) |
| `dest_zero_balance` | Binaire | Signe de compte mule (corrélation 0.109) |

### Colonnes exclues de la modélisation

| Colonne | Motif |
|---|---|
| `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` | Data leakage métier |
| `isFlaggedFraud` | Data leakage direct (100 % corrélé quand = 1) |
| `nameOrig`, `nameDest` | Cardinalité trop élevée |
| `step` | Remplacé par `hour`, `day`, `week` |
| `type` | Remplacé par `is_transfer_or_cashout` |

### Décisions impactant les notebooks suivants

1. **Déséquilibre 1:774** → SMOTE ou `class_weight='balanced'` obligatoire dans les modèles ML
2. **Baseline F1 ≈ 0.8 %** → barre minimale que tous les modèles doivent dépasser
3. **8 features finales** → pipeline de preprocessing standardisé dans `03_data_preparation.ipynb`
4. **Rapport EDA en JSON** → référence documentaire pour les audits et pour nourrir les rapports automatiques
