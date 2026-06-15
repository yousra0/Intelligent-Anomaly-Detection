# Notebook 07 — Intégration LLM : Analyse détaillée

**Notebook :** `notebooks/07_llm_integration.ipynb`
**Objectif :** Générer automatiquement des explications en langage naturel pour chaque transaction détectée comme frauduleuse par l'AutoEncoder, en utilisant un LLM via API cloud.

---

## Table des matières

1. [Pourquoi un LLM dans ce pipeline](#1-pourquoi-un-llm-dans-ce-pipeline)
2. [Pourquoi une API cloud plutôt qu'un LLM local](#2-pourquoi-une-api-cloud-plutôt-quun-llm-local)
3. [Pourquoi Groq — Choix du provider API](#3-pourquoi-groq--choix-du-provider-api)
4. [Pourquoi le modèle llama-3.1-8b-instant](#4-pourquoi-le-modèle-llama-31-8b-instant)
5. [Configuration & Paramètres LLM](#5-configuration--paramètres-llm)
6. [Architecture de configuration (YAML + .env)](#6-architecture-de-configuration-yaml--env)
7. [Chargement des références et artefacts](#7-chargement-des-références-et-artefacts)
8. [Identification des transactions frauduleuses](#8-identification-des-transactions-frauduleuses)
9. [Calcul des erreurs de reconstruction par feature](#9-calcul-des-erreurs-de-reconstruction-par-feature)
10. [Initialisation du LLMHelper](#10-initialisation-du-llmhelper)
11. [Test sur une seule transaction](#11-test-sur-une-seule-transaction)
12. [Génération batch — toutes les fraudes détectées](#12-génération-batch--toutes-les-fraudes-détectées)
13. [Analyse des explications générées](#13-analyse-des-explications-générées)
14. [Visualisations](#14-visualisations)
15. [Sauvegarde des artefacts](#15-sauvegarde-des-artefacts)
16. [Synthèse finale](#16-synthèse-finale)

---

## 1. Pourquoi un LLM dans ce pipeline

### Le problème de la boîte noire

Les modèles ML (AutoEncoder, XGBoost, Random Forest) produisent des scores numériques — `"score AE = 313.65"` ou `"P(fraude) = 0.988"` — mais un auditeur PwC Tunisie ne peut pas agir sur un chiffre brut sans comprendre le *pourquoi* de l'alerte.

Le notebook 06 (SHAP + LIME) produit des explications structurées en JSON :
```json
{"feature": "balance_diff_orig", "error": 65.73, "direction": "↑ risque fraude"}
```

Mais cette structure JSON reste **technique** — elle n'est pas directement utilisable par un auditeur financier non-développeur.

### Ce que le LLM apporte

Le LLM transforme les données techniques en **rapport d'audit en français** :

```
Entrée (JSON technique)                         Sortie (texte auditeur)
────────────────────────────────────────────    ──────────────────────────────────────
ae_score: 313.65                          →     "Risque ÉLEVÉ"
balance_diff_orig error: 65.73            →     "Le compte émetteur a été entièrement
log_amount error: 5.87                          vidé — anomalie majeure sur le solde"
threshold: 1.753                          →     "Actions : vérifier les transactions
                                                 récentes du compte émetteur"
```

Le LLM ne **décide pas** si c'est une fraude (le seuil AE fait cette décision) — il **explique** pourquoi le modèle a décidé que c'est une fraude, en langage compréhensible pour l'auditeur.

---

## 2. Pourquoi une API cloud plutôt qu'un LLM local

### Le problème des LLM locaux (Ollama, llama.cpp)

Les LLM locaux comme **Ollama + Llama 3** sont une option valide en théorie, mais présentent des contraintes matérielles critiques dans ce contexte :

| Contrainte | LLM local (Ollama) | API cloud (Groq) |
|---|---|---|
| **RAM requise** | 8–16 Go minimum pour Llama 3 8B | 0 Go (calcul côté serveur) |
| **GPU requis** | GPU dédié recommandé (sinon 10–30s/token sur CPU) | GPU H100 de Groq (côté serveur) |
| **Temps par explication** | 30–120 secondes sur CPU | **0.5–0.8 secondes** |
| **Impact système** | Monopolise CPU/RAM pendant le calcul | Aucun impact local |
| **Installation** | Télécharger ~4 Go de modèle + configurer Ollama | `pip install groq` |
| **Disponibilité offline** | Oui | Non |

**Sur la machine de développement (laptop PwC Tunisie, NVIDIA MX450, 8 Go RAM) :**
- Ollama avec Llama 3 8B consomme ~7 Go RAM → le système devient inutilisable
- L'entraînement de l'AutoEncoder (NB05) prend 191s sur GPU — un Ollama en parallèle serait impossible
- Pour 20 transactions en batch → 20 × 60s = **20 minutes** en local vs **86 secondes** via Groq API

**Décision :** L'API cloud est le seul choix viable avec les ressources disponibles. Le mode fallback rule-based (sans réseau) reste disponible si l'API est inaccessible.

### Fallback rule-based intégré
Si la clé API est absente ou invalide, `LLMHelper` bascule automatiquement sur un mode rule-based déterministe qui génère une explication structurée depuis les valeurs SHAP/erreurs AE — sans aucun appel réseau. Cela garantit que le pipeline fonctionne toujours, même sans connexion.

---

## 3. Pourquoi Groq — Choix du provider API

### Comparaison des providers LLM API disponibles

| Critère | **Groq** | OpenAI (GPT-4o) | Gemini (Google) | Ollama (local) |
|---|---|---|---|---|
| **Vitesse d'inférence** | ⚡ Très rapide (LPU) | Moyenne | Rapide | Très lente (CPU) |
| **Coût** | **Gratuit** (tier gratuit) | Payant ($0.005/1K tokens) | Gratuit limité | Gratuit |
| **Qualité Llama 3.1** | ✅ Excellente | Non applicable | Non applicable | ✅ Identique |
| **Tier gratuit** | **100k tokens/jour** | 0 | 1M tokens/jour | Illimité |
| **Latence** | ~200–500 ms | ~800 ms | ~400 ms | 30–120s |
| **Modèles disponibles** | Llama 3.1, Mixtral, Gemma | GPT-4o, GPT-3.5 | Gemini Pro | Tout modèle |
| **Confidentialité données** | Données envoyées à Groq | Données envoyées à OpenAI | Données à Google | 100% local |

**Raisons du choix Groq :**

1. **Vitesse d'inférence unique :** Groq utilise des puces LPU (Language Processing Unit) conçues spécifiquement pour l'inférence LLM — 10× plus rapides que les GPU classiques pour l'inférence. Résultat : 0.5–0.8s par explication vs 3–5s sur d'autres APIs.

2. **Tier gratuit suffisant :** 100 000 tokens/jour gratuits. Avec max_tokens=400 par explication et 20 explications → 8 000 tokens/exécution. Le tier gratuit couvre largement les besoins de développement et d'audit.

3. **Accès à Llama 3.1 :** Groq héberge les modèles Meta Llama — open-source, sans restrictions de redistribution, identiques aux modèles locaux Ollama mais exécutés sur infrastructure Groq.

4. **OpenAI exclu :** Payant dès le premier token, coût prohibitif pour une expérimentation PwC sans budget dédié API.

5. **Gemini exclu :** Tier gratuit généreux mais latence plus élevée et modèle moins performant sur les tâches de génération structurée JSON en français.

---

## 4. Pourquoi le modèle llama-3.1-8b-instant

### Comparaison des modèles disponibles sur Groq

| Modèle | Taille | Vitesse | Qualité JSON | Français | Tier gratuit |
|---|---|---|---|---|---|
| **llama-3.1-8b-instant** | 8B | ⚡⚡⚡ | ✅ Bonne | ✅ Bonne | ✅ |
| llama-3.1-70b-versatile | 70B | ⚡ | ✅✅ Excellente | ✅✅ | ✅ (limité) |
| mixtral-8x7b-32768 | 47B | ⚡⚡ | ✅ Bonne | ✅ | ✅ |
| gemma-7b-it | 7B | ⚡⚡⚡ | ⚠️ Variable | ⚠️ Partielle | ✅ |

**Raisons du choix `llama-3.1-8b-instant` :**

1. **Vitesse prioritaire en batch :** Avec 20 transactions à expliquer, la vitesse est critique. Le 8B génère en 0.5–0.8s vs 2–4s pour le 70B.

2. **Qualité suffisante pour le JSON structuré :** Le 8B gère parfaitement la génération JSON avec les champs imposés (`risk_level`, `resume`, `raisons`, `actions_recommandees`). Le 70B ne serait pas significativement meilleur sur cette tâche contrainte.

3. **Température 0.1 compense la taille :** Une température très basse (0.1) réduit la variabilité du 8B et le rend quasi-déterministe — les explications sont stables et reproductibles sans avoir besoin de la puissance du 70B.

4. **"instant" dans le nom :** Désigne la variante optimisée pour la faible latence sur l'infrastructure Groq.

---

## 5. Configuration & Paramètres LLM

### Paramètres de génération et justifications

| Paramètre | Valeur | Justification |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Vitesse + tier gratuit (voir section 3) |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Vitesse + qualité suffisante (voir section 4) |
| `LLM_TIMEOUT` | 30 s (90 s pour le test initial) | Au-delà de 30s → timeout → fallback rule-based. 90s pour le premier test unitaire (sécurité) |
| `LLM_TEMPERATURE` | **0.1** | Proche de 0 → sortie quasi-déterministe. Indispensable pour un rapport d'audit : la même transaction doit produire la même explication à chaque exécution |
| `LLM_MAX_TOKENS` | **400** | Suffisant pour : résumé (50 tokens) + 3 raisons (100) + 2 actions (80) + JSON overhead (170). Évite les réponses tronquées ou les hallucinations longues |
| `LLM_TOP_K_FEATURES` | **3** | Top 3 features les plus anormales (par erreur AE) transmises au LLM. Au-delà de 3, le LLM dilue le message — 3 est le nombre optimal pour un résumé actionnable |
| `LLM_MAX_EXPLAIN` | **20** | Maximum 20 transactions expliquées par exécution. Limite de sécurité pour le tier gratuit (20 × 400 = 8 000 tokens << 100k limite quotidienne) |
| `LLM_BATCH_VERBOSE` | `True` | Affiche le statut en temps réel pendant le batch — permet de surveiller la progression et de détecter les timeouts |

### Pourquoi temperature=0.1 et non 0.0

- `temperature=0.0` (greedy) peut produire des boucles répétitives sur certains modèles
- `temperature=0.1` conserve une micro-variance qui évite les artefacts de génération tout en restant quasi-déterministe
- Pour un rapport d'audit, la stabilité est primordiale : deux exécutions sur la même transaction doivent produire des explications identiques

### Pourquoi max_tokens=400

Le prompt structuré JSON impose 6 champs minimum. 400 tokens permettent :
```
{
  "risk_level": "ELEVÉ",           ← 3 tokens
  "risk_score": 0.178,             ← 5 tokens
  "confidence_score": 0.78,        ← 5 tokens
  "resume": "...",                 ← ~40 tokens
  "raisons": ["...", "...", "..."],← ~80 tokens × 3
  "actions_recommandees": ["...", "..."] ← ~50 tokens × 2
}                                  ← Total : ~300 tokens utiles
```
Marge de 100 tokens pour les variations. Au-delà de 400, les modèles ont tendance à ajouter du texte non-JSON qui casse le parser.

---

## 6. Architecture de configuration (YAML + .env)

### Ce qui est fait
La configuration LLM est entièrement externalisée dans deux fichiers, jamais codée en dur dans le notebook :

```
config/
├── config.yaml       → chemins données, modèles, figures
└── llm_config.yaml   → provider, modèle, paramètres génération, chemins output
.env                  → GROQ_API_KEY=gsk_xxx (jamais committé)
```

### Résolution sécurisée de la clé API
```python
load_dotenv()  # charge .env avant os.getenv
LLM_API_KEY = resolve_env_vars(
    llm_config.get('api_keys', {}).get(LLM_PROVIDER, '')
)
# resolve_env_vars : "${GROQ_API_KEY}" → os.getenv("GROQ_API_KEY")
```

**Sécurité :** La clé API n'est jamais écrite dans `llm_config.yaml` (qui est versionné) — seulement dans `.env` (ignoré par git). Le placeholder `${GROQ_API_KEY}` est résolu au runtime depuis les variables d'environnement.

### Variables chargées depuis llm_config.yaml

```
LLM_PROVIDER   : groq
LLM_MODEL      : llama-3.1-8b-instant
API Key found  : True
```

---

## 7. Chargement des références et artefacts

### Ce qui est fait
Chargement des métriques de référence depuis les rapports JSON produits par NB05 (AutoEncoder) et NB04 (baselines ML), pour les afficher dans les explications et les résumés.

### Références chargées

| Référence | Source | Valeur |
|---|---|---|
| Seuil AE optimal | `autoencoder_report.json` | **1.753022** |
| AE Recall test | `autoencoder_report.json` | 0.3590 |
| AE F1 test | `autoencoder_report.json` | 0.4516 |
| RF_smote Recall | `baseline_report.json` | 0.7949 |
| RF_smote F1 | `baseline_report.json` | 0.8052 |

### Datasets chargés

| Dataset | Shape | Fraudes |
|---|---|---|
| `X_test` | (30 001, 14) | 39 |
| `X_val` | (30 000, 14) | 38 |

### Scores AE pré-calculés (NB05)

| Fichier | Shape | Min | Max |
|---|---|---|---|
| `ae_scores_test.npy` | (30 001,) | 0.0146 | **313.6496** |
| `ae_scores_val.npy` | (30 000,) | 0.0151 | 314.1599 |

**Choix : réutiliser les scores précalculés du NB05**
Recalculer les scores AE dans NB07 prendrait ~0.2s mais introduirait une dépendance à l'AutoEncoder PyTorch. Les scores pré-calculés garantissent la cohérence exacte avec NB05 (même valeurs, même ordre) — traçabilité essentielle pour l'audit.

---

## 8. Identification des transactions frauduleuses

### Ce qui est fait
Application du seuil AE optimal (1.753022) sur les scores du test set pour identifier les 23 transactions à expliquer.

### Résultats

```
Seuil AE       : 1.753022
Transactions flaggées : 23
  TP (vraies fraudes détectées) : 14
  FP (fausses alertes)          :  9
  FN (fraudes manquées)         : 25
  Recall : 0.3590
```

### Composition des 23 transactions à expliquer

| Catégorie | Nombre | Description |
|---|---|---|
| TP (Vrais Positifs) | **14** | Fraudes réelles correctement détectées par l'AE |
| FP (Faux Positifs) | **9** | Transactions légitimes classées à tort comme anomalies |
| Total flaggé | **23** | Candidats pour l'explication LLM |

**Note importante :** Le LLM explique **toutes** les transactions flaggées, y compris les 9 FP. C'est intentionnel — l'auditeur reçoit une explication pour chaque alerte et peut décider lui-même si elle est justifiée après investigation. Le LLM ne connaît pas le label réel (`y_test`) — il génère une explication factuelle basée uniquement sur les features et les erreurs AE.

---

## 9. Calcul des erreurs de reconstruction par feature

### Ce qui est fait
Reconstruction de toutes les 30 001 transactions du test set par l'AutoEncoder, puis calcul de l'erreur absolue par feature — en **0.2 secondes** sur GPU.

### Formule
```python
feature_errors = |X_test - AutoEncoder(X_test)|  # shape (30001, 14)
```

Chaque ligne = 14 erreurs, une par feature. L'erreur totale AE = MSE(X, X̂) = mean(feature_errors²).

### Transaction la plus anormale (idx=29340, score=313.65)

| Rang | Feature | Erreur de reconstruction | Interprétation |
|---|---|---|---|
| 1 | `balance_diff_orig` | **65.7344** | L'AE ne peut pas reconstruire la valeur de 66.07 (66 std dev au-dessus de la moyenne) — anomalie extrême |
| 2 | `log_amount` | 5.8698 | Le montant log est anormalement élevé pour ce type de transaction |
| 3 | `type_PAYMENT` | 1.6641 | L'AE reconstruit un pattern PAYMENT non-nul alors que tx_PAYMENT=0 — confusion avec une autre transaction |

### Rôle dans le prompt LLM
Ces top-3 erreurs par feature constituent le **signal principal** envoyé au LLM :
```json
"top_features": [
  {"feature": "balance_diff_orig", "error": 65.73, "value": 66.07},
  {"feature": "log_amount",        "error": 5.87,  "value": 2.91},
  {"feature": "type_PAYMENT",      "error": 1.66,  "value": 0.0}
]
```
Le LLM transforme ces chiffres en langage naturel : "La différence de solde du compte émetteur est anormalement élevée (erreur=65.73)".

---

## 10. Initialisation du LLMHelper

### Ce qui est fait
Instanciation du `LLMHelper` (module `src/llm_integration/llm_helper.py`) avec les paramètres de `llm_config.yaml`.

```
Provider       : groq
Modele         : llama-3.1-8b-instant
API disponible : True
✅ Pret a generer des explications LLM via GROQ
```

### Architecture du LLMHelper

```
LLMHelper
├── provider = "groq"
├── model = "llama-3.1-8b-instant"
├── temperature = 0.1
├── max_tokens = 400
├── timeout = 30s
├── explain_fraud(transaction, feature_errors, ae_score, threshold, top_k)
│     → Appel API unique → JSON structuré
├── explain_batch(transactions[], feature_errors_batch[], ae_scores[], ...)
│     → Boucle sur N transactions → Liste de JSON
└── is_available() → bool (clé API valide et réseau accessible)
```

### Sécurité : chargement dynamique du module
Le `LLMHelper` est chargé via `spec_from_file_location` pour forcer le rechargement de la dernière version du code source, indépendamment du cache Python. Cela permet de modifier `llm_helper.py` entre deux exécutions sans redémarrer le kernel.

---

## 11. Test sur une seule transaction

### Ce qui est fait
Test unitaire sur la transaction TP avec le score AE le plus élevé (idx=29340, score=313.6496) pour valider que le LLM génère une explication correcte avant le batch.

### Transaction testée

| Feature | Valeur standardisée | Signal |
|---|---|---|
| `balance_diff_orig` | **66.0686** | ↑ Compte source entièrement vidé (+66 std dev) |
| `dest_zero_balance` | 1.0000 | ↑ Compte destination "mule" (balance 0/0) |
| `type_TRANSFER` | 1.0000 | ↑ Type TRANSFER (le seul lié à ce schéma) |
| `is_transfer_or_cashout` | 1.0000 | ↑ Confirme TRANSFER ou CASH_OUT |
| `log_amount` | 2.9061 | Montant relativement modeste en log-scale |
| `high_risk_hour` | 0.0000 | Heure normale (paradoxalement) |

### Explication générée par Groq (llama-3.1-8b-instant)

**Durée :** 0.6 secondes | **Status :** ok | **Modèle :** llama-3.1-8b-instant

```json
{
  "risk_level": "ELEVÉ",
  "risk_score": 0.178,
  "confidence_score": 0.78,
  "resume": "La transaction présente des anomalies concernant le solde
             du compte emetteur, le montant de la transaction et
             le type de transaction.",
  "raisons": [
    "La différence de solde du compte emetteur est anormalement
     élevée (erreur=65.7344)",
    "Le montant de la transaction est anormalement élevé
     (log-transforme) (erreur=5.8698)",
    "Le type PAYMENT présente une reconstruction anormale
     (erreur=1.6641)"
  ],
  "actions_recommandees": [
    "Vérifier le solde du compte emetteur et les transactions récentes",
    "Analyser les transactions similaires pour identifier
     des modèles anormaux"
  ],
  "ae_score": 313.649628,
  "threshold": 1.753022,
  "top_features": [
    {"feature": "balance_diff_orig", "error": 65.734421, "value": 66.06855},
    {"feature": "log_amount",        "error": 5.869781,  "value": 2.906082},
    {"feature": "type_PAYMENT",      "error": 1.664119,  "value": 0.0}
  ],
  "duration_s": 0.62,
  "status": "ok",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "_audit": {
    "hash": "04a1c8234d63ad5d748081fbaba2db87c38f0c679d1d438d6d7e482f13815dfb",
    "timestamp_utc": "2026-06-12T10:56:41.400912+00:00",
    "provider": "groq",
    "model_version": "llama-3.1-8b-instant",
    "hash_algo": "sha256"
  }
}
```

### Analyse de la réponse

**Points forts :**
- `risk_level: ELEVÉ` correctement attribué (score 313.65 >> seuil 1.75)
- Les 3 raisons correspondent exactement aux top-3 features passées en entrée
- Actions recommandées concrètes et actionnables pour un auditeur
- `confidence_score: 0.78` — confiance modérée (le LLM signale une incertitude)

**Note sur `_audit` :**
Un hash SHA-256 du couple (prompt, réponse LLM brute) est calculé à chaque appel avec timestamp UTC. Cela permet de prouver à l'auditeur que l'explication n'a pas été modifiée après génération — traçabilité essentielle pour la conformité.

> **Couverture complète du trail d'audit (backend) :** En production, le mécanisme d'audit couvre également le mode fallback rule-based. Lorsque le LLM est indisponible, `_fallback()` hache le contenu de l'explication déterministe elle-même (pas de prompt/réponse LLM — le contenu JSON signé est le résultat) et l'écrit dans `outputs/audit_trail/explanations_audit.jsonl`. Toutes les explications — LLM ou rule-based — sont donc tracées avec hash SHA-256 + timestamp UTC, sans exception.

**Note sur `risk_score: 0.178` :**
Il s'agit du score normalisé `ae.predict_score()` [0,1] et non du score brut MSE (313.65). Le LLM reçoit les deux valeurs pour contextualiser correctement.

---

## 12. Génération batch — toutes les fraudes détectées

### Ce qui est fait
Génération d'explications pour les 20 premières transactions flaggées (sur 23), triées par score AE décroissant — les plus anormales d'abord.

### Paramètres batch

| Paramètre | Valeur | Justification |
|---|---|---|
| `max_explain` | 20 | Limite le coût API et le temps d'exécution |
| `verbose` | True | Affichage en temps réel — permet de surveiller |
| `top_k` | 3 | Top 3 features par transaction |
| Ordre | Score AE décroissant | Les anomalies les plus graves sont expliquées en priorité |

### Log d'exécution batch

| # | Score AE | Status | Durée | Risque |
|---|---|---|---|---|
| 1/20 | 313.65 | ok | 0.5s | ELEVÉ |
| 2/20 | 260.77 | ok | 0.5s | ELEVÉ |
| 3/20 | 67.41 | ok | 0.5s | ELEVÉ |
| 4/20 | 54.53 | ok | 0.7s | ELEVÉ |
| 5/20 | 12.73 | ok | 0.6s | ELEVÉ |
| 6/20 | 11.80 | ok | 0.5s | ELEVÉ |
| 7/20 | 5.68 | ok | 0.8s | ELEVÉ |
| 8/20 | 4.39 | ok | 0.6s | ELEVÉ |
| 9/20 | 4.36 | ok | 0.6s | ELEVÉ |
| 10/20 | 3.84 | ok | 0.5s | ELEVÉ |
| 11/20 | 3.78 | ok | 0.4s | ELEVÉ |
| 12/20 | 3.28 | ok | 0.5s | ELEVÉ |
| 13/20 | 3.09 | ok | 0.5s | ELEVÉ |
| 14/20 | 2.95 | ok | 0.7s | ELEVÉ |
| 15/20 | 2.63 | ok | 0.6s | ELEVÉ |
| 16/20 | 2.42 | ok | 0.8s | ELEVÉ |
| 17/20 | 2.14 | ok | 0.6s | ELEVÉ |
| 18/20 | 1.98 | ok | 0.5s | ELEVÉ |
| 19/20 | 1.97 | ok | 0.5s | ELEVÉ |
| 20/20 | 1.92 | ok | 0.6s | ELEVÉ |

### Statistiques du batch

| Indicateur | Valeur |
|---|---|
| **Explications générées** | **20 / 20** |
| **Durée totale** | **86.3 secondes** |
| Durée moyenne par explication | ~4.3 s (incluant les pauses anti-rate-limit) |
| Durée pure inférence | ~0.6 s/explication |
| Statut ok | 20 (100 %) |
| Statut fallback | 0 |
| Statut erreur | 0 |

**Comparaison avec LLM local (Ollama) :** 20 × 60s = **20 minutes** vs **86 secondes** via Groq — gain de **×14 en vitesse**.

---

## 13. Analyse des explications générées

### Ce qui est fait
Agrégation des statistiques sur les 20 explications pour identifier les tendances globales.

### Statistiques de statut

| Statut | Nombre | % |
|---|---|---|
| **ok** | **20** | **100 %** |
| fallback | 0 | 0 % |
| erreur | 0 | 0 % |

Taux de succès API : 100 % — aucun timeout, aucun échec de parsing JSON.

### Distribution des niveaux de risque

| Niveau | Nombre | % |
|---|---|---|
| **ELEVÉ** | **20** | **100 %** |

**Analyse :** Toutes les transactions flaggées reçoivent un niveau de risque ÉLEVÉ. Ce résultat est attendu et cohérent : le seuil AE de 1.753 a été optimisé pour ne flaguer que les anomalies les plus significatives. Les 20 transactions ont toutes un score AE >> seuil (1.92 au minimum, 313.65 au maximum).

### Features les plus citées par le LLM (top 7)

| Rang | Feature | Citations | % des 20 explications |
|---|---|---|---|
| 1 | `balance_diff_orig` | **20** | **100 %** |
| 2 | `hour` | 15 | 75 % |
| 3 | `log_amount` | 14 | 70 % |
| 4 | `step` | 4 | 20 % |
| 5 | `day` | 2 | 10 % |
| 6 | `week` | 1 | 5 % |
| 7 | `high_risk_hour` | 1 | 5 % |

**Interprétation :**

1. **`balance_diff_orig` citée dans 100 % des explications** : confirme qu'elle est le signal dominant pour toutes les fraudes détectées par l'AE — cohérent avec sa corrélation EDA de 0.3662 et son importance SHAP de 37-38 %.

2. **`hour` citée dans 75 % des cas** : l'AutoEncoder a appris une forte structure temporelle. Les fraudes surviennent à des heures non-représentatives des patterns normaux — l'AE génère une forte erreur de reconstruction sur `hour` pour ces transactions.

3. **`log_amount` dans 70 % des cas** : les montants des fraudes sont atypiques pour l'heure et le type de transaction observés — l'AE signale cette anomalie combinée.

4. **`type_PAYMENT`, `type_CASH_IN`, etc.** ne sont pas dans le top-7 : l'AE confond parfois les types one-hot lors de la reconstruction, mais ces erreurs sont secondaires par rapport aux features continues.

---

## 14. Visualisations

### Figure 35 — Analyse des explications LLM (3 panneaux)

**Panneau 1 — Niveau de risque :**
Graphique en barres : ÉLEVÉ = 20 transactions. Couleur orange (#E67E22). Confirme que toutes les alertes AE sont classées ÉLEVÉ.

**Panneau 2 — Features les plus anormales :**
Graphique en barres horizontales (top 8). `balance_diff_orig` domine avec 20 citations, suivi de `hour` (15) et `log_amount` (14). Visuelle directement lisible par l'auditeur PwC.

**Panneau 3 — Distribution des scores AE (fraudes expliquées) :**
Histogramme des 20 scores AE expliqués avec le seuil (1.753) marqué en rouge. Distribution très asymétrique : 2 transactions avec score > 100, la majorité entre 1.9 et 15. Illustre la grande variabilité des anomalies détectées.

**Fichier :** `35_ollama_analysis.png`

### Figure 36 — Exemples d'explications LLM (2 panneaux texte)

Affichage côte-à-côte de 2 exemples d'explications complètes en format texte monospace, avec :
- Niveau de risque + score + confiance
- Résumé en français
- Liste numérotée des raisons
- Actions recommandées

Couleur de fond adaptée au niveau de risque (orange pour ÉLEVÉ). Format directement exportable dans un rapport PDF.

**Fichier :** `36_explanation_examples.png`

---

## 15. Sauvegarde des artefacts

### Ce qui est fait
Export de toutes les explications et statistiques en JSON pour réutilisation par l'API FastAPI et les rapports de génération du notebook suivant.

### Fichiers produits (`outputs/reports/`)

**`explanations.json` — 20 explications complètes**
```json
[
  {
    "risk_level": "ELEVÉ",
    "risk_score": 0.178,
    "confidence_score": 0.78,
    "resume": "La transaction présente des anomalies...",
    "raisons": ["...", "...", "..."],
    "actions_recommandees": ["...", "..."],
    "ae_score": 313.649628,
    "threshold": 1.753022,
    "top_features": [...],
    "duration_s": 0.62,
    "status": "ok",
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
    "_audit": {"hash": "04a1c8...", "timestamp_utc": "2026-06-12T10:56:41Z"}
  },
  ...
]
```

**`llm_summary.json` — Statistiques globales**
```json
{
  "n_flagged_total": 23,
  "n_explained": 20,
  "n_tp": 14,
  "n_fp": 9,
  "ae_threshold": 1.753022,
  "ae_recall": 0.3590,
  "ae_f1": 0.4516,
  "rf_smote_recall": 0.7949,
  "rf_smote_f1": 0.8052,
  "risk_distribution": {"ELEVÉ": 20},
  "status_distribution": {"ok": 20},
  "top_features_cited": [
    {"feature": "balance_diff_orig", "count": 20},
    {"feature": "hour", "count": 15},
    {"feature": "log_amount", "count": 14}
  ],
  "llm_model": "llama-3.1-8b-instant",
  "llm_provider": "groq",
  "llm_available": true
}
```

### Choix : hash SHA-256 dans `_audit`

Chaque explication contient un hash SHA-256 + timestamp UTC, écrit dans `outputs/audit_trail/explanations_audit.jsonl` (fichier JSONL append-only, une ligne par explication).

| Chemin LLM | Contenu haché | Couvert |
|------------|--------------|---------|
| LLM disponible (`status: ok`) | SHA-256(prompt + réponse brute LLM) | ✅ |
| Fallback rule-based | SHA-256(contenu JSON de l'explication déterministe) | ✅ |
| Erreur HTTP / timeout | Fallback déclenché → même mécanisme | ✅ |

Permet à l'auditeur PwC de vérifier qu'aucune explication n'a été altérée après génération, quelle que soit la voie (LLM cloud ou règles métier offline) — conformité aux exigences d'intégrité des rapports d'audit.

---

## 16. Synthèse finale

### Récapitulatif de l'exécution

| Indicateur | Valeur |
|---|---|
| Provider LLM | **GROQ** |
| Modèle | **llama-3.1-8b-instant** |
| Transactions flaggées (AE) | 23 (seuil = 1.753022) |
| Explications générées | **20 / 20** (100 % de succès) |
| Durée totale batch | **86.3 secondes** |
| Durée par explication (inférence) | ~0.6 secondes |
| Statuts ok / fallback / erreur | 20 / 0 / 0 |
| Niveau de risque | ÉLEVÉ = 20 (100 %) |
| **Feature #1 la plus citée** | **`balance_diff_orig`** (20/20 = 100 %) |

### Décisions d'architecture justifiées

| Décision | Choix retenu | Alternative rejetée | Raison |
|---|---|---|---|
| LLM runtime | **API Groq** | Ollama local | Ollama monopolise 7 Go RAM, 60s/explication — machine inutilisable |
| Provider API | **Groq** | OpenAI, Gemini | Tier gratuit + vitesse LPU unique (0.6s vs 3s OpenAI) |
| Modèle | **llama-3.1-8b-instant** | llama-3.1-70b | 10× plus rapide, qualité JSON suffisante pour la tâche |
| temperature | **0.1** | 0.7 (créatif) | Rapport d'audit = reproductibilité requise |
| max_tokens | **400** | 1000 | 400 suffisant pour JSON structuré, évite hallucinations longues |
| top_k features | **3** | 5 | 3 raisons = message clair et actionnable pour l'auditeur |
| Configuration | **YAML + .env** | Codé en dur | Sécurité clé API + flexibilité sans modifier le notebook |
| Hash audit | **SHA-256** | Aucun | Conformité PwC — intégrité vérifiable des rapports |

### Pipeline complet — bilan final

```
NB02 EDA
  → NB03 Préparation (14 features, splits 70/15/15, SMOTE)
  → NB04 Baselines ML (XGB_smote : Recall=0.846, F1=0.835)
  → NB05 AutoEncoder (14→[4]→14, Recall=0.359, 9 FP sur 30 001)
  → NB06 SHAP + LIME (balance_diff_orig dominant, cohérence 2/3)
  → NB07 LLM Groq   (20 explications, 100% ok, 86s batch)
                      ↓
              explanations.json + llm_summary.json
                      ↓
              API FastAPI + Rapport PwC (NB08)
```

### Fichiers produits (accès API et rapport)

| Fichier | Consommateur |
|---|---|
| `explanations.json` | API FastAPI → endpoint `/explain/{transaction_id}` |
| `llm_summary.json` | Rapport PwC → section "Résultats d'analyse LLM" |
| `35_ollama_analysis.png` | Rapport PwC → graphiques statistiques |
| `36_explanation_examples.png` | Rapport PwC → exemples d'explications |
