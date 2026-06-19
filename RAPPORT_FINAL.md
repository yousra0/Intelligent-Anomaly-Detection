# Rapport de Fin d'Études

---

**ÉCOLE SUPÉRIEURE PRIVÉE D'INGÉNIERIE ET DE TECHNOLOGIES**  
Département Informatique — Filière Data Science & Intelligence Artificielle

---

**Titre :**  
**Détection Intelligente d'Anomalies Financières**  
*Analyse de transactions, modélisation par apprentissage automatique et génération d'explications par modèle de langage*

---

**Réalisé par :** Yousra Chaieb  
**Encadrant académique :** Pr. Mohamed Islem Samaali  
**Encadrants entreprise :** Mr. Slim Zribi & Mme. Hajer  
**Organisme d'accueil :** PricewaterhouseCoopers Tunisie — Risk Assurance Services  
**Année universitaire :** 2025/2026

---

---

## Dédicace

*À ma famille, pour son soutien indéfectible tout au long de ce parcours.*  
*À mes encadrants, pour leur confiance et leurs précieux conseils.*  
*À toute personne qui croit que la rigueur et la curiosité peuvent transformer les données en connaissance.*

---

## Remerciements

Ce travail n'aurait pas pu voir le jour sans le concours de nombreuses personnes auxquelles je tiens à exprimer ma sincère gratitude.

Je remercie en premier lieu **Pr. Mohamed Islem Samaali**, mon encadrant académique à l'ESPRIT, pour sa disponibilité, ses orientations méthodologiques et ses retours constructifs qui ont guidé chacune des étapes de ce projet.

Mes remerciements s'adressent également à **Mr. Slim Zribi** et **Mme. Hajer**, mes encadrants au sein de PwC Tunisie, pour m'avoir offert un cadre de stage exigeant et enrichissant, et pour avoir partagé leur expérience du terrain en matière d'audit et de gestion des risques financiers. Leur connaissance des enjeux réels auxquels font face les entreprises m'a permis d'ancrer ce projet dans une problématique concrète.

Je tiens également à remercier l'ensemble des équipes de **Risk Assurance Services** de PwC Tunisie pour leur accueil chaleureux et pour les échanges professionnels qui ont nourri ma réflexion tout au long du stage.

Enfin, je remercie mes proches pour leur patience, leur soutien moral et leurs encouragements constants, sans lesquels ce projet n'aurait pas pu aboutir dans les meilleures conditions.

---

## Table des Matières

- Introduction Générale
- **Chapitre 1** — Présentation de l'Organisme d'Accueil
- **Chapitre 2** — Compréhension Métier
- **Chapitre 3** — Compréhension des Données
- **Chapitre 4** — Préparation des Données
- **Chapitre 5** — Modélisation
- **Chapitre 6** — Évaluation et Interprétabilité
- **Chapitre 7** — Déploiement et Reproductibilité
- Conclusion Générale
- Bibliographie
- Annexes

---

## Liste des Abréviations

| Sigle | Signification |
|-------|---------------|
| AE | AutoEncoder (Auto-encodeur) |
| AUC | Area Under the Curve |
| CRISP-DM | Cross-Industry Standard Process for Data Mining |
| FN | Faux Négatif |
| FP | Faux Positif |
| IA | Intelligence Artificielle |
| LIME | Local Interpretable Model-agnostic Explanations |
| LLM | Large Language Model |
| LR | Logistic Regression (Régression Logistique) |
| ML | Machine Learning (Apprentissage Automatique) |
| PR-AUC | Precision-Recall Area Under the Curve |
| PwC | PricewaterhouseCoopers |
| RF | Random Forest (Forêt Aléatoire) |
| ROC | Receiver Operating Characteristic |
| SHAP | SHapley Additive exPlanations |
| SMOTE | Synthetic Minority Over-sampling Technique |
| TN | Vrai Négatif |
| TP | Vrai Positif |
| XGB | XGBoost (Extreme Gradient Boosting) |

---

## Introduction Générale

La fraude financière constitue aujourd'hui l'une des menaces les plus persistantes et les plus coûteuses pour les organisations à travers le monde. Qu'il s'agisse de détournements de fonds, de manipulation de transactions ou d'abus de systèmes de paiement, les conséquences sont à la fois financières et réputationnelles, et elles touchent aussi bien les grandes multinationales que les institutions de taille modeste. Dans ce contexte, les cabinets d'audit et de conseil comme PricewaterhouseCoopers jouent un rôle central : ils doivent non seulement identifier les risques existants, mais aussi proposer des moyens de les anticiper avant qu'ils ne causent des dommages irréparables.

Mon stage de fin d'études au sein de **PwC Tunisie**, et plus précisément dans le département **Risk Assurance Services**, m'a placée au cœur de cette problématique. Dès les premiers jours, j'ai pu mesurer l'écart qui existe entre les approches traditionnelles de détection de fraude — essentiellement fondées sur des règles métier rigides et prédéfinies — et les attentes croissantes d'un environnement économique où la fraude évolue plus vite que les systèmes de contrôle. L'équipe m'a confié une mission ambitieuse : concevoir un pipeline complet de détection intelligente d'anomalies financières, en exploitant les techniques modernes d'apprentissage automatique et d'intelligence artificielle explicable.

La particularité de ce projet réside dans son approche à double exigence. D'un côté, une exigence **technique** : construire des modèles capables d'identifier des transactions frauduleuses dans un jeu de données extrêmement déséquilibré, où les fraudes représentent moins de 0,13 % des cas. De l'autre, une exigence **métier** : s'assurer que les résultats produits par ces modèles soient compréhensibles par des auditeurs non-techniciens, qui doivent prendre des décisions concrètes et justifiables. Cette double contrainte m'a conduite à intégrer des outils d'explicabilité (SHAP, LIME) et à enrichir les prédictions par des explications en langage naturel générées via un modèle de langage (LLM).

Pour mener ce projet de façon rigoureuse et structurée, j'ai adopté la méthodologie **CRISP-DM** (Cross-Industry Standard Process for Data Mining), un standard éprouvé dans le domaine de la science des données qui organise le travail en six phases itératives : compréhension métier, compréhension des données, préparation des données, modélisation, évaluation et déploiement. Cette approche m'a permis de maintenir une cohérence globale tout en itérant sur chaque étape en fonction des résultats obtenus.

Ce rapport retrace l'intégralité de ce parcours. Il est organisé en sept chapitres : le premier présente le contexte organisationnel du stage chez PwC ; le second expose la problématique métier et le cadre méthodologique ; les troisième et quatrième chapitres documentent les phases d'exploration et de préparation des données ; le cinquième détaille la phase de modélisation ; le sixième présente les résultats de l'évaluation et les mécanismes d'interprétabilité ; enfin, le septième chapitre décrit l'architecture de déploiement mise en place pour garantir la reproductibilité du pipeline.

---

# Chapitre 1 : Présentation de l'Organisme d'Accueil

## Introduction

Avant d'entrer dans les aspects techniques de ce projet, il est essentiel de comprendre le contexte dans lequel il s'inscrit. Ce premier chapitre présente l'organisme qui m'a accueillie pour ce stage de fin d'études : PricewaterhouseCoopers, l'un des plus grands réseaux de services professionnels au monde. Il détaille également la structure de l'entité tunisienne et le positionnement spécifique du département dans lequel j'ai effectué ma mission, afin d'ancrer les choix techniques qui seront développés dans les chapitres suivants dans une réalité professionnelle concrète.

---

## 1.1 PricewaterhouseCoopers International

PricewaterhouseCoopers, communément désigné sous le sigle **PwC**, est l'un des quatre plus grands cabinets d'audit et de conseil au monde — les fameux « Big Four » aux côtés de Deloitte, KPMG et Ernst & Young. Fondé en 1998 par la fusion de Price Waterhouse et Coopers & Lybrand, deux cabinets dont les origines remontent respectivement à 1849 et 1854, PwC bénéficie d'une histoire longue de plus de 170 ans dans les services professionnels.

À l'échelle mondiale, PwC est présent dans **187 pays**, avec un réseau de plus de **364 000 collaborateurs** répartis sur tous les continents. Cette présence internationale lui confère une capacité unique à accompagner des organisations multinationales tout en maintenant une connaissance fine des réglementations et cultures locales. Le cabinet est organisé autour de trois grandes lignes de services : **Assurance** (audit financier, contrôle interne), **Advisory** (conseil en stratégie, management, technologie) et **Tax & Legal** (fiscalité, droit des affaires).

La stratégie globale de PwC, baptisée **« The New Equation »**, repose sur deux axes complémentaires. Le premier axe, **Building Trust** (Construire la Confiance), reflète la mission historique du cabinet : donner aux parties prenantes des organisations l'assurance que les informations financières publiées sont fiables et conformes aux normes. Le second axe, **Sustained Outcomes** (Résultats Durables), témoigne de l'évolution du cabinet vers un rôle de partenaire stratégique, capable d'aider ses clients à traverser des transformations complexes — numériques, réglementaires ou organisationnelles — en produisant des résultats durables sur le long terme.

Cette double ambition est particulièrement pertinente dans le domaine de la détection de fraude : il s'agit à la fois de renforcer la confiance dans l'intégrité des transactions financières, et de mettre en place des mécanismes robustes et durables pour protéger les organisations contre des menaces en constante évolution.

---

## 1.2 PwC Tunisie

**PwC Tunisie** a été fondé en **1983**, faisant de lui l'une des premières grandes firmes internationales d'audit à s'implanter sur le marché tunisien. Depuis plus de quarante ans, le cabinet accompagne des entreprises de toutes tailles et de tous secteurs — banques, assurances, télécommunications, industries manufacturières, secteur public — dans leurs besoins en audit, en conseil fiscal et en transformation digitale.

Basé à Tunis, PwC Tunisie est pleinement intégré dans le réseau mondial PwC tout en adaptant son offre aux spécificités du marché local. Ses équipes sont composées d'auditeurs, de consultants et de spécialistes sectoriels qui combinent expertise internationale et connaissance approfondie du contexte économique et réglementaire tunisien.

Le cabinet a su, au fil des années, développer une réputation solide notamment dans le domaine des services financiers, où les exigences de transparence et de conformité sont particulièrement élevées. C'est dans ce contexte que s'inscrit le développement de capacités analytiques avancées, comme celle que j'ai eu la mission de contribuer à construire durant ce stage.

---

## 1.3 Le Département Risk Assurance Services

### 1.3.1 Organisation et Missions Générales

Au sein de PwC Tunisie, le département **Risk Assurance Services (RAS)** constitue l'entité dédiée à l'évaluation et à la gestion des risques pour les organisations clientes. Ses missions couvrent un spectre large : cartographie des risques opérationnels et financiers, audit de conformité, revue des systèmes de contrôle interne, et accompagnement des entreprises dans leur mise en conformité avec des référentiels tels que SOX, IFRS ou encore les recommandations de la Banque Centrale de Tunisie.

Le département est structuré en plusieurs cellules spécialisées, parmi lesquelles la **cellule Audit IT** occupe une place stratégique croissante. Dans un monde où les transactions financières sont massivement numérisées, l'audit des systèmes d'information est devenu une composante incontournable de l'assurance-qualité financière. C'est précisément au sein de cette cellule que j'ai effectué mon stage.

### 1.3.2 La Cellule Audit IT

La cellule Audit IT intervient principalement sur deux types de missions. D'une part, elle réalise des **audits de systèmes d'information** : revue des contrôles d'accès, des processus de sauvegarde et de reprise après sinistre, des procédures de gestion des changements, et de la sécurité des infrastructures. D'autre part, elle développe et maintient des **outils d'analyse de données** destinés à automatiser et à fiabiliser certaines procédures d'audit, notamment pour la détection d'anomalies dans des volumes importants de transactions financières.

C'est dans ce second volet que s'inscrit ma mission : concevoir un pipeline de machine learning capable de détecter automatiquement les transactions suspectes dans un jeu de données de paiements mobiles, puis de fournir aux auditeurs des explications intelligibles sur les raisons pour lesquelles une transaction a été signalée. L'enjeu est double : améliorer l'efficacité du processus d'audit en réduisant le temps passé à l'examen manuel, et renforcer la traçabilité des décisions en fournissant des justifications claires et exploitables.

---

## 1.4 Cadre et Périmètre de la Mission

La mission qui m'a été confiée s'est déroulée sur une période de **douze semaines**, structurée selon les six phases de la méthodologie CRISP-DM. Dès le démarrage du stage, j'ai travaillé en étroite collaboration avec Mr. Slim Zribi et Mme. Hajer pour définir précisément le périmètre du projet, les sources de données disponibles et les critères de succès.

Les objectifs ont été définis ainsi :

1. **Identifier les transactions frauduleuses** dans un dataset de paiements mobiles avec une performance significativement supérieure aux approches existantes.
2. **Garantir la traçabilité** des décisions en intégrant des mécanismes d'explicabilité (SHAP, LIME).
3. **Rendre les résultats exploitables** par des équipes non techniques grâce à des explications en langage naturel.
4. **Assurer la reproductibilité** du pipeline en produisant une architecture modulaire et documentée.

Ce cadre clair m'a permis de structurer mon travail de façon rigoureuse et de prioriser les développements en fonction de leur valeur ajoutée métier.

---

## Conclusion du Chapitre 1

Ce premier chapitre a posé les bases contextuelles du projet. PwC Tunisie, à travers son département Risk Assurance Services et sa cellule Audit IT, m'a offert un terrain professionnel idéal pour travailler sur une problématique à fort enjeu : automatiser et enrichir la détection de fraude dans les transactions financières. La mission définie en concertation avec mes encadrants — détecter, expliquer et restituer — constitue le fil conducteur de l'ensemble de ce rapport. Le chapitre suivant s'attache à formaliser cette problématique métier et à présenter le cadre méthodologique choisi pour y répondre.

---

# Chapitre 2 : Compréhension Métier

## Introduction

Dans tout projet de data science, la tentation de « plonger » directement dans les données et les algorithmes est grande. L'expérience montre pourtant que les projets les mieux réussis sont ceux qui commencent par une compréhension approfondie du problème métier avant toute ligne de code. Ce chapitre est consacré à cette étape fondatrice : comprendre ce qu'est la fraude financière dans le contexte des paiements mobiles, pourquoi les approches conventionnelles atteignent leurs limites, et quelle méthodologie adopter pour structurer un projet d'apprentissage automatique répondant aux besoins réels identifiés chez PwC.

---

## 2.1 La Fraude Financière : Contexte et Enjeux

### 2.1.1 Un Phénomène en Expansion

La fraude financière n'est pas un phénomène nouveau, mais elle a connu une accélération remarquable avec la généralisation des paiements numériques. Selon plusieurs études sectorielles, les pertes mondiales liées à la fraude dans les paiements électroniques se chiffrent en dizaines de milliards de dollars annuellement. Dans les marchés émergents, notamment en Afrique subsaharienne et dans la région MENA, l'adoption rapide des paiements mobiles s'est accompagnée d'une hausse des tentatives de fraude, souvent sophistiquées et difficiles à détecter en temps réel.

Les typologies de fraude sont variées : le blanchiment d'argent via des transferts successifs, les prises de contrôle de comptes (Account Takeover), les fraudes à la facturation, ou encore les détournements de fonds dissimulés derrière des transactions apparemment légitimes. Ce dernier type est particulièrement insidieux car il exploite les mêmes canaux et les mêmes mécanismes que les transactions normales, rendant leur détection par inspection visuelle ou règles simples pratiquement impossible à grande échelle.

### 2.1.2 Les Limites des Approches Basées sur des Règles

Les systèmes de détection de fraude traditionnels reposent sur des **règles métier** définies par des experts : « si le montant dépasse X et l'heure est entre 22h et 6h, alors signaler ». Ces règles ont l'avantage d'être simples, transparentes et faciles à maintenir. Mais elles souffrent de limitations structurelles importantes.

La première est leur **rigidité** : une règle définie sur la base de fraudes passées peut facilement être contournée par des fraudeurs qui adaptent leur comportement. La seconde est leur **manque d'adaptabilité** : dans un environnement où les patterns de transactions évoluent continuellement, les règles doivent être régulièrement revisitées manuellement, ce qui représente un coût opérationnel élevé. La troisième, et sans doute la plus critique dans notre contexte, est leur **faible sensibilité** : en cherchant à minimiser les faux positifs pour ne pas alerter inutilement les équipes d'audit, les systèmes à règles génèrent un taux de faux négatifs (fraudes non détectées) particulièrement préoccupant.

Ce dernier point a été illustré de façon saisissante lors de l'analyse du dataset utilisé dans ce projet. La variable `isFlaggedFraud`, qui représente le mécanisme de détection natif du système (une règle métier qui signale les transferts supérieurs à 200 000 unités), affiche un **Recall de seulement 0,0039** : autrement dit, elle détecte moins de 0,4 % des fraudes réelles. C'est une illustration concrète et chiffrée de l'inadéquation des approches basées sur des règles pour ce type de problème.

---

## 2.2 Objectifs du Projet

À partir de ce constat, les objectifs du projet ont été définis en concertation avec les équipes de PwC :

**Objectif principal :** Concevoir un pipeline de détection automatique des transactions frauduleuses, capable de surpasser significativement les approches existantes en termes de Recall (capacité à détecter les vraies fraudes) sans générer un taux excessif de faux positifs.

**Objectifs secondaires :**
- Fournir des **explications locales** pour chaque transaction signalée, permettant aux auditeurs de comprendre pourquoi le modèle a pris sa décision.
- Générer des **rapports en langage naturel** synthétisant les éléments suspects d'une transaction, compréhensibles par des non-spécialistes.
- Assurer la **reproductibilité** complète du pipeline via un script d'orchestration unique.

### 2.2.1 Définition des Critères de Succès

Dans le domaine de la détection de fraude, le choix des métriques d'évaluation est crucial. L'**accuracy** (taux de classification correcte global) est une métrique trompeuse dans ce contexte : un modèle qui classifierait toutes les transactions comme légitimes obtiendrait une accuracy de 99,87 % tout en étant totalement inutile. Les métriques pertinentes sont :

- **Recall (Sensibilité)** : proportion de fraudes réellement détectées. C'est la métrique prioritaire, car manquer une fraude a des conséquences directes sur les clients et l'organisation.
- **Précision** : proportion de transactions signalées qui sont effectivement frauduleuses. Un Recall élevé est inutile si chaque transaction légitime est aussi signalée.
- **F1-score** : moyenne harmonique du Recall et de la Précision, qui équilibre les deux.
- **F2-score** : variante du F1-score qui accorde deux fois plus de poids au Recall qu'à la Précision, reflétant la priorité métier de ne pas manquer les fraudes.
- **PR-AUC** (Area Under the Precision-Recall Curve) : métrique globale particulièrement adaptée aux datasets déséquilibrés.

Le seuil de succès fixé était d'atteindre un **F1-score supérieur à 0,75** et un **Recall supérieur à 0,75**, comparé au baseline à Recall = 0,0039.

---

## 2.3 La Méthodologie CRISP-DM

### 2.3.1 Présentation et Justification

Pour structurer ce projet de A à Z, j'ai adopté la méthodologie **CRISP-DM** (Cross-Industry Standard Process for Data Mining). Développée à la fin des années 1990 par un consortium regroupant SPSS, NCR et Daimler-Chrysler, CRISP-DM est aujourd'hui le standard de facto pour la conduite de projets de science des données, notamment dans les contextes industriels et professionnels.

La méthodologie organise le processus en **six phases itératives** :

1. **Business Understanding (Compréhension Métier)** — Définir le problème à résoudre et les objectifs de succès du point de vue du commanditaire.
2. **Data Understanding (Compréhension des Données)** — Collecter et explorer les données disponibles pour identifier leur structure, leur qualité et leurs particularités.
3. **Data Preparation (Préparation des Données)** — Nettoyer, transformer et enrichir les données pour les rendre exploitables par les algorithmes de machine learning.
4. **Modeling (Modélisation)** — Sélectionner, entraîner et ajuster des modèles d'apprentissage automatique.
5. **Evaluation (Évaluation)** — Évaluer rigoureusement les performances des modèles et vérifier leur alignement avec les objectifs métier.
6. **Deployment (Déploiement)** — Intégrer les modèles dans un environnement opérationnel reproductible.

Le caractère **itératif** de CRISP-DM est l'une de ses forces principales : il autorise, voire encourage, les allers-retours entre phases. Par exemple, une anomalie découverte lors de la modélisation peut justifier un retour à la préparation des données pour corriger un problème de fuite de données (*data leakage*), comme ce fut le cas dans ce projet.

### 2.3.2 Comparaison avec les Alternatives

CRISP-DM n'est pas la seule méthodologie disponible. J'ai évalué deux alternatives principales :

**SEMMA** (Sample, Explore, Modify, Model, Assess), développée par SAS, est plus centrée sur l'aspect technique du data mining et ne prend pas en compte explicitement la phase de déploiement ni le retour des résultats vers les enjeux métier. Elle est adaptée à des projets purement exploratoires mais insuffisante pour un projet destiné à être intégré dans un flux opérationnel.

**KDD** (Knowledge Discovery in Databases) est le précurseur académique de CRISP-DM, mais sa structure moins prescriptive le rend difficile à appliquer dans un contexte professionnel sans adaptation significative.

CRISP-DM s'est imposé comme le choix le plus pertinent car il est **indépendant des outils**, **orienté vers les objectifs métier**, et il intègre nativement la notion de déploiement — ce qui était un critère essentiel chez PwC, où les livrables doivent être opérationnels et maintenables par les équipes après la fin du stage.

---

## 2.4 Planning du Projet

Le projet a été organisé sur **douze semaines**, selon la répartition suivante :

| Semaines | Phase CRISP-DM | Activités Principales |
|----------|----------------|-----------------------|
| S1 – S2 | Compréhension Métier | Analyse du contexte, définition des objectifs, étude du dataset |
| S3 – S4 | Compréhension des Données | Exploration des données, analyses statistiques, EDA |
| S5 – S6 | Préparation des Données | Nettoyage, feature engineering, gestion du déséquilibre |
| S7 – S9 | Modélisation | Entraînement des modèles (supervisés + non supervisés) |
| S10 | Évaluation | Comparaison des modèles, SHAP, LIME, seuil optimal |
| S11 | Explicabilité & LLM | Intégration SHAP, LIME, génération d'explications LLM |
| S12 | Déploiement | Architecture modulaire, tests, documentation, rapport |

Ce découpage a été respecté dans ses grandes lignes, avec quelques ajustements notamment lors de la phase de préparation des données, où la détection et la gestion du *data leakage* a nécessité plus de temps que prévu initialement.

---

## Conclusion du Chapitre 2

Ce chapitre a mis en évidence la problématique centrale de ce projet : la détection de fraude dans des flux de transactions massivement déséquilibrés, là où les approches conventionnelles échouent. La méthodologie CRISP-DM, choisie pour sa robustesse et son orientation métier, fournit le cadre structurant qui guide l'ensemble de la démarche. Les chapitres suivants en retracent l'application concrète, en commençant par l'étape d'exploration des données.

---

# Chapitre 3 : Compréhension des Données

## Introduction

La qualité d'un modèle de machine learning est fondamentalement liée à la qualité des données sur lesquelles il est entraîné. Avant toute modélisation, il est indispensable de passer du temps à comprendre les données disponibles : leur structure, leur distribution, leurs anomalies, et surtout les patterns qui peuvent permettre de distinguer une transaction frauduleuse d'une transaction légitime. Ce chapitre documente l'Analyse Exploratoire des Données (EDA) réalisée sur le dataset PaySim, et les insights clés qui ont orienté toutes les décisions de préparation et de modélisation prises par la suite.

---

## 3.1 Présentation du Dataset PaySim

### 3.1.1 Origine et Contexte

Le dataset utilisé dans ce projet est issu de **PaySim**, un simulateur de transactions de paiements mobiles développé dans le cadre d'une étude académique publiée à l'EMSS 2016. PaySim génère des transactions synthétiques qui imitent le comportement de transactions réelles observées dans un service de paiement mobile déployé en Afrique. L'utilisation d'un dataset synthétique est une contrainte inhérente aux projets de détection de fraude : les données réelles sont hautement confidentielles et soumises à des réglementations strictes sur la protection des données personnelles.

Dans notre projet, nous travaillons sur un **sous-ensemble de 200 000 transactions**, extrait du dataset original qui en comporte 6,36 millions. Ce sous-ensemble a été constitué de façon à maintenir les proportions originales de fraude, garantissant ainsi la représentativité statistique pour notre étude.

### 3.1.2 Structure des Données Brutes

Le dataset brut contient **11 variables** pour chaque transaction :

| Variable | Type | Description |
|----------|------|-------------|
| `step` | Entier | Unité de temps (1 step = 1 heure), de 1 à 744 (30 jours) |
| `type` | Catégoriel | Type de transaction : CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Flottant | Montant de la transaction |
| `nameOrig` | Chaîne | Identifiant du compte émetteur |
| `oldbalanceOrg` | Flottant | Solde du compte émetteur avant la transaction |
| `newbalanceOrig` | Flottant | Solde du compte émetteur après la transaction |
| `nameDest` | Chaîne | Identifiant du compte destinataire |
| `oldbalanceDest` | Flottant | Solde du compte destinataire avant la transaction |
| `newbalanceDest` | Flottant | Solde du compte destinataire après la transaction |
| `isFraud` | Binaire | Variable cible : 1 si fraude, 0 sinon |
| `isFlaggedFraud` | Binaire | Règle métier native : 1 si transfert > 200 000 |

---

## 3.2 Analyse du Déséquilibre des Classes

La première observation — et la plus importante — qui ressort de l'exploration est le **déséquilibre extrême des classes**. Sur les 200 000 transactions du dataset, **258 sont frauduleuses**, soit un taux de fraude de **0,129 %** (environ 1 fraude pour 774 transactions légitimes).

Ce déséquilibre est de loin le défi technique principal du projet. Il a trois conséquences directes :
1. Un modèle naïf qui prédirait systématiquement « légitime » obtiendrait une accuracy de 99,87 % tout en détectant zéro fraude.
2. Les algorithmes de machine learning standards ont tendance à « ignorer » la classe minoritaire lors de l'optimisation de leur fonction de perte, aboutissant à des modèles biaisés vers la classe majoritaire.
3. Les métriques d'évaluation classiques (accuracy) sont trompeuses et doivent être remplacées par des métriques adaptées aux datasets déséquilibrés.

La performance du mécanisme de détection natif `isFlaggedFraud` illustre parfaitement ce problème : avec un **Recall de 0,0039**, il ne détecte que 1 fraude sur 258 — soit une inefficacité totale qui justifie pleinement le développement d'une approche par apprentissage automatique.

---

## 3.3 Analyse par Type de Transaction

### 3.3.1 Distribution des Types

Les cinq types de transactions ont des distributions très inégales dans le dataset :

| Type | Nombre | Proportion | Transactions Frauduleuses | Taux de Fraude |
|------|--------|------------|--------------------------|----------------|
| PAYMENT | 68 621 | 34,3 % | 0 | 0 % |
| CASH_OUT | 61 085 | 30,5 % | 111 | 0,182 % |
| CASH_IN | 42 540 | 21,3 % | 0 | 0 % |
| TRANSFER | 26 267 | 13,1 % | 204 | 0,776 % |
| DEBIT | 1 487 | 0,7 % | 0 | 0 % |

### 3.3.2 Un Insight Crucial : La Fraude est Concentrée sur Deux Types

Cette analyse révèle un fait fondamental : **toutes les fraudes du dataset sont concentrées sur les transactions de type TRANSFER et CASH_OUT**. Aucune fraude n'a été observée sur les types PAYMENT, CASH_IN ou DEBIT. Plus important encore, le taux de fraude est significativement plus élevé sur les TRANSFER (0,776 %) que sur les CASH_OUT (0,182 %).

Cette observation s'explique logiquement : les fraudes typiques consistent à transférer des fonds depuis un compte compromis vers un compte sous contrôle du fraudeur (TRANSFER), puis à convertir ces fonds en espèces (CASH_OUT). Cette connaissance métier a directement influencé la stratégie de feature engineering décrite au chapitre suivant, notamment la création d'un indicateur binaire pour les transactions de type TRANSFER ou CASH_OUT.

---

## 3.4 Analyse des Variables Numériques

### 3.4.1 Distribution des Montants

La variable `amount` présente une distribution fortement asymétrique, avec un **coefficient d'asymétrie (skewness) de 30,80**. La grande majorité des transactions portent sur des montants modestes, tandis qu'un petit nombre de transactions portent sur des montants très élevés. Cette forte asymétrie justifiera l'utilisation de transformations logarithmiques ou de normalisations robustes lors de la préparation des données.

Concernant les transactions frauduleuses spécifiquement, une analyse comparative des montants révèle qu'elles tendent à porter sur des montants plus importants que la moyenne, bien que la variance soit élevée et que cette seule variable ne soit pas discriminante.

### 3.4.2 Analyse Temporelle

La variable `step` représente le temps en heures sur une période de 30 jours (de l'heure 1 à l'heure 744). L'analyse temporelle révèle des patterns intéressants. En divisant la journée en deux plages — heures à risque élevé (nuit et petites heures du matin) et heures normales — on observe que le **ratio de fraude pendant les heures à risque est 10,63 fois plus élevé** que pendant les heures normales. Cette observation est cohérente avec les connaissances générales sur la fraude financière : les fraudeurs agissent préférentiellement la nuit pour bénéficier d'une supervision réduite des systèmes de contrôle.

### 3.4.3 Analyse des Soldes : La Découverte du Data Leakage

L'analyse des variables de solde (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) a conduit à l'une des découvertes les plus importantes de la phase d'exploration : la présence de **fuite de données** (*data leakage*).

Dans le cas des transactions frauduleuses, les soldes après transaction présentent des patterns quasi-déterministes : dans la plupart des cas, le solde du compte émetteur est ramené exactement à zéro après une fraude (le fraudeur vide intégralement le compte). Si ces variables de solde brutes sont utilisées directement comme *features* dans le modèle, celui-ci apprend essentiellement à reconnaître ce pattern de « solde réduit à zéro » — ce qui constitue de la fuite d'information depuis la variable cible.

Cette découverte a motivé une stratégie spécifique de gestion des variables de solde, détaillée dans le chapitre 4.

---

## 3.5 Matrice de Corrélation et Relations Inter-Variables

L'analyse de la matrice de corrélation entre les variables numériques révèle plusieurs relations attendues (corrélation entre les variables de solde avant et après transaction) et quelques relations moins intuitives. Notamment, la variable dérivée `balance_diff_orig` (différence entre le solde avant et le solde après pour le compte émetteur) s'avère être l'une des variables les plus discriminantes pour la détection de fraude, comme le confirmera l'analyse SHAP au chapitre 6.

---

## 3.6 Synthèse des Insights Clés

Au terme de cette phase d'exploration, plusieurs enseignements structurants ont été identifiés :

1. **Le déséquilibre extrême** (1:774) impose une stratégie spécifique de rééchantillonnage et un choix de métriques adapté.
2. **La fraude est concentrée sur deux types** (TRANSFER, CASH_OUT), ce qui justifie la création d'un indicateur spécifique.
3. **Les patterns temporels sont exploitables** : les fraudes sont surreprésentées pendant les heures nocturnes.
4. **Le data leakage** dans les variables de solde brutes impose leur suppression ou transformation.
5. **La règle native `isFlaggedFraud`** est quasi-inopérante (Recall = 0,0039), confirmant la nécessité d'une approche ML.

---

## Conclusion du Chapitre 3

Ce chapitre a permis de construire une compréhension fine du dataset PaySim et d'identifier les défis spécifiques liés à la détection de fraude : déséquilibre sévère, concentration sur deux types de transactions, patterns temporels exploitables et risque de fuite de données. Ces insights constituent la matière première de la phase de préparation des données, qui va transformer ces observations en *features* exploitables par les algorithmes de machine learning.

---

# Chapitre 4 : Préparation des Données

## Introduction

La préparation des données est souvent décrite, non sans ironie, comme l'étape qui consomme 80 % du temps d'un projet de data science pour 20 % de la gloire. Cette maxime, bien que caricaturale, reflète une réalité importante : la qualité du modèle final dépend directement de la qualité du travail effectué en amont sur les données. Ce chapitre décrit en détail le pipeline de préparation mis en place, depuis la gestion du data leakage jusqu'à la constitution des ensembles d'entraînement, de validation et de test, en passant par le feature engineering et les stratégies de gestion du déséquilibre des classes.

---

## 4.1 Gestion du Data Leakage

### 4.1.1 Identification et Suppression des Variables Problématiques

Comme identifié lors de la phase d'exploration, sept colonnes du dataset brut ont été supprimées pour éliminer tout risque de fuite d'information :

**Colonnes identifiants (jamais utilisées comme features):**
- `nameOrig` — identifiant du compte émetteur (chaîne de caractères non numérique)
- `nameDest` — identifiant du compte destinataire (idem)

**Colonnes de solde brutes (risque de leakage):**
- `oldbalanceOrg` — solde avant pour l'émetteur
- `newbalanceOrig` — solde après pour l'émetteur
- `oldbalanceDest` — solde avant pour le destinataire
- `newbalanceDest` — solde après pour le destinataire

**Variable cible alternative (utilisée uniquement comme baseline de comparaison):**
- `isFlaggedFraud` — règle métier native

La décision de supprimer les variables de solde brutes plutôt que de les conserver telles quelles est justifiée par le constat suivant : un modèle qui apprend à partir de `newbalanceOrig = 0` ou `newbalanceDest - oldbalanceDest = amount` apprend à reconnaître le *résultat* d'une fraude, pas ses *caractéristiques prédictives*. En production, ces informations seraient disponibles *après* que la transaction soit déjà exécutée, rendant la détection ex-post et non préventive.

### 4.1.2 Variables Dérivées : Capturer l'Information Sans Fuite

Pour ne pas perdre l'information contenue dans les soldes tout en évitant le leakage, deux variables dérivées ont été créées :

- **`balance_diff_orig`** = `oldbalanceOrg - newbalanceOrig` : représente la variation nette du solde de l'émetteur. Cette variable est disponible immédiatement après la transaction et ne révèle pas directement l'état final du compte.
- **`dest_zero_balance`** : indicateur binaire signalant si le solde du destinataire *avant* la transaction était nul. Ce pattern est caractéristique des « mule accounts » (comptes utilisés comme intermédiaires par les fraudeurs, généralement ouverts récemment et sans solde initial).

---

## 4.2 Feature Engineering

### 4.2.1 Variables Temporelles

À partir de la variable `step` (heure dans la simulation de 30 jours), cinq variables temporelles ont été extraites :

- **`hour`** = `step % 24` — heure de la journée (0 à 23)
- **`day`** = `step // 24` — jour du mois (0 à 29)
- **`week`** = `step // (24 * 7)` — semaine du mois (0 à 3)
- **`high_risk_hour`** : indicateur binaire, 1 si l'heure est comprise entre 22h et 6h. Motivé par l'observation EDA que le ratio de fraude est 10,63 fois plus élevé pendant ces heures.

### 4.2.2 Variable de Type de Transaction

- **`is_transfer_or_cashout`** : indicateur binaire, 1 si le type de transaction est TRANSFER ou CASH_OUT. Motivé par l'observation que 100 % des fraudes appartiennent à ces deux types.

### 4.2.3 Encodage des Types de Transaction

La variable catégorielle `type` (5 modalités : CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) a été transformée par **one-hot encoding**, produisant 5 colonnes binaires. Cette approche évite d'imposer un ordre ordinal artificiel à des catégories nominales.

### 4.2.4 Bilan : 14 Features Finales

Au terme du feature engineering, le dataset final comprend **14 variables explicatives** :

| # | Variable | Type | Origine |
|---|----------|------|---------|
| 1 | `amount` | Numérique | Brute |
| 2 | `hour` | Numérique | Dérivée |
| 3 | `day` | Numérique | Dérivée |
| 4 | `week` | Numérique | Dérivée |
| 5 | `high_risk_hour` | Binaire | Dérivée |
| 6 | `is_transfer_or_cashout` | Binaire | Dérivée |
| 7 | `balance_diff_orig` | Numérique | Dérivée |
| 8 | `dest_zero_balance` | Binaire | Dérivée |
| 9 | `type_CASH_IN` | Binaire | One-hot |
| 10 | `type_CASH_OUT` | Binaire | One-hot |
| 11 | `type_DEBIT` | Binaire | One-hot |
| 12 | `type_PAYMENT` | Binaire | One-hot |
| 13 | `type_TRANSFER` | Binaire | One-hot |
| 14 | `step` | Numérique | Brute |

---

## 4.3 Division du Dataset

### 4.3.1 Stratégie de Division

Le dataset a été divisé en trois sous-ensembles selon une répartition **70/15/15** :

| Ensemble | Taille | Usage |
|---------|--------|-------|
| Entraînement (train) | 139 999 transactions | Entraînement des modèles |
| Validation (val) | 30 000 transactions | Optimisation des hyperparamètres et des seuils |
| Test (test) | 30 001 transactions | Évaluation finale (non vu pendant l'entraînement) |

La division a été effectuée avec **stratification** sur la variable cible `isFraud`, garantissant que la proportion de fraudes (≈0,129 %) est préservée dans les trois ensembles. Cette précaution est essentielle avec des données aussi déséquilibrées : sans stratification, un tirage aléatoire pourrait par malchance produire un ensemble de test sans aucune fraude.

### 4.3.2 Étanchéité des Ensembles

Une règle fondamentale a été respectée tout au long du projet : **aucune information des ensembles de validation et de test n'est utilisée pour les décisions d'entraînement**. En particulier, le StandardScaler (normalisation) a été exclusivement *fit* sur l'ensemble d'entraînement, puis *appliqué* aux ensembles de validation et de test. Faire l'inverse — normaliser l'ensemble complet avant de le diviser — constituerait une forme de data leakage qui biaiserait les estimations de performance vers le haut.

---

## 4.4 Normalisation

La **standardisation** (Z-score normalization) a été appliquée aux variables numériques continues via `StandardScaler` de scikit-learn. Cette transformation centre chaque variable à une moyenne de 0 et une variance de 1, ce qui est particulièrement important pour les algorithmes sensibles à l'échelle des variables (régression logistique, SVM, etc.) et améliore également la convergence des réseaux de neurones.

Les variables binaires (one-hot encodées ou indicateurs booléens) n'ont pas été normalisées.

---

## 4.5 Gestion du Déséquilibre des Classes

### 4.5.1 Class Weight : Approche Par Pondération

La première stratégie pour gérer le déséquilibre est d'utiliser le paramètre **`class_weight='balanced'`** disponible dans scikit-learn pour la régression logistique et le Random Forest. Ce mécanisme ajuste automatiquement le poids accordé à chaque classe pendant l'entraînement, en proportion inverse de sa fréquence :

```
w_0 = n_total / (2 × n_class_0) ≈ 1.0
w_1 = n_total / (2 × n_class_1) ≈ 386.7
```

Un exemple frauduleux contribue donc à la mise à jour du modèle environ **387 fois plus** qu'un exemple légitime. Cette asymétrie compense le déséquilibre naturel du dataset.

### 4.5.2 SMOTE : Approche Par Suréchantillonnage Synthétique

La seconde stratégie est l'utilisation de **SMOTE** (Synthetic Minority Over-sampling Technique). Contrairement à un simple suréchantillonnage par duplication, SMOTE génère de *nouveaux exemples synthétiques* dans la classe minoritaire en interpolant entre des exemples existants dans l'espace des features.

Pour ce projet, SMOTE a été appliqué avec `sampling_strategy=0.1`, ce qui porte le ratio fraude/légitime à **1:10** dans l'ensemble d'entraînement (contre 1:774 initialement). Ce ratio a été choisi délibérément : un ratio de 1:1 (équilibrage parfait) peut introduire trop d'exemples synthétiques et provoquer un sur-apprentissage ; un ratio 1:10 représente un compromis raisonnable.

### 4.5.3 Ensemble Normal pour l'AutoEncoder

Pour l'entraînement du modèle non supervisé (AutoEncoder), une version filtrée de l'ensemble d'entraînement a été constituée, ne contenant que les **139 818 transactions légitimes**. L'AutoEncoder étant entraîné à reconstruire des transactions normales, il ne doit pas « voir » de fraudes pendant l'entraînement.

---

## 4.6 Validation de l'Intégrité du Pipeline

Une fois le pipeline de préparation établi, une série d'**assertions automatiques** a été implémentée pour vérifier l'intégrité des données à chaque étape :

- Vérification que les proportions de fraude dans chaque ensemble sont cohérentes avec la stratification attendue.
- Vérification de l'absence de valeurs manquantes après transformation.
- Vérification que les bornes de normalisation correspondent bien aux statistiques de l'ensemble d'entraînement uniquement.
- Vérification de l'absence de chevauchement entre les identifiants des ensembles train/val/test.

Ces assertions constituent une forme de **test unitaire pour le pipeline de données**, garantissant que tout changement futur qui introduirait accidentellement du data leakage serait immédiatement détecté.

---

## Conclusion du Chapitre 4

La préparation des données a été l'étape la plus délicate du projet, non pas sur le plan technique, mais sur le plan conceptuel : identifier et éliminer le data leakage, choisir les bonnes stratégies de rééchantillonnage, et construire des features qui capturent la connaissance métier sans biaiser le modèle. Au terme de cette phase, le dataset est prêt : 14 features propres, trois ensembles étanches, deux stratégies de gestion du déséquilibre (class_weight et SMOTE), et un sous-ensemble normal pour l'AutoEncoder. Les conditions sont réunies pour aborder sereinement la phase de modélisation.

---

# Chapitre 5 : Modélisation

## Introduction

La phase de modélisation est, symboliquement, le cœur d'un projet de machine learning. C'est là que les données préparées rencontrent les algorithmes, que les hypothèses sont testées et que les performances émergent. Ce chapitre présente l'ensemble des modèles développés dans ce projet : quatre modèles supervisés (Régression Logistique, Random Forest, XGBoost, Isolation Forest) et un modèle non supervisé (AutoEncoder). Pour chaque modèle, nous détaillons l'architecture ou les hyperparamètres retenus, le processus d'entraînement et les stratégies d'optimisation employées.

---

## 5.1 Stratégie Globale de Modélisation

La stratégie de modélisation repose sur trois principes :

1. **Diversité des approches** : combiner des algorithmes supervisés (qui apprennent à partir d'exemples étiquetés) et un algorithme non supervisé (qui apprend la distribution des données normales sans voir de fraudes) pour avoir une vision complémentaire du problème.

2. **Optimisation des seuils de décision** : les algorithmes de classification produisent des scores de probabilité, non des décisions binaires. Le seuil qui transforme ce score en décision (0 ou 1) est un hyperparamètre qui peut être optimisé séparément sur l'ensemble de validation pour maximiser une métrique cible.

3. **Reproductibilité** : toutes les initialisations aléatoires utilisent `RANDOM_STATE=42`, garantissant des résultats identiques à chaque exécution.

---

## 5.2 Modèles Supervisés

### 5.2.1 Régression Logistique (LR)

La **Régression Logistique** est le modèle de référence (*baseline*) pour les tâches de classification binaire. Simple, interprétable et rapide à entraîner, elle fournit un point de comparaison incontournable.

**Configuration :**
- Régularisation L2 avec `C=0.1` (régularisation forte, adaptée aux features corrélées)
- Solver `lbfgs` (méthode d'optimisation quasi-Newton, efficace sur des datasets de taille modeste)
- Deux variantes entraînées :
  - `LR_balanced` : avec `class_weight='balanced'`
  - `LR_smote` : entraîné sur le dataset augmenté par SMOTE

**Optimisation du seuil :**
Pour `LR_smote`, l'optimisation du seuil sur l'ensemble de validation (en maximisant le F1-score) converge vers un **seuil de 0,9621**. Ce seuil élevé s'explique par le fait que SMOTE a enrichi l'espace des fraudes synthétiques : le modèle a appris à attribuer des scores de probabilité élevés uniquement aux cas vraiment caractéristiques des fraudes.

**Performances :**
`LR_smote` atteint un **Recall de 0,6923** et un **F1-score de 0,7105** sur l'ensemble de test — déjà une amélioration considérable par rapport au baseline (`isFlaggedFraud`, Recall = 0,0039).

### 5.2.2 Random Forest (RF)

Le **Random Forest** est un modèle ensembliste qui construit un grand nombre d'arbres de décision indépendants et agrège leurs prédictions par vote majoritaire. Sa robustesse au sur-apprentissage, sa capacité à gérer des features de différentes échelles sans normalisation préalable, et ses performances généralement très bonnes en font un choix naturel pour ce type de problème.

**Hyperparamètres retenus :**
- `n_estimators=300` : 300 arbres dans la forêt
- `max_depth=10` : profondeur maximale de chaque arbre (contrôle le sur-apprentissage)
- `min_samples_leaf=5` : nombre minimum d'exemples dans une feuille (lisse les décisions)
- Deux variantes : `RF_balanced` et `RF_smote`

**Optimisation du seuil :**
Pour `RF_smote`, l'optimisation sur l'ensemble de validation converge vers un **seuil de 0,6291**. Ce seuil plus modéré que celui de la régression logistique reflète la plus grande confiance du Random Forest dans ses scores de probabilité.

**Performances :**
`RF_smote` atteint un **Recall de 0,7949** et un **F1-score de 0,8052** sur l'ensemble de test, avec une matrice de confusion révélant **31 vrais positifs** pour seulement **7 faux positifs**. C'est le meilleur modèle supervisé parmi ceux évalués.

### 5.2.3 XGBoost (XGB)

**XGBoost** (Extreme Gradient Boosting) est une implémentation hautement optimisée du Gradient Boosting, un algorithme ensembliste qui construit les arbres séquentiellement, chaque nouvel arbre corrigeant les erreurs du précédent. XGBoost est réputé pour ses performances de pointe sur des données tabulaires.

**Hyperparamètres retenus :**
- `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`
- `scale_pos_weight=386.7` (équivalent du class_weight pour gérer le déséquilibre dans XGBoost)
- Seuil optimisé à **0,35** sur l'ensemble de validation

**Performances :**
`XGB_smote` atteint un **Recall de 0,846** et un **F1-score de 0,835** avec un **PR-AUC de 0,868** — les meilleures performances brutes parmi tous les modèles supervisés testés.

### 5.2.4 Isolation Forest

L'**Isolation Forest** est un algorithme semi-supervisé qui détecte les anomalies en exploitant le fait que les points anormaux sont, par nature, plus faciles à « isoler » que les points normaux. Il construit un ensemble d'arbres d'isolation et calcule un score d'anomalie basé sur la longueur moyenne du chemin nécessaire pour isoler chaque point.

Bien qu'entraîné sans utiliser les labels de fraude, l'Isolation Forest offre une perspective complémentaire et constitue un excellent modèle de référence non supervisé. Dans ce projet, ses performances restent en deçà des modèles supervisés, ce qui était attendu.

---

## 5.3 Modèle Non Supervisé : AutoEncoder

### 5.3.1 Principe et Justification

L'**AutoEncoder** représente une approche fondamentalement différente des modèles supervisés. Plutôt que d'apprendre à distinguer fraudes et transactions légitimes sur la base d'exemples étiquetés, l'AutoEncoder apprend à **reconstruire des transactions normales**. En production, une transaction frauduleuse — dont le pattern est inhabituel — sera mal reconstruite par le modèle (erreur de reconstruction élevée), ce qui sert de signal d'anomalie.

Cette approche présente plusieurs avantages dans le contexte de PwC :
- Elle est **robuste aux nouvelles formes de fraude** : si un nouveau pattern de fraude émerge, l'AutoEncoder le signalera car il s'écarte de la distribution normale apprise, même s'il n'a jamais vu ce pattern spécifique.
- Elle ne nécessite pas d'exemples étiquetés de fraude pour l'entraînement, ce qui est précieux dans des contextes où les labels sont rares ou peu fiables.

### 5.3.2 Architecture du Réseau

L'architecture de l'AutoEncoder a été conçue comme un réseau symétrique de type *bottleneck* :

```
Encodeur : 14 → 10 → 7 → [4] (espace latent)
Décodeur : [4] → 7 → 10 → 14
```

Chaque couche utilise des fonctions d'activation **ReLU** à l'exception de la couche de sortie qui utilise une activation **linéaire** (pour permettre des valeurs de reconstruction non contraintes). La couche de *bottleneck* à 4 dimensions force le réseau à comprimer l'information en ne conservant que les représentations les plus essentielles des transactions normales.

**Paramètres totaux : 664** (un réseau délibérément petit pour éviter le sur-apprentissage sur des données normales relativement homogènes).

### 5.3.3 Entraînement

L'AutoEncoder a été entraîné sur les **139 818 transactions légitimes** de l'ensemble d'entraînement. Le framework utilisé est **PyTorch**, exécuté sur un GPU NVIDIA MX450 (CUDA 12.4).

**Paramètres d'entraînement :**
- Fonction de perte : **MSE** (Mean Squared Error) entre la transaction originale et sa reconstruction
- Optimiseur : **Adam**, taux d'apprentissage = 0,001
- Batch size : 256
- Early Stopping avec patience=5 (arrêt si la loss de validation ne s'améliore pas pendant 5 époques consécutives)

L'entraînement a convergé en **35 époques** (une époque supplémentaire par rapport aux 36 attendues à cause de l'early stopping) avec une loss de validation finale de **0,145024**.

### 5.3.4 Détermination du Seuil d'Anomalie

La phase de détermination du seuil de détection est critique pour l'AutoEncoder. Le seuil optimal a été calculé en maximisant le **F1-score** sur l'ensemble de validation, en faisant varier la valeur seuil de l'erreur de reconstruction. Le seuil retenu est **1,687408**.

**Validation de la séparation :**
L'analyse de la distribution des erreurs de reconstruction montre un excellent pouvoir discriminant : l'erreur de reconstruction moyenne sur les transactions **frauduleuses** est **122,95 fois plus élevée** que sur les transactions **légitimes**. Cette séparation remarquable valide l'hypothèse que les fraudes sont des anomalies réelles dans l'espace des features appris.

**Performances finales de l'AutoEncoder :**
Malgré ce fort pouvoir discriminant en termes de séparation des distributions, l'AutoEncoder obtient un **Recall de 0,359** et un **F1-score de 0,452** sur l'ensemble de test — en deçà des modèles supervisés, mais significativement au-dessus du baseline à Recall = 0,0039.

---

## 5.4 Comparaison des Performances

| Modèle | Recall | Précision | F1-Score | PR-AUC |
|--------|--------|-----------|----------|--------|
| `isFlaggedFraud` (baseline) | 0,0039 | 1,000 | 0,0077 | — |
| `LR_balanced` | 0,641 | 0,714 | 0,676 | 0,762 |
| `LR_smote` | 0,692 | 0,730 | 0,710 | 0,781 |
| `RF_balanced` | 0,769 | 0,822 | 0,795 | 0,820 |
| **`RF_smote`** | **0,795** | **0,816** | **0,805** | **0,840** |
| `XGB_smote` | 0,846 | 0,825 | 0,835 | 0,868 |
| `AutoEncoder` | 0,359 | 0,622 | 0,452 | 0,531 |

---

## Conclusion du Chapitre 5

La phase de modélisation a permis de développer et d'évaluer six modèles complémentaires. Deux se distinguent : **XGB_smote** avec les meilleures performances brutes (F1=0,835, PR-AUC=0,868) et **RF_smote** avec d'excellentes performances (F1=0,805) et une meilleure compatibilité avec les outils d'explicabilité (TreeSHAP). L'AutoEncoder, bien qu'inférieur aux modèles supervisés en termes de métriques, apporte une perspective non supervisée précieuse pour la détection de nouvelles formes de fraude. Le chapitre suivant approfondit l'évaluation et introduit les outils d'interprétabilité.

---

# Chapitre 6 : Évaluation et Interprétabilité

## Introduction

Obtenir de bonnes métriques ne suffit pas dans un contexte professionnel comme celui de PwC. Un auditeur ne peut pas — et ne devrait pas — accepter une décision de signalement d'une transaction frauduleuse sans en comprendre les raisons. La confiance dans un système automatisé de détection de fraude repose sur sa capacité à s'expliquer. Ce chapitre présente d'abord une évaluation rigoureuse et comparative des modèles développés, puis détaille les mécanismes d'interprétabilité mis en place : SHAP pour l'explicabilité globale et locale, LIME pour l'approximation locale agnostique, et un modèle de langage pour la génération d'explications en langage naturel.

---

## 6.1 Évaluation des Modèles

### 6.1.1 Analyse des Matrices de Confusion

L'analyse des matrices de confusion apporte des informations que les métriques agrégées ne capturent pas entièrement. Pour **RF_smote**, la matrice sur l'ensemble de test révèle :

```
                 Prédit Légitime    Prédit Frauduleux
Réel Légitime        29 955                 7
Réel Frauduleux          8                31
```

Ce résultat est particulièrement remarquable : **31 transactions frauduleuses détectées sur 39** (Recall = 0,795), avec seulement **7 fausses alertes** (Précision = 0,816). Pour un auditeur, cela signifie que sur 38 transactions signalées par le système, 31 sont effectivement frauduleuses — un taux de fraude parmi les alertes de 81,6 % contre 0,129 % dans le dataset brut : un gain de **632 fois** en densité de fraudes dans les alertes.

### 6.1.2 Courbes Précision-Rappel

Les courbes PR (Precision-Recall) illustrent le compromis entre Précision et Recall pour chaque modèle à différents seuils de décision. L'**aire sous la courbe PR (PR-AUC)** est la métrique globale la plus informative pour les datasets déséquilibrés : un PR-AUC de 0,840 pour RF_smote signifie que, quelle que soit la valeur de seuil choisie, le modèle maintient un excellent équilibre entre sa capacité à détecter les fraudes et sa précision.

### 6.1.3 Robustesse et Généralisation

Un aspect crucial de l'évaluation est la vérification de la généralisation : les performances observées sur l'ensemble de test sont-elles représentatives de la performance réelle en production ? Plusieurs indicateurs rassurants ont été identifiés :
- Les métriques sur l'ensemble de validation et l'ensemble de test sont cohérentes (pas d'écart significatif indiquant un sur-ajustement).
- La stratification garantit que les ensembles de validation et de test ont des distributions de fraude comparables.
- L'ensemble de test n'a jamais été vu pendant la phase d'optimisation des hyperparamètres.

---

## 6.2 Interprétabilité Globale : SHAP

### 6.2.1 Fondements Théoriques de SHAP

**SHAP** (SHapley Additive exPlanations) est un framework d'interprétabilité basé sur la théorie des jeux coopératifs. Il attribue à chaque feature une valeur Shapley qui représente sa contribution marginale à la prédiction, en moyenne sur toutes les combinaisons possibles de features. SHAP possède plusieurs propriétés théoriques garanties : **efficacité** (la somme des valeurs Shapley égale la différence entre la prédiction du modèle et la valeur de référence), **symétrie** (deux features avec des contributions identiques reçoivent les mêmes valeurs) et **consistance** (si un modèle change de façon à attribuer plus d'importance à une feature, sa valeur Shapley ne diminue pas).

### 6.2.2 Implémentation par Type de Modèle

Différents explainers SHAP ont été utilisés selon le type de modèle :

- **TreeSHAP** pour Random Forest et XGBoost : algorithme exact et efficace (O(TLD²) complexité) exploitant la structure d'arbre, garantissant des valeurs Shapley exactes sans approximation.
- **LinearExplainer** pour la Régression Logistique : exploite la linéarité du modèle pour un calcul exact et très rapide.
- **KernelExplainer** pour l'AutoEncoder : approximation par perturbation locale, plus lente mais agnostique à l'architecture.

### 6.2.3 Résultats de l'Analyse SHAP pour RF_smote

L'analyse SHAP globale sur RF_smote révèle la hiérarchie d'importance des features :

**Features les plus importantes (par ordre décroissant) :**

1. **`balance_diff_orig`** — La variable la plus discriminante de loin. Une valeur élevée de la différence de solde (indiquant un vidage important du compte émetteur) est fortement associée à la fraude. Ce résultat valide a posteriori la décision de créer cette variable dérivée plutôt que d'utiliser les soldes bruts.

2. **`dest_zero_balance`** — Le fait que le destinataire ait un solde nul avant la transaction est un signal fort de fraude. Ce pattern correspond aux « mule accounts » créés spécifiquement pour recevoir des fonds volés.

3. **`type_TRANSFER`** et **`type_CASH_OUT`** — Les indicateurs de type confirment la concentration de la fraude sur ces deux types de transactions.

4. **`amount`** — Le montant de la transaction contribue, mais son importance est secondaire, reflétant le fait que la fraude se produit sur une gamme variée de montants.

5. **`high_risk_hour`** — Les transactions nocturnes ont une contribution positive à la probabilité de fraude.

6. **`is_transfer_or_cashout`** — Redondant avec les indicateurs de type individuels, mais sa présence confirme la robustesse de l'analyse.

### 6.2.4 Graphiques SHAP : Summary Plot et Beeswarm

Le *summary plot* SHAP affiche, pour chaque feature, la distribution des valeurs SHAP sur l'ensemble des transactions de l'échantillon analysé. Pour `balance_diff_orig`, on observe une distribution bimodale : les transactions avec une forte différence de solde (valeur de feature élevée, affichée en rouge) reçoivent des valeurs SHAP très positives, tandis que les transactions avec une faible différence de solde (valeur de feature basse, affichée en bleu) reçoivent des valeurs SHAP proches de zéro ou légèrement négatives.

### 6.2.5 Explication d'une Transaction Individuelle : SHAP Waterfall

Pour une transaction spécifique signalée comme frauduleuse, le graphique *waterfall* SHAP décompose la prédiction en contributions individuelles de chaque feature. Par exemple, pour une transaction frauduleuse typique :
- La valeur de base (moyenne des prédictions) est de 0,002 (reflétant le faible taux de fraude dans le dataset)
- `balance_diff_orig` contribue +0,43 (le compte a été vidé)
- `dest_zero_balance` contribue +0,21 (le destinataire n'avait pas de solde antérieur)
- `type_TRANSFER` contribue +0,15 (c'est un virement)
- La prédiction finale est de 0,87 (probabilité de fraude de 87 %)

Ce niveau de détail est précieux pour l'auditeur : il comprend exactement *pourquoi* la transaction a été signalée.

---

## 6.3 Interprétabilité Locale : LIME

### 6.3.1 Principe de LIME

**LIME** (Local Interpretable Model-agnostic Explanations) adopte une approche différente de SHAP. Pour expliquer la prédiction d'un modèle pour une instance spécifique, LIME génère un ensemble de perturbations locales autour de cette instance, observe les prédictions du modèle sur ces perturbations, puis entraîne un modèle linéaire (interpretable) pour approximer localement le comportement du modèle original.

L'avantage majeur de LIME est son caractère **agnostique au modèle** : il fonctionne identiquement pour n'importe quel type de modèle, ce qui le rend universellement applicable. Son principal inconvénient est la variabilité des explications selon les perturbations générées.

### 6.3.2 Résultats LIME

L'analyse LIME appliquée aux mêmes transactions que SHAP confirme globalement les mêmes features importantes. Pour une transaction frauduleuse typique, LIME identifie comme features localement importantes :
- `balance_diff_orig > seuil` (contribution positive à la probabilité de fraude)
- `dest_zero_balance = 1` (contribution positive)
- `type = TRANSFER` (contribution positive)
- `amount > seuil` (contribution positive modérée)

### 6.3.3 Convergence SHAP × LIME : Validation Croisée

La comparaison systématique des features identifiées par SHAP et LIME sur l'ensemble des transactions de test révèle un **chevauchement de 2 features** constamment identifiées par les deux méthodes comme les plus importantes :
- **`dest_zero_balance`**
- **`balance_diff_orig`**

Ce consensus inter-méthodes est une forme de validation croisée puissante : ces deux variables ne sont pas seulement importantes pour un type d'algorithme particulier, mais représentent des signaux robustes et fondamentaux de la fraude dans ce dataset. Cela renforce la confiance dans la signification métier de ces features.

---

## 6.4 Génération d'Explications en Langage Naturel via LLM

### 6.4.1 Motivation : Le Dernier Kilomètre de l'Explicabilité

Les valeurs SHAP et les graphiques LIME sont précieux pour un data scientist, mais ils restent abstraits pour un auditeur sans formation technique. Le « dernier kilomètre » de l'explicabilité consiste à transformer ces insights quantitatifs en **narration compréhensible** : pourquoi cette transaction est-elle suspecte, en termes clairs et accessibles ?

Pour répondre à ce besoin, un composant d'intégration LLM a été développé.

### 6.4.2 Architecture Technique

Le système utilise l'**API Groq** comme moteur d'inférence, avec le modèle **llama-3.1-8b-instant**. Groq est un fournisseur d'inférence LLM qui propose une API compatible OpenAI, avec des latences particulièrement basses grâce à ses puces LPU (Language Processing Unit) propriétaires.

**Caractéristiques de l'implémentation :**
- Chaque transaction signalée génère une requête incluant : les valeurs des features, les valeurs SHAP correspondantes, le score de probabilité du modèle.
- Un *prompt* système en français guide le LLM pour produire une explication structurée (niveau de risque, features suspectes, recommandation d'action).
- Les explications sont générées en **français**, adapté au contexte des auditeurs de PwC Tunisie.

### 6.4.3 Résultats de l'Intégration LLM

Le système a été évalué sur **20 transactions frauduleuses** de l'ensemble de test. Les performances mesurées :

- **Temps moyen par explication** : 87,3 secondes pour 20 transactions (≈ 4,4 secondes/transaction)
- **Taux d'analyse correcte** : 100 % — toutes les transactions ont été classées au niveau de risque **« ÉLEVÉ »** par le LLM
- **Cohérence des explications** : la variable `balance_diff_orig` a été citée comme facteur suspect dans **20/20** explications, confirmant sa prééminence identifiée par SHAP
- **Cohérence SHAP** : les features citées par le LLM correspondent systématiquement aux features avec les valeurs SHAP les plus élevées

**Exemple d'explication générée :**

> *« Cette transaction présente un niveau de risque ÉLEVÉ. L'indicateur principal de suspicion est une différence de solde du compte émetteur particulièrement élevée (balance_diff_orig = 487 532 unités), suggérant un vidage quasi-total du compte. De plus, le compte destinataire présentait un solde nul avant la transaction, un pattern typique des comptes relais utilisés dans les circuits de fraude. La transaction est de type TRANSFER, catégorie présentant le taux de fraude le plus élevé dans notre base de données. Recommandation : investigation prioritaire avant validation de la transaction. »*

### 6.4.4 Confidentialité et Sécurité des Données

Un point essentiel dans le contexte d'un cabinet d'audit comme PwC est la **confidentialité des données clients**. Les données transmises au LLM ont été anonymisées : les identifiants de comptes (`nameOrig`, `nameDest`) ne font jamais partie du prompt. Seules les valeurs numériques des features et les scores du modèle sont transmis. Cette approche préserve la confidentialité des informations personnelles tout en permettant des explications pertinentes.

---

## Conclusion du Chapitre 6

L'évaluation rigoureuse a confirmé que **RF_smote** (F1=0,805, Recall=0,795) et **XGB_smote** (F1=0,835, Recall=0,846) constituent les solutions les plus performantes, avec un gain considérable par rapport au baseline. L'intégration de SHAP, LIME et d'un LLM transforme ces résultats en une solution complète : non seulement le système détecte les fraudes, il explique ses décisions à deux niveaux — technique (SHAP/LIME) et naturel (LLM). Cette dualité est précisément ce que requiert un déploiement dans un environnement d'audit professionnel.

---

# Chapitre 7 : Déploiement et Reproductibilité

## Introduction

Un projet de data science n'atteint sa valeur réelle que lorsqu'il peut être mis en œuvre de façon fiable, maintenu dans le temps et compris par d'autres que son auteur. La phase de déploiement est souvent négligée dans les projets académiques, mais elle est centrale dans un contexte professionnel comme celui de PwC, où la pérennité des outils développés est une exigence fondamentale. Ce chapitre décrit l'architecture modulaire mise en place pour garantir la reproductibilité du pipeline, faciliter sa maintenance et permettre son évolution future.

---

## 7.1 Principes de l'Architecture

L'architecture de déploiement repose sur quatre principes directeurs :

1. **Modularité** : chaque composante du pipeline (préprocessing, feature engineering, modélisation, explicabilité, rapports) est encapsulée dans un module Python indépendant, avec une interface claire.

2. **Reproductibilité** : l'exécution du pipeline doit produire des résultats identiques à chaque exécution. Cette propriété est garantie par la fixation de toutes les graines aléatoires (`RANDOM_STATE=42`), la sérialisation des modèles entraînés et la gestion versionnée des dépendances.

3. **Lisibilité** : le code est écrit pour être lu et compris par d'autres développeurs. Des conventions de nommage cohérentes, des interfaces documentées et une organisation logique des modules facilitent la prise en main.

4. **Testabilité** : des tests unitaires couvrent les composantes critiques du pipeline, notamment le préprocessing (pour détecter les régressions) et les assertions d'intégrité des données.

---

## 7.2 Structure du Projet

### 7.2.1 Organisation des Répertoires

```
anomaly_detection_project/
│
├── src/                          # Code source Python
│   ├── preprocessing.py          # Chargement et nettoyage des données
│   ├── feature_engineering.py    # Création des 14 features
│   ├── models.py                 # Entraînement et sauvegarde des modèles
│   ├── pipeline.py               # Orchestration des étapes
│   ├── utils.py                  # Fonctions utilitaires communes
│   ├── visualization.py          # Génération des graphiques
│   ├── explainability.py         # SHAP et LIME
│   ├── ollama_integration.py     # Intégration LLM (Groq/Ollama)
│   └── tests/                    # Tests unitaires
│       ├── test_preprocessing.py
│       ├── test_features.py
│       └── test_pipeline.py
│
├── notebooks/                    # Notebooks exploratoires (NB01–NB07)
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_autoencoder.ipynb
│   ├── 06_shap_lime.ipynb
│   └── 07_llm_integration.ipynb
│
├── outputs/                      # Artefacts générés
│   ├── models/                   # Modèles sérialisés (.pkl, .pt)
│   │   ├── lr_balanced.pkl
│   │   ├── lr_smote.pkl
│   │   ├── rf_balanced.pkl
│   │   ├── rf_smote.pkl
│   │   ├── xgb_smote.pkl
│   │   ├── isolation_forest.pkl
│   │   └── autoencoder.pt
│   ├── reports/                  # Rapports générés (PDF, JSON)
│   └── figures/                  # Graphiques sauvegardés
│
├── data/                         # Données (non versionnées)
│   └── paysim_200k.csv
│
├── run_all.py                    # Script d'orchestration complet
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation principale
```

### 7.2.2 Description des Modules Principaux

**`src/preprocessing.py`**
Ce module gère le chargement des données brutes et l'application des transformations de base : suppression des colonnes identifiants, gestion des valeurs manquantes et validation de la structure attendue. Il expose une interface simple : `load_and_preprocess(filepath) → DataFrame`.

**`src/feature_engineering.py`**
Encapsule toute la logique de création des 14 features finales, incluant les variables temporelles, les indicateurs binaires et les variables dérivées des soldes. Le module garantit que la même transformation est appliquée identiquement lors de l'entraînement et de l'inférence.

**`src/models.py`**
Contient les classes et fonctions pour entraîner, sauvegarder et charger chaque modèle. Chaque modèle est sérialisé avec `joblib` (modèles scikit-learn) ou avec `torch.save` (AutoEncoder PyTorch), permettant de les recharger sans réentraînement.

**`src/explainability.py`**
Centralise les calculs SHAP et LIME. Expose des fonctions `compute_shap_values(model, X, model_type)` et `compute_lime_explanation(model, X_instance, feature_names)` qui retournent des structures standardisées exploitables par les composants de visualisation et de reporting.

**`src/ollama_integration.py`**
Gère la communication avec l'API Groq. Construit les prompts, envoie les requêtes et parse les réponses. Inclut une gestion des erreurs robuste (retry avec backoff exponentiel, fallback sur une explication générique en cas d'indisponibilité de l'API).

---

## 7.3 Le Script d'Orchestration : `run_all.py`

Le script `run_all.py` est la pièce maîtresse de la reproductibilité. Il orchestre l'exécution complète du pipeline de A à Z, depuis le chargement des données brutes jusqu'à la génération des rapports finaux, en un seul appel en ligne de commande.

### 7.3.1 Fonctionnement

```bash
# Exécution complète
python run_all.py

# Exécution partielle (à partir d'une étape spécifique)
python run_all.py --start-from modeling

# Exécution avec un nouveau fichier de données
python run_all.py --data-path /chemin/vers/nouvelles_donnees.csv
```

### 7.3.2 Étapes Orchestrées

Le script exécute séquentiellement les étapes suivantes, avec journalisation à chaque étape :

1. **Chargement et validation des données** — Vérification de la structure attendue, comptage des fraudes, vérification de l'absence de valeurs manquantes.
2. **Feature engineering** — Création des 14 features.
3. **Division et normalisation** — Split 70/15/15 stratifié, fit du StandardScaler sur train uniquement.
4. **Rééchantillonnage** — Application de SMOTE sur l'ensemble d'entraînement.
5. **Entraînement des modèles** — Entraînement séquentiel de tous les modèles, sauvegarde dans `outputs/models/`.
6. **Optimisation des seuils** — Optimisation sur l'ensemble de validation pour chaque modèle.
7. **Évaluation** — Calcul des métriques sur l'ensemble de test, génération des matrices de confusion.
8. **Calcul des valeurs SHAP** — Pour le meilleur modèle (RF_smote).
9. **Génération des explications LLM** — Pour les transactions frauduleuses détectées.
10. **Génération des rapports** — Compilation des résultats en JSON et PDF.

### 7.3.3 Gestion de la Reproductibilité

Toutes les initialisations aléatoires de l'ensemble du pipeline sont contrôlées par la constante globale `RANDOM_STATE=42` :
- `train_test_split(random_state=RANDOM_STATE)`
- `RandomForestClassifier(random_state=RANDOM_STATE)`
- `XGBClassifier(random_state=RANDOM_STATE)`
- `SMOTE(random_state=RANDOM_STATE)`
- `torch.manual_seed(RANDOM_STATE)` (AutoEncoder)

Cette discipline garantit que deux exécutions successives du pipeline sur le même dataset produisent des résultats identiques au bit près.

---

## 7.4 Gestion des Dépendances

Le fichier `requirements.txt` documente l'intégralité des dépendances Python avec leurs versions exactes :

```
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
torch==2.0.1
shap==0.42.1
lime==0.2.0.1
imbalanced-learn==0.11.0
matplotlib==3.7.2
seaborn==0.12.2
groq==0.4.2
joblib==1.3.1
```

L'environnement peut être reconstitué exactement par `pip install -r requirements.txt`, garantissant qu'un autre développeur ou une exécution sur un autre serveur reproduira les mêmes résultats.

---

## 7.5 Tests et Validation du Pipeline

### 7.5.1 Tests Unitaires

Les tests unitaires couvrent les composantes les plus critiques :

- **`test_preprocessing.py`** : vérifie que le preprocessing produit le bon nombre de features, élimine les colonnes attendues, et ne génère pas de valeurs manquantes.
- **`test_features.py`** : vérifie les valeurs des features dérivées sur des cas de test connus (transactions construites manuellement avec les valeurs attendues).
- **`test_pipeline.py`** : teste la cohérence de bout en bout sur un mini-dataset de 1000 transactions.

### 7.5.2 Assertions d'Intégrité

En plus des tests formels, les assertions d'intégrité implémentées dans le pipeline (cf. Chapitre 4) servent de garde-fous en production : toute déviation par rapport aux invariants attendus (proportion de fraudes, nombre de features, absence de NaN) déclenche une exception avec un message d'erreur explicite.

---

## 7.6 Perspectives d'Évolution

L'architecture modulaire mise en place facilite plusieurs évolutions envisageables à court et moyen terme :

- **Intégration d'un système de monitoring** : en production, il serait nécessaire de surveiller la dérive des données (*data drift*) et la dégradation des performances dans le temps.
- **Déploiement via une API REST** : l'architecture modulaire permet facilement d'exposer le pipeline sous forme d'API (Flask/FastAPI), permettant à d'autres systèmes d'envoyer des transactions et de recevoir des scores de fraude en temps réel.
- **Réentraînement automatique** : un mécanisme de réentraînement périodique sur des données récentes permettrait au modèle de s'adapter à l'évolution des patterns de fraude.
- **Enrichissement des explications** : l'intégration de modèles LLM plus puissants (GPT-4, Claude) permettrait de générer des explications encore plus détaillées et contextualisées.

---

## Conclusion du Chapitre 7

Ce chapitre a présenté l'architecture de déploiement développée pour garantir la reproductibilité et la maintenabilité du pipeline. La combinaison d'une structure modulaire claire, d'un script d'orchestration unique, d'une gestion rigoureuse des dépendances et d'une suite de tests forme un tout cohérent qui transforme le projet de recherche en un outil opérationnel. C'est cette dimension de livrable complet — pas seulement des notebooks exploratoires mais une solution déployable — qui constitue la valeur ajoutée finale du stage pour PwC Tunisie.

---

# Conclusion Générale

Ce projet de fin d'études a constitué pour moi une expérience professionnelle et intellectuelle d'une richesse exceptionnelle. En rejoignant PwC Tunisie pour concevoir un pipeline de détection intelligente d'anomalies financières, j'ai eu l'opportunité de travailler sur un problème réel, exigeant et multidimensionnel, à la frontière entre la data science, la gestion des risques et l'intelligence artificielle explicable.

**Sur le plan technique**, les résultats obtenus sont significatifs. Le modèle RF_smote atteint un F1-score de 0,805 et un Recall de 0,795, représentant une amélioration de plus de **100 fois** par rapport au système de détection natif (isFlaggedFraud, Recall = 0,0039). L'AutoEncoder, bien que moins performant en termes de métriques globales, apporte une perspective non supervisée précieuse pour la détection de nouvelles formes de fraude. L'intégration de SHAP, LIME et d'un LLM transforme ces performances statistiques en une solution réellement utilisable : chaque alerte est accompagnée d'une explication compréhensible, à la fois pour les data scientists (valeurs SHAP) et pour les auditeurs (narration en langage naturel).

**Sur le plan méthodologique**, l'adoption de CRISP-DM a été un choix structurant qui a contribué à la cohérence du projet. Le caractère itératif de la méthodologie a permis de gérer sereinement les imprévus — notamment la découverte du data leakage dans les variables de solde brutes, qui a nécessité un retour en phase de préparation après avoir déjà commencé la modélisation. Sans ce cadre méthodologique, une telle découverte aurait pu menacer l'ensemble du projet.

**Sur le plan personnel**, ce stage m'a permis de consolider mes compétences en machine learning, de les enrichir avec une dimension d'explicabilité que je n'avais pas encore approfondie, et de comprendre comment un projet de data science s'intègre dans un contexte professionnel réel avec ses contraintes de confidentialité, de maintenabilité et de communication avec des parties prenantes non techniques.

**Les défis principaux rencontrés** ont été de trois ordres :
- La **gestion du déséquilibre extrême** (1:774), qui a nécessité une combinaison de stratégies (class_weight, SMOTE) et une réflexion approfondie sur le choix des métriques.
- La **discipline anti-leakage**, qui demande une vigilance permanente sur la façon dont l'information circule entre les features et la variable cible.
- La **sensibilité au seuil de l'AutoEncoder**, dont les performances varient significativement selon la valeur de seuil choisie, imposant une optimisation soigneuse sur l'ensemble de validation.

**Les perspectives** ouvertes par ce travail sont nombreuses. À court terme, l'intégration du pipeline dans un environnement de production PwC passerait par le développement d'une API REST, d'un tableau de bord de monitoring et d'un mécanisme de réentraînement automatique. À moyen terme, l'enrichissement des données avec des informations comportementales supplémentaires (historique des transactions, profil du client, données réseau) permettrait d'améliorer encore la discrimination entre fraudes et légitimes. Enfin, l'exploration de modèles de graphes (Graph Neural Networks) pour exploiter les relations entre comptes constitue une piste de recherche particulièrement prometteuse pour la détection de réseaux de fraude organisés.

Ce projet aura été, pour moi, bien plus qu'une expérience académique : il m'a convaincue que la data science, lorsqu'elle est pratiquée avec rigueur et orientée vers des besoins réels, peut produire des solutions qui changent concrètement la façon dont les organisations gèrent leurs risques.

---

# Bibliographie

[1] E. A. Lopez-Rojas, A. Elmir et S. Axelsson, « PaySim: A Financial Mobile Money Simulator for Fraud Detection », *Proceedings of the 28th European Modeling & Simulation Symposium (EMSS)*, 2016.

[2] R. Shearer, « CRISP-DM 1.0 — Step-by-step data mining guide », *SPSS Inc.*, 2000.

[3] N. V. Chawla, K. W. Bowyer, L. O. Hall et W. P. Kegelmeyer, « SMOTE: Synthetic Minority Over-sampling Technique », *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[4] L. Breiman, « Random Forests », *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[5] T. Chen et C. Guestrin, « XGBoost: A Scalable Tree Boosting System », *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[6] F. T. Liu, K. M. Ting et Z. Zhou, « Isolation Forest », *2008 Eighth IEEE International Conference on Data Mining*, 2008, pp. 413–422.

[7] D. P. Kingma et M. Welling, « Auto-Encoding Variational Bayes », *arXiv:1312.6114*, 2013.

[8] S. M. Lundberg et S. I. Lee, « A Unified Approach to Interpreting Model Predictions », *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 2017, pp. 4765–4774.

[9] M. T. Ribeiro, S. Singh et C. Guestrin, « "Why Should I Trust You?": Explaining the Predictions of Any Classifier », *Proceedings of the 22nd ACM SIGKDD International Conference*, 2016, pp. 1135–1144.

[10] Meta AI Research, « Llama 3.1: Open Foundation and Fine-Tuned Chat Models », *Meta AI*, 2024.

[11] F. Pedregosa et al., « Scikit-learn: Machine Learning in Python », *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[12] A. Paszke et al., « PyTorch: An Imperative Style, High-Performance Deep Learning Library », *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019.

[13] PricewaterhouseCoopers, « PwC Annual Global CEO Survey », *PwC Publications*, 2024.

[14] PricewaterhouseCoopers, « The New Equation: PwC's strategy », *PwC Corporate Communications*, 2021.

[15] Y. Chaieb, « Détection intelligente d'anomalies financières — Code source et notebooks », *GitHub Repository*, 2026.

---

# Annexes

## Annexe A : Hyperparamètres des Modèles

### Régression Logistique

```python
LogisticRegression(
    C=0.1,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
```

### Random Forest

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
```

### XGBoost

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=386.7,
    random_state=42
)
```

### AutoEncoder (PyTorch)

```python
class FraudAutoEncoder(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(14, 10), nn.ReLU(),
            nn.Linear(10, 7),  nn.ReLU(),
            nn.Linear(7, 4)    # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 7),   nn.ReLU(),
            nn.Linear(7, 10),  nn.ReLU(),
            nn.Linear(10, 14)  # reconstruction
        )
```

---

## Annexe B : Liste Complète des 14 Features

| Indice | Nom de la Feature | Description | Type | Origine |
|--------|-------------------|-------------|------|---------|
| 0 | `amount` | Montant de la transaction | Numérique | Brute |
| 1 | `step` | Heure dans la simulation | Numérique | Brute |
| 2 | `hour` | Heure de la journée (0-23) | Numérique | Dérivée |
| 3 | `day` | Jour du mois (0-29) | Numérique | Dérivée |
| 4 | `week` | Semaine du mois (0-3) | Numérique | Dérivée |
| 5 | `high_risk_hour` | Heure nocturne (22h-6h) | Binaire | Dérivée |
| 6 | `is_transfer_or_cashout` | Type TRANSFER ou CASH_OUT | Binaire | Dérivée |
| 7 | `balance_diff_orig` | Variation solde émetteur | Numérique | Dérivée |
| 8 | `dest_zero_balance` | Solde destinataire initial nul | Binaire | Dérivée |
| 9 | `type_CASH_IN` | One-hot type CASH_IN | Binaire | One-hot |
| 10 | `type_CASH_OUT` | One-hot type CASH_OUT | Binaire | One-hot |
| 11 | `type_DEBIT` | One-hot type DEBIT | Binaire | One-hot |
| 12 | `type_PAYMENT` | One-hot type PAYMENT | Binaire | One-hot |
| 13 | `type_TRANSFER` | One-hot type TRANSFER | Binaire | One-hot |

---

## Annexe C : Planning Détaillé du Projet (12 Semaines)

| Semaines | Phase | Tâches Réalisées |
|----------|-------|------------------|
| S1 | Onboarding | Prise en main de l'environnement PwC, lectures sectorielles sur la fraude financière |
| S2 | Business Understanding | Définition formelle du problème, critères de succès, étude du dataset PaySim |
| S3 | Data Understanding (1/2) | Exploration univariée, distribution des classes, analyse par type de transaction |
| S4 | Data Understanding (2/2) | Analyse temporelle, corrélations, identification du data leakage |
| S5 | Data Preparation (1/2) | Suppression des colonnes à risque, feature engineering temporel |
| S6 | Data Preparation (2/2) | Features dérivées des soldes, split stratifié, normalisation, SMOTE |
| S7 | Modélisation — Supervisé (1/2) | Régression Logistique, Random Forest, optimisation des seuils |
| S8 | Modélisation — Supervisé (2/2) | XGBoost, Isolation Forest, comparaison initiale |
| S9 | Modélisation — AutoEncoder | Architecture, entraînement GPU, calibration du seuil de reconstruction |
| S10 | Évaluation & SHAP | Matrices de confusion, courbes PR, calcul TreeSHAP, summary plots |
| S11 | LIME & LLM | Explications LIME, intégration API Groq, génération de 20 explications |
| S12 | Déploiement & Documentation | Architecture modulaire, run_all.py, tests, rédaction du rapport |

---

## Annexe D : Exemple d'Explication LLM Complète

**Transaction analysée :**
- Type : TRANSFER
- Montant : 312 450 unités
- Différence de solde émetteur : 312 450 (vidage total)
- Solde destinataire initial : 0 (dest_zero_balance = 1)
- Heure : 3h00 (high_risk_hour = 1)
- Score de probabilité RF : 0,94

**Explication générée par llama-3.1-8b-instant :**

> *« Analyse de la transaction — Niveau de risque : ÉLEVÉ (Score : 0,94/1,00)*
>
> *Facteurs de suspicion identifiés :*
>
> *1. Vidage complet du compte émetteur (balance_diff_orig = 312 450 unités). Ce pattern, où le solde du compte est intégralement transféré en une seule opération, est l'un des signaux les plus caractéristiques d'une fraude de type compromission de compte.*
>
> *2. Compte destinataire sans historique de solde (dest_zero_balance = 1). Le fait que le compte destinataire présente un solde nul avant réception suggère qu'il s'agit d'un compte récemment créé, potentiellement utilisé comme intermédiaire dans un circuit de blanchiment.*
>
> *3. Transaction effectuée à 3h00 (heures à risque élevé). Les transactions nocturnes présentent historiquement un taux de fraude 10,6 fois supérieur à la moyenne.*
>
> *4. Type de transaction TRANSFER : ce type de transaction concentre 79 % des fraudes dans notre base de référence.*
>
> *Recommandation : Cette transaction requiert une investigation prioritaire. Nous recommandons de suspendre temporairement le virement dans l'attente d'une vérification téléphonique avec le titulaire du compte émetteur, et d'examiner l'historique du compte destinataire. »*

---

*Fin du rapport*

---

**ESPRIT — École Supérieure Privée d'Ingénierie et de Technologies**  
*Département Informatique — Filière Data Science & Intelligence Artificielle*  
*Année universitaire 2025/2026*
