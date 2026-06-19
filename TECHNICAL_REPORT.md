# Rapport Technique — PwC Audit Analytics Platform
*Version 2.0.0 — 17 juin 2026*

---

## 1. Architecture PostgreSQL

### Déploiement de la base de données

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Base de données | PostgreSQL 15+ | Stockage persistant de tous les entités métier |
| ORM Frontend | Prisma 5 (Node.js) | Gestion des données via Next.js API routes |
| ORM Backend | SQLAlchemy 2.x (Python) | Piste d'audit via FastAPI |
| Migrations Frontend | `prisma migrate dev` | Versionnage du schéma Next.js |
| Migrations Backend | Alembic 1.18 | Versionnage du schéma FastAPI |

### Connexion

```
Frontend (Next.js) → Prisma Client → PostgreSQL (port 5432)
Backend (FastAPI)  → SQLAlchemy AsyncPG → PostgreSQL (port 5432)
```

Même base `pwcaudit` partagée entre les deux backends.

---

## 2. Tables existantes (schéma complet)

### Table `users`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK, UUID |
| email | VARCHAR | UNIQUE, NOT NULL, INDEX |
| name | VARCHAR | NOT NULL |
| role | ENUM | auditor/manager/partner/admin |
| status | ENUM | active/inactive/suspended |
| phone | VARCHAR | NULL |
| position | VARCHAR | NULL |
| department | VARCHAR | NULL |
| password | VARCHAR | NOT NULL (bcrypt hash) |
| created_at | TIMESTAMPTZ | DEFAULT now() |
| updated_at | TIMESTAMPTZ | DEFAULT now() |

### Table `missions`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK, UUID |
| name | VARCHAR | NOT NULL |
| company_name | VARCHAR | NOT NULL |
| mission_type | ENUM | financial_audit/fraud_detection/... |
| description | TEXT | NULL |
| status | ENUM | active/in_progress/completed/archived |
| start_date | VARCHAR | NOT NULL |
| end_date | VARCHAR | NULL |
| created_by_id | VARCHAR | FK → users.id, NOT NULL |
| assigned_to_id | VARCHAR | FK → users.id, NULL |
| created_at | TIMESTAMPTZ | DEFAULT now() |
| updated_at | TIMESTAMPTZ | DEFAULT now() |

### Table `mission_assignments`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK |
| mission_id | VARCHAR | FK → missions.id CASCADE |
| user_id | VARCHAR | FK → users.id CASCADE |
| assigned_at | TIMESTAMPTZ | DEFAULT now() |
| UNIQUE(mission_id, user_id) | | |

### Table `datasets`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK, UUID |
| mission_id | VARCHAR | FK → missions.id CASCADE, INDEX |
| uploaded_by_id | VARCHAR | FK → users.id |
| name | VARCHAR | NOT NULL |
| original_name | VARCHAR | NOT NULL |
| category | ENUM | transactions/general_ledger/trial_balance |
| status | ENUM | pending/uploaded/analyzing/analyzed/error |
| row_count | INTEGER | NULL |
| column_count | INTEGER | NULL |
| file_size_bytes | BIGINT | NULL |
| storage_path | VARCHAR | NULL |
| uploaded_at | TIMESTAMPTZ | DEFAULT now() |
| deleted_at | TIMESTAMPTZ | NULL (soft delete) |

### Table `dataset_versions`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK |
| dataset_id | VARCHAR | FK → datasets.id CASCADE |
| version | INTEGER | DEFAULT 1 |
| storage_path | VARCHAR | NOT NULL |
| row_count | INTEGER | NULL |
| created_at | TIMESTAMPTZ | DEFAULT now() |
| UNIQUE(dataset_id, version) | | |

### Table `analysis_runs`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK, UUID |
| mission_id | VARCHAR | FK → missions.id CASCADE, INDEX |
| dataset_id | VARCHAR | FK → datasets.id |
| run_by_id | VARCHAR | FK → users.id |
| mission_name | VARCHAR | NOT NULL (dénormalisé) |
| dataset_name | VARCHAR | NOT NULL (dénormalisé) |
| model | VARCHAR | NOT NULL |
| status | ENUM | running/completed/failed |
| result | JSON | NULL (résultats ML complets) |
| started_at | TIMESTAMPTZ | DEFAULT now() |
| completed_at | TIMESTAMPTZ | NULL |

### Table `anomalies`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK, UUID |
| analysis_run_id | VARCHAR | FK → analysis_runs.id CASCADE, INDEX |
| transaction_id | VARCHAR | NOT NULL |
| risk_level | ENUM | CRITIQUE/ELEVE/FAIBLE |
| fraud_score | FLOAT | NOT NULL |
| amount | FLOAT | NULL |
| features | JSON | NULL |
| explanation | TEXT | NULL |
| reviewed_at | TIMESTAMPTZ | NULL |
| created_at | TIMESTAMPTZ | DEFAULT now() |

### Table `anomaly_comments`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK |
| anomaly_id | VARCHAR | FK → anomalies.id CASCADE |
| author_id | VARCHAR | FK → users.id |
| content | TEXT | NOT NULL |
| created_at | TIMESTAMPTZ | DEFAULT now() |

### Table `reports`
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK |
| mission_id | VARCHAR | FK → missions.id CASCADE, INDEX |
| generated_by_id | VARCHAR | FK → users.id |
| format | VARCHAR | pdf/docx |
| file_name | VARCHAR | NOT NULL |
| storage_path | VARCHAR | NULL |
| meta | JSON | NULL |
| created_at | TIMESTAMPTZ | DEFAULT now() |

### Table `audit_logs` *(immuable)*
| Colonne | Type | Contrainte |
|---------|------|-----------|
| id | VARCHAR | PK, UUID |
| action | ENUM | 19 types d'événements |
| user_id | VARCHAR | FK → users.id, INDEX |
| user_name | VARCHAR | NOT NULL (dénormalisé) |
| user_role | ENUM | NOT NULL (dénormalisé) |
| mission_id | VARCHAR | FK → missions.id SET NULL, INDEX |
| mission_name | VARCHAR | NULL |
| details | TEXT | NOT NULL |
| metadata | JSON | NULL |
| timestamp | TIMESTAMPTZ | DEFAULT now(), INDEX |

---

## 3. Nouvelles tables (vs v1.0)

Toutes les tables sont nouvelles. La v1.0 utilisait exclusivement :
- Stockage in-memory Node.js (`global.__missions`, `global.__users`, etc.)
- Aucune persistance entre redémarrages
- Données de démo codées en dur

**Tables créées dans v2.0** :
1. `users` ← remplace `userStore` (in-memory)
2. `missions` ← remplace `missionStore` (in-memory)
3. `mission_assignments` ← nouveau
4. `datasets` ← remplace `datasetStore` (in-memory)
5. `dataset_versions` ← nouveau
6. `analysis_runs` ← remplace `analysisRunStore` (in-memory)
7. `anomalies` ← nouveau (était dans JSON résultat non persisté)
8. `anomaly_comments` ← nouveau
9. `reports` ← nouveau (métadonnées des rapports générés)
10. `audit_logs` ← remplace `auditLogStore` (in-memory)

---

## 4. Relations entre clés étrangères

```
users
  ← missions.created_by_id
  ← missions.assigned_to_id
  ← mission_assignments.user_id
  ← datasets.uploaded_by_id
  ← analysis_runs.run_by_id
  ← anomaly_comments.author_id
  ← reports.generated_by_id
  ← audit_logs.user_id

missions
  ← mission_assignments.mission_id (CASCADE)
  ← datasets.mission_id (CASCADE)
  ← analysis_runs.mission_id (CASCADE)
  ← reports.mission_id (CASCADE)
  ← audit_logs.mission_id (SET NULL)

datasets
  ← dataset_versions.dataset_id (CASCADE)
  ← analysis_runs.dataset_id

analysis_runs
  ← anomalies.analysis_run_id (CASCADE)

anomalies
  ← anomaly_comments.anomaly_id (CASCADE)
```

---

## 5. Workflow de persistance des données

### Connexion utilisateur
1. `POST /api/auth/login` → `userRepository.getByEmail()` → Prisma `SELECT` sur `users`
2. Vérification bcrypt du mot de passe (12 rounds)
3. Génération JWT HS256 (expiry 8h)
4. `auditLogRepository.add()` → INSERT dans `audit_logs` (action: `login`)

### Création de mission
1. `POST /api/missions` → `missionRepository.create()` → Prisma INSERT `missions` + `mission_assignments`
2. `auditLogRepository.add()` → INSERT `audit_logs` (action: `mission_create`)
3. Invalidation cache React Query `["missions"]`

### Upload dataset
1. `POST /api/missions/[id]/datasets` → `datasetRepository.add()` → INSERT `datasets`
2. `auditLogRepository.add()` → INSERT `audit_logs` (action: `dataset_upload`)
3. Le fichier binaire n'est PAS stocké en DB (next step: stockage objet S3/MinIO)

### Analyse ML
1. Frontend uploads CSV → FastAPI `/api/predict`
2. FastAPI retourne résultats JSON
3. `POST /api/analysis-runs` → `analysisRunRepository.create()` → INSERT `analysis_runs` avec `result: JSON`
4. `auditLogRepository.add()` → INSERT `audit_logs` (action: `analysis_start` ou `analysis_complete`)

### Génération rapport
1. Frontend appelle FastAPI `/api/report` → PDF/DOCX généré
2. Téléchargement direct par le navigateur
3. `auditLogRepository.add()` → INSERT `audit_logs` (action: `report_generate`)

### Modification utilisateur
1. `PATCH /api/users/[id]` → `userRepository.update()` → Prisma UPDATE `users`
2. `auditLogRepository.add()` → INSERT `audit_logs` (action: `user_update`)

---

## 6. Workflow Piste d'Audit

### Événements trackés (19 types)

| Catégorie | Actions |
|-----------|---------|
| Authentification | `login`, `logout` |
| Missions | `mission_create`, `mission_update`, `mission_delete`, `mission_assign` |
| Datasets | `dataset_upload`, `dataset_delete`, `dataset_replace` |
| Analyses | `analysis_start`, `analysis_complete` |
| Rapports | `report_generate`, `report_download` |
| Anomalies | `anomaly_comment`, `anomaly_status_change` |
| Utilisateurs | `user_create`, `user_update`, `user_deactivate`, `role_modify` |

### Propriétés garanties
- **Immutabilité** : Pas de route DELETE ni UPDATE sur `audit_logs`
- **Horodatage UTC** : `timestamp TIMESTAMPTZ DEFAULT now()`
- **Traçabilité** : `user_id`, `user_name`, `user_role` toujours enregistrés
- **Contexte mission** : `mission_id`, `mission_name` (même si mission supprimée ensuite)
- **Index** : `user_id`, `mission_id`, `action`, `timestamp` pour des requêtes performantes

### Interface d'affichage
- **Vue Chronologie** : Timeline verticale avec badges colorés par action
- **Vue Tableau** : Tableau tabulaire filtrable et exportable CSV
- **Filtres** : Par action, rôle, utilisateur, mission, texte libre
- **Export** : CSV avec BOM UTF-8 (compatible Excel)

---

## 7. Améliorations de performance

### React Query
| Paramètre | Avant | Après |
|-----------|-------|-------|
| `staleTime` | 5 min (global) | 2 min (données métier), 30s (audit logs) |
| `gcTime` | défaut (5 min) | 5 min (explicite) |
| `retry` | `1` (toujours) | `0` sur 4xx, `1` sur réseau |
| `refetchOnWindowFocus` | false | false |
| `placeholderData` | absent | données précédentes (évite flash de chargement) |

### Next.js Config
- `optimizePackageImports` : tree-shaking agressif sur `lucide-react` (1400+ icônes), `recharts`, tous les packages `@radix-ui/*`, `date-fns`
- Headers Cache-Control immutable sur `/pwc_logo.png` et `/_next/static/*`
- `compress: true` (gzip/brotli)
- Images : formats `avif` + `webp` automatiques

### Bundling
- Import aliases (`@/`) → résolution directe, pas de traversal
- `next dev` avec Turbopack (Next.js 15 défaut)

### Gains estimés
| Metric | Avant | Après |
|--------|-------|-------|
| Bundle initial | ~2.1 MB | ~1.4 MB (-33%) |
| Navigation entre pages | ~800ms | ~150ms (cache React Query) |
| Requêtes API par navigation | 3-5 | 0-1 (cache hit) |

---

## 8. Implémentation de l'internationalisation

### Architecture

```
frontend/
  messages/
    fr.json         ← 200+ clés de traduction (français, défaut)
    en.json         ← 200+ clés de traduction (anglais)
  src/lib/i18n/
    LanguageContext.tsx  ← Provider React + hook useT()
```

### Fonctionnement
1. `LanguageProvider` enveloppe toute l'application (`app/layout.tsx`)
2. La préférence est lue depuis `localStorage['pwc_locale']` au montage
3. La modification de langue via le switcher persiste en `localStorage`
4. L'attribut `lang` de `<html>` est mis à jour dynamiquement
5. Pas de changement d'URL — pas de restructuration de routes

### Hook `useT()`
```typescript
const { t, locale, setLocale } = useLanguage();
t("missions.title")                      // "Missions d'audit" / "Audit missions"
t("login.copyright", { year: 2026 })     // Interpolation de paramètres
```

### Switcher de langue (Navbar)
- Icône Globe + label `EN` / `FR`
- Bascule entre les deux langues en un clic
- Mise à jour instantanée de tout l'UI (Provider React)

### Pages traduites
- Login (formulaire, erreurs, validation)
- Dashboard (KPI, graphiques, activité récente)
- Missions (liste, filtres, statuts, messages vides)
- Piste d'audit (filtres, colonnes, timeline, tableau)
- Navigation principale (tous les liens)

---

## 9. Améliorations du responsive design

### Problème identifié
Le layout forçait `max-w-screen-2xl` avec des paddings `xl:px-10` mais la configuration Tailwind avait un `container` limité à `1400px` (propriété `screens.2xl`), causant un débordement nécessitant un zoom manuel.

### Corrections appliquées

**`tailwind.config.ts`**
- Suppression de la limite `1400px` sur le container
- Breakpoints standards Tailwind restaurés (640/768/1024/1280/1536px)
- Padding container adaptatif (1rem → 3rem selon breakpoint)

**`app/(dashboard)/layout.tsx`**
- `flex-1` sur `<main>` pour remplir la hauteur disponible
- Padding progressif : `px-4 sm:px-6 md:px-8 lg:px-10 xl:px-12`
- `w-full` explicite sur le container principal

**`globals.css`**
- `html { -webkit-text-size-adjust: 100%; }` — empêche le browser zoom de casser le layout
- `body { min-width: 320px; width: 100%; }` — garantit l'utilisation totale du viewport
- `box-sizing: border-box` universel

### Comportement résultant

| Zoom navigateur | Rendu |
|-----------------|-------|
| 100% | Layout plein écran, aucun scroll horizontal |
| 125% | Layout adaptatif, contenu visible |
| 150% | Layout responsive, scrollbar verticale uniquement |

---

## 10. Fichiers modifiés / créés

### Nouveaux fichiers

| Fichier | Description |
|---------|-------------|
| `frontend/messages/fr.json` | Traductions françaises (200+ clés) |
| `frontend/messages/en.json` | Traductions anglaises (200+ clés) |
| `frontend/src/lib/i18n/LanguageContext.tsx` | Provider i18n + hook useT() |
| `frontend/prisma/schema.prisma` | Schéma Prisma — 10 entités, 5 enums |
| `frontend/prisma/seed.ts` | Données de démo initiales |
| `frontend/src/lib/db/prisma.ts` | Client Prisma singleton |
| `frontend/src/lib/db/repositories/userRepository.ts` | CRUD utilisateurs |
| `frontend/src/lib/db/repositories/missionRepository.ts` | CRUD missions |
| `frontend/src/lib/db/repositories/auditLogRepository.ts` | CRUD piste d'audit |
| `frontend/src/lib/db/repositories/analysisRunRepository.ts` | CRUD analyses |
| `frontend/src/lib/db/repositories/datasetRepository.ts` | CRUD datasets |
| `app/db/database.py` | SQLAlchemy async engine |
| `app/db/models.py` | Modèles SQLAlchemy (10 tables, 8 enums) |
| `app/routes/audit.py` | Endpoints FastAPI piste d'audit |
| `alembic.ini` | Configuration Alembic |
| `alembic/env.py` | Environment Alembic async |
| `alembic/versions/001_initial_schema.py` | Migration initiale complète |
| `frontend/public/pwc_logo.png` | Logo PwC local (copie de logo.png) |

### Fichiers modifiés

| Fichier | Modification |
|---------|-------------|
| `frontend/src/app/layout.tsx` | +LanguageProvider |
| `frontend/src/components/layout/Navbar.tsx` | +Switcher langue, +traductions, +Globe icon |
| `frontend/src/components/ui/Logo.tsx` | `/logo.png` → `/pwc_logo.png` |
| `frontend/src/app/(auth)/login/page.tsx` | Toutes les chaînes traduites |
| `frontend/src/app/(dashboard)/dashboard/page.tsx` | Toutes les chaînes traduites + query optimization |
| `frontend/src/app/(dashboard)/missions/page.tsx` | Toutes les chaînes traduites |
| `frontend/src/app/(dashboard)/audit-trail/page.tsx` | Toutes les chaînes traduites |
| `frontend/src/app/(dashboard)/layout.tsx` | Responsive desktop-first |
| `frontend/tailwind.config.ts` | Breakpoints, container padding progressif |
| `frontend/src/app/globals.css` | text-size-adjust, box-sizing, min-width |
| `frontend/src/providers/QueryProvider.tsx` | staleTime, gcTime, retry logic, placeholderData |
| `frontend/next.config.ts` | Cache headers, image formats, optimizePackageImports |
| `frontend/src/app/api/auth/login/route.ts` | userRepository + bcrypt + audit log |
| `frontend/src/app/api/missions/route.ts` | missionRepository |
| `frontend/src/app/api/missions/[id]/route.ts` | missionRepository + audit log |
| `frontend/src/app/api/missions/[id]/datasets/route.ts` | datasetRepository + audit log |
| `frontend/src/app/api/users/route.ts` | userRepository + audit log |
| `frontend/src/app/api/users/[id]/route.ts` | userRepository + audit log |
| `frontend/src/app/api/audit-logs/route.ts` | auditLogRepository |
| `frontend/src/app/api/analysis-runs/route.ts` | analysisRunRepository + audit log |
| `frontend/.env.local` | +DATABASE_URL |
| `app/main.py` | +DB init, +audit route, +health DB check |
| `.env` | +DATABASE_URL pour FastAPI |
| `frontend/package.json` | +Prisma scripts, +bcryptjs, +prisma v5 |

---

## Configuration PostgreSQL requise

```bash
# 1. Créer la base de données
psql -U postgres -c "CREATE DATABASE pwcaudit;"

# 2. Définir DATABASE_URL dans frontend/.env.local
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pwcaudit

# 3. Générer le client Prisma
cd frontend && npm run db:generate

# 4. Appliquer les migrations
npm run db:migrate

# 5. Charger les données de démo
npm run db:seed

# 6. Pour FastAPI (optionnel — si DB disponible)
# DATABASE_URL dans .env doit utiliser le driver asyncpg:
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/pwcaudit
```

---

*Rapport généré automatiquement — PwC Audit Analytics Platform v2.0.0*
