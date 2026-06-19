# Guide Complet — Frontend de la Plateforme PwC Audit Analytics

## Table des matières

1. [Vue d'ensemble de l'architecture](#1-vue-densemble-de-larchitecture)
2. [Installation et configuration initiale](#2-installation-et-configuration-initiale)
3. [Structure du projet frontend](#3-structure-du-projet-frontend)
4. [Système de routage (Next.js App Router)](#4-système-de-routage-nextjs-app-router)
5. [Système d'authentification](#5-système-dauthentification)
6. [Couche base de données — Prisma](#6-couche-base-de-données--prisma)
7. [Clients API et communication avec le backend](#7-clients-api-et-communication-avec-le-backend)
8. [Gestion d'état — React Query et Zustand](#8-gestion-détat--react-query-et-zustand)
9. [Composants principaux](#9-composants-principaux)
10. [Intégration des modèles ML depuis les notebooks](#10-intégration-des-modèles-ml-depuis-les-notebooks)
11. [Internationalisation (FR/EN)](#11-internationalisation-fren)
12. [Système de rapports](#12-système-de-rapports)
13. [Piste d'audit (Audit Trail)](#13-piste-daudit-audit-trail)
14. [Développer une nouvelle fonctionnalité](#14-développer-une-nouvelle-fonctionnalité)
15. [Variables d'environnement](#15-variables-denvironnement)
16. [Déploiement avec Docker](#16-déploiement-avec-docker)

---

## 1. Vue d'ensemble de l'architecture

La plateforme utilise une architecture **full-stack découplée** avec deux serveurs distincts qui partagent la même base de données PostgreSQL.

```
┌─────────────────────────────────────────────────────────────────┐
│                         NAVIGATEUR                              │
│                    (React 19 / Next.js 15)                      │
└────────────────────┬───────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌─────────────────┐   ┌─────────────────────────┐
│   Next.js API   │   │   FastAPI Backend        │
│  (port 3000)    │   │  (port 8000)             │
│                 │   │                          │
│  /api/auth/*    │   │  /api/predict            │
│  /api/missions/*│   │  /api/explain/*          │
│  /api/users/*   │   │  /api/report             │
│  /api/datasets/*│   │  /api/models             │
│  /api/audit-    │   │  /api/profile            │
│  logs/*         │   │                          │
│                 │   │  ← XGBoost, AutoEncoder  │
│  ← Prisma ORM   │   │    IsoForest, SHAP       │
│                 │   │    LIME, LLM (Groq)      │
└────────┬────────┘   └──────────┬───────────────┘
         │                       │
         └──────────┬────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  PostgreSQL (port     │
        │  5432) — pwcaudit     │
        │                       │
        │  users, missions,     │
        │  datasets, analysis_  │
        │  runs, audit_logs...  │
        └───────────────────────┘
```

**Rôles de chaque serveur :**

| Serveur | Responsabilité |
|---------|---------------|
| **Next.js `/api/*`** | Authentification JWT, CRUD métier (missions, datasets, users), écriture en base via Prisma, traçabilité audit |
| **FastAPI `/api/*`** | Pipeline ML (prédiction, explication), génération de rapports PDF/DOCX, registre de modèles |
| **PostgreSQL** | Source unique de vérité — les deux serveurs lisent/écrivent la même DB |

**Principe de communication :** le navigateur envoie les requêtes métier vers Next.js (`appClient`) et les requêtes ML vers FastAPI via le proxy `/ml/*` (`mlClient`). Le proxy est configuré dans `next.config.ts` pour que le CORS ne soit jamais un problème en production.

---

## 2. Installation et configuration initiale

### Prérequis

- Node.js ≥ 18.x
- npm ≥ 9.x (ou pnpm/yarn)
- PostgreSQL 13+ en cours d'exécution sur le port 5432
- Python 3.10+ avec le backend FastAPI démarré (voir `BACKEND_GUIDE.md`)

### Étapes d'installation

```bash
# 1. Se placer dans le dossier frontend
cd frontend

# 2. Installer les dépendances Node
npm install

# 3. Créer le fichier d'environnement local
cp .env.local.example .env.local
# Puis éditer .env.local :
# NEXT_PUBLIC_API_URL=http://localhost:8000

# 4. S'assurer que DATABASE_URL est défini dans .env (à la racine frontend/)
# DATABASE_URL=postgresql://postgres:123@localhost:5432/pwcaudit

# 5. Générer le client Prisma et synchroniser le schéma
npx prisma generate
npx prisma db push          # ou npx prisma migrate dev

# 6. (Optionnel) Peupler la base avec des données initiales
npx prisma db seed

# 7. Démarrer le serveur de développement
npm run dev
# → http://localhost:3000
```

### Vérifier que tout fonctionne

1. Ouvrir `http://localhost:3000` → redirige vers `/login`
2. Se connecter avec les identifiants du seed
3. S'assurer que le backend FastAPI tourne sur `http://localhost:8000` (sinon les prédictions ML échoueront)

---

## 3. Structure du projet frontend

```
frontend/
├── prisma/
│   ├── schema.prisma          ← Schéma de la base (source de vérité BDD)
│   ├── migrations/            ← Historique des migrations SQL
│   └── seed.ts                ← Script de peuplement initial
│
├── public/                    ← Fichiers statiques servis directement
│
├── src/
│   ├── app/                   ← Next.js App Router (pages + API routes)
│   │   ├── (auth)/            ← Groupe de routes publiques
│   │   ├── (dashboard)/       ← Groupe de routes protégées
│   │   ├── api/               ← API routes Next.js (Server-side)
│   │   ├── layout.tsx         ← Layout racine (providers, polices, styles)
│   │   ├── page.tsx           ← Page d'accueil (redirige vers /dashboard)
│   │   └── globals.css        ← Styles globaux + variables CSS
│   │
│   ├── components/            ← Composants React réutilisables
│   │   ├── analysis/          ← Wizard, dashboard résultats, tableaux
│   │   ├── datasets/          ← Upload, liste des jeux de données
│   │   ├── explanations/      ← Cartes SHAP / AE / LIME / LLM
│   │   ├── layout/            ← Navbar, TopBar
│   │   ├── missions/          ← Cartes missions, modal création
│   │   ├── reports/           ← Génération et téléchargement rapports
│   │   └── ui/                ← Composants Shadcn/UI (boutons, cards...)
│   │
│   ├── lib/
│   │   ├── api/               ← Services Axios (appels HTTP)
│   │   │   ├── client.ts      ← 2 instances Axios (appClient, mlClient)
│   │   │   ├── analysisService.ts
│   │   │   ├── authService.ts
│   │   │   ├── missionService.ts
│   │   │   ├── userService.ts
│   │   │   ├── datasetService.ts
│   │   │   ├── reportService.ts
│   │   │   ├── auditLogService.ts
│   │   │   ├── llmService.ts
│   │   │   └── analysisRunService.ts
│   │   │
│   │   ├── auth/
│   │   │   └── AuthContext.tsx ← Contexte React d'authentification
│   │   │
│   │   ├── db/
│   │   │   ├── prisma.ts      ← Singleton PrismaClient
│   │   │   └── repositories/  ← Accès BDD encapsulé par entité
│   │   │
│   │   ├── hooks/
│   │   │   └── usePermissions.ts ← Vérification des droits par rôle
│   │   │
│   │   ├── i18n/
│   │   │   └── LanguageContext.tsx ← Contexte FR/EN
│   │   │
│   │   ├── store/             ← Stores Zustand (état global client)
│   │   └── utils.ts           ← Fonctions utilitaires
│   │
│   ├── providers/
│   │   ├── QueryProvider.tsx  ← TanStack React Query
│   │   └── ThemeProvider.tsx  ← Dark mode (next-themes)
│   │
│   ├── types/
│   │   └── index.ts           ← Tous les types TypeScript du projet
│   │
│   └── middleware.ts          ← Protection JWT des routes
│
├── .env                       ← DATABASE_URL pour Prisma
├── .env.local                 ← NEXT_PUBLIC_API_URL (non commité)
├── next.config.ts             ← Config Next.js (proxy /ml/*, rewrites)
├── tailwind.config.ts         ← Config Tailwind (couleurs PwC)
└── tsconfig.json
```

---

## 4. Système de routage (Next.js App Router)

Next.js 15 utilise le **App Router** : chaque dossier dans `src/app/` correspond à une route, et les fichiers `page.tsx` rendent la page correspondante.

### Groupes de routes

Les parenthèses créent des **groupes** qui n'apparaissent pas dans l'URL mais permettent de partager un layout.

```
(auth)/login          → /login          (layout sans navbar)
(dashboard)/dashboard → /dashboard      (layout avec navbar + sidebar)
(dashboard)/missions  → /missions
(dashboard)/missions/[id]            → /missions/abc-123
(dashboard)/missions/[id]/analysis   → /missions/abc-123/analysis
(dashboard)/admin/users              → /admin/users
(dashboard)/audit-trail              → /audit-trail
(dashboard)/history                  → /history
(dashboard)/reports                  → /reports
```

### Protection des routes — Middleware

Le fichier `src/middleware.ts` s'exécute **avant** chaque requête et vérifie la présence du cookie JWT :

```typescript
// src/middleware.ts
export function middleware(request: NextRequest) {
  const token = request.cookies.get("pwc_token")?.value;

  // Si pas de token ET route non publique → redirection vers /login
  if (!token && !pathname.startsWith("/api")) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("from", pathname); // mémorise la destination
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}
```

**Remarque :** le middleware lit le cookie HTTP-only `pwc_token`, mais les services Axios lisent aussi `localStorage.pwc_token`. Les deux sont synchronisés lors du login.

### API Routes (Server-side)

Les fichiers dans `src/app/api/` sont des **API Routes Next.js** — elles tournent côté serveur Node.js et ont accès à Prisma (PostgreSQL). Elles ne sont **jamais** exposées au navigateur directement.

```
src/app/api/
├── auth/login/route.ts      POST  /api/auth/login
├── auth/logout/route.ts     POST  /api/auth/logout
├── auth/me/route.ts         GET   /api/auth/me
├── missions/route.ts        GET   /api/missions  |  POST /api/missions
├── missions/[id]/route.ts   GET/PUT/DELETE /api/missions/:id
├── missions/[id]/datasets/  GET   /api/missions/:id/datasets
├── users/route.ts           GET/POST /api/users
├── users/[id]/route.ts      GET/PUT /api/users/:id
├── analysis-runs/route.ts   GET/POST /api/analysis-runs
└── audit-logs/route.ts      GET   /api/audit-logs
```

**Exemple — création d'une mission (`src/app/api/missions/route.ts`) :**

```typescript
export async function POST(request: Request) {
  // 1. Vérifier l'authentification
  const currentUser = await getCurrentUser(request);
  if (!currentUser) return new Response("Unauthorized", { status: 401 });

  // 2. Lire et valider le body JSON
  const body = await request.json();

  // 3. Appeler le repository Prisma
  const mission = await missionRepository.create({
    ...body,
    created_by_id: currentUser.id,
  });

  // 4. Logger l'action dans l'audit trail
  await auditLogRepository.add({
    action: "mission_create",
    user_id: currentUser.id,
    user_name: currentUser.name,
    mission_id: mission.id,
    details: `Mission "${mission.name}" créée`,
  });

  return Response.json(mission, { status: 201 });
}
```

---

## 5. Système d'authentification

### Flux de connexion complet

```
1. Utilisateur soumet email + password dans LoginPage
   ↓
2. authService.login() → POST /api/auth/login (Next.js API)
   ↓
3. L'API route:
   a. Cherche l'utilisateur en BDD via userRepository
   b. Compare le hash bcrypt du password
   c. Génère un JWT signé avec JWT_SECRET (HS256, expiry 8h)
   d. Place le JWT dans un cookie HTTP-only "pwc_token"
   e. Retourne { user, token } en JSON
   ↓
4. authService stocke token + user dans localStorage
   (pour les appels Axios côté client)
   ↓
5. AuthContext met à jour l'état global
   ↓
6. Middleware détecte le cookie → accès autorisé à toutes les routes protégées
```

### AuthContext (`src/lib/auth/AuthContext.tsx`)

```typescript
// Fournit l'état d'auth à tous les composants enfants
const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);

  // Au montage : restaurer la session depuis localStorage
  useEffect(() => {
    const savedToken = localStorage.getItem("pwc_token");
    const savedUser = localStorage.getItem("pwc_user");
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const login = async (credentials) => {
    const response = await authService.login(credentials);
    setToken(response.token);
    setUser(response.user);
    localStorage.setItem("pwc_token", response.token);
    localStorage.setItem("pwc_user", JSON.stringify(response.user));
  };

  const logout = async () => {
    await authService.logout();
    setToken(null);
    setUser(null);
    localStorage.removeItem("pwc_token");
    localStorage.removeItem("pwc_user");
    window.location.href = "/login";
  };
}

// Hook d'utilisation
export const useAuth = () => useContext(AuthContext);
```

### Rôles et permissions (`src/lib/hooks/usePermissions.ts`)

```typescript
export function usePermissions() {
  const { user } = useAuth();

  return {
    canCreateMission: ["manager", "partner", "admin"].includes(user?.role),
    canAssignAuditor: ["manager", "partner", "admin"].includes(user?.role),
    canViewAuditTrail: ["partner", "admin"].includes(user?.role),
    canManageUsers: user?.role === "admin",
    canRunAnalysis: ["auditor", "manager", "partner", "admin"].includes(user?.role),
    canGenerateReport: ["manager", "partner", "admin"].includes(user?.role),
  };
}
```

**Utilisation dans un composant :**
```typescript
const { canCreateMission } = usePermissions();

return (
  <>
    {canCreateMission && (
      <Button onClick={() => setShowCreateModal(true)}>
        Nouvelle mission
      </Button>
    )}
  </>
);
```

---

## 6. Couche base de données — Prisma

### Pourquoi Prisma dans Next.js ?

FastAPI utilise SQLAlchemy (Python) pour ses propres opérations ML. Next.js utilise **Prisma** (TypeScript) pour toutes les opérations métier (CRUD). Les deux partagent la **même base PostgreSQL**.

### Schéma principal (`frontend/prisma/schema.prisma`)

Le schéma définit 10 tables principales :

```prisma
// Utilisateurs (auth + profil)
model User {
  id         String     @id @default(uuid())
  email      String     @unique
  name       String
  role       UserRole               // auditor | manager | partner | admin
  status     UserStatus @default(active)
  password   String                 // hash bcrypt
  // Relations → missions, datasets, analysisRuns, auditLogs...
}

// Missions d'audit
model Mission {
  id          String        @id @default(uuid())
  name        String
  companyName String        @map("company_name")
  missionType MissionType   @map("mission_type")
  status      MissionStatus @default(active)
  startDate   String        @map("start_date")
  // FK → createdBy (User), assignedTo (User)
  // Relations → datasets, analysisRuns, reports, auditLogs
}

// Jeux de données uploadés
model Dataset {
  id           String          @id @default(uuid())
  missionId    String          @map("mission_id")
  name         String
  storagePath  String?         @map("storage_path") // chemin physique du CSV
  rowCount     Int?            @map("row_count")
  // FK → mission, uploadedBy
}

// Exécutions d'analyse ML
model AnalysisRun {
  id          String         @id @default(uuid())
  missionId   String
  datasetId   String
  model       String         // "paysim", "ae_isoforest", etc.
  status      AnalysisStatus // running | completed | failed
  result      Json?          // PredictResponse complet sérialisé
  // FK → mission, dataset, runBy
}

// Anomalies détectées
model Anomaly {
  id            String    @id @default(uuid())
  analysisRunId String
  transactionId String
  riskLevel     RiskLevel // CRITIQUE | ELEVE | FAIBLE
  fraudScore    Float
  amount        Float?
  features      Json?
  explanation   String?
}

// Piste d'audit
model AuditLog {
  id        String      @id @default(uuid())
  action    AuditAction // login, mission_create, analysis_start...
  userId    String
  userName  String
  details   String
  metadata  Json?
  timestamp DateTime    @default(now())
}
```

### Singleton PrismaClient (`src/lib/db/prisma.ts`)

```typescript
// Évite de créer une nouvelle connexion à chaque rechargement en dev
const globalForPrisma = globalThis as unknown as { prisma: PrismaClient };

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: process.env.NODE_ENV === "development" ? ["query", "error"] : ["error"],
  });

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
```

### Repositories (`src/lib/db/repositories/`)

Chaque entité a son repository qui encapsule les requêtes Prisma :

**Exemple — `missionRepository.ts` :**
```typescript
export const missionRepository = {
  async findAll(userId?: string): Promise<Mission[]> {
    return prisma.mission.findMany({
      where: userId ? { OR: [
        { createdById: userId },
        { assignedToId: userId },
        { assignments: { some: { userId } } }
      ]} : undefined,
      include: { createdBy: true, assignedTo: true, assignments: true },
      orderBy: { createdAt: "desc" },
    });
  },

  async create(data: CreateMissionPayload): Promise<Mission> {
    return prisma.mission.create({
      data: {
        name: data.name,
        companyName: data.company_name,
        missionType: data.mission_type,
        startDate: data.start_date,
        createdById: data.created_by_id,
      },
    });
  },

  async findById(id: string): Promise<Mission | null> {
    return prisma.mission.findUnique({
      where: { id },
      include: { datasets: true, analysisRuns: true },
    });
  },
};
```

### Commandes Prisma essentielles

```bash
# Régénérer le client TypeScript après modification du schéma
npx prisma generate

# Appliquer les changements de schéma en développement (sans migration)
npx prisma db push

# Créer une migration nommée
npx prisma migrate dev --name "add_column_x_to_missions"

# Appliquer les migrations en production
npx prisma migrate deploy

# Visualiser la base dans l'interface Prisma Studio
npx prisma studio

# Peupler la base avec les données initiales
npx prisma db seed
```

### Ajouter une colonne — exemple complet

**Objectif :** ajouter un champ `priority` (HIGH/MEDIUM/LOW) sur `Mission`.

**Étape 1 — Modifier `schema.prisma` :**
```prisma
enum MissionPriority {
  HIGH
  MEDIUM
  LOW
}

model Mission {
  // ... champs existants ...
  priority MissionPriority @default(MEDIUM)
}
```

**Étape 2 — Créer la migration :**
```bash
npx prisma migrate dev --name "add_priority_to_missions"
```

**Étape 3 — Mettre à jour les types TypeScript (`src/types/index.ts`) :**
```typescript
export type MissionPriority = "HIGH" | "MEDIUM" | "LOW";

export interface Mission {
  // ... champs existants ...
  priority: MissionPriority;
}
```

**Étape 4 — Mettre à jour le repository et l'API route.**

---

## 7. Clients API et communication avec le backend

### Les deux instances Axios (`src/lib/api/client.ts`)

```typescript
// Client vers les API routes Next.js (/api/...)
export const appClient = axios.create({
  baseURL: "/api",
  timeout: 30_000,
});

// Client vers le backend FastAPI (via proxy /ml/...)
export const mlClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL
    ? `${process.env.NEXT_PUBLIC_API_URL}/ml`
    : "/ml",
  timeout: 120_000, // ML peut prendre du temps
});

// Intercepteur : injecte automatiquement le JWT dans chaque requête
function injectToken(config) {
  const token = localStorage.getItem("pwc_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
}

appClient.interceptors.request.use(injectToken);
mlClient.interceptors.request.use(injectToken);

// Intercepteur : redirige vers /login si 401
function onResponseError(error) {
  if (error.response?.status === 401) {
    localStorage.removeItem("pwc_token");
    window.location.href = "/login";
  }
  return Promise.reject(error);
}
```

### Proxy Next.js vers FastAPI (`next.config.ts`)

```typescript
// next.config.ts
async rewrites() {
  return [
    {
      source: "/ml/:path*",
      destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/:path*`,
    },
  ];
},
```

Cela signifie que `mlClient.post("/api/predict")` devient en réalité une requête vers `http://localhost:8000/api/predict`, sans problème de CORS.

### Services disponibles

| Service | Client utilisé | Endpoint cible |
|---------|---------------|----------------|
| `authService` | `appClient` | Next.js `/api/auth/*` |
| `missionService` | `appClient` | Next.js `/api/missions/*` |
| `userService` | `appClient` | Next.js `/api/users/*` |
| `datasetService` | `appClient` | Next.js `/api/missions/:id/datasets` |
| `auditLogService` | `appClient` | Next.js `/api/audit-logs` |
| `analysisRunService` | `appClient` | Next.js `/api/analysis-runs` |
| `analysisService` | `mlClient` | FastAPI `/api/predict`, `/api/explain/*`, `/api/profile` |
| `reportService` | `mlClient` | FastAPI `/api/report`, `/api/report/docx` |
| `llmService` | `mlClient` | FastAPI `/api/explain` |

### Exemple d'utilisation dans un composant

```typescript
// Dans une page React
import { analysisService } from "@/lib/api/analysisService";
import { missionService } from "@/lib/api/missionService";

function AnalysisPage() {
  const [result, setResult] = useState<PredictResponse | null>(null);

  async function handlePredict(file: File) {
    // Appel ML FastAPI
    const prediction = await analysisService.predict(file, (pct) => {
      console.log(`Upload: ${pct}%`);
    });
    setResult(prediction);

    // Sauvegarder l'exécution en base via Next.js API
    await analysisRunService.create({
      mission_id: missionId,
      dataset_id: datasetId,
      model_mode: prediction.prediction_mode,
      result: prediction,
    });
  }
}
```

### Ajouter un nouveau service API

Créer `src/lib/api/newService.ts` :
```typescript
import { appClient } from "./client"; // ou mlClient selon la cible
import type { NewEntity, CreateNewEntityPayload } from "@/types";

export const newService = {
  async getAll(): Promise<NewEntity[]> {
    const { data } = await appClient.get<NewEntity[]>("/new-entities");
    return data;
  },

  async create(payload: CreateNewEntityPayload): Promise<NewEntity> {
    const { data } = await appClient.post<NewEntity>("/new-entities", payload);
    return data;
  },
};
```

---

## 8. Gestion d'état — React Query et Zustand

### TanStack React Query (état serveur)

React Query gère le **cache des données venant du backend**. Toutes les données qui nécessitent une synchronisation avec la base passent par React Query.

**Configuration (`src/providers/QueryProvider.tsx`) :**
```typescript
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // données fraîches pendant 5 min
      retry: 1,
    },
  },
});
```

**Exemple — liste des missions avec React Query :**
```typescript
function MissionsPage() {
  const { data: missions, isLoading, error } = useQuery({
    queryKey: ["missions"],
    queryFn: () => missionService.getAll(),
  });

  const createMission = useMutation({
    mutationFn: missionService.create,
    onSuccess: () => {
      // Invalide le cache → re-fetch automatique
      queryClient.invalidateQueries({ queryKey: ["missions"] });
    },
  });

  if (isLoading) return <Skeleton />;
  if (error) return <ErrorMessage />;

  return (
    <div>
      {missions?.map(m => <MissionCard key={m.id} mission={m} />)}
      <Button onClick={() => createMission.mutate(payload)}>
        Nouvelle mission
      </Button>
    </div>
  );
}
```

### Zustand (état client local)

Zustand gère l'**état UI persistant côté client** (sélection courante, filtres, résultats d'analyse en cours).

**Exemple — store d'analyse (`src/lib/store/analysisRunStore.ts`) :**
```typescript
import { create } from "zustand";

interface AnalysisRunStore {
  currentRun: AnalysisRun | null;
  setCurrentRun: (run: AnalysisRun | null) => void;
  predictionResult: PredictResponse | null;
  setPredictionResult: (result: PredictResponse | null) => void;
}

export const useAnalysisRunStore = create<AnalysisRunStore>((set) => ({
  currentRun: null,
  setCurrentRun: (run) => set({ currentRun: run }),
  predictionResult: null,
  setPredictionResult: (result) => set({ predictionResult: result }),
}));
```

**Règle de décision :**
- Données qui viennent du backend et sont partagées entre composants → **React Query**
- État UI local (wizard step, modal ouvert/fermé, résultat ML en cours) → **Zustand**
- État local à un seul composant → **useState**

---

## 9. Composants principaux

### AnalysisWizard (`src/components/analysis/AnalysisWizard.tsx`)

Wizard multi-étapes pour lancer une analyse ML :

```
Étape 1: Sélection du dataset
   ↓ Upload CSV ou sélection depuis la liste
Étape 2: Profilage (appel /api/profile)
   ↓ Affiche : qualité données, colonnes détectées, mapping PaySim
Étape 3: Lancement (appel /api/predict)
   ↓ Barre de progression, affichage du mode de prédiction choisi
Étape 4: Résultats → ResultsDashboard
```

**Props :**
```typescript
interface AnalysisWizardProps {
  missionId: string;
  datasetId?: string;
  onComplete: (result: PredictResponse) => void;
}
```

### ResultsDashboard (`src/components/analysis/ResultsDashboard.tsx`)

Tableau de bord complet des résultats d'analyse :

- **KPICards** : taux de fraude, montant à risque, nb transactions CRITIQUE/ELEVE/FAIBLE
- **RiskPieChart** : répartition des niveaux de risque
- **AnomalyBarChart** : distribution par catégorie
- **ScoreDistributionChart** : histogramme des scores
- **AnomalyTable** : liste filtrée/triée avec pagination
- **ExplanationCard** : détails SHAP / AE / LIME / LLM pour chaque transaction

**Utilisation :**
```typescript
<ResultsDashboard
  result={predictionResult}
  missionId={missionId}
  onExplain={(txId) => handleExplain(txId)}
/>
```

### ExplanationCard (`src/components/explanations/ExplanationCard.tsx`)

Affiche les 4 niveaux d'explication pour une transaction :

```typescript
interface ExplanationCardProps {
  explanation: ExplainResponse;
}

// Affiche :
// 1. Score XGB + score AE + niveau de risque
// 2. SHAP values : barres horizontales (contribution de chaque feature)
// 3. AE feature errors : reconstruction error par variable
// 4. LIME rules : règles textuelles (si activé)
// 5. LLM summary : résumé en français + raisons + actions recommandées
```

### Navbar (`src/components/layout/Navbar.tsx`)

Navigation principale avec :
- Logo PwC
- Liens : Dashboard, Missions, Historique, Rapports, Audit Trail (conditionnel au rôle)
- Menu utilisateur (profil, déconnexion)
- Sélecteur de langue FR/EN

---

## 10. Intégration des modèles ML depuis les notebooks

### Architecture globale

Les modèles sont **entraînés dans des notebooks Jupyter** (hors du projet), puis **exportés en fichiers `.pkl` ou dossiers** dans `outputs/models/`. FastAPI les charge au démarrage.

```
notebooks/                          outputs/models/
├── 01_eda.ipynb                    ├── xgb_smote.pkl
├── 02_feature_engineering.ipynb    ├── iso_forest.pkl
├── 03_train_xgb.ipynb    ──────►  ├── lr_balanced.pkl
├── 04_train_autoencoder.ipynb ──► ├── autoencoder/
│                                  │   ├── model.pt
│                                  │   └── config.json
└── 05_evaluate.ipynb   ──────────►├── scaler.pkl
                                   ├── iso_forest_scaler.pkl
                                   ├── features.json
                                   └── optimal_thresholds.json
```

### Exporter un modèle depuis un notebook

```python
# Dans le notebook d'entraînement
import joblib
import json

# Sauvegarder XGBoost
joblib.dump(xgb_model, "../../outputs/models/xgb_smote.pkl")

# Sauvegarder le scaler
joblib.dump(scaler, "../../outputs/models/scaler.pkl")

# Sauvegarder les seuils optimaux
thresholds = {
    "XGB_smote": 0.42,
    "IsoForest": -0.1,
}
with open("../../outputs/models/optimal_thresholds.json", "w") as f:
    json.dump(thresholds, f)

# Sauvegarder les noms de features utilisées
with open("../../outputs/models/features.json", "w") as f:
    json.dump({"feature_cols": FEATURE_COLS}, f)

# Pour PyTorch (AutoEncoder)
import torch
torch.save(autoencoder.state_dict(), "../../outputs/models/autoencoder/model.pt")
```

### Comment FastAPI charge les modèles

Dans `app/main.py`, la fonction `lifespan` charge tout au démarrage du serveur :

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement au démarrage
    models = load_all_models(PROJECT_ROOT)   # depuis app/services/predictor.py
    app.state.models = models                # stocké dans l'état global FastAPI

    llm = get_llm_helper()                   # depuis app/services/llm_service.py
    app.state.llm = llm

    app.state.results_cache = {}             # cache des résultats (max 20 runs)

    yield  # ← serveur actif ici

    # Nettoyage à l'arrêt
    app.state.results_cache.clear()
```

Dans `app/services/predictor.py` :
```python
def load_all_models(project_root: Path) -> dict:
    models_dir = project_root / "outputs" / "models"

    return {
        "xgb": joblib.load(models_dir / "xgb_smote.pkl"),
        "scaler": joblib.load(models_dir / "scaler.pkl"),
        "iso_forest": joblib.load(models_dir / "iso_forest.pkl"),
        "iso_forest_scaler": joblib.load(models_dir / "iso_forest_scaler.pkl"),
        "autoencoder": load_autoencoder(models_dir / "autoencoder"),
        "thresholds": json.loads((models_dir / "optimal_thresholds.json").read_text()),
        "feature_cols": json.loads((models_dir / "features.json").read_text())["feature_cols"],
    }
```

### Pipeline de prédiction complet

```
POST /api/predict (CSV file)
         │
         ▼
1. DatasetProfiler.profile(df)
   → Détecte types colonnes, qualité, colonnes manquantes
         │
         ▼
2. ColumnMapper.map(df.columns)
   → Mappe les noms de colonnes vers le schéma PaySim
   → Ex: "montant" → "amount", "type_op" → "type"
         │
         ▼
3. SchemaDetector.detect(df, column_mapping)
   → Choisit le mode:
   ├── paysim       : XGBoost + AutoEncoder (schéma complet)
   ├── ae_isoforest : AutoEncoder + IsoForest (colonne amount présente)
   ├── ae_only      : AutoEncoder seul (quelques numériques)
   └── isoforest    : IsoForest seul (schéma inconnu)
         │
         ▼
4. DynamicFeatureBuilder.build(df, mapping)
   → Construit les 14 features pour AE:
   step, hour, day, week, high_risk_hour,
   log_amount, type_CASH_IN, type_CASH_OUT,
   type_DEBIT, type_PAYMENT, type_TRANSFER,
   balance_diff, dest_zero_balance, amount_scaled
         │
         ▼
5. FeatureEngineer.transform(df)
   → Enrichit avec features génériques
   (datetime extraction, encoding, patterns)
         │
         ▼
6. Prédictions selon le mode:
   paysim       → xgb.predict_proba() + ae.reconstruct_error()
   ae_isoforest → ae.reconstruct_error() + iso_forest.score_samples()
   ae_only      → ae.reconstruct_error()
   isoforest    → iso_forest.score_samples()
         │
         ▼
7. Attribution des niveaux de risque:
   score > threshold × 2  → CRITIQUE
   score > threshold      → ELEVE
   sinon                  → FAIBLE
         │
         ▼
8. Cache des résultats (run_id → liste transactions)
         │
         ▼
9. Retour PredictResponse (JSON)
   + log monitoring en BDD
```

### Sélection automatique du mode de prédiction

Le `SchemaDetector` analyse les colonnes du CSV et choisit le meilleur modèle :

```python
# app/services/schema_detector.py
PAYSIM_REQUIRED = {"step", "type", "amount", "nameOrig", "nameDest",
                   "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}

def detect_mode(df, column_mapping):
    mapped_cols = {v.canonical for v in column_mapping.values()}
    n_mapped = len(mapped_cols & PAYSIM_REQUIRED)

    if n_mapped >= 6:
        return "paysim"     # XGB + AE
    elif "amount" in mapped_cols:
        return "ae_isoforest"
    elif len(df.select_dtypes("number").columns) >= 3:
        return "ae_only"
    else:
        return "isoforest"
```

### Intégrer un nouveau modèle

**Étape 1 — Entraîner et exporter depuis le notebook :**
```python
joblib.dump(new_model, "../../outputs/models/new_model.pkl")
```

**Étape 2 — Charger dans `predictor.py` :**
```python
def load_all_models(project_root):
    return {
        # ... modèles existants ...
        "new_model": joblib.load(models_dir / "new_model.pkl"),
    }
```

**Étape 3 — Ajouter le mode dans `schema_detector.py` :**
```python
def detect_mode(df, mapping):
    # ... logique existante ...
    elif some_condition:
        return "new_mode"
```

**Étape 4 — Implémenter les prédictions dans `predict.py` :**
```python
if schema.mode == "new_mode":
    scores = app.state.models["new_model"].predict_proba(features)[:, 1]
```

**Étape 5 — Déclarer les métriques dans `outputs/reports/baseline_report.json` :**
```json
{
  "new_model": {
    "recall": 0.91,
    "precision": 0.88,
    "f1": 0.895,
    "roc_auc": 0.97
  }
}
```

**Étape 6 — Ajouter dans les types TypeScript (`src/types/index.ts`) :**
```typescript
export type AnalysisModel = "paysim" | "ae_isoforest" | "ae_only" | "isoforest" | "new_mode";
```

---

## 11. Internationalisation (FR/EN)

Le contexte `LanguageContext` (`src/lib/i18n/LanguageContext.tsx`) fournit un hook `useTranslation()` à tous les composants.

```typescript
const { t, language, setLanguage } = useTranslation();

// Utilisation
<h1>{t("dashboard.title")}</h1>
<Button onClick={() => setLanguage(language === "fr" ? "en" : "fr")}>
  {language === "fr" ? "English" : "Français"}
</Button>
```

**Ajouter une nouvelle clé de traduction :**
```typescript
// Dans src/lib/i18n/LanguageContext.tsx, dans l'objet translations :
const translations = {
  fr: {
    "ma_nouvelle_section.titre": "Mon Titre",
    "ma_nouvelle_section.description": "Ma description en français",
  },
  en: {
    "ma_nouvelle_section.titre": "My Title",
    "ma_nouvelle_section.description": "My English description",
  },
};
```

---

## 12. Système de rapports

### Génération PDF (`reportService.generateReport`)

```typescript
// src/lib/api/reportService.ts
async generateReport(payload: ReportPayload): Promise<Blob> {
  const { data } = await mlClient.post("/api/report", payload, {
    responseType: "blob",
  });
  return data;
}

// Dans le composant ReportSection
async function handleGenerate() {
  const blob = await reportService.generateReport({
    mission_name: mission.name,
    company_name: mission.company_name,
    transactions: predictionResult.transactions,
    fraud_stats: { n_fraud, fraud_rate_pct, amount_at_risk },
    schema_detection: predictionResult.schema_detection,
    explanations: batchExplanations,
  });

  // Déclencher le téléchargement
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `rapport_audit_${Date.now()}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}
```

### Structure du rapport PDF (7 pages)

1. Page de garde PwC (logo, nom mission, date)
2. Résumé exécutif (KPIs : taux fraude, montant à risque)
3. Méthodologie (modèles utilisés, seuils, mode de détection)
4. Analyse des risques (graphiques : pie, bar, distribution scores)
5. Tableau des anomalies CRITIQUE et ELEVE
6. Explications détaillées des top 5 anomalies (SHAP + LLM)
7. Recommandations et conclusion

---

## 13. Piste d'audit (Audit Trail)

Chaque action significative est enregistrée automatiquement dans la table `audit_logs`.

### Logger une action (dans une API route Next.js)

```typescript
// Après chaque opération importante
await auditLogRepository.add({
  action: "dataset_upload",           // AuditAction enum
  user_id: currentUser.id,
  user_name: currentUser.name,
  user_role: currentUser.role,
  mission_id: missionId,
  mission_name: mission.name,
  details: `Dataset "${filename}" uploadé (${rowCount} lignes)`,
  metadata: { file_size: fileSizeBytes, row_count: rowCount },
});
```

### Actions auditées

| Action | Déclencheur |
|--------|-------------|
| `login` / `logout` | Connexion/déconnexion utilisateur |
| `mission_create` / `update` / `delete` | CRUD missions |
| `mission_assign` | Assignation d'un auditeur |
| `dataset_upload` / `delete` | Gestion datasets |
| `analysis_start` / `analysis_complete` | Pipeline ML |
| `report_generate` / `report_download` | Rapports |
| `anomaly_comment` / `anomaly_status_change` | Revue anomalies |
| `user_create` / `user_update` / `user_deactivate` | Admin utilisateurs |

### Consulter l'audit trail (page `/audit-trail`)

```typescript
// GET /api/audit-logs?mission_id=xxx&limit=100
const { data: logs } = useQuery({
  queryKey: ["audit-logs", missionId],
  queryFn: () => auditLogService.getAuditLogs({ mission_id: missionId }),
});
```

---

## 14. Développer une nouvelle fonctionnalité

### Exemple complet : ajouter une page "Anomalies favoris"

**1. Type TypeScript** (`src/types/index.ts`) :
```typescript
export interface FavoriteAnomaly {
  id: string;
  anomaly_id: string;
  user_id: string;
  note?: string;
  created_at: string;
}
```

**2. Schéma Prisma** (`prisma/schema.prisma`) :
```prisma
model FavoriteAnomaly {
  id        String   @id @default(uuid())
  anomalyId String   @map("anomaly_id")
  userId    String   @map("user_id")
  note      String?
  createdAt DateTime @default(now()) @map("created_at")

  anomaly Anomaly @relation(fields: [anomalyId], references: [id], onDelete: Cascade)
  user    User    @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([anomalyId, userId])
  @@map("favorite_anomalies")
}
```

```bash
npx prisma migrate dev --name "add_favorite_anomalies"
```

**3. Repository** (`src/lib/db/repositories/favoriteAnomalyRepository.ts`) :
```typescript
export const favoriteAnomalyRepository = {
  async add(userId: string, anomalyId: string, note?: string) {
    return prisma.favoriteAnomaly.create({
      data: { userId, anomalyId, note },
    });
  },

  async findByUser(userId: string) {
    return prisma.favoriteAnomaly.findMany({
      where: { userId },
      include: { anomaly: true },
      orderBy: { createdAt: "desc" },
    });
  },
};
```

**4. API Route** (`src/app/api/favorites/route.ts`) :
```typescript
export async function GET(request: Request) {
  const user = await getCurrentUser(request);
  const favorites = await favoriteAnomalyRepository.findByUser(user.id);
  return Response.json(favorites);
}

export async function POST(request: Request) {
  const user = await getCurrentUser(request);
  const { anomaly_id, note } = await request.json();
  const fav = await favoriteAnomalyRepository.add(user.id, anomaly_id, note);
  return Response.json(fav, { status: 201 });
}
```

**5. Service** (`src/lib/api/favoriteService.ts`) :
```typescript
export const favoriteService = {
  async getAll(): Promise<FavoriteAnomaly[]> {
    const { data } = await appClient.get<FavoriteAnomaly[]>("/favorites");
    return data;
  },
  async add(anomalyId: string, note?: string): Promise<FavoriteAnomaly> {
    const { data } = await appClient.post<FavoriteAnomaly>("/favorites", { anomaly_id: anomalyId, note });
    return data;
  },
};
```

**6. Page** (`src/app/(dashboard)/favorites/page.tsx`) :
```typescript
"use client";
import { useQuery } from "@tanstack/react-query";
import { favoriteService } from "@/lib/api/favoriteService";

export default function FavoritesPage() {
  const { data: favorites, isLoading } = useQuery({
    queryKey: ["favorites"],
    queryFn: favoriteService.getAll,
  });

  if (isLoading) return <div>Chargement...</div>;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-pwc-dark">Anomalies favorites</h1>
      <div className="mt-4 space-y-2">
        {favorites?.map(fav => (
          <div key={fav.id} className="card">
            <p>Transaction #{fav.anomaly.transactionId}</p>
            <p className="text-gray-500">{fav.note}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

**7. Lien dans la Navbar** (`src/components/layout/Navbar.tsx`) :
```typescript
const navLinks = [
  { href: "/dashboard", label: t("nav.dashboard") },
  { href: "/missions", label: t("nav.missions") },
  { href: "/favorites", label: "Favoris" },  // ← ajouter ici
  // ...
];
```

---

## 15. Variables d'environnement

### `frontend/.env` (jamais commité — contient des secrets)

```bash
# Connexion PostgreSQL pour Prisma (côté serveur Next.js)
DATABASE_URL=postgresql://postgres:123@localhost:5432/pwcaudit
```

### `frontend/.env.local` (jamais commité — settings locaux)

```bash
# URL du backend FastAPI (accessible depuis le navigateur via le proxy)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### `.env` (racine du projet — côté FastAPI)

```bash
# Clés LLM
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
HF_API_KEY=hf_...

# JWT — doit être le même secret qu'utilisé par Next.js
JWT_SECRET=your-super-secret-jwt-key-at-least-32-chars

# Base de données (asyncpg pour FastAPI async)
DATABASE_URL=postgresql+asyncpg://postgres:123@localhost:5432/pwcaudit

# Auto-migration au démarrage (dev only)
AUTO_MIGRATE=true
```

**Important :** le `JWT_SECRET` doit être **identique** côté Next.js et côté FastAPI, car les tokens émis par Next.js sont vérifiés par FastAPI.

---

## 16. Déploiement avec Docker

### Architecture Docker

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: pwcaudit
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: 123
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  fastapi:
    build: .           # Dockerfile à la racine
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - postgres

  frontend:
    build: ./frontend  # frontend/Dockerfile
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:123@postgres:5432/pwcaudit
      NEXT_PUBLIC_API_URL: http://fastapi:8000
    depends_on:
      - postgres
      - fastapi
```

### Lancer avec Docker

```bash
# Construire et démarrer tous les services
docker-compose up --build

# Lancer en arrière-plan
docker-compose up -d

# Voir les logs
docker-compose logs -f frontend
docker-compose logs -f fastapi

# Arrêter
docker-compose down
```

### Sans Docker (développement)

```bash
# Terminal 1 — PostgreSQL (si installé localement)
# S'assurer que PostgreSQL tourne sur le port 5432

# Terminal 2 — Backend FastAPI
cd anomaly_detection_project
pip install -r requirements_app.txt
uvicorn app.main:app --reload --port 8000

# Terminal 3 — Frontend Next.js
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Récapitulatif des flux principaux

### Flux 1 : Connexion utilisateur

```
LoginPage → authService.login() → POST /api/auth/login (Next.js)
→ userRepository.findByEmail() → bcrypt.compare()
→ JWT signé → cookie pwc_token + localStorage
→ AuthContext mis à jour → redirect vers /dashboard
```

### Flux 2 : Création de mission

```
CreateMissionModal → missionService.create() → POST /api/missions (Next.js)
→ getCurrentUser() → missionRepository.create() (Prisma → PostgreSQL)
→ auditLogRepository.add("mission_create")
→ React Query invalide ["missions"] → re-fetch → MissionCard affiché
```

### Flux 3 : Upload et analyse ML

```
UploadDropzone → File sélectionné
→ datasetService.uploadDataset() → POST /api/missions/:id/datasets (Next.js)
→ datasetRepository.create() → fichier stocké sur disque (file_storage.py)
→ AnalysisWizard.predict() → analysisService.predict(file)
→ POST /ml/api/predict (proxy → FastAPI port 8000)
→ Pipeline ML : profile → map → detect → build → predict
→ PredictResponse → ResultsDashboard affiché
→ analysisRunService.create() → résultats persistés en BDD
```

### Flux 4 : Explication d'une anomalie

```
Clic sur une ligne du AnomalyTable
→ analysisService.explain(txId) → GET /ml/api/explain/:tx_id (FastAPI)
→ FastAPI : cherche tx dans cache → compute_shap() → compute_ae_errors()
→ llm.generate_explanation() (Groq API)
→ ExplainResponse → ExplanationCard affiché
```

### Flux 5 : Génération de rapport

```
ReportSection → reportService.generateReport(payload)
→ POST /ml/api/report (FastAPI)
→ report_gen.py : FPDF2 → PDF binaire (7 pages, thème PwC)
→ Blob téléchargé par le navigateur
→ auditLogRepository.add("report_generate") via Next.js API
```
