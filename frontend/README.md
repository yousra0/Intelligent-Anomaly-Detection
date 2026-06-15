# PwC Audit Analytics Platform — Frontend

Enterprise-grade Next.js 15 frontend for the Financial Anomaly Detection Platform.

## Stack

- **Next.js 15** (App Router) + TypeScript
- **TailwindCSS** + **shadcn/ui** + Lucide Icons
- **Recharts** for data visualization
- **TanStack Query** (React Query) for data fetching
- **React Hook Form** + Zod validation
- **Sonner** for toast notifications

## Quick Start

```bash
cd frontend
cp .env.local.example .env.local
# Edit .env.local with your FastAPI URL and JWT secret
npm install
npm run dev
```

Open http://localhost:3000. Login with:
- `auditeur@pwc.com` / `pwc2024`
- `manager@pwc.com` / `pwc2024`

## Environment Variables

| Variable | Description |
|---|---|
| `FASTAPI_URL` | FastAPI backend URL (server-side proxy) |
| `NEXT_PUBLIC_API_URL` | Public base URL for the Next.js app |
| `JWT_SECRET` | 32+ char secret for signing JWT tokens |
| `JWT_EXPIRY` | JWT expiry in seconds (default: 28800 = 8h) |

## Architecture

```
src/
├── app/
│   ├── (auth)/login/          ← Login page (JWT, @pwc.com only)
│   ├── (dashboard)/
│   │   ├── missions/          ← Mission list + create
│   │   └── missions/[id]/
│   │       ├── page.tsx       ← Mission detail + datasets
│   │       └── analysis/      ← Analysis wizard + results
│   └── api/
│       ├── auth/              ← JWT login/logout API routes
│       └── missions/          ← Mission CRUD API routes
├── components/
│   ├── analysis/              ← Wizard, KPIs, charts, anomaly table
│   ├── datasets/              ← Upload dropzone, dataset cards
│   ├── explanations/          ← SHAP + LIME + LLM explanation card
│   ├── layout/                ← TopBar
│   ├── missions/              ← Mission card + create modal
│   ├── reports/               ← PDF/DOCX report generation
│   └── ui/                    ← shadcn-style UI primitives
├── lib/
│   ├── api/                   ← Service layer (FastAPI + app)
│   └── auth/                  ← AuthContext
└── types/index.ts             ← All TypeScript interfaces
```

## API Integration

### FastAPI ML Backend (via `/ml/*` proxy)
| Frontend call | FastAPI endpoint | Description |
|---|---|---|
| `analysisService.predict(file)` | `POST /api/predict` | Run full anomaly detection |
| `llmService.getExplanation(txId)` | `GET /api/explain/{tx_id}` | SHAP + LIME + LLM explanation |
| `llmService.getBatchExplanations(ids)` | `POST /api/explain/batch` | Batch explanations |
| `reportService.generatePDF()` | `POST /api/report` | PDF audit report |
| `reportService.generateDOCX()` | `POST /api/report/docx` | Word audit report |
| `analysisService.getModels()` | `GET /api/models` | Model metrics |

### Next.js API Routes (in-memory → replace with PostgreSQL)
| Route | Description |
|---|---|
| `POST /api/auth/login` | JWT authentication |
| `POST /api/auth/logout` | Clear JWT cookie |
| `GET/POST /api/missions` | List / create missions |
| `GET/PUT/DELETE /api/missions/[id]` | Mission CRUD |
| `GET/POST /api/missions/[id]/datasets` | Dataset management |

## Production Notes

1. Replace in-memory stores in API routes with PostgreSQL queries
2. Use a real password hashing library (bcrypt) for user passwords
3. Set `NODE_ENV=production` for secure cookies
4. Configure `JWT_SECRET` with a strong random value
5. Update `next.config.ts` `FASTAPI_URL` to point to production FastAPI

## PwC Design System

- **Primary orange**: `#D04A02` (used for CTAs, highlights)
- **Dark**: `#293854` (headers)
- **Background**: `#F7F7F7`
- **Risk CRITIQUE**: Red `#C00000`
- **Risk ÉLEVÉ**: Orange `#D04A02`
- **Risk FAIBLE**: Green `#008246`
