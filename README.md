[![CI](https://github.com/vjdonald1117-eng/parlaypal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vjdonald1117-eng/parlaypal/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)](https://react.dev/)

# Parlay Pal

**Parlay Pal** is an NBA betting analytics platform that blends feature-engineered **XGBoost** player-prop modeling with **Monte Carlo simulation** to rank edges and expected value. It turns schedule, injury, pace, and market context into a disciplined workflow for props, game projections, and multi-leg parlays—backed by a **FastAPI** service and a **React** operator console.

![ParlayPal Dashboard](./assets/dashboard-preview.png)

> Add your own screenshot at `assets/dashboard-preview.png` (recommended width 1200–1600px) so contributors and stakeholders see the live UI at a glance.

---

## At a glance

| Area | What you get |
|------|----------------|
| **Modeling** | XGBoost-assisted means fused with domain heuristics before simulation. |
| **Risk engine** | Vectorized NumPy sampling and aggregation across large trial counts per prop. |
| **Operations** | PostgreSQL (Supabase) for slate, box scores, injuries, and graded prediction history. |
| **Client** | Vite + React + TypeScript + Tailwind, calling versioned REST endpoints under `/api`. |

---

## Tech Stack

| Layer | Technologies |
|-------|----------------|
| **Runtime** | Python 3.11, Uvicorn |
| **API** | FastAPI, Pydantic v2, Starlette / HTTPX ecosystem |
| **Data** | SQLAlchemy 2.x, psycopg2, PostgreSQL (Supabase-hosted) |
| **ML & numerics** | XGBoost, scikit-learn, NumPy, Pandas, SciPy |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS |
| **Quality** | pytest, GitHub Actions (`ubuntu-latest`, workflow `ci.yml`) |

---

## Architecture & Performance

- **Vectorized Monte Carlo** — Player and slate-level draws use contiguous NumPy arrays for sampling, clipping, and summary statistics (over/under mass, percentiles, verdicts) without Python-level inner loops over trials.
- **Concurrent CPU work** — The API fans out independent per-player simulation units through a **`ThreadPoolExecutor`**. Much of the Monte Carlo path runs in **NumPy’s native code**, which releases the CPython GIL during heavy numerical work, improving slate throughput while preserving per-player outputs for logging and the UI.
- **Async-first HTTP surface** — FastAPI **`lifespan`** hooks and **`async` route handlers** (for example health and read-mostly projections) keep the event loop responsive while long-running refresh work is isolated behind **job IDs**, in-process locks, and structured logging instead of blocking the whole server.
- **Unified slate pipeline** — A single refresh path materializes prop tables, game-level projections, EV-ranked parlays, and persistence into **`prediction_log`** for downstream calibration and history grading.

---

## Environment setup

Configuration is loaded from a **`.env` file at the repository root** (see `python-dotenv` usage in `database.py` and `api.py`). Do **not** commit `.env`. Consider adding a committed **`.env.example`** (no secrets) that mirrors the variables below for onboarding.

### 1. Create `.env` in the repo root

```bash
# Windows (PowerShell): New-Item .env -ItemType File
# macOS / Linux:        touch .env
```

### 2. Supabase (PostgreSQL) — required

Parlay Pal expects a standard **SQLAlchemy** connection string in **`DATABASE_URL`**.

1. In the [Supabase Dashboard](https://supabase.com/dashboard), open **Project Settings → Database**.
2. Under **Connection string**, choose the **URI** format.
3. Prefer the **Transaction pooler** (PgBouncer, typically port **6543**) for API-style workloads unless you rely on session-scoped features incompatible with transaction pooling.
4. Copy the URI and set:

```env
# Supabase / Postgres (required). Use sslmode=require in production.
# Example shape (replace host, user, password, db name with yours):
DATABASE_URL=postgresql://postgres.[PROJECT_REF]:[YOUR_PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require
```

The application enables **`sslmode=require`** and UTC session options in `database.py` / `migrate.py` to align with Supabase defaults.

### 3. External APIs — optional but recommended for full slate features

```env
# The Odds API — live lines, sync-game-lines, and some parlay paths
ODDS_API_KEY=your_odds_api_key
```

### 4. Tunables (optional)

```env
# History / lookback windows (scripts and some API subprocess helpers)
LOOKBACK_HOURS=48
PARLAY_LOOKBACK_HOURS=168

# Default American -110 style pricing when deriving EV (see routers/simulations.py)
PARLAY_DEFAULT_DECIMAL_ODDS=1.9091

# Feature flags
PARLAY_DISABLE_NBA_RESOLVE=false
```

### 5. Frontend (`parlay-ui`)

For local development you usually rely on the **Vite dev proxy** to `/api`. If you must call the API directly (for example cross-origin testing), set in **`parlay-ui/.env.local`** or **`.env.development`**:

```env
# Only when not using the proxy; on Windows prefer 127.0.0.1 over localhost if IPv6 causes issues.
VITE_API_BASE=http://127.0.0.1:8000
```

---

## How to run locally

### Backend (FastAPI)

```powershell
cd "c:\Users\Jayyy\OneDrive\Desktop\parlaypal"
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Health check: `http://127.0.0.1:8000/api/health`

### Frontend (Vite + React)

Open a **second** terminal:

```powershell
cd "c:\Users\Jayyy\OneDrive\Desktop\parlaypal\parlay-ui"
npm install
npm run dev -- --host 127.0.0.1
```

Application URL: `http://127.0.0.1:5173/`

---

## Repository layout (quick reference)

| Path | Role |
|------|------|
| `api.py` | FastAPI application entry and orchestration |
| `routers/` | Modular route groups (simulations, jobs, history, health) |
| `services/` | Simulation engine and domain services |
| `models/` | Parlay builder, projections, XGBoost training helpers |
| `scripts/` | ETL, schedule sync, odds fetch, operational tooling |
| `parlay-ui/` | React client |
| `tests/` | pytest suite (run in CI via `.github/workflows/ci.yml`) |
