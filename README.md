# Parlay Pal

Parlay Pal is an NBA betting analytics platform that combines feature-engineered XGBoost player-prop modeling with Monte Carlo simulation to surface high-confidence edges and EV-ranked parlays. It helps bettors make faster, more disciplined decisions by translating large volumes of player and game context into actionable +EV opportunities.

## Tech Stack

- Python 3 + FastAPI + Uvicorn
- XGBoost + NumPy + Pandas + SQLAlchemy
- Monte Carlo simulation engine for player props and game-level outcomes
- PostgreSQL/Supabase-backed data and prediction history workflow
- React 19 + TypeScript + Vite + Tailwind CSS

## Architecture & Performance

- FastAPI orchestrates the end-to-end prediction workflow: data refresh, projection generation, simulation runs, caching, and history grading endpoints consumed by the React client.
- XGBoost models are loaded into the core prediction pipeline and blended with heuristics/game-context features to improve expected stat means before simulation.
- Monte Carlo execution is optimized with NumPy vectorization, enabling fast probabilistic sampling and metric aggregation across thousands of trials per prop.
- The backend uses `ThreadPoolExecutor` to run many player simulations concurrently, reducing slate refresh time while preserving per-player simulation detail.
- Unified slate runs produce both player-prop outputs and game-level projections, then persist results for downstream EV ranking, parlay generation, and calibration analysis.

## How to Run Locally

### 1) Backend (FastAPI)

```powershell
cd "c:\Users\Jayyy\OneDrive\Desktop\parlaypal"
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

### 2) Frontend (Vite + React)

Open a second terminal:

```powershell
cd "c:\Users\Jayyy\OneDrive\Desktop\parlaypal\parlay-ui"
npm install
npm run dev -- --host 127.0.0.1
```

Frontend: `http://127.0.0.1:5173`  
Backend health check: `http://127.0.0.1:8000/api/health`
