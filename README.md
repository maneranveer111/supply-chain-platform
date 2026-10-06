# Supply Chain Intelligence Platform

A web-based platform that forecasts per-store daily demand, clusters
stores by sales behaviour, generates purchase order recommendations,
and produces a natural-language weekly summary for procurement
managers. Built on top of an LSTM vs. Prophet demand forecasting
pipeline trained on the Rossmann retail dataset.

## Structure

- `backend/` — FastAPI API, Celery workers, ML inference, Postgres via SQLAlchemy + Alembic
- `frontend/` — Next.js dashboard

## Quickstart (Docker)

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
docker compose up --build
```

Then, in a new terminal, run migrations and seed demo data:

```bash
docker compose exec backend alembic upgrade head
docker compose exec backend python -m scripts.seed_data
```

- API: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:3000

## Running without Docker

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Create all tables
alembic upgrade head

# Populate demo data (real data if processed_df.parquet is present
# in saved_models/, synthetic otherwise) so endpoints return real
# results instead of empty/mocked ones
python -m scripts.seed_data

uvicorn app.main:app --reload
```

In a second terminal, for background jobs:
```bash
celery -A app.workers.celery_app worker --loglevel=info
celery -A app.workers.celery_app beat --loglevel=info
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

## Wiring up your trained model

Copy these four files (produced by the Rossmann notebook's checkpoint)
into `backend/saved_models/`:

```
lstm_model.keras
store_scalers.pkl
comp_scaler.pkl
feature_cols.pkl
processed_df.parquet   (optional — lets scripts/seed_data.py seed REAL data)
```

Without `lstm_model.keras`, `/stores/{id}/forecast` still runs and
returns mock predictions, so the API is demoable end-to-end
immediately — see `app/services/forecast_service.py`.

Without `processed_df.parquet`, `scripts/seed_data.py` falls back to
generating synthetic daily features for 3 demo stores, so the DB-backed
parts of the pipeline (real forecasts, real purchase order math) are
still runnable without your full notebook output.

## Natural-language weekly summaries (optional)

Set in `backend/.env`:
```
LLM_PROVIDER=groq          # or "gemini"
GROQ_API_KEY=...           # https://console.groq.com/keys (free tier)
# or
GEMINI_API_KEY=...         # https://aistudio.google.com/app/apikey (free tier)
```
Leave both blank and `/reports/weekly-summary` still works — it returns
a deterministic plain-text summary instead of an LLM-generated one.

## What's implemented vs. scaffolded

| Component | Status |
|---|---|
| FastAPI app, routing, auth (JWT), RBAC | Implemented |
| Rate limiting (Redis sliding window) | Implemented |
| Database schema + Alembic migrations | Implemented (`alembic upgrade head` creates all 7 tables) |
| Demo data seeding | Implemented (`scripts/seed_data.py` — real or synthetic) |
| Forecast endpoint + Redis caching | Implemented — pulls real history from `daily_store_features` |
| Purchase order recommendation logic | Implemented — reads real `inventory` table, persists each PO row |
| PO approval endpoint | Implemented — updates the DB row's status |
| Clustering (KMeans on store behaviour) | Implemented as a pure function; writing results back to `stores.cluster_id` is still a TODO in the Celery task |
| NL weekly summary (Gemini or Groq) | Implemented, falls back to plain text with no key set |
| Celery workers + scheduled beat jobs | Scaffolded — retry logic is real, task bodies still have TODOs |
| Next.js dashboard | Implemented, calls the live API |
| Tests | A few starter tests — extend before relying on this |

## Remaining TODOs (smaller than before, but real)

- `workers/cluster_tasks.py` — should call `ml/clustering.fit_clusters()`
  against real sales data and write `cluster_id` back onto `Store` rows
  (currently a placeholder print statement)
- `workers/retrain_tasks.py` — should call an actual training pipeline
  and write new artifacts to `saved_models/` (currently a placeholder)
- An ingestion job to keep `daily_store_features` updated with each
  new day's real sales as they come in (currently only populated by
  the one-off `seed_data.py` script)
