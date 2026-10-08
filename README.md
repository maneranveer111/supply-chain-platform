# Supply Chain Intelligence Platform

An enterprise-grade, multi-tenant demand forecasting and inventory intelligence platform. Built on FastAPI, PostgreSQL, Redis, Celery, and Next.js, the system predicts daily retail demand using deep learning (LSTM) and time-series clustering, tracks inventory health runways, automates purchase order generation with explainable AI reasoning, and dispatches automated transactional notifications.

---

## 🏗️ Architecture & Core Components

```
                              ┌─────────────────────────────────────────┐
                              │             Next.js Frontend            │
                              │           (Port 3000 Dashboard)         │
                              └────────────────────┬────────────────────┘
                                                   │ HTTP / REST
                                                   ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FastAPI Backend (Port 8000)                                     │
│                                                                                                       │
│  ┌─────────────────────────┐   ┌──────────────────────────┐   ┌────────────────────────────────────┐  │
│  │   Request Correlation   │   │  Multi-Tenant RBAC Auth  │   │     Fail-Open Rate Limiting        │  │
│  │   (X-Request-ID Header) │   │ (Admin, PM, Analyst)     │   │      (Redis Sliding Window)        │  │
│  └────────────┬────────────┘   └─────────────┬────────────┘   └──────────────────┬─────────────────┘  │
│               │                              │                                   │                    │
│               └──────────────────────────────┼───────────────────────────────────┘                    │
│                                              ▼                                                        │
│  ┌────────────────────────┐    ┌───────────────────────────┐    ┌──────────────────────────────────┐  │
│  │  Multi-Tier Forecast   │    │   Inventory Telemetry     │    │   Explainable PO Engine          │  │
│  │  - Benchmark LSTM      │    │   - Real-time Stockout    │    │   - Lead-time buffer math        │  │
│  │  - Custom Scaler LSTM  │    │   - Days of Supply Runway │    │   - Multi-tenant approvals       │  │
│  │  - KMeans Clustering   │    │   - Dynamic Reorder Point │    │   - Gemini / Groq reasoning      │  │
│  │  - Metadata Cohort     │    │   - Risk Scoring (0.0-1.0)│    │   - Domain fallback engine       │  │
│  └────────────┬───────────┘    └─────────────┬─────────────┘    └──────────────────┬───────────────┘  │
└───────────────┼──────────────────────────────┼─────────────────────────────────────┼──────────────────┘
                │                              │                                     │
                ▼                              ▼                                     ▼
┌───────────────────────────────┐ ┌──────────────────────────────┐ ┌────────────────────────────────────┐
│      PostgreSQL Database      │ │         Redis Cache          │ │        Celery Worker & Beat        │
│  - Stores & User Tenants      │ │  - Non-blocking Scan Eviction│ │  - Weekly Executive Summaries      │
│  - Raw Daily Sales & Features │ │  - Sliding Window Limits     │ │  - Automated Low-Stock Alerting    │
│  - Scalers & Cluster Profiles │ │  - Cache Key Versioning (v2) │ │  - Subscription Reminders          │
│  - Purchase Orders & Inventory│ │  - Task Retry Deduplication  │ │  - Store Feature Onboarding        │
└───────────────────────────────┘ └──────────────────────────────┘ └─────────────────┬──────────────────┘
                                                                                     │
                                                                                     ▼
                                                                   ┌────────────────────────────────────┐
                                                                   │       External Integrations        │
                                                                   │  - Gemini / Groq LLMs (AI Insights)│
                                                                   │  - Brevo REST API (Email Delivery) │
                                                                   └────────────────────────────────────┘
```

---

## 🚀 Key Capabilities

### 1. Multi-Tier Demand Forecasting Pipeline
* **LSTM Benchmark Inference:** Stores mapped to Rossmann benchmark stores utilize pre-trained LSTM weights with standardized feature transformations.
* **Custom Onboarding Pipeline:** Upload historical sales CSVs (`POST /stores/{id}/upload-sales`) to dynamically fit store-specific standard scalers and generate 30-day lag features.
* **Cold-Start Fallbacks:** Stores without sufficient history route to KMeans behavioral cluster averages or metadata cohort profiles (`store_type` + `assortment`).
* **Non-Blocking Invalidation:** Caching layer operates on Redis with non-blocking key iteration (`SCAN`), isolating production response times from cache purges.

### 2. Multi-Tenant Authorization & RBAC
* **Role Hierarchy:** Supports `admin`, `procurement_manager`, and `store_analyst`.
* **Object-Level Isolation:** Non-admin tenants are strictly forbidden (`HTTP 403`) from accessing or modifying another tenant's sales history, forecasts, inventory, or purchase orders via `verify_store_access`.

### 3. Inventory Intelligence & Telemetry
* Real-time metrics computed per store:
  * **Days of Supply Runway:** `current_stock / forecast_daily_mean`.
  * **Dynamic Reorder Point:** Based on supplier lead time and safety stock buffers.
  * **Stockout Risk Scoring:** Normalizes inventory health (`0.0` safe to `1.0` imminent stockout).

### 4. Explainable Purchase Orders & Generative AI
* **Automated PO Generation:** Calculates optimal order quantities: `max(0, forecasted_demand + safety_stock - current_stock)`.
* **AI Business Justifications:** Generates domain explanations and executive store reviews using Google Gemini (or Groq), with automatic deterministic fallback if LLM keys are absent.

### 5. Asynchronous Background Automation (Celery + Redis)
* Periodic Celery Beat schedules:
  * Weekly platform executive summary dispatch (`summary_tasks.py`).
  * Daily low-stock scanning and owner alerting with retry deduplication (`notification_tasks.py`).
  * Daily user subscription renewal notifications.
  * Weekly cluster profile recalculation.

### 6. Production Security & Observability
* **Correlation Tracing:** Injected `X-Request-ID` headers trace incoming requests through structured JSON application logs with duration logging (`duration_ms`).
* **Error Sanitization:** Dedicated `SQLAlchemyError` handlers ensure database schemas and query details never leak in API responses.
* **Fail-Open Rate Limiter:** Sliding-window rate limiter fails open safely if Redis becomes unavailable, preserving API uptime.

---

## 🛠️ Project Structure

```
supply-chain-platform/
├── backend/
│   ├── app/
│   │   ├── api/v1/             # REST endpoints (auth, stores, inventory, forecasts, pos, admin)
│   │   ├── core/               # Configuration, security (JWT), authorization, dependencies
│   │   ├── db/                 # Database session, migrations (Alembic), Redis client
│   │   ├── ml/                 # LSTM inference, KMeans clustering, forecast router
│   │   ├── models/             # SQLAlchemy ORM models (Store, User, Inventory, PO, etc.)
│   │   ├── schemas/            # Pydantic v2 validation models
│   │   ├── services/           # Business logic (forecast, inventory, PO engine, email, LLM)
│   │   └── workers/            # Celery app, scheduled tasks, and event workers
│   ├── saved_models/           # Pretrained ML artifacts (LSTM weights, scalers, feature cols)
│   ├── scripts/                # Data seeding and migration utilities
│   ├── tests/                  # Pytest test suite (105 automated unit and integration tests)
│   ├── alembic.ini             # Database migration configuration
│   ├── requirements.txt        # Backend Python dependencies
│   └── .env.example            # Environment configuration template
├── frontend/                   # Next.js user dashboard
└── README.md                   # Platform documentation
```

---

## ⚙️ Environment Configuration

Create a `.env` file in `backend/` based on `.env.example`:

```ini
# Application
APP_NAME=supply-chain-platform
ENVIRONMENT=development
SECRET_KEY=change-this-to-a-long-random-string
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/supplychain

# Redis & Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Machine Learning Parameters
MODEL_DIR=saved_models
FORECAST_WINDOW=30
FORECAST_HORIZON=7

# Generative AI Intelligence Layer (Optional - falls back to domain heuristics)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

# Transactional Email Alerts via Brevo REST API (Optional)
BREVO_API_KEY=your_brevo_api_key_here
BREVO_SENDER_EMAIL=alerts@yourdomain.com
BREVO_SENDER_NAME=Supply Chain Platform

# Production Security & Logging
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
LOG_LEVEL=INFO
```

---

## 💻 Local Setup & Execution

### 1. Prerequisites
* Python 3.11+
* PostgreSQL 14+
* Redis 6+
* Node.js 18+ (for frontend dashboard)

### 2. Backend Setup
```bash
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Apply database migrations to head
alembic upgrade head

# Seed initial store data and demo users
python -m scripts.seed_data

# Start FastAPI application server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Background Workers (Celery)
In separate terminals:
```bash
# Start Celery Worker
celery -A app.workers.celery_app worker --loglevel=info

# Start Celery Beat Scheduler
celery -A app.workers.celery_app beat --loglevel=info
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The API documentation is interactively available at `http://localhost:8000/docs`.

---

## 🧪 Testing & Verification

The platform maintains a comprehensive automated test suite covering all operational components:

```bash
cd backend
python -m pytest tests/ -v
```

### Test Suite Coverage (105 Tests Passing):
* **Phase 1–5 Base Tests (70 tests):** Authentication, Store lifecycle, Forecast routing across all tiers, Redis caching resilience, Custom scaler uploads, Celery background tasks, Liveness & readiness probes.
* **Phase 6 Intelligence & Telemetry (23 tests):** Gemini & Groq multi-provider fallback, Brevo REST API email dispatch with safety timeouts, Inventory telemetry, Multi-tenant inventory boundaries, Explainable PO reasoning, Admin overview & inspection.
* **Phase 7 Production Hardening (12 tests):** `X-Request-ID` correlation middleware, Database error sanitization, Object-level PO authorization checks, Sliding-window fail-open rate limiting, Upload file size limits, Admin matrix outer-join queries, Celery task retry deduplication.

---

## 🔒 Security & Production Guidelines

1. **Secret Safety:** The `.env` file is excluded from revision control via `.gitignore`. Never commit API keys or database connection strings.
2. **CORS Headers:** In production, specify explicit domain origins using `CORS_ALLOWED_ORIGINS` (e.g. `https://supplychain.example.com`).
3. **Database Maintenance:** High-frequency query columns (`stores.onboarding_status`, `purchase_orders.status`, `stores.user_id`, `stores.benchmark_store_id`) are indexed in Alembic migration `68c71bd7b578`.
