import logging
import os
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.v1 import api_router
from app.core.config import settings
from app.db.redis import get_redis
from app.db.session import SessionLocal
from app.ml.inference import (
    FEATURE_COLS_PATH,
    MODEL_PATH,
    STORE_SCALERS_PATH,
)

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def health_check():
    """
    Liveness probe: returns 200 as long as the application process is running.
    Does not depend on external services so pod/container restarts are not
    triggered falsely when backing services are transiently unavailable.
    """
    return {"status": "ok", "environment": settings.environment}


@app.get("/readiness")
async def readiness_check():
    """
    Readiness probe: validates availability of required backing infrastructure:
      - Database connectivity
      - Redis connectivity
      - ML artifact availability
    Returns 200 when ready/degraded, or 503 when primary datastore is down.
    Does not run expensive model inference.
    """
    checks = {}

    # 1. Database check
    db_ok = False
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            checks["database"] = "ok"
            db_ok = True
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Database readiness check failed: %s", exc)
        checks["database"] = "unreachable"

    # 2. Redis check
    try:
        r = await get_redis()
        await r.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.warning("Redis readiness check failed: %s", exc)
        checks["redis"] = "unreachable"

    # 3. ML model artifacts file check
    artifacts_present = (
        os.path.exists(MODEL_PATH)
        and os.path.exists(STORE_SCALERS_PATH)
        and os.path.exists(FEATURE_COLS_PATH)
    )
    checks["ml_artifacts"] = "ok" if artifacts_present else "missing"

    if not db_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "checks": checks},
        )

    overall_status = "ready" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall_status, "checks": checks}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Standardize unhandled application exceptions to prevent leaking
    stack traces, credentials, or internal details to API consumers.
    """
    logger.error("Unhandled exception processing request %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )
