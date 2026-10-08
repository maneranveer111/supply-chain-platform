import logging
import os
import uuid
import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

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

# Basic logging configuration that can be imported/used by other modules
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

app = FastAPI(title=settings.app_name)

@app.middleware("http")
async def request_correlation_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    start_time = time.perf_counter()
    
    # Simple structlog-like prefix for log correlation (using standard logging)
    # We prefix logs in handlers if needed, or simply pass req_id to context
    logger.info("request_started method=%s path=%s req_id=%s", request.method, request.url.path, req_id)
    
    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "request_finished method=%s path=%s status=%d duration_ms=%s req_id=%s",
            request.method, request.url.path, response.status_code, duration_ms, req_id
        )
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            "request_failed method=%s path=%s duration_ms=%s req_id=%s error=%s",
            request.method, request.url.path, duration_ms, req_id, type(exc).__name__,
        )
        raise

allowed_origins = settings.get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["https://localhost"],
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
        logger.warning("Database readiness check failed: %s", type(exc).__name__)
        checks["database"] = "unreachable"

    # 2. Redis check
    try:
        r = await get_redis()
        await r.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.warning("Redis readiness check failed: %s", type(exc).__name__)
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


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """
    Prevent internal database errors, schema details, or SQL statements
    from leaking to the client.
    """
    logger.error("Database exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "A database error occurred while processing your request."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Standardize unhandled application exceptions to prevent leaking
    stack traces, credentials, or internal details to API consumers.
    """
    logger.error("Unhandled exception processing request %s: %s", request.url.path, type(exc).__name__, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )
