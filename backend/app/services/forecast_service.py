"""
Forecast Service: coordinates caching, routing, and the API layer.

Responsibilities:
  - Versioned Redis cache layer with safe degradation on cache failure.
  - Granular cache invalidation per store or globally.
  - Delegates all routing decisions to app.ml.forecast_router.
  - Converts ForecastResult → serializable dict.
  - Does NOT contain any scaler-lookup or ML-inference logic.

Cache key strategy:
  Key format: forecast:{CACHE_KEY_VERSION}:{store_id}:{horizon}
  Example:    forecast:v2:1:7

  - Versioned namespace prevents stale formats from colliding across deployments.
  - store_id segment prevents cross-store or cross-user data leakage.
  - horizon segment isolates forecasts of different lengths (e.g. 7 vs 14 vs 30 days).
  - Invalidation uses pattern matching:
      forecast:v2:{store_id}:*  -> sweeps all horizons for the specified store.
      forecast:v2:*             -> sweeps all stores when clusters or models refresh.
"""

import json
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.redis import get_redis
from app.ml.forecast_router import StoreNotFoundError, route_forecast
from app.ml.inference import ModelNotLoadedError

logger = logging.getLogger(__name__)

CACHE_KEY_VERSION = "v2"
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours


async def get_forecast(store_id: int, horizon: int, db: Session) -> dict:
    """
    Public entry point. Checks the Redis cache first; on miss or Redis failure,
    computes the forecast directly. If Redis is unavailable, inference continues
    safely without crashing.
    """
    cache_key = f"forecast:{CACHE_KEY_VERSION}:{store_id}:{horizon}"
    redis = None

    try:
        redis = await get_redis()
        cached = await redis.get(cache_key)
        if cached:
            logger.info("Forecast cache HIT: store_id=%d, key=%s", store_id, cache_key)
            return json.loads(cached)
        logger.info("Forecast cache MISS: store_id=%d, key=%s", store_id, cache_key)
    except Exception as exc:
        logger.warning(
            "Redis cache unavailable for store_id=%d (%s). Falling back to direct forecast.",
            store_id,
            exc,
        )
        redis = None

    forecast = await _generate_forecast(store_id, horizon, db)

    if redis is not None:
        try:
            await redis.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(forecast))
            logger.debug("Forecast cached: key=%s, ttl=%ds", cache_key, CACHE_TTL_SECONDS)
        except Exception as exc:
            logger.warning(
                "Failed to write forecast to Redis for store_id=%d: %s",
                store_id,
                exc,
            )

    return forecast


async def invalidate_forecast_cache(store_id: int | None = None) -> int:
    """
    Asynchronously invalidates forecast cache keys.
    - If store_id is given: deletes all horizons for this store (`forecast:v2:{store_id}:*`).
    - If store_id is None: deletes all forecast cache keys (`forecast:v2:*`).
    Returns the count of deleted keys.
    Gracefully handles Redis downtime.
    """
    pattern = (
        f"forecast:{CACHE_KEY_VERSION}:{store_id}:*"
        if store_id is not None
        else f"forecast:{CACHE_KEY_VERSION}:*"
    )
    try:
        redis = await get_redis()
        if hasattr(redis, "scan_iter"):
            keys = [k async for k in redis.scan_iter(match=pattern, count=100)]
        else:
            keys = await redis.keys(pattern)
        if keys:
            count = await redis.delete(*keys)
            logger.info("Invalidated %d cache keys matching '%s'", count, pattern)
            return count

        logger.debug("No keys found to invalidate for pattern '%s'", pattern)
        return 0
    except Exception as exc:
        logger.warning(
            "Redis error during cache invalidation (pattern=%s): %s", pattern, exc
        )
        return 0


def invalidate_forecast_cache_sync(store_id: int | None = None) -> int:
    """
    Synchronous cache invalidator for Celery tasks or non-async contexts.
    """
    import redis as sync_redis

    pattern = (
        f"forecast:{CACHE_KEY_VERSION}:{store_id}:*"
        if store_id is not None
        else f"forecast:{CACHE_KEY_VERSION}:*"
    )
    try:
        r = sync_redis.from_url(settings.redis_url, socket_timeout=2.0, socket_connect_timeout=2.0)
        keys = list(r.scan_iter(match=pattern, count=100))
        if keys:
            count = r.delete(*keys)
            logger.info("Sync invalidated %d cache keys matching '%s'", count, pattern)
            return count
        return 0
    except Exception as exc:
        logger.warning(
            "Sync Redis error during cache invalidation (pattern=%s): %s", pattern, exc
        )
        return 0


async def _generate_forecast(store_id: int, horizon: int, db: Session) -> dict:
    """
    Delegates to the forecast router and converts the result to a dict.

    Error handling contract:
      - StoreNotFoundError  → re-raised; API layer converts to HTTP 404.
      - ModelNotLoadedError → re-raised; API layer converts to HTTP 503.
      - All routing failures → ForecastResult(insufficient_data) is returned
                               as a valid dict (never an exception).
    """
    result = route_forecast(
        app_store_id=store_id,
        horizon=horizon,
        forecast_window=settings.forecast_window,
        db=db,
    )
    return result.to_dict()
