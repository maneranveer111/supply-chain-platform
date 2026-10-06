"""
Forecast Service: coordinates caching, routing, and the API layer.

Responsibilities:
  - Redis cache layer (versioned key to invalidate Phase 1 cache entries).
  - Delegates all routing decisions to app.ml.forecast_router.
  - Converts ForecastResult → serializable dict.
  - Does NOT contain any scaler-lookup or ML-inference logic.

Cache key version:
  "v2" is appended to all keys in Phase 2 so that any old cache entries
  (which lack forecast_method / forecast_confidence) are automatically
  bypassed.  Old keys will expire on their original TTL without intervention.
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

# Cache key version — increment whenever response schema changes to prevent
# stale cache entries with missing fields from being served.
CACHE_KEY_VERSION = "v2"
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours


async def get_forecast(store_id: int, horizon: int, db: Session) -> dict:
    """
    Public entry point.  Checks the Redis cache first; on miss, calls
    _generate_forecast() and stores the result.

    The cache key is versioned so that Phase 1 cache entries (which do
    not contain forecast_method / forecast_confidence) are bypassed.
    """
    redis = await get_redis()
    cache_key = f"forecast:{CACHE_KEY_VERSION}:{store_id}:{horizon}"

    cached = await redis.get(cache_key)
    if cached:
        logger.debug("Cache HIT: %s", cache_key)
        return json.loads(cached)

    logger.debug("Cache MISS: %s", cache_key)
    forecast = await _generate_forecast(store_id, horizon, db)
    await redis.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(forecast))
    return forecast


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
