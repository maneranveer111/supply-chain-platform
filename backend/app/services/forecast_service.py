import json
import logging
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.redis import get_redis
from app.ml.inference import (
    ModelNotLoadedError,
    build_forecast_dates,
    load_feature_cols,
    predict_for_store,
)
from app.models.daily_store_feature import DailyStoreFeature

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours


async def get_forecast(store_id: int, horizon: int, db: Session) -> dict:
    redis = await get_redis()
    cache_key = f"forecast:{store_id}:{horizon}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    forecast = await _generate_forecast(store_id, horizon, db)
    await redis.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(forecast))
    return forecast


async def _generate_forecast(store_id: int, horizon: int, db: Session) -> dict:
    recent_window = _load_recent_window(store_id, db)

    try:
        predictions = predict_for_store(store_id, recent_window, horizon)
    except ModelNotLoadedError:
        # Graceful fallback so the API is runnable/demoable before a
        # trained model is dropped into saved_models/.
        predictions = _mock_predictions(horizon)

    dates = build_forecast_dates(horizon)

    return {
        "store_id": store_id,
        "horizon": horizon,
        "window_used": settings.forecast_window,
        "forecast": [
            {"day": i + 1, "date": dates[i], "predicted_sales": round(predictions[i], 2)}
            for i in range(horizon)
        ],
        "generated_at": datetime.utcnow().isoformat(),
    }


def _load_recent_window(store_id: int, db: Session) -> np.ndarray:
    """
    Queries the last `settings.forecast_window` days of engineered
    features for this store from daily_store_features, re-orders each
    row's JSON dict to match feature_cols (the exact order the model
    was trained on), and returns a (window_size, num_features) array.

    If the store has no rows yet (nothing has been seeded/ingested),
    falls back to a zero-filled window so the endpoint stays runnable
    out of the box -- predictions in that case are meaningless and the
    mock-prediction fallback in ml/inference.py will likely kick in
    anyway if no model is loaded either.
    """
    feature_cols = load_feature_cols()
    window = settings.forecast_window

    rows = (
        db.query(DailyStoreFeature)
        .filter(DailyStoreFeature.store_id == store_id)
        .order_by(DailyStoreFeature.date.desc())
        .limit(window)
        .all()
    )
    rows = list(reversed(rows))  # chronological order, oldest first

    if not rows:
        logger.warning(
            "No daily_store_features rows for store_id=%s -- "
            "returning a zero-filled window (seed data first for real forecasts).",
            store_id,
        )
        return np.zeros((window, len(feature_cols)), dtype=np.float32)

    if len(rows) < window:
        logger.warning(
            "Only %d of %d days of history available for store_id=%s -- "
            "padding the start of the window with zeros.",
            len(rows), window, store_id,
        )

    matrix = np.zeros((window, len(feature_cols)), dtype=np.float32)
    # Right-align: if we have fewer rows than the window, the most
    # recent data sits at the END of the array (closest to "today"),
    # matching how the model expects the window to end right before
    # the forecast period.
    offset = window - len(rows)
    for i, row in enumerate(rows):
        for j, col in enumerate(feature_cols):
            matrix[offset + i, j] = row.features.get(col, 0.0)

    return matrix


def _mock_predictions(horizon: int) -> list[float]:
    rng = np.random.default_rng(42)
    base = 5000
    return [float(base + rng.normal(0, 300)) for _ in range(horizon)]
