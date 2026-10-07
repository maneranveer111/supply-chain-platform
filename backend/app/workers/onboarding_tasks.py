"""
Celery background tasks for store onboarding data processing.
"""

import logging
from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
from app.services.onboarding_service import process_store_features_and_clustering
from app.services.forecast_service import invalidate_forecast_cache_sync

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    name="app.workers.onboarding_tasks.process_store_onboarding",
)
def process_store_onboarding(self, store_id: int):
    """
    Idempotent background job to process historical sales, fit scalers,
    generate feature rows, and assign clusters for a store.
    """
    logger.info(
        "Celery task [process_store_onboarding] started for store_id=%d (attempt %d/%d)",
        store_id,
        self.request.retries + 1,
        self.max_retries + 1,
    )
    db = SessionLocal()
    try:
        result = process_store_features_and_clustering(store_id, db)
        # Invalidate cache for this store across all horizons
        invalidate_forecast_cache_sync(store_id)
        logger.info(
            "Celery task [process_store_onboarding] completed successfully for store_id=%d: %s",
            store_id,
            result,
        )
        return result
    except Exception as exc:
        logger.error(
            "Celery task [process_store_onboarding] failed for store_id=%d: %s",
            store_id,
            exc,
            exc_info=True,
        )
        if self.request.retries < self.max_retries:
            countdown = 30 * (2 ** self.request.retries)
            logger.info(
                "Retrying process_store_onboarding for store_id=%d in %d seconds...",
                store_id,
                countdown,
            )
            raise self.retry(exc=exc, countdown=countdown)
        raise exc
    finally:
        db.close()
