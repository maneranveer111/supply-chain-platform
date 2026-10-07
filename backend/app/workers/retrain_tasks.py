import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def retrain_model(self):
    """
    Kicks off LSTM/Prophet retraining pipeline.
    """
    logger.info(
        "Celery task [retrain_model] started (attempt %d/%d)",
        self.request.retries + 1,
        self.max_retries + 1,
    )
    try:
        # Placeholder for full training pipeline
        logger.info("Retraining triggered (placeholder — pipeline integration hook).")
        return "Retraining triggered successfully."
    except Exception as exc:
        logger.error("Celery task [retrain_model] failed: %s", exc, exc_info=True)
        if self.request.retries < self.max_retries:
            countdown = 60 * (2 ** self.request.retries)
            raise self.retry(exc=exc, countdown=countdown)
        raise exc
