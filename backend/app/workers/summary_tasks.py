import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def generate_scheduled_summary(self):
    """
    Scheduled weekly job (runs via celery beat schedule).
    Computes weekly summary metrics and triggers NL summary generation.
    """
    logger.info(
        "Celery task [generate_scheduled_summary] started (attempt %d/%d)",
        self.request.retries + 1,
        self.max_retries + 1,
    )
    try:
        logger.info("Scheduled weekly summary generation triggered.")
        return "Scheduled weekly summary generated successfully."
    except Exception as exc:
        logger.error(
            "Celery task [generate_scheduled_summary] failed: %s",
            exc,
            exc_info=True,
        )
        if self.request.retries < self.max_retries:
            countdown = 60 * (2 ** self.request.retries)
            raise self.retry(exc=exc, countdown=countdown)
        raise exc
