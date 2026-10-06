from app.workers.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3)
def generate_scheduled_summary(self):
    """
    Scheduled weekly job (see celery_app.py beat_schedule). Computes the
    week's stats and generates + stores the NL summary so it's ready
    before procurement managers check Monday morning, rather than being
    generated on-demand at request time.
    """
    try:
        # TODO: compute real weekly stats from DB, call
        # summary_service.generate_weekly_summary, persist the result.
        print("Scheduled weekly summary generation triggered (placeholder)")
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))
