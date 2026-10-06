from app.workers.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3)
def refresh_clusters(self):
    """
    Re-runs clustering (app/ml/clustering.py) against the latest sales
    data and writes updated cluster assignments back to the stores table.
    """
    try:
        # TODO: pull recent sales from DB, call ml.clustering.fit_clusters,
        # persist cluster_id back onto the Store rows.
        print("Cluster refresh triggered (placeholder)")
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))
