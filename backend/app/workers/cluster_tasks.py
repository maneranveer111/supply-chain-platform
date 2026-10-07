import logging
import pandas as pd
from app.workers.celery_app import celery_app
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def refresh_clusters(self):
    """
    Re-runs clustering profile generation against the latest store
    assignments and writes updated cluster profiles to the clusters table.
    Invalidates all global forecast caches upon success.
    """
    logger.info(
        "Celery task [refresh_clusters] started (attempt %d/%d)",
        self.request.retries + 1,
        self.max_retries + 1,
    )
    db = SessionLocal()
    try:
        from app.models.store import Store
        from app.models.cluster import Cluster
        from app.ml.profile_builder import build_cluster_profiles
        from app.services.forecast_service import invalidate_forecast_cache_sync

        store_rows = (
            db.query(Store.id, Store.cluster_id)
            .filter(Store.cluster_id.isnot(None))
            .all()
        )
        if not store_rows:
            logger.info("No stores have cluster assignments. Nothing to refresh.")
            return "No stores have cluster assignments."

        df_assignments = pd.DataFrame(store_rows, columns=["Store", "cluster_id"])
        profiles = build_cluster_profiles(df_assignments)

        for cid, profile_dict in profiles.items():
            cluster_obj = db.query(Cluster).filter(Cluster.id == cid).first()
            if cluster_obj:
                cluster_obj.profile = profile_dict
                cluster_obj.store_count = profile_dict.get("n_stores", 0)

        db.commit()
        logger.info("Persisted %d cluster profiles to database.", len(profiles))

        # Invalidate all forecast caches globally
        deleted_count = invalidate_forecast_cache_sync(None)
        logger.info(
            "Task [refresh_clusters] finished: updated %d profiles, cleared %d cache keys.",
            len(profiles),
            deleted_count,
        )
        return f"Refreshed {len(profiles)} cluster profiles and cleared cache."

    except Exception as exc:
        db.rollback()
        logger.error(
            "Task [refresh_clusters] failed: %s", exc, exc_info=True
        )
        if self.request.retries < self.max_retries:
            countdown = 60 * (2 ** self.request.retries)
            logger.info("Retrying refresh_clusters in %d seconds...", countdown)
            raise self.retry(exc=exc, countdown=countdown)
        raise exc
    finally:
        db.close()
