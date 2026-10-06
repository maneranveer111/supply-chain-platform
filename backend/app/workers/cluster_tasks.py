from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
import pandas as pd

@celery_app.task(bind=True, max_retries=3)
def refresh_clusters(self):
    """
    Re-runs clustering (app/ml/clustering.py) against the latest sales
    data and writes updated cluster assignments back to the stores table.
    """
    try:
        from app.models.store import Store
        from app.models.cluster import Cluster
        from app.ml.profile_builder import build_cluster_profiles
        from app.core.config import settings
        import redis
        
        db = SessionLocal()
        try:
            # In Phase 3, we skip the dynamic fitting and just build profiles 
            # from the existing store mappings + parquet data, since real fitting 
            # requires Phase 4 CSV ingestion.
            
            # Get current store assignments
            store_rows = db.query(Store.id, Store.cluster_id).filter(Store.cluster_id.isnot(None)).all()
            if not store_rows:
                return "No stores have cluster assignments."
                
            df_assignments = pd.DataFrame(store_rows, columns=["Store", "cluster_id"])
            
            # Build profiles
            profiles = build_cluster_profiles(df_assignments)
            
            # Persist to DB
            for cid, profile_dict in profiles.items():
                cluster_obj = db.query(Cluster).filter(Cluster.id == cid).first()
                if cluster_obj:
                    cluster_obj.profile = profile_dict
                    cluster_obj.store_count = profile_dict.get("n_stores", 0)
            
            db.commit()
            
            # Invalidate all forecast caches
            r = redis.from_url(settings.redis_url)
            keys = r.keys("forecast:v2:*")
            if keys:
                r.delete(*keys)
                
            return f"Refreshed {len(profiles)} cluster profiles and cleared cache."
        finally:
            db.close()
            
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))
