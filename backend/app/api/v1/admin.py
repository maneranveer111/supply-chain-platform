from fastapi import APIRouter, Depends

from app.core.dependencies import Role, rate_limiter, require_role
from app.db.session import get_db

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/retrain")
async def trigger_retrain(
    user=Depends(require_role(Role.ADMIN)),
    _=Depends(rate_limiter(max_requests=1, window_seconds=3600)),
):
    from app.workers.retrain_tasks import retrain_model

    retrain_model.delay()
    return {"status": "retraining queued"}


@router.post("/refresh-clusters")
async def trigger_cluster_refresh(user=Depends(require_role(Role.ADMIN))):
    from app.workers.cluster_tasks import refresh_clusters

    refresh_clusters.delay()
    return {"status": "cluster refresh queued"}


from pydantic import BaseModel
class BenchmarkMappingRequest(BaseModel):
    benchmark_store_id: int | None

@router.patch("/stores/{store_id}/benchmark")
async def map_benchmark_store(
    store_id: int,
    payload: BenchmarkMappingRequest,
    user=Depends(require_role(Role.ADMIN)),
    db=Depends(get_db),
):
    from app.models.store import Store
    from app.ml.inference import load_rossmann_scaler, BenchmarkScalerNotFoundError
    from fastapi import HTTPException
    
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    if payload.benchmark_store_id is not None:
        try:
            # Validate that the ID actually exists in the pkl
            load_rossmann_scaler(payload.benchmark_store_id)
        except BenchmarkScalerNotFoundError:
            raise HTTPException(
                status_code=400,
                detail=f"Rossmann benchmark ID {payload.benchmark_store_id} not found in model artifacts."
            )

    store.benchmark_store_id = payload.benchmark_store_id
    db.commit()
    
    # Invalidate forecast cache
    from app.db.redis import get_redis
    from app.services.forecast_service import CACHE_KEY_VERSION
    from app.core.config import settings
    redis = await get_redis()
    cache_key = f"forecast:{CACHE_KEY_VERSION}:{store_id}:{settings.forecast_horizon}"
    await redis.delete(cache_key)

    return {"status": "success", "store_id": store_id, "benchmark_store_id": payload.benchmark_store_id}

