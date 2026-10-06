from fastapi import APIRouter, Depends

from app.core.dependencies import Role, rate_limiter, require_role

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
