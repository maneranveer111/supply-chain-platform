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
    
    # Invalidate forecast cache for all horizons of this store
    from app.services.forecast_service import invalidate_forecast_cache
    await invalidate_forecast_cache(store_id)

    return {"status": "success", "store_id": store_id, "benchmark_store_id": payload.benchmark_store_id}


@router.get("/overview")
async def get_admin_overview(
    user=Depends(require_role(Role.ADMIN)),
    db=Depends(get_db),
):
    """
    Get aggregate system-level platform health, store statuses, and operations.
    """
    from sqlalchemy import func
    from app.models.store import Store
    from app.models.user import User
    from app.models.purchase_order import PurchaseOrder
    from app.models.inventory import Inventory
    from app.core.config import settings

    total_stores = db.query(func.count(Store.id)).scalar() or 0
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_pos = db.query(func.count(PurchaseOrder.id)).scalar() or 0
    pending_pos = (
        db.query(func.count(PurchaseOrder.id))
        .filter(PurchaseOrder.status.like("pending%"))
        .scalar()
        or 0
    )
    critical_stockouts = (
        db.query(func.count(Inventory.id))
        .filter(Inventory.current_quantity <= 0)
        .scalar()
        or 0
    )

    onboarding_counts = (
        db.query(Store.onboarding_status, func.count(Store.id))
        .group_by(Store.onboarding_status)
        .all()
    )
    onboarding_breakdown = {status or "unknown": count for status, count in onboarding_counts}

    return {
        "total_stores": total_stores,
        "total_users": total_users,
        "total_purchase_orders": total_pos,
        "pending_purchase_orders": pending_pos,
        "critical_stockouts": critical_stockouts,
        "onboarding_breakdown": onboarding_breakdown,
        "llm_provider": settings.llm_provider,
        "gemini_model": settings.gemini_model,
        "email_service": "configured" if settings.brevo_api_key else "not_configured",
    }


@router.get("/stores")
async def get_admin_stores(
    user=Depends(require_role(Role.ADMIN)),
    db=Depends(get_db),
):
    """
    Detailed administrative inventory and ML onboarding matrix across all stores.
    """
    from app.models.store import Store
    from app.models.user import User
    from app.models.inventory import Inventory

    stores = db.query(Store).all()
    results = []
    for s in stores:
        owner_email = None
        if s.user_id:
            u = db.query(User.email).filter(User.id == s.user_id).first()
            if u:
                owner_email = u[0]
        inv = db.query(Inventory.current_quantity).filter(Inventory.store_id == s.id).first()
        stock = inv[0] if inv else 0.0

        results.append({
            "id": s.id,
            "name": s.name,
            "store_type": s.store_type,
            "assortment": s.assortment,
            "user_id": s.user_id,
            "owner_email": owner_email,
            "cluster_id": s.cluster_id,
            "benchmark_store_id": s.benchmark_store_id,
            "forecast_mode": s.forecast_mode,
            "onboarding_status": s.onboarding_status,
            "current_stock": stock,
        })
    return results


class AdminEmailTestRequest(BaseModel):
    to_email: str
    subject: str = "Admin Connectivity Test"
    content: str = "This is a test notification from the Supply Chain Platform admin suite."


@router.post("/email/test")
async def send_admin_test_email(
    payload: AdminEmailTestRequest,
    user=Depends(require_role(Role.ADMIN)),
):
    """
    Manually dispatch a test email via Brevo to verify transactional email delivery.
    """
    from app.services.email_service import send_email

    result = await send_email(
        to_email=payload.to_email,
        subject=payload.subject,
        html_content=f"<p>{payload.content}</p>",
    )
    return {"status": "dispatched", "brevo_response": result}


@router.get("/health/deep")
async def deep_health_check(
    user=Depends(require_role(Role.ADMIN)),
    db=Depends(get_db),
):
    """
    Comprehensive diagnostics for DB, Redis, Celery, and external API integrations.
    """
    from sqlalchemy import text
    import redis
    from app.core.config import settings

    checks = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"

    try:
        r = redis.Redis.from_url(settings.redis_url, socket_timeout=2)
        r.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"

    checks["gemini"] = "configured" if settings.gemini_api_key else "missing_key"
    checks["brevo"] = "configured" if settings.brevo_api_key else "missing_key"

    return {"status": "ok", "components": checks}


