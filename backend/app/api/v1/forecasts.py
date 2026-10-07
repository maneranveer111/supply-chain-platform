import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.core.authorization import verify_store_access
from app.db.session import get_db
from app.ml.forecast_router import StoreNotFoundError
from app.ml.inference import ModelNotLoadedError
from app.models.store import Store
from app.models.store_sale import StoreSale
from app.models.daily_store_feature import DailyStoreFeature
from app.models.store_scaler import StoreScaler
from app.schemas.forecast import ForecastResponse, ForecastStatusResponse
from app.services import forecast_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stores", tags=["forecasts"])


@router.get("/{store_id}/forecast", response_model=ForecastResponse)
async def get_forecast(
    store_id: int,
    horizon: int = Query(
        settings.forecast_horizon,
        ge=1,
        le=90,
        description="Forecast horizon in days (1 to 90)",
    ),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Generate a sales forecast for a given application store.

    Phase 2-5 routing outcomes:
      - LSTM with Rossmann benchmark scaler: forecast_method="lstm", confidence="high"
      - LSTM with custom DB scaler:          forecast_method="lstm", confidence="medium"
      - Cluster forecast:                    forecast_method="cluster_average", confidence="low"
      - Metadata cohort:                     forecast_method="metadata_cohort", confidence="low"
      - Insufficient data / no scaler:       forecast_method="insufficient_data",
                                             confidence="none", forecast=[]

    HTTP status codes:
      200  — forecast produced (including insufficient_data responses, which are
             valid business outcomes, not errors).
      401  — authentication failure.
      403  — user does not have permission for this store.
      404  — application store_id does not exist in the database.
      422  — invalid request parameters (e.g. invalid horizon).
      503  — ML model artifacts are not loaded (deployment issue).
    """
    try:
        store = db.query(Store).filter(Store.id == store_id).first()
        if not store:
            raise StoreNotFoundError()
        verify_store_access(store, user)

        return await forecast_service.get_forecast(store_id, horizon, db)
    except StoreNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Store with id={store_id} not found.",
        )
    except ModelNotLoadedError as exc:
        logger.error("ML model not loaded for store_id=%d: %s", store_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The ML model is not currently loaded. "
                "Ensure lstm_model.keras is present in saved_models/."
            ),
        )


@router.get("/{store_id}/forecast/status", response_model=ForecastStatusResponse)
@router.get("/{store_id}/status", response_model=ForecastStatusResponse)
async def get_forecast_status(
    store_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Returns the routing and onboarding readiness status of a store without
    running ML inference.
    Accurately distinguishes:
      - no history
      - metadata cohort ready
      - cluster ready
      - custom scaler ready
      - LSTM ready
      - benchmark LSTM ready
      - processing
      - failed
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Store with id={store_id} not found.",
        )

    verify_store_access(store, user)

    # 1. Check distinct history days (never count duplicate dates)
    sales_days = (
        db.query(func.count(func.distinct(StoreSale.date)))
        .filter(StoreSale.store_id == store_id)
        .scalar()
        or 0
    )
    feature_days = (
        db.query(func.count(func.distinct(DailyStoreFeature.date)))
        .filter(DailyStoreFeature.store_id == store_id)
        .scalar()
        or 0
    )
    history_days = max(sales_days, feature_days)

    # 2. Check custom scaler
    has_custom_scaler = (
        db.query(StoreScaler).filter(StoreScaler.store_id == store_id).first() is not None
    )

    # 3. Determine status & method
    if store.onboarding_status == "processing":
        status_val = "processing"
        method = "pending"
        confidence = "none"
    elif store.onboarding_status == "failed":
        status_val = "failed"
        method = "failed"
        confidence = "none"
    elif store.benchmark_store_id is not None:
        if feature_days >= 30:
            status_val = "benchmark_lstm_ready"
            method = "lstm"
            confidence = "high"
        elif history_days == 0:
            status_val = "no_history"
            method = "insufficient_data"
            confidence = "none"
        else:
            status_val = "insufficient_data"
            method = "insufficient_data"
            confidence = "none"
    elif has_custom_scaler:
        if feature_days >= 30:
            status_val = "lstm_ready"
            method = "lstm"
            confidence = "medium"
        else:
            status_val = "custom_scaler_ready"
            method = "insufficient_data"
            confidence = "none"
    elif store.cluster_id is not None:
        status_val = "cluster_ready"
        method = "cluster_average"
        confidence = "low"
    else:
        from app.ml.profile_builder import resolve_store_cohort_key
        from app.ml.cluster_forecast import get_metadata_cohort_profile

        cohort_key = resolve_store_cohort_key(store)
        if cohort_key and get_metadata_cohort_profile(cohort_key):
            status_val = "metadata_cohort_ready"
            method = "metadata_cohort"
            confidence = "low"
        else:
            status_val = "no_history" if history_days == 0 else "insufficient_data"
            method = "insufficient_data"
            confidence = "none"

    return ForecastStatusResponse(
        store_id=store.id,
        forecast_method=method,
        forecast_confidence=confidence,
        benchmark_store_id=store.benchmark_store_id,
        has_custom_scaler=has_custom_scaler,
        history_days=history_days,
        has_cluster=store.cluster_id is not None,
        status=status_val,
        onboarding_status=store.onboarding_status or "ready",
    )
