import logging

from fastapi import APIRouter, Depends, HTTPException, status
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
    horizon: int = settings.forecast_horizon,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Generate a sales forecast for a given application store.

    Phase 2 routing outcomes:
      - LSTM with Rossmann benchmark scaler: forecast_method="lstm", confidence="high"
      - LSTM with custom DB scaler:          forecast_method="lstm", confidence="high"
      - Insufficient data / no scaler:       forecast_method="insufficient_data",
                                             confidence="none", forecast=[]

    HTTP status codes:
      200  — forecast produced (including insufficient_data responses, which are
             valid business outcomes, not errors).
      404  — application store_id does not exist in the database.
      503  — ML model artifacts are not loaded (deployment issue).
    """
    try:
        # Phase 3: Check authorization before routing
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
        logger.error("ML model not loaded: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The ML model is not currently loaded. "
                "Ensure lstm_model.keras is present in saved_models/."
            ),
        )


@router.get("/{store_id}/forecast/status", response_model=ForecastStatusResponse)
async def get_forecast_status(
    store_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Returns the routing status of a store without actually running LSTM inference.
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Store with id={store_id} not found.",
        )
        
    verify_store_access(store, user)

    # 1. Check history days
    sales_days = db.query(StoreSale).filter(StoreSale.store_id == store_id).count()
    feature_days = db.query(DailyStoreFeature).filter(DailyStoreFeature.store_id == store_id).count()
    history_days = max(sales_days, feature_days)

    # 2. Check custom scaler
    has_custom_scaler = db.query(StoreScaler).filter(StoreScaler.store_id == store_id).first() is not None

    # Determine status & method
    if store.benchmark_store_id is not None:
        if feature_days >= 30:
            status_val = "lstm_ready"
            method = "lstm"
            confidence = "high"
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
            status_val = "insufficient_data"
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
            status_val = "cold_start"
            method = "metadata_cohort"
            confidence = "low"
        else:
            status_val = "insufficient_data"
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
    )
