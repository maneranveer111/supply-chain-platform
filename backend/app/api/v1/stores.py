import logging
from io import BytesIO
from datetime import datetime
import pandas as pd
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import get_db
from app.core.dependencies import get_current_user, CurrentUser, Role
from app.core.authorization import verify_store_access
from app.models.store import Store
from app.models.store_sale import StoreSale
from app.models.store_scaler import StoreScaler
from app.models.daily_store_feature import DailyStoreFeature
from app.models.cluster import Cluster
from app.schemas.store import StoreCreateRequest, StoreResponse
from app.api.v1.forecasts import get_forecast_status
from app.services.forecast_service import invalidate_forecast_cache
from app.services.onboarding_service import (
    process_features,
    process_store_features_and_clustering,
)
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stores", tags=["stores"])


@router.post("", response_model=StoreResponse, status_code=status.HTTP_201_CREATED)
def create_store(
    store_in: StoreCreateRequest,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    logger.info(
        "Creating store name='%s' for user_id=%s role=%s",
        store_in.name,
        user.user_id,
        user.role,
    )
    store = Store(
        name=store_in.name,
        store_type=store_in.store_type,
        assortment=store_in.assortment,
        competition_distance=store_in.competition_distance,
        promo2_active=store_in.promo2_active,
        user_id=int(user.user_id),
        benchmark_store_id=None,
        onboarding_status="ready",
    )
    db.add(store)
    db.commit()
    db.refresh(store)
    logger.info("Store created successfully with id=%d", store.id)
    return store


@router.get("", response_model=list[StoreResponse])
def get_stores(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    if user.role == Role.ADMIN.value:
        stores = db.query(Store).all()
    else:
        stores = db.query(Store).filter(Store.user_id == int(user.user_id)).all()
    return stores


@router.get("/{store_id}")
async def get_store(
    store_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    verify_store_access(store, user)

    status_response = await get_forecast_status(store_id=store_id, db=db, user=user)

    # For backward compatibility with Phase 4 tests, "metadata_cohort_ready" maps to "cold_start"
    forecast_status_val = (
        "cold_start"
        if status_response.status == "metadata_cohort_ready"
        else status_response.status
    )

    return {
        "id": store.id,
        "name": store.name,
        "store_type": store.store_type,
        "assortment": store.assortment,
        "competition_distance": store.competition_distance,
        "promo2_active": store.promo2_active,
        "benchmark_store_id": store.benchmark_store_id,
        "user_id": store.user_id,
        "cluster_id": store.cluster_id,
        "forecast_mode": store.forecast_mode,
        "forecast_status": forecast_status_val,
        "readiness_status": status_response.status,
        "onboarding_status": status_response.onboarding_status,
        "forecast_method": status_response.forecast_method,
        "history_days": status_response.history_days,
        "has_custom_scaler": status_response.has_custom_scaler,
        "has_cluster": status_response.has_cluster,
    }


@router.post("/{store_id}/upload-sales")
async def upload_sales(
    store_id: int,
    file: UploadFile = File(...),
    background: bool = Query(
        False, description="Whether to dispatch processing to a Celery background worker"
    ),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """
    Upload historical sales CSV for store onboarding.
    Validates CSV format, stores sales records, and triggers feature engineering.
    """
    logger.info("Upload sales requested for store_id=%d by user_id=%s", store_id, user.user_id)
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Limit is 10MB.")

    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    verify_store_access(store, user)

    try:
        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    if len(df) > 5000:
        raise HTTPException(status_code=400, detail="CSV exceeds maximum allowed rows (5,000).")

    required_cols = {"date", "sales", "promo", "school_holiday", "state_holiday"}
    if not required_cols.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail=f"Missing required columns. Found: {list(df.columns)}")

    if df[list(required_cols)].isnull().any().any():
        raise HTTPException(status_code=400, detail="Missing or null values found in required columns.")

    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format in CSV.")

    try:
        df["sales"] = pd.to_numeric(df["sales"])
    except Exception:
        raise HTTPException(status_code=400, detail="Sales column must contain numeric values.")

    if df["sales"].min() < 0:
        raise HTTPException(status_code=400, detail="Sales cannot be negative.")

    if not df["promo"].isin([0, 1]).all():
        raise HTTPException(status_code=400, detail="promo must be 0 or 1.")

    if not df["school_holiday"].isin([0, 1]).all():
        raise HTTPException(status_code=400, detail="school_holiday must be 0 or 1.")

    if not set(df["state_holiday"].astype(str).unique()).issubset({"0", "a", "b", "c"}):
        raise HTTPException(status_code=400, detail="state_holiday must be one of 0, a, b, c.")

    if df["date"].duplicated().any():
        raise HTTPException(status_code=400, detail="Duplicate dates found in CSV.")

    df = df.sort_values("date")

    # Validate contiguous dates
    n_days = (df["date"].max() - df["date"].min()).days + 1
    if len(df) != n_days:
        raise HTTPException(status_code=400, detail="Dates are not contiguous. Missing days in history.")

    try:
        # Idempotently save to store_sales (replace previous sales for this store)
        db.query(StoreSale).filter(StoreSale.store_id == store_id).delete()
        sales_records = [
            StoreSale(
                store_id=store_id,
                date=row["date"].date(),
                sales=float(row["sales"]),
                promo=int(row["promo"]),
                school_holiday=int(row["school_holiday"]),
                state_holiday=str(row["state_holiday"]),
            )
            for _, row in df.iterrows()
        ]
        db.add_all(sales_records)
        store.onboarding_status = "pending"
        db.commit()
        logger.info("Saved %d raw sales rows for store_id=%d", len(sales_records), store_id)

    except Exception as e:
        db.rollback()
        logger.error("Database error saving sales for store_id=%d: %s", store_id, e)
        raise HTTPException(status_code=500, detail=f"Database error during upload: {str(e)}")

    # Process features: either synchronous or via Celery background worker
    if background:
        try:
            from app.workers.onboarding_tasks import process_store_onboarding

            task = process_store_onboarding.delay(store_id)
            logger.info("Dispatched background onboarding task %s for store_id=%d", task.id, store_id)
            return {
                "message": f"Successfully uploaded {len(df)} days. Background processing initiated.",
                "task_id": str(task.id),
                "status": "processing",
            }
        except Exception as e:
            logger.warning("Could not dispatch Celery task (%s), running synchronously.", e)
            process_store_features_and_clustering(store_id, db)
    else:
        process_store_features_and_clustering(store_id, db)

    # Invalidate forecast cache for all horizons of this store
    await invalidate_forecast_cache(store_id)

    return {"message": f"Successfully processed {len(df)} days of historical data."}


@router.post("/{store_id}/process")
async def trigger_store_processing(
    store_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """
    Manually trigger background onboarding processing for a store.
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    verify_store_access(store, user)

    try:
        from app.workers.onboarding_tasks import process_store_onboarding

        task = process_store_onboarding.delay(store_id)
        return {
            "message": "Background processing task triggered.",
            "task_id": str(task.id),
            "store_id": store_id,
        }
    except Exception as e:
        logger.warning("Celery dispatch failed (%s); processing synchronously.", e)
        result = process_store_features_and_clustering(store_id, db)
        await invalidate_forecast_cache(store_id)
        return {
            "message": "Synchronous processing completed.",
            "store_id": store_id,
            "result": result,
        }
