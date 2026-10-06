import logging
from io import BytesIO
import pandas as pd
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime

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
from app.ml.preprocessing import add_cyclical_day_of_week, add_lag_rolling_features
from app.ml.inference import load_feature_cols, COMP_SCALER_PATH
import pickle
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stores", tags=["stores"])

@router.post("", response_model=StoreResponse, status_code=status.HTTP_201_CREATED)
def create_store(
    store_in: StoreCreateRequest,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    store = Store(
        name=store_in.name,
        store_type=store_in.store_type,
        assortment=store_in.assortment,
        competition_distance=store_in.competition_distance,
        promo2_active=store_in.promo2_active,
        user_id=int(user.user_id),
        benchmark_store_id=None
    )
    db.add(store)
    db.commit()
    db.refresh(store)
    return store

@router.get("", response_model=list[StoreResponse])
def get_stores(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
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
    user: CurrentUser = Depends(get_current_user)
):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    verify_store_access(store, user)
    
    status_response = await get_forecast_status(store_id=store_id, db=db, user=user)
    
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
        "forecast_status": status_response.status,
        "forecast_method": status_response.forecast_method,
        "history_days": status_response.history_days,
        "has_custom_scaler": status_response.has_custom_scaler,
        "has_cluster": status_response.has_cluster,
    }

def process_features(df_sales: pd.DataFrame, store: Store, db: Session):
    # This generates the 22 feature columns
    # We expect df_sales to be sorted by date
    feature_cols = load_feature_cols()
    
    # Fit scaler
    mean_log = float(np.mean(np.log1p(df_sales['sales'])))
    scale_log = float(np.std(np.log1p(df_sales['sales'])))
    if scale_log == 0.0 or np.isnan(scale_log):
        scale_log = 1.0
    n_samples = len(df_sales)
    
    # Store scaler
    scaler_row = db.query(StoreScaler).filter(StoreScaler.store_id == store.id).first()
    if not scaler_row:
        scaler_row = StoreScaler(store_id=store.id)
        db.add(scaler_row)
    scaler_row.mean_log = mean_log
    scaler_row.scale_log = scale_log
    scaler_row.n_samples = n_samples
    scaler_row.updated_at = datetime.utcnow()
    db.flush()

    df = df_sales.copy()
    df.rename(columns={'sales': 'Sales', 'date': 'Date'}, inplace=True)
    df['Store'] = store.id
    df['Sales_scaled'] = (np.log1p(df['Sales']) - mean_log) / scale_log
    
    df['DayOfWeek'] = df['Date'].dt.weekday
    df = add_cyclical_day_of_week(df)
    
    df = add_lag_rolling_features(df, sales_col="Sales_scaled")
    
    df['Promo'] = df['promo']
    df['SchoolHoliday'] = df['school_holiday']
    df['IsPromo2Active'] = 1 if store.promo2_active else 0
    
    with open(COMP_SCALER_PATH, "rb") as f:
        comp_scaler = pickle.load(f)
    comp_dist = store.competition_distance if store.competition_distance is not None else 5000.0
    comp_scaled = comp_scaler.transform([[comp_dist]])[0][0]
    df['CompetitionDistance_scaled'] = comp_scaled
    
    for c in ['a', 'b', 'c', 'd']:
        df[f'StoreType_{c}'] = 1 if store.store_type == c else 0
    for c in ['a', 'b', 'c']:
        df[f'Assortment_{c}'] = 1 if store.assortment == c else 0
        
    for c in ['0', 'a', 'b', 'c']:
        df[f'StateHoliday_{c}'] = (df['state_holiday'].astype(str) == c).astype(int)

    # Delete existing
    db.query(DailyStoreFeature).filter(DailyStoreFeature.store_id == store.id).delete()
    
    features_to_insert = []
    records = df.to_dict('records')
    for row in records:
        feat_dict = {}
        for col in feature_cols:
            if col in row:
                feat_dict[col] = float(row[col])
            else:
                feat_dict[col] = 0.0
        
        # We need Date object for SQLAlchemy
        row_date = row['Date']
        if isinstance(row_date, pd.Timestamp):
            row_date = row_date.date()
            
        features_to_insert.append(DailyStoreFeature(
            store_id=store.id,
            date=row_date,
            features=feat_dict
        ))
        
    db.add_all(features_to_insert)

@router.post("/{store_id}/upload-sales")
async def upload_sales(
    store_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
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
        
    required_cols = {'date', 'sales', 'promo', 'school_holiday', 'state_holiday'}
    if not required_cols.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail=f"Missing required columns. Found: {list(df.columns)}")

    if df[list(required_cols)].isnull().any().any():
        raise HTTPException(status_code=400, detail="Missing or null values found in required columns.")
        
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format in CSV.")

    try:
        df['sales'] = pd.to_numeric(df['sales'])
    except Exception:
        raise HTTPException(status_code=400, detail="Sales column must contain numeric values.")

    if df['sales'].min() < 0:
        raise HTTPException(status_code=400, detail="Sales cannot be negative.")
        
    if not df['promo'].isin([0, 1]).all():
        raise HTTPException(status_code=400, detail="promo must be 0 or 1.")
        
    if not df['school_holiday'].isin([0, 1]).all():
        raise HTTPException(status_code=400, detail="school_holiday must be 0 or 1.")
        
    if not set(df['state_holiday'].astype(str).unique()).issubset({'0', 'a', 'b', 'c'}):
        raise HTTPException(status_code=400, detail="state_holiday must be one of 0, a, b, c.")
        
    if df['date'].duplicated().any():
        raise HTTPException(status_code=400, detail="Duplicate dates found in CSV.")
        
    df = df.sort_values('date')
    
    # Validate contiguous dates
    n_days = (df['date'].max() - df['date'].min()).days + 1
    if len(df) != n_days:
        raise HTTPException(status_code=400, detail="Dates are not contiguous. Missing days in history.")
        
    try:
        # Save to store_sales
        db.query(StoreSale).filter(StoreSale.store_id == store_id).delete()
        sales_records = []
        for _, row in df.iterrows():
            sales_records.append(StoreSale(
                store_id=store_id,
                date=row['date'].date(),
                sales=float(row['sales']),
                promo=int(row['promo']),
                school_holiday=int(row['school_holiday']),
                state_holiday=str(row['state_holiday'])
            ))
        db.add_all(sales_records)
        
        history_days = len(df)
        
        if history_days >= 60:
            process_features(df, store, db)
            
        elif history_days >= 14:
            from app.ml.clustering import build_store_features
            df_for_clustering = df.copy()
            df_for_clustering.rename(columns={'date': 'Date', 'sales': 'Sales', 'promo': 'Promo'}, inplace=True)
            df_for_clustering['Store'] = store.id
            agg = build_store_features(df_for_clustering)
            if not agg.empty:
                mean_sales = float(agg.iloc[0]['mean_sales'])
                promo_uplift = float(agg.iloc[0]['promo_uplift'])
                
                # Assign nearest cluster by mean_daily_sales & promo_uplift difference
                clusters = db.query(Cluster).filter(Cluster.profile.isnot(None)).all()
                best_cluster = None
                best_dist = float('inf')
                for cl in clusters:
                    prof = cl.profile
                    c_mean = prof.get('mean_daily_sales', 0.0)
                    c_uplift = prof.get('promo_uplift', 0.0)
                    # Simple heuristic distance since we lack scaler/kmeans
                    dist = ((mean_sales - c_mean) / (c_mean + 1))**2 + (promo_uplift - c_uplift)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = cl
                
                if best_cluster:
                    store.cluster_id = best_cluster.id
                else:
                    store.cluster_id = None
        else:
            store.cluster_id = None
            
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error during upload: {str(e)}")
        
    # Invalidate cache
    try:
        r = redis.from_url(settings.redis_url)
        keys = await r.keys(f"forecast:v2:{store_id}:*")
        if keys:
            await r.delete(*keys)
    except Exception as e:
        logger.warning(f"Failed to invalidate cache: {e}")

    return {"message": f"Successfully processed {len(df)} days of historical data."}
