from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.schemas.forecast import ForecastResponse
from app.services import forecast_service

router = APIRouter(prefix="/stores", tags=["forecasts"])


@router.get("/{store_id}/forecast", response_model=ForecastResponse)
async def get_forecast(
    store_id: int,
    horizon: int = settings.forecast_horizon,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return await forecast_service.get_forecast(store_id, horizon, db)
