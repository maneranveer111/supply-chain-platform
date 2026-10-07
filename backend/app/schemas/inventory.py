from datetime import datetime
from pydantic import BaseModel, Field


class InventoryUpdate(BaseModel):
    current_quantity: float = Field(..., ge=0.0, description="New stock level, must be non-negative")


class InventoryResponse(BaseModel):
    id: int
    store_id: int
    current_quantity: float
    updated_at: datetime

    class Config:
        from_attributes = True


class InventorySignalResponse(BaseModel):
    store_id: int
    current_quantity: float
    daily_demand_mean: float
    days_of_supply: float | None
    reorder_point: float
    status: str  # "critical_stockout", "low_stock", "healthy", "overstocked", "insufficient_forecast"
    stockout_risk_score: float  # 0.0 (safe) to 1.0 (imminent stockout)
    recommended_reorder_qty: float
    forecast_method: str | None
    explanation: str | None = None
    updated_at: datetime
