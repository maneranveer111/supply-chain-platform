from typing import Literal

from pydantic import BaseModel


class ForecastDay(BaseModel):
    day: int
    date: str
    predicted_sales: float


class ForecastResponse(BaseModel):
    store_id: int
    horizon: int
    window_used: int
    forecast: list[ForecastDay]

    # Phase 2: routing provenance fields
    # forecast_method values:
    #   "lstm"               — LSTM ran with a valid scaler and sufficient history
    #   "insufficient_data"  — no usable path available; forecast list is empty
    forecast_method: str

    forecast_confidence: str

    generated_at: str


class ForecastStatusResponse(BaseModel):
    store_id: int
    forecast_method: str
    forecast_confidence: str
    benchmark_store_id: int | None
    has_custom_scaler: bool
    history_days: int
    has_cluster: bool
    status: str
    onboarding_status: str | None = None
