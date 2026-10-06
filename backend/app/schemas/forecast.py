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
    generated_at: str
