from pydantic import BaseModel, Field
from typing import Literal

class StoreCreateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    store_type: Literal["a", "b", "c", "d"]
    assortment: Literal["a", "b", "c"]
    competition_distance: float | None = Field(default=None, ge=0)
    promo2_active: bool = False

class StoreResponse(BaseModel):
    id: int
    name: str | None
    store_type: str | None
    assortment: str | None
    competition_distance: float | None
    promo2_active: bool
    benchmark_store_id: int | None
    user_id: int | None
    cluster_id: int | None
    forecast_mode: str

    class Config:
        orm_mode = True
        from_attributes = True
