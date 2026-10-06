from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer

from app.db.session import Base


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False, index=True)
    forecast_date = Column(DateTime, nullable=False)
    horizon_day = Column(Integer, nullable=False)  # 1..horizon
    predicted_sales = Column(Float, nullable=False)
    model_version = Column(Integer, default=1)
    created_at = Column(DateTime, nullable=False)
