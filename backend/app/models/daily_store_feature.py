from sqlalchemy import JSON, Column, Date, Integer, UniqueConstraint

from app.db.session import Base


class DailyStoreFeature(Base):
    """
    One row per store per day, holding the engineered feature vector
    used as LSTM input (DayOfWeek_sin/cos, Promo, Sales_scaled,
    Sales_lag_7, rolling stats, etc. -- same set as feature_cols.pkl).

    `features` stores the engineered values as a JSON dict keyed by
    feature name, so the exact column set can evolve without a schema
    migration every time the notebook's feature list changes. The
    inference layer re-orders this dict using feature_cols.pkl to
    match what the model was trained on.
    """

    __tablename__ = "daily_store_features"
    __table_args__ = (
        UniqueConstraint("store_id", "date", name="uq_daily_store_features_store_id_date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    features = Column(JSON, nullable=False)
