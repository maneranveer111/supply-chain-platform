from sqlalchemy import Column, Integer, JSON, String

from app.db.session import Base


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False)  # e.g. "high-volume stable"
    description = Column(String, nullable=True)
    store_count = Column(Integer, default=0)

    # Phase 3: persisted cluster profile for cold-start forecasting.
    # JSON structure:
    # {
    #   "mean_daily_sales": float,       — average real daily sales for this cluster
    #   "dow_multipliers": [float x 7],  — day-of-week multipliers (Mon=0 .. Sun=6)
    #   "promo_uplift": float,           — fractional promo uplift (e.g. 0.35 = +35%)
    #   "n_stores": int,                 — number of training stores in cluster
    #   "n_samples": int                 — total rows used to build the profile
    # }
    # Populated by the cluster refresh task, NOT invented.
    profile = Column(JSON, nullable=True)
