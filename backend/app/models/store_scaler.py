"""
StoreScaler: per-application-store custom StandardScaler parameters.

This table stores the mean_log and scale_log fitted on a user-uploaded
historical sales CSV for a given application store.  It is completely
separate from saved_models/store_scalers.pkl, which holds the original
Rossmann benchmark scalers and is NOT stored here.

Design decisions:
- One row per application store (UNIQUE constraint on store_id).
- Populated only by the CSV-upload / feature-engineering pipeline
  (to be implemented in Phase 4).
- DO NOT pre-populate from store_scalers.pkl.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.session import Base


class StoreScaler(Base):
    __tablename__ = "store_scalers"
    __table_args__ = (
        UniqueConstraint("store_id", name="uq_store_scalers_store_id"),
    )

    id = Column(Integer, primary_key=True, index=True)

    # FK to application stores table (NOT to Rossmann store IDs)
    store_id = Column(
        Integer,
        ForeignKey("stores.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # StandardScaler parameters fitted on log1p(Sales) for this store
    mean_log = Column(Float, nullable=False)  # scaler.mean_[0]
    scale_log = Column(Float, nullable=False)  # scaler.scale_[0]

    # Number of training samples seen when scaler was fitted
    n_samples = Column(Integer, nullable=False)

    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # ORM relationship
    store = relationship("Store", back_populates="scaler")
