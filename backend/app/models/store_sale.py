"""
StoreSale: raw daily sales records for a user-owned application store.

This table stores the raw transactional input that a user uploads via a
CSV file (to be implemented in Phase 4).  It is the source of truth for
the feature-engineering pipeline that produces daily_store_features rows
and ultimately a fitted StoreScaler.

Design decisions:
- One row per (store_id, date) — enforced by a UNIQUE constraint.
- store_id is a FK to the application stores table.
- Columns mirror the minimum required Rossmann field set so that the
  existing preprocessing.py logic can be applied without modification.
"""

from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.session import Base


class StoreSale(Base):
    __tablename__ = "store_sales"
    __table_args__ = (
        # Prevent duplicate daily records for the same store
        UniqueConstraint("store_id", "date", name="uq_store_sales_store_date"),
    )

    id = Column(Integer, primary_key=True, index=True)

    # FK to application stores table
    store_id = Column(
        Integer,
        ForeignKey("stores.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Calendar date of the sales record
    date = Column(Date, nullable=False, index=True)

    # Raw (non-log-scaled) daily sales amount
    sales = Column(Float, nullable=False)

    # Binary promotional flag (0 or 1)
    promo = Column(Integer, nullable=False, default=0)

    # Binary school-holiday flag (0 or 1)
    school_holiday = Column(Integer, nullable=False, default=0)

    # State-holiday flag: '0' = no holiday, 'a' = public, 'b' = Easter, 'c' = Christmas
    state_holiday = Column(String(1), nullable=False, default="0")

    # ORM relationship
    store = relationship("Store", back_populates="sales")
