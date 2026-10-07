from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer

from app.db.session import Base


class Inventory(Base):
    """
    Current stock level per store. In a real deployment this would be
    synced from a warehouse/ERP system; here it's a simple table the
    PO engine reads from, seeded manually or via scripts/seed_data.py.
    """

    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, index=True, unique=True, nullable=False)
    current_quantity = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
