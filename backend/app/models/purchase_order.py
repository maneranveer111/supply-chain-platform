from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from app.db.session import Base


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False, index=True)
    recommended_qty = Column(Float, nullable=False)
    current_inventory = Column(Float, nullable=False)
    forecasted_demand = Column(Float, nullable=False)
    status = Column(String, default="pending", index=True)  # pending, approved, rejected

    created_at = Column(DateTime, nullable=False)
