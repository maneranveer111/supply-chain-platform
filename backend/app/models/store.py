from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.session import Base


class Store(Base):
    __tablename__ = "stores"

    id = Column(Integer, primary_key=True, index=True)

    # --- Multi-tenant ownership ---
    # NULL means this is a demo / system store with no explicit owner.
    # A normal user-created store must have user_id set.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Human-readable name (optional for legacy/demo stores)
    name = Column(String, nullable=True)

    # --- Rossmann benchmark mapping ---
    # The ONLY authoritative link to a Rossmann training store.
    # NULL  →  this application store is NOT mapped to any Rossmann training store.
    # Non-NULL →  benchmark_store_id is a key into store_scalers.pkl (1..1115).
    # This field must NEVER be set automatically from Store.id.
    # It is an admin-controlled assignment only.
    benchmark_store_id = Column(Integer, nullable=True, index=True)

    # --- Existing store metadata (unchanged) ---
    store_type = Column(String, nullable=True)
    assortment = Column(String, nullable=True)
    competition_distance = Column(Float, nullable=True)
    promo2_active = Column(Boolean, nullable=False, server_default="false")
    cluster_id = Column(Integer, ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True, index=True)

    # --- Forecast routing hint (Phase 2 will implement the router) ---
    # "auto"              →  system decides at forecast time (default)
    # "lstm"              →  force LSTM path (requires benchmark_store_id or custom scaler)
    # "cluster_average"   →  force cluster-average cold-start path
    forecast_mode = Column(String, nullable=False, server_default="auto")

    # --- Phase 5: Store onboarding / background processing status ---
    # "pending", "processing", "ready", "failed"
    onboarding_status = Column(String, nullable=True, server_default="ready")

    # --- SQLAlchemy relationships ---
    owner = relationship("User", back_populates="stores")

    # One-to-one: custom per-store scaler (Phase 4 — populated by CSV pipeline)
    scaler = relationship("StoreScaler", back_populates="store", uselist=False, passive_deletes=True)

    # One-to-many: raw daily sales rows (Phase 4 — populated by CSV upload)
    sales = relationship("StoreSale", back_populates="store", passive_deletes=True)

