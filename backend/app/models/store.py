from sqlalchemy import Column, Float, Integer, String

from app.db.session import Base


class Store(Base):
    __tablename__ = "stores"

    id = Column(Integer, primary_key=True, index=True)
    store_type = Column(String, nullable=True)
    assortment = Column(String, nullable=True)
    competition_distance = Column(Float, nullable=True)
    cluster_id = Column(Integer, nullable=True)
