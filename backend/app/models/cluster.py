from sqlalchemy import Column, Integer, String

from app.db.session import Base


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False)  # e.g. "high-volume stable"
    description = Column(String, nullable=True)
    store_count = Column(Integer, default=0)
