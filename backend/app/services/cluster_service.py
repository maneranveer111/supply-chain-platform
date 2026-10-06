from sqlalchemy.orm import Session

from app.models.cluster import Cluster
from app.models.store import Store


async def get_all_clusters(db: Session) -> list[dict]:
    clusters = db.query(Cluster).all()
    return [
        {
            "cluster_id": c.id,
            "label": c.label,
            "description": c.description,
            "store_count": c.store_count,
        }
        for c in clusters
    ]


async def get_store_cluster(store_id: int, db: Session) -> dict | None:
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store or store.cluster_id is None:
        return None

    cluster = db.query(Cluster).filter(Cluster.id == store.cluster_id).first()
    peers = (
        db.query(Store.id)
        .filter(Store.cluster_id == store.cluster_id, Store.id != store_id)
        .limit(10)
        .all()
    )
    return {
        "store_id": store_id,
        "cluster_id": store.cluster_id,
        "cluster_label": cluster.label if cluster else "unknown",
        "peer_store_ids": [p.id for p in peers],
    }
