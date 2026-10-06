from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.services import cluster_service

router = APIRouter(prefix="/stores", tags=["clusters"])


@router.get("/clusters")
async def list_clusters(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return await cluster_service.get_all_clusters(db)


@router.get("/{store_id}/cluster")
async def get_store_cluster(
    store_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)
):
    result = await cluster_service.get_store_cluster(store_id, db)
    if result is None:
        raise HTTPException(404, "Store not found or not yet clustered")
    return result
