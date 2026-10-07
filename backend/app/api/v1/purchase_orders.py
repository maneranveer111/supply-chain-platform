from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import Role, require_role
from app.core.authorization import verify_store_access
from app.db.session import get_db
from app.models.purchase_order import PurchaseOrder
from app.models.store import Store
from app.schemas.purchase_order import PurchaseOrderApproval
from app.services import po_engine

router = APIRouter(prefix="/purchase-orders", tags=["purchase-orders"])


@router.get("/recommendations")
async def get_recommendations(
    store_id: int | None = None,
    db: Session = Depends(get_db),
    user=Depends(require_role(Role.PROCUREMENT_MANAGER, Role.ADMIN)),
):
    """
    Generate purchase order recommendations based on forecasted demand.
    - If store_id is specified: verifies store access for current user.
    - If store_id is omitted:
        - Admin receives recommendations for all active stores.
        - Non-admin procurement managers receive recommendations only for
          stores they have permission to access.
    """
    if store_id is not None:
        store = db.query(Store).filter(Store.id == store_id).first()
        if not store:
            raise HTTPException(404, "Store not found")
        verify_store_access(store, user)
        return await po_engine.get_recommendations(store_id, db)

    # store_id is None: respect multi-tenant boundaries
    if user.role == Role.ADMIN.value:
        return await po_engine.get_recommendations(None, db)

    # Non-admin user: list only owned stores
    user_stores = (
        db.query(Store.id).filter(Store.user_id == int(user.user_id)).all()
    )
    user_store_ids = [s[0] for s in user_stores]
    return await po_engine.get_recommendations_for_stores(user_store_ids, db)


@router.post("/{po_id}/approve")
async def approve_order(
    po_id: int,
    payload: PurchaseOrderApproval,
    db: Session = Depends(get_db),
    user=Depends(require_role(Role.PROCUREMENT_MANAGER)),
):
    if payload.status not in ("approved", "rejected"):
        raise HTTPException(400, "status must be 'approved' or 'rejected'")

    order = db.query(PurchaseOrder).filter(PurchaseOrder.id == po_id).first()
    if order is None:
        raise HTTPException(404, f"Purchase order {po_id} not found")

    order.status = payload.status
    db.commit()
    db.refresh(order)

    return {"po_id": order.id, "status": order.status}
