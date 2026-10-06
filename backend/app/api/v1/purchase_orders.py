from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import Role, require_role
from app.db.session import get_db
from app.models.purchase_order import PurchaseOrder
from app.schemas.purchase_order import PurchaseOrderApproval
from app.services import po_engine

router = APIRouter(prefix="/purchase-orders", tags=["purchase-orders"])


@router.get("/recommendations")
async def get_recommendations(
    store_id: int | None = None,
    db: Session = Depends(get_db),
    user=Depends(require_role(Role.PROCUREMENT_MANAGER, Role.ADMIN)),
):
    return await po_engine.get_recommendations(store_id, db)


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
