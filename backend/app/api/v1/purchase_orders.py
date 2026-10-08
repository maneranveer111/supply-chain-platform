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
    user=Depends(require_role(Role.PROCUREMENT_MANAGER, Role.ADMIN)),
):
    if payload.status not in ("approved", "rejected"):
        raise HTTPException(400, "status must be 'approved' or 'rejected'")

    order = db.query(PurchaseOrder).filter(PurchaseOrder.id == po_id).first()
    if order is None:
        raise HTTPException(404, f"Purchase order {po_id} not found")

    store = db.query(Store).filter(Store.id == order.store_id).first()
    if not store:
        raise HTTPException(404, "Associated store not found")
    verify_store_access(store, user)

    order.status = payload.status
    db.commit()
    db.refresh(order)

    # Optional notification hook
    try:
        if store and store.user_id:
            from app.models.user import User
            from app.services.email_service import send_po_approval_notification
            owner = db.query(User).filter(User.id == store.user_id).first()
            if owner and owner.email:
                await send_po_approval_notification(
                    to_email=owner.email,
                    po_id=order.id,
                    store_id=order.store_id,
                    status=order.status,
                    recommended_qty=order.recommended_qty,
                )
    except Exception:
        pass

    return {"po_id": order.id, "status": order.status}


@router.post("/{po_id}/explain")
async def explain_order(
    po_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_role(Role.PROCUREMENT_MANAGER, Role.ADMIN, Role.STORE_ANALYST)),
):
    """
    Generate an explainable AI business reasoning breakdown for a purchase order.
    """
    from app.services.llm_service import explain_purchase_order

    order = db.query(PurchaseOrder).filter(PurchaseOrder.id == po_id).first()
    if order is None:
        raise HTTPException(404, f"Purchase order {po_id} not found")

    store = db.query(Store).filter(Store.id == order.store_id).first()
    if not store:
        raise HTTPException(404, "Associated store not found")
    verify_store_access(store, user)

    safety_stock = (
        round(max(0.0, order.forecasted_demand * 0.15), 2)
        if order.forecasted_demand > 0
        else 0.0
    )

    explanation = await explain_purchase_order(
        store_id=order.store_id,
        recommended_qty=order.recommended_qty,
        forecasted_demand=order.forecasted_demand,
        current_inventory=order.current_inventory,
        safety_stock=safety_stock,
    )

    return {
        "po_id": order.id,
        "store_id": order.store_id,
        "recommended_qty": order.recommended_qty,
        "forecasted_demand": order.forecasted_demand,
        "current_inventory": order.current_inventory,
        "safety_stock": safety_stock,
        "status": order.status,
        "explanation": explanation,
    }
