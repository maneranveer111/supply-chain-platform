"""
Inventory API endpoints.

Provides inventory telemetry, manual adjustments, stockout signals,
and automated low-stock alerting.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, CurrentUser, Role, require_role
from app.core.authorization import verify_store_access
from app.db.session import get_db
from app.models.store import Store
from app.models.user import User
from app.schemas.inventory import (
    InventoryResponse,
    InventoryUpdate,
    InventorySignalResponse,
)
from app.services import inventory_service
from app.services.email_service import send_stock_alert_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inventory", tags=["inventory"])


@router.get("/{store_id}", response_model=InventoryResponse)
def get_inventory(
    store_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """
    Fetch current inventory for a specific store.
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    verify_store_access(store, user)

    return inventory_service.get_or_create_inventory(store_id, db)


@router.put("/{store_id}", response_model=InventoryResponse)
def update_inventory(
    store_id: int,
    payload: InventoryUpdate,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """
    Update stock on hand for a store.
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    verify_store_access(store, user)

    return inventory_service.update_inventory_level(store_id, payload.current_quantity, db)


@router.get("/{store_id}/signals", response_model=InventorySignalResponse)
async def get_inventory_signals(
    store_id: int,
    include_explanation: bool = Query(False, description="Generate natural-language LLM explanation"),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """
    Get real-time stockout risk, days of supply runway, and reorder recommendation.
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    verify_store_access(store, user)

    signals = await inventory_service.analyze_inventory_signals(
        store_id=store_id,
        db=db,
        include_llm_explanation=include_explanation,
    )
    return signals


@router.post("/{store_id}/alert-test")
async def trigger_stock_alert(
    store_id: int,
    recipient_email: str | None = Query(None, description="Optional override recipient"),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.ADMIN, Role.PROCUREMENT_MANAGER)),
):
    """
    Send a stock alert email notification for a store based on its current signals.
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    verify_store_access(store, user)

    signals = await inventory_service.analyze_inventory_signals(store_id=store_id, db=db)

    # Determine target recipient
    target_email = recipient_email
    if not target_email and store.user_id:
        owner = db.query(User).filter(User.id == store.user_id).first()
        if owner and owner.email:
            target_email = owner.email
    if not target_email:
        target_email = user.email or "manager@example.com"

    result = await send_stock_alert_email(
        to_email=target_email,
        store_id=store_id,
        store_name=store.name or f"Store #{store_id}",
        current_stock=signals["current_quantity"],
        reorder_point=signals["reorder_point"],
        days_of_supply=signals["days_of_supply"] or 0.0,
    )
    return {"status": "alert_dispatched", "email_result": result, "target_email": target_email}
