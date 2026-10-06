"""
Purchase Order Recommendation Engine.

Phase 2 changes:
  - get_forecast() now returns dicts that may have forecast_method="insufficient_data"
    and an empty forecast list.
  - The PO engine must detect this state and NOT compute a recommendation
    using an empty forecast list (which would crash with a ZeroDivisionError
    or produce a misleading recommended_qty=0 with no reason).
  - When a store returns insufficient_data, a PO record is still written to
    the database with status="pending_forecast" so the store is not silently
    skipped.
  - The PurchaseOrderRecommendation is still returned but with explicit
    fields indicating the forecast is unavailable.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.models.inventory import Inventory
from app.models.purchase_order import PurchaseOrder
from app.models.store import Store
from app.schemas.purchase_order import PurchaseOrderRecommendation
from app.services.forecast_service import get_forecast

SAFETY_STOCK_FACTOR = 0.15  # 15% buffer over forecasted demand
LEAD_TIME_DAYS = 3
DEFAULT_INVENTORY_FALLBACK = 2000.0  # used only if a store has no inventory row at all

# Sentinel value used in PO records when the forecast is unavailable.
# Using -1 makes it unambiguous that this is NOT a real recommendation.
FORECAST_UNAVAILABLE_SENTINEL = -1.0


async def get_recommendations(
    store_id: int | None, db: Session
) -> list[PurchaseOrderRecommendation]:
    store_ids = [store_id] if store_id else await _all_active_store_ids(db)
    results = []

    for sid in store_ids:
        forecast = await get_forecast(sid, horizon=LEAD_TIME_DAYS, db=db)

        # Phase 2: detect insufficient_data state
        forecast_method = forecast.get("forecast_method", "unknown")
        forecast_days = forecast.get("forecast", [])

        if forecast_method == "insufficient_data" or not forecast_days:
            # Cannot compute a meaningful recommendation — write a placeholder PO.
            po_row = PurchaseOrder(
                store_id=sid,
                recommended_qty=FORECAST_UNAVAILABLE_SENTINEL,
                current_inventory=round(await _get_current_inventory(sid, db), 2),
                forecasted_demand=FORECAST_UNAVAILABLE_SENTINEL,
                status="pending_forecast",
                created_at=datetime.utcnow(),
            )
            db.add(po_row)
            db.commit()
            db.refresh(po_row)

            results.append(
                PurchaseOrderRecommendation(
                    po_id=po_row.id,
                    store_id=sid,
                    forecasted_demand=FORECAST_UNAVAILABLE_SENTINEL,
                    current_inventory=po_row.current_inventory,
                    safety_stock=FORECAST_UNAVAILABLE_SENTINEL,
                    recommended_qty=FORECAST_UNAVAILABLE_SENTINEL,
                    reason=(
                        "Forecast unavailable: this store has insufficient historical data "
                        "or no configured scaler. Complete the store onboarding process "
                        "before generating purchase order recommendations."
                    ),
                )
            )
            continue

        # Normal path: forecast is valid — compute recommendation.
        forecasted_demand = sum(day["predicted_sales"] for day in forecast_days)

        current_inventory = await _get_current_inventory(sid, db)
        safety_stock = forecasted_demand * SAFETY_STOCK_FACTOR
        recommended_qty = max(0, forecasted_demand + safety_stock - current_inventory)

        po_row = PurchaseOrder(
            store_id=sid,
            recommended_qty=round(recommended_qty, 2),
            current_inventory=round(current_inventory, 2),
            forecasted_demand=round(forecasted_demand, 2),
            status="pending",
            created_at=datetime.utcnow(),
        )
        db.add(po_row)
        db.commit()
        db.refresh(po_row)

        results.append(
            PurchaseOrderRecommendation(
                po_id=po_row.id,
                store_id=sid,
                forecasted_demand=round(forecasted_demand, 2),
                current_inventory=round(current_inventory, 2),
                safety_stock=round(safety_stock, 2),
                recommended_qty=round(recommended_qty, 2),
                reason=(
                    f"{LEAD_TIME_DAYS}-day forecasted demand plus "
                    f"{int(SAFETY_STOCK_FACTOR * 100)}% safety stock, minus current inventory"
                ),
            )
        )

    return results


async def _all_active_store_ids(db: Session) -> list[int]:
    rows = db.query(Store.id).all()
    if rows:
        return [r[0] for r in rows]
    # No stores loaded yet (e.g. fresh DB before seeding) — return empty list.
    # Do NOT fall back to hardcoded [1, 2, 3]; those IDs may not exist.
    return []


async def _get_current_inventory(store_id: int, db: Session) -> float:
    row = db.query(Inventory).filter(Inventory.store_id == store_id).first()
    if row is None:
        return DEFAULT_INVENTORY_FALLBACK
    return row.current_quantity
