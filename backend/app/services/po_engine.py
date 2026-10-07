"""
Purchase Order Recommendation Engine.

Requirements & Safety invariants:
  - Works across all forecast methods: LSTM, cluster_average, and metadata_cohort.
  - Detects insufficient_data and maintains PO status="pending_forecast" with sentinel (-1.0).
  - Sentinel values (-1.0) or negative predictions are NEVER treated as real demand.
  - Recommended order quantities are strictly clamped: recommended_qty >= 0.0.
  - Structured logging for traceability and diagnostics.
"""

import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.inventory import Inventory
from app.models.purchase_order import PurchaseOrder
from app.models.store import Store
from app.schemas.purchase_order import PurchaseOrderRecommendation
from app.services.forecast_service import get_forecast

logger = logging.getLogger(__name__)

SAFETY_STOCK_FACTOR = 0.15  # 15% buffer over forecasted demand
LEAD_TIME_DAYS = 3
DEFAULT_INVENTORY_FALLBACK = 2000.0  # used only if a store has no inventory row at all

# Sentinel value used in PO records when the forecast is unavailable.
# Using -1.0 makes it unambiguous that this is NOT a real recommendation.
FORECAST_UNAVAILABLE_SENTINEL = -1.0


async def get_recommendations(
    store_id: int | None, db: Session
) -> list[PurchaseOrderRecommendation]:
    """
    Public entry point for calculating PO recommendations.
    If store_id is None, processes all active stores.
    """
    store_ids = [store_id] if store_id else await _all_active_store_ids(db)
    return await get_recommendations_for_stores(store_ids, db)


async def get_recommendations_for_stores(
    store_ids: list[int], db: Session
) -> list[PurchaseOrderRecommendation]:
    """
    Computes recommendations for an explicit list of authorized store IDs.
    """
    results = []
    logger.info("Generating purchase order recommendations for %d store(s)", len(store_ids))

    for sid in store_ids:
        forecast = await get_forecast(sid, horizon=LEAD_TIME_DAYS, db=db)

        forecast_method = forecast.get("forecast_method", "unknown")
        forecast_days = forecast.get("forecast", [])

        # Filter out sentinel or invalid days
        valid_predictions = [
            float(day["predicted_sales"])
            for day in forecast_days
            if day.get("predicted_sales") is not None
            and float(day["predicted_sales"]) >= 0.0
            and float(day["predicted_sales"]) != FORECAST_UNAVAILABLE_SENTINEL
        ]

        if forecast_method == "insufficient_data" or not valid_predictions:
            logger.info(
                "Store store_id=%d has insufficient_data or no valid predictions; creating pending PO.",
                sid,
            )
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

        # Normal path: forecast is valid (LSTM, cluster_average, or metadata_cohort)
        forecasted_demand = sum(valid_predictions)
        current_inventory = await _get_current_inventory(sid, db)
        safety_stock = forecasted_demand * SAFETY_STOCK_FACTOR
        # Never produce negative order quantity
        recommended_qty = max(0.0, forecasted_demand + safety_stock - current_inventory)

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

        logger.info(
            "Created PO recommendation id=%d for store_id=%d: method=%s, demand=%.2f, qty=%.2f",
            po_row.id,
            sid,
            forecast_method,
            forecasted_demand,
            recommended_qty,
        )

        results.append(
            PurchaseOrderRecommendation(
                po_id=po_row.id,
                store_id=sid,
                forecasted_demand=round(forecasted_demand, 2),
                current_inventory=round(current_inventory, 2),
                safety_stock=round(safety_stock, 2),
                recommended_qty=round(recommended_qty, 2),
                reason=(
                    f"{LEAD_TIME_DAYS}-day forecasted demand ({forecast_method}) plus "
                    f"{int(SAFETY_STOCK_FACTOR * 100)}% safety stock, minus current inventory"
                ),
            )
        )

    return results


async def _all_active_store_ids(db: Session) -> list[int]:
    rows = db.query(Store.id).all()
    if rows:
        return [r[0] for r in rows]
    return []


async def _get_current_inventory(store_id: int, db: Session) -> float:
    row = db.query(Inventory).filter(Inventory.store_id == store_id).first()
    if row is None:
        return DEFAULT_INVENTORY_FALLBACK
    return float(row.current_quantity)
