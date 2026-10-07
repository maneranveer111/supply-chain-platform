"""
Inventory Intelligence Service.

Computes real-time inventory telemetry, stockout risk signals,
days of supply projections, and automated alert triggering.
"""

import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.inventory import Inventory
from app.models.store import Store
from app.services.forecast_service import get_forecast
from app.services.llm_service import explain_inventory_health

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_STOCK = 2000.0
LEAD_TIME_DAYS = 3
SAFETY_STOCK_FACTOR = 0.15


def get_or_create_inventory(store_id: int, db: Session) -> Inventory:
    """
    Fetch current inventory for a store. If no inventory record exists yet,
    creates a default baseline entry so telemetry is never broken.
    """
    row = db.query(Inventory).filter(Inventory.store_id == store_id).first()
    if not row:
        logger.info("Initializing baseline inventory for store_id=%d", store_id)
        row = Inventory(
            store_id=store_id,
            current_quantity=DEFAULT_INITIAL_STOCK,
            updated_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def update_inventory_level(store_id: int, new_quantity: float, db: Session) -> Inventory:
    """
    Update on-hand stock for a store.
    """
    row = get_or_create_inventory(store_id, db)
    row.current_quantity = float(new_quantity)
    row.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(row)
    logger.info("Updated inventory for store_id=%d to %.2f", store_id, new_quantity)
    return row


async def analyze_inventory_signals(
    store_id: int,
    db: Session,
    include_llm_explanation: bool = False,
) -> dict:
    """
    Analyze store stock level against multi-day demand forecast to evaluate:
    - Average daily demand
    - Projected days of supply (runway)
    - Reorder point (Lead time demand + 15% safety stock)
    - Stock status: critical_stockout, low_stock, healthy, overstocked, insufficient_forecast
    - Stockout risk score (0.0 to 1.0)
    - Optional natural language LLM intelligence explanation
    """
    inv = get_or_create_inventory(store_id, db)
    current_qty = float(inv.current_quantity)

    # 7-day forecast horizon for inventory runway analysis
    forecast_data = await get_forecast(store_id=store_id, horizon=7, db=db)
    forecast_method = forecast_data.get("forecast_method")
    forecast_days = forecast_data.get("forecast", [])

    valid_predictions = [
        float(d["predicted_sales"])
        for d in forecast_days
        if d.get("predicted_sales") is not None and float(d["predicted_sales"]) >= 0.0
    ]

    if forecast_method == "insufficient_data" or not valid_predictions:
        return {
            "store_id": store_id,
            "current_quantity": current_qty,
            "daily_demand_mean": 0.0,
            "days_of_supply": None,
            "reorder_point": 0.0,
            "status": "insufficient_forecast",
            "stockout_risk_score": 0.5,
            "recommended_reorder_qty": 0.0,
            "forecast_method": forecast_method or "none",
            "explanation": "Store requires historical sales onboarding before inventory runway can be computed.",
            "updated_at": inv.updated_at,
        }

    daily_mean = sum(valid_predictions) / len(valid_predictions)
    daily_mean = max(0.01, daily_mean)  # avoid division by zero

    # Reorder point = (Lead time * daily demand) + safety stock factor
    reorder_point = (daily_mean * LEAD_TIME_DAYS) * (1.0 + SAFETY_STOCK_FACTOR)

    days_of_supply = round(current_qty / daily_mean, 1)

    # Status classification & Risk scoring
    if current_qty <= 0:
        status = "critical_stockout"
        risk_score = 1.0
    elif days_of_supply < LEAD_TIME_DAYS:
        status = "critical_stockout"
        risk_score = 0.92
    elif days_of_supply < 7.0:
        status = "low_stock"
        risk_score = 0.65
    elif days_of_supply > 35.0:
        status = "overstocked"
        risk_score = 0.05
    else:
        status = "healthy"
        risk_score = 0.15

    # Target runway is 14 days
    target_stock = daily_mean * 14 * (1.0 + SAFETY_STOCK_FACTOR)
    recommended_reorder = max(0.0, round(target_stock - current_qty, 1))

    explanation = None
    if include_llm_explanation:
        explanation = await explain_inventory_health(
            store_id=store_id,
            current_stock=current_qty,
            forecasted_demand=round(sum(valid_predictions), 1),
            days_of_supply=days_of_supply,
            status=status,
        )

    return {
        "store_id": store_id,
        "current_quantity": current_qty,
        "daily_demand_mean": round(daily_mean, 2),
        "days_of_supply": days_of_supply,
        "reorder_point": round(reorder_point, 1),
        "status": status,
        "stockout_risk_score": risk_score,
        "recommended_reorder_qty": recommended_reorder,
        "forecast_method": forecast_method,
        "explanation": explanation,
        "updated_at": inv.updated_at,
    }
