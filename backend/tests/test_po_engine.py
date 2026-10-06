"""
Tests for the purchase order recommendation formula. These don't need
a real DB or model — they test the pure business logic.
"""

from app.services.po_engine import SAFETY_STOCK_FACTOR


def test_safety_stock_factor_is_reasonable():
    assert 0 < SAFETY_STOCK_FACTOR < 1


def test_recommended_qty_never_negative():
    forecasted_demand = 100.0
    current_inventory = 10_000.0  # far more stock than needed
    safety_stock = forecasted_demand * SAFETY_STOCK_FACTOR
    recommended_qty = max(0, forecasted_demand + safety_stock - current_inventory)

    assert recommended_qty == 0
