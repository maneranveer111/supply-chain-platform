from pydantic import BaseModel


class PurchaseOrderRecommendation(BaseModel):
    po_id: int
    store_id: int
    forecasted_demand: float
    current_inventory: float
    safety_stock: float
    recommended_qty: float
    reason: str


class PurchaseOrderApproval(BaseModel):
    status: str  # "approved" or "rejected"
