from pydantic import BaseModel


class ClusterAssignment(BaseModel):
    store_id: int
    cluster_id: int
    cluster_label: str


class ClusterSummary(BaseModel):
    cluster_id: int
    label: str
    description: str
    store_count: int
    avg_daily_sales: float
