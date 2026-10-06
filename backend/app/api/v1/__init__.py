from fastapi import APIRouter

from app.api.v1 import admin, auth, clusters, forecasts, purchase_orders, stores, summaries

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(forecasts.router)
api_router.include_router(clusters.router)
api_router.include_router(purchase_orders.router)
api_router.include_router(summaries.router)
api_router.include_router(stores.router)
api_router.include_router(admin.router)
