from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.services.summary_service import generate_weekly_summary

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/weekly-summary")
async def weekly_summary(db: Session = Depends(get_db), user=Depends(get_current_user)):
    # TODO: replace with real aggregates pulled from the DB for the past week.
    stats = {
        "week_start": "2026-09-29",
        "week_end": "2026-10-05",
        "top_stores": [12, 45, 78],
        "bottom_stores": [301, 512],
        "total_forecasted_demand": 1_245_000,
        "low_stock_alerts": [301, 512, 88],
    }
    summary_text = await generate_weekly_summary(stats)
    return {"summary": summary_text, "stats": stats}
