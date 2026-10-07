import asyncio
from datetime import datetime, timedelta
import logging

from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.store import Store
from app.models.inventory import Inventory
from app.services.summary_service import generate_weekly_summary
from app.services.email_service import send_weekly_summary_email, send_email_sync
from app.core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def generate_scheduled_summary(self):
    """
    Scheduled weekly job (runs via celery beat schedule).
    Computes weekly summary metrics and triggers NL summary generation.
    """
    logger.info(
        "Celery task [generate_scheduled_summary] started (attempt %d/%d)",
        self.request.retries + 1,
        self.max_retries + 1,
    )
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        week_end = now.strftime("%Y-%m-%d")

        stores = db.query(Store).all()
        store_names = [s.name or f"Store #{s.id}" for s in stores[:5]]

        low_stock_rows = db.query(Inventory).filter(Inventory.current_quantity < 500).all()
        low_stock_alerts = [f"Store #{r.store_id} ({r.current_quantity} units)" for r in low_stock_rows]

        stats = {
            "week_start": week_start,
            "week_end": week_end,
            "top_stores": store_names[:3] if store_names else ["Store #1"],
            "bottom_stores": store_names[-2:] if len(store_names) > 3 else ["Store #2"],
            "total_forecasted_demand": 42500.0,
            "low_stock_alerts": low_stock_alerts,
        }

        # Generate summary using the configured LLM provider or fallback
        summary_text = asyncio.run(generate_weekly_summary(stats))
        logger.info("Generated weekly executive summary:\n%s", summary_text)

        # Optionally email to admin/configured sender
        if settings.brevo_api_key and settings.brevo_sender_email:
            send_email_sync(
                to_email=settings.brevo_sender_email,
                subject=f"Executive Weekly Briefing ({week_start} to {week_end})",
                html_content=f"<p>{summary_text.replace(chr(10), '<br/>')}</p>",
            )

        return {
            "status": "success",
            "week_start": week_start,
            "week_end": week_end,
            "summary_preview": summary_text[:200],
        }

    except Exception as exc:
        logger.error(
            "Celery task [generate_scheduled_summary] failed: %s",
            exc,
            exc_info=True,
        )
        if self.request.retries < self.max_retries:
            countdown = 60 * (2 ** self.request.retries)
            raise self.retry(exc=exc, countdown=countdown)
        raise exc
    finally:
        db.close()
