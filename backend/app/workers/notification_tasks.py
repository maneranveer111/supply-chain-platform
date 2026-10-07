"""
Celery background tasks for operational alerting and transactional notifications.
"""

import asyncio
import logging
from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.store import Store
from app.models.user import User
from app.models.inventory import Inventory
from app.services.inventory_service import analyze_inventory_signals
from app.services.email_service import send_email_sync, send_subscription_reminder_email

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def check_low_stock_alerts(self):
    """
    Periodic task scanning all stores for critical stockouts or low inventory.
    Dispatches Brevo email notifications to store owners when risk thresholds are breached.
    """
    logger.info("Starting low stock alert scan (task id: %s)", self.request.id)
    db = SessionLocal()
    alerts_sent = 0
    checked_stores = 0

    try:
        stores = db.query(Store).all()
        for store in stores:
            checked_stores += 1
            # Run inventory analysis
            signals = asyncio.run(analyze_inventory_signals(store.id, db, include_llm_explanation=False))
            status = signals.get("status")

            if status in ("critical_stockout", "low_stock"):
                # Determine recipient email
                owner = db.query(User).filter(User.id == store.user_id).first() if store.user_id else None
                recipient = owner.email if owner and owner.email else None

                if recipient:
                    subject = f"⚠️ Stockout Alert: {store.name or f'Store #{store.id}'}"
                    html = (
                        f"<h3>Stock Health Warning</h3>"
                        f"<p>Store <strong>{store.name or f'#{store.id}'}</strong> stock level is <strong>{signals['current_quantity']}</strong> units.</p>"
                        f"<p>Projected runway: <strong>{signals['days_of_supply']} days</strong> ({status}).</p>"
                        f"<p>Recommended reorder: <strong>{signals['recommended_reorder_qty']} units</strong>.</p>"
                    )
                    res = send_email_sync(to_email=recipient, subject=subject, html_content=html)
                    logger.info("Dispatched low stock email to %s for store_id=%d: %s", recipient, store.id, res.get("status"))
                    alerts_sent += 1

        db.commit()
        logger.info("Completed low stock scan: %d stores checked, %d alerts sent.", checked_stores, alerts_sent)
        return {"checked_stores": checked_stores, "alerts_sent": alerts_sent}

    except Exception as exc:
        logger.error("Error running low stock check: %s", exc, exc_info=True)
        db.rollback()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60)
        raise exc
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=3)
def send_subscription_reminders(self):
    """
    Periodic task sending subscription renewal reminders to active platform users.
    """
    logger.info("Starting subscription reminders scan (task id: %s)", self.request.id)
    db = SessionLocal()
    sent_count = 0

    try:
        users = db.query(User).all()
        for user in users:
            if user.email:
                subject = "Notice: Your Supply Chain Platform Subscription"
                html = (
                    f"<h3>Subscription Renewal Notice</h3>"
                    f"<p>Hello {user.email},</p>"
                    f"<p>Your automated procurement & forecasting subscription is active and scheduled for regular renewal.</p>"
                    f"<p>Thank you for using the Supply Chain Intelligence Platform.</p>"
                )
                res = send_email_sync(to_email=user.email, subject=subject, html_content=html)
                if res.get("status") == "sent":
                    sent_count += 1

        logger.info("Completed subscription reminders: %d emails processed.", sent_count)
        return {"reminders_processed": len(users), "sent_count": sent_count}

    except Exception as exc:
        logger.error("Error in subscription reminders: %s", exc, exc_info=True)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60)
        raise exc
    finally:
        db.close()
