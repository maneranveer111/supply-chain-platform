from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

celery_app = Celery(
    "supply_chain_platform",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.retrain_tasks",
        "app.workers.cluster_tasks",
        "app.workers.summary_tasks",
        "app.workers.onboarding_tasks",
        "app.workers.notification_tasks",
    ],
)

celery_app.conf.beat_schedule = {
    "weekly-summary-every-monday": {
        "task": "app.workers.summary_tasks.generate_scheduled_summary",
        "schedule": crontab(hour=6, minute=0, day_of_week="monday"),
    },
    "refresh-clusters-weekly": {
        "task": "app.workers.cluster_tasks.refresh_clusters",
        "schedule": crontab(hour=5, minute=0, day_of_week="sunday"),
    },
    "check-low-stock-alerts-daily": {
        "task": "app.workers.notification_tasks.check_low_stock_alerts",
        "schedule": crontab(hour=7, minute=0),
    },
    "subscription-reminders-daily": {
        "task": "app.workers.notification_tasks.send_subscription_reminders",
        "schedule": crontab(hour=8, minute=0),
    },
}
celery_app.conf.timezone = "UTC"
