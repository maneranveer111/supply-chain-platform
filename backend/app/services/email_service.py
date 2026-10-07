"""
Brevo Transactional Email Service.

Provides robust, production-grade email delivery via the Brevo v3 REST API.
Features:
  - Non-blocking failure handling: never crashes caller upon HTTP/network issues
  - 10-second request timeout
  - Graceful fallback when BREVO_API_KEY is unset or dummy
  - Beautiful, responsive HTML email templates for alerts, POs, and summaries
  - Synchronous and asynchronous helpers for API and Celery workers
"""

import logging
from typing import Any
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
TIMEOUT_SECONDS = 10.0


async def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    to_name: str | None = None,
) -> dict[str, Any]:
    """
    Send a transactional email using Brevo's v3 REST API.
    Returns status dict: {'status': 'sent' | 'skipped' | 'failed', 'message_id': ...}
    """
    if not settings.brevo_api_key or settings.brevo_api_key.startswith("your_"):
        logger.warning(
            "BREVO_API_KEY is not configured. Email to %s skipped.", to_email
        )
        return {"status": "skipped", "reason": "brevo_api_key not configured"}

    headers = {
        "api-key": settings.brevo_api_key,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    payload = {
        "sender": {
            "name": settings.brevo_sender_name,
            "email": settings.brevo_sender_email,
        },
        "to": [
            {"email": to_email, "name": to_name or to_email}
        ],
        "subject": subject,
        "htmlContent": html_content,
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(BREVO_API_URL, headers=headers, json=payload)
            if response.status_code in (200, 201, 202):
                data = response.json()
                message_id = data.get("messageId")
                logger.info("Email successfully sent to %s (messageId=%s)", to_email, message_id)
                return {"status": "sent", "message_id": message_id}
            else:
                logger.error(
                    "Brevo API error %d sending to %s: %s",
                    response.status_code,
                    to_email,
                    response.text,
                )
                return {
                    "status": "failed",
                    "status_code": response.status_code,
                    "error": response.text,
                }
    except httpx.TimeoutException:
        logger.error("Brevo API timeout after %ss sending to %s", TIMEOUT_SECONDS, to_email)
        return {"status": "failed", "error": "Brevo request timed out"}
    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to_email, exc, exc_info=True)
        return {"status": "failed", "error": str(exc)}


def send_email_sync(
    to_email: str,
    subject: str,
    html_content: str,
    to_name: str | None = None,
) -> dict[str, Any]:
    """
    Synchronous version of send_email for use inside Celery workers.
    """
    if not settings.brevo_api_key or settings.brevo_api_key.startswith("your_"):
        logger.warning("BREVO_API_KEY not set. Email to %s skipped.", to_email)
        return {"status": "skipped", "reason": "brevo_api_key not configured"}

    headers = {
        "api-key": settings.brevo_api_key,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    payload = {
        "sender": {
            "name": settings.brevo_sender_name,
            "email": settings.brevo_sender_email,
        },
        "to": [{"email": to_email, "name": to_name or to_email}],
        "subject": subject,
        "htmlContent": html_content,
    }

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(BREVO_API_URL, headers=headers, json=payload)
            if response.status_code in (200, 201, 202):
                data = response.json()
                return {"status": "sent", "message_id": data.get("messageId")}
            else:
                logger.error("Brevo API error %d: %s", response.status_code, response.text)
                return {"status": "failed", "status_code": response.status_code, "error": response.text}
    except Exception as exc:
        logger.error("Failed to send sync email to %s: %s", to_email, exc)
        return {"status": "failed", "error": str(exc)}


async def send_stock_alert_email(
    to_email: str,
    store_id: int,
    store_name: str,
    current_stock: float,
    reorder_point: float,
    days_of_supply: float,
) -> dict[str, Any]:
    """
    Alert procurement and operations team of imminent stockout risk.
    """
    subject = f"[URGENT] Stockout Alert: {store_name or f'Store #{store_id}'}"
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc; padding: 24px; color: #1e293b;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="background-color: #ef4444; color: #ffffff; padding: 18px 24px;">
                <h2 style="margin: 0; font-size: 20px;">⚠️ Low Stock Alert</h2>
            </div>
            <div style="padding: 24px;">
                <p style="font-size: 16px; line-height: 1.5;">
                    Store <strong>{store_name or f'#{store_id}'}</strong> (ID: {store_id}) has fallen below its safety threshold.
                </p>
                <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 10px 0; color: #64748b;">Current Stock</td>
                        <td style="padding: 10px 0; font-weight: bold; text-align: right; color: #ef4444;">{current_stock:,.1f} units</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 10px 0; color: #64748b;">Reorder Point</td>
                        <td style="padding: 10px 0; font-weight: bold; text-align: right;">{reorder_point:,.1f} units</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 10px 0; color: #64748b;">Estimated Days of Supply</td>
                        <td style="padding: 10px 0; font-weight: bold; text-align: right; color: #f59e0b;">{days_of_supply:.1f} days</td>
                    </tr>
                </table>
                <p style="font-size: 14px; color: #64748b;">
                    Immediate action recommended: Check Purchase Order recommendations to initiate restocking.
                </p>
            </div>
            <div style="background: #f8fafc; padding: 16px 24px; font-size: 12px; color: #94a3b8; text-align: center;">
                Supply Chain Intelligence Platform &bull; Automated Operations Engine
            </div>
        </div>
    </body>
    </html>
    """
    return await send_email(to_email=to_email, subject=subject, html_content=html)


async def send_po_approval_notification(
    to_email: str,
    po_id: int,
    store_id: int,
    status: str,
    recommended_qty: float,
) -> dict[str, Any]:
    """
    Send purchase order status change update (approved / rejected).
    """
    is_approved = status.lower() == "approved"
    header_color = "#10b981" if is_approved else "#64748b"
    status_label = "Approved" if is_approved else "Rejected"
    subject = f"Purchase Order #{po_id} Status: {status_label}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc; padding: 24px; color: #1e293b;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="background-color: {header_color}; color: #ffffff; padding: 18px 24px;">
                <h2 style="margin: 0; font-size: 20px;">PO #{po_id} &mdash; {status_label}</h2>
            </div>
            <div style="padding: 24px;">
                <p style="font-size: 16px; line-height: 1.5;">
                    Purchase order <strong>#{po_id}</strong> for Store <strong>#{store_id}</strong> has been marked as <strong>{status}</strong>.
                </p>
                <div style="background: #f8fafc; border-radius: 6px; padding: 16px; margin: 16px 0;">
                    <p style="margin: 4px 0; color: #64748b;">Quantity: <strong>{recommended_qty:,.1f} units</strong></p>
                    <p style="margin: 4px 0; color: #64748b;">Status: <strong style="color: {header_color};">{status.upper()}</strong></p>
                </div>
            </div>
            <div style="background: #f8fafc; padding: 16px 24px; font-size: 12px; color: #94a3b8; text-align: center;">
                Supply Chain Intelligence Platform &bull; Procurement Workflow
            </div>
        </div>
    </body>
    </html>
    """
    return await send_email(to_email=to_email, subject=subject, html_content=html)


async def send_weekly_summary_email(
    to_email: str,
    summary_text: str,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Send executive weekly procurement & forecasting intelligence summary.
    """
    subject = "Weekly Procurement & Inventory Intelligence Briefing"
    stats = stats or {}

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc; padding: 24px; color: #1e293b;">
        <div style="max-width: 650px; margin: 0 auto; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: #ffffff; padding: 22px 24px;">
                <h2 style="margin: 0; font-size: 22px;">📊 Weekly Supply Chain Intelligence</h2>
                <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 14px;">Executive Operational Briefing</p>
            </div>
            <div style="padding: 24px;">
                <div style="background: #f0fdf4; border-left: 4px solid #22c55e; padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
                    <p style="margin: 0; font-size: 14px; color: #166534;">
                        <strong>Key Focus:</strong> AI-powered demand forecasting and multi-echelon stock health.
                    </p>
                </div>
                <div style="font-size: 15px; line-height: 1.6; color: #334155;">
                    {summary_text.replace(chr(10), '<br/>')}
                </div>
            </div>
            <div style="background: #f8fafc; padding: 16px 24px; font-size: 12px; color: #94a3b8; text-align: center;">
                Supply Chain Intelligence Platform &bull; Automated Executive Briefing
            </div>
        </div>
    </body>
    </html>
    """
    return await send_email(to_email=to_email, subject=subject, html_content=html)


async def send_subscription_reminder_email(
    to_email: str,
    user_name: str | None = None,
    days_remaining: int = 7,
) -> dict[str, Any]:
    """
    Send subscription / renewal reminder to users.
    """
    subject = f"Friendly Reminder: Your Platform Plan Renews in {days_remaining} Days"
    name_greeting = f"Hello {user_name}," if user_name else "Hello,"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc; padding: 24px; color: #1e293b;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; overflow: hidden;">
            <div style="background-color: #4f46e5; color: #ffffff; padding: 20px 24px;">
                <h2 style="margin: 0; font-size: 20px;">🔔 Subscription Notice</h2>
            </div>
            <div style="padding: 24px;">
                <p style="font-size: 16px; line-height: 1.5;">{name_greeting}</p>
                <p style="font-size: 15px; line-height: 1.6; color: #475569;">
                    This is a notification that your Supply Chain Intelligence subscription is scheduled for renewal in <strong>{days_remaining} days</strong>.
                </p>
                <p style="font-size: 15px; line-height: 1.6; color: #475569;">
                    Your access to AI forecasts, automated purchase order engine, and inventory telemetry remains fully active.
                </p>
            </div>
            <div style="background: #f8fafc; padding: 16px 24px; font-size: 12px; color: #94a3b8; text-align: center;">
                Supply Chain Intelligence Platform &bull; Account Billing
            </div>
        </div>
    </body>
    </html>
    """
    return await send_email(to_email=to_email, subject=subject, html_content=html)
