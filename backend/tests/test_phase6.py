"""
Phase 6 Comprehensive Test Suite: Intelligent Supply Chain Platform.

Validates:
  1. Multi-provider LLM intelligence layer (Gemini, Groq, Fallback, Domain explanations)
  2. Transactional email service (Brevo REST API, safety timeouts, graceful degradation)
  3. Inventory intelligence & telemetry (Stockout signals, days of supply, reorder points, CRUD)
  4. Multi-tenant authorization on inventory & insight routes (403 for unauthorized stores)
  5. Explainable Purchase Order engine (PO breakdown, LLM reasoning)
  6. Store-level holistic AI insights (/stores/{store_id}/insights)
  7. Celery scheduled notification & summary tasks
  8. Administrative platform inspection & deep health checks (/admin/overview, /admin/stores, etc.)
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.dependencies import get_current_user, CurrentUser, Role
from app.db.session import Base, get_db
from app.main import app
from app.models.inventory import Inventory
from app.models.purchase_order import PurchaseOrder
from app.models.store import Store
from app.models.user import User
from app.services import email_service, inventory_service, llm_service
from app.services.llm_service import (
    FallbackProvider,
    GeminiProvider,
    GroqProvider,
    explain_forecast,
    explain_inventory_health,
    explain_purchase_order,
    get_llm_provider,
    safe_generate_insight,
)
from tests.test_phase3 import (
    TestingSessionLocal,
    db_session,
    engine,
    override_get_current_user_admin,
    override_get_current_user_normal,
    override_get_db,
)


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    yield TestClient(app)


@pytest.fixture
def admin_client(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    yield TestClient(app)


# ===========================================================================
# 1. LLM Service & Intelligence Layer Tests
# ===========================================================================

def test_llm_provider_selection():
    # Fallback provider when no keys
    with patch.object(settings, "gemini_api_key", ""):
        provider = get_llm_provider()
        assert isinstance(provider, FallbackProvider)

    # Gemini provider when key present and provider="gemini"
    with patch.object(settings, "gemini_api_key", "mock-gemini-key"):
        with patch.object(settings, "llm_provider", "gemini"):
            provider = get_llm_provider()
            assert isinstance(provider, GeminiProvider)

    # Groq provider when key present and provider="groq"
    with patch.object(settings, "groq_api_key", "mock-groq-key"):
        with patch.object(settings, "llm_provider", "groq"):
            provider = get_llm_provider()
            assert isinstance(provider, GroqProvider)


@pytest.mark.asyncio
async def test_llm_fallback_provider_generation():
    provider = FallbackProvider()
    resp = await provider.generate_insight("Analyze inventory", fallback_text="Fallback text")
    assert resp == "Fallback text"


@pytest.mark.asyncio
async def test_llm_domain_explanations():
    # Test deterministic fallback mode
    with patch.object(settings, "gemini_api_key", ""):
        # Test explain_forecast
        fc_text = await explain_forecast(
            store_name_or_id=1,
            total_demand=3500.0,
            horizon_days=7,
            avg_daily=500.0,
        )
        assert "Store #1" in fc_text
        assert "3,500.0" in fc_text

        # Test explain_inventory_health
        inv_text = await explain_inventory_health(
            store_name_or_id=2,
            current_quantity=300.0,
            days_of_supply=1.8,
            status="critical_stockout",
        )
        assert "Store #2" in inv_text
        assert "critical_stockout" in inv_text

        # Test explain_purchase_order
        po_text = await explain_purchase_order(
            store_name_or_id=3,
            recommended_qty=850.0,
            forecasted_demand=1000.0,
            current_inventory=300.0,
            safety_stock=150.0,
        )
        assert "Store #3" in po_text
        assert "850.0" in po_text


@pytest.mark.asyncio
async def test_safe_generate_insight_resilience():
    # Mocking a failing provider to ensure it returns fallback
    mock_failing_provider = MagicMock()
    mock_failing_provider.generate_text = AsyncMock(side_effect=Exception("API connection timeout"))

    with patch("app.services.llm_service.get_llm_provider", return_value=mock_failing_provider):
        res = await safe_generate_insight("Prompt", fallback_text="Graceful Fallback")
        assert res == "Graceful Fallback"


# ===========================================================================
# 2. Brevo Transactional Email Service Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_email_skipped_when_key_unconfigured():
    with patch.object(settings, "brevo_api_key", ""):
        res = await email_service.send_email(
            to_email="test@example.com",
            subject="Test",
            html_content="<p>Hello</p>",
        )
        assert res["status"] == "skipped"


@pytest.mark.asyncio
async def test_email_send_success_mock():
    with patch.object(settings, "brevo_api_key", "live-or-mock-key"):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"messageId": "<test-msg-id-123>"}

        with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_response)):
            res = await email_service.send_email(
                to_email="test@example.com",
                subject="Test Subject",
                html_content="<p>Test Content</p>",
            )
            assert res["status"] == "sent"
            assert res["message_id"] == "<test-msg-id-123>"


@pytest.mark.asyncio
async def test_email_send_timeout_resilience():
    import httpx

    with patch.object(settings, "brevo_api_key", "valid-key"):
        with patch("httpx.AsyncClient.post", AsyncMock(side_effect=httpx.TimeoutException("Timeout"))):
            res = await email_service.send_email(
                to_email="test@example.com",
                subject="Test Subject",
                html_content="<p>Timeout test</p>",
            )
            assert res["status"] == "failed"
            assert "timed out" in res["error"]


def test_email_send_sync_helper():
    with patch.object(settings, "brevo_api_key", "valid-key"):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messageId": "sync-123"}

        with patch("httpx.Client.post", return_value=mock_response):
            res = email_service.send_email_sync(
                to_email="test@example.com",
                subject="Sync Test",
                html_content="<p>Sync</p>",
            )
            assert res["status"] == "sent"
            assert res["message_id"] == "sync-123"


@pytest.mark.asyncio
async def test_specialized_email_dispatchers():
    with patch("app.services.email_service.send_email", AsyncMock(return_value={"status": "sent"})) as mock_send:
        # Stock alert
        await email_service.send_stock_alert_email("ops@example.com", 1, "Store Alpha", 150.0, 500.0, 1.5)
        assert mock_send.called
        assert "Stockout Alert" in mock_send.call_args.kwargs["subject"]

        # PO notification
        await email_service.send_po_approval_notification("mgr@example.com", 101, 1, "approved", 450.0)
        assert "Approved" in mock_send.call_args.kwargs["subject"]

        # Weekly summary
        await email_service.send_weekly_summary_email("exec@example.com", "Executive Briefing Text")
        assert "Weekly Procurement" in mock_send.call_args.kwargs["subject"]

        # Subscription reminder
        await email_service.send_subscription_reminder_email("user@example.com", "Alice", 5)
        assert "Renews in 5 Days" in mock_send.call_args.kwargs["subject"]


# ===========================================================================
# 3. Inventory Service & API Telemetry Tests
# ===========================================================================

def test_inventory_get_or_create_and_update(db_session):
    store = Store(id=501, name="Inventory Store", store_type="a", assortment="a", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    # Initial get creates baseline
    inv = inventory_service.get_or_create_inventory(501, db_session)
    assert inv.store_id == 501
    assert inv.current_quantity == 2000.0

    # Update level
    updated = inventory_service.update_inventory_level(501, 450.0, db_session)
    assert updated.current_quantity == 450.0


@pytest.mark.asyncio
async def test_inventory_signal_calculations(db_session):
    store = Store(id=502, name="Signal Store", store_type="a", assortment="a", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    inventory_service.update_inventory_level(502, 100.0, db_session)

    # Mock get_forecast returning 7 days of 100 sales each (mean = 100/day)
    mock_forecast = {
        "forecast_method": "lstm",
        "forecast": [{"date": f"2026-01-0{i}", "predicted_sales": 100.0} for i in range(1, 8)],
    }

    with patch("app.services.inventory_service.get_forecast", AsyncMock(return_value=mock_forecast)):
        signals = await inventory_service.analyze_inventory_signals(502, db_session)
        assert signals["store_id"] == 502
        assert signals["daily_demand_mean"] == 100.0
        assert signals["days_of_supply"] == 1.0  # 100 stock / 100 daily = 1.0 day
        assert signals["status"] == "critical_stockout"
        assert signals["stockout_risk_score"] > 0.8
        assert signals["reorder_point"] > 0


def test_inventory_api_routes(client, db_session):
    store = Store(id=503, name="Route Store", store_type="a", assortment="a", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    # GET inventory
    resp = client.get("/api/v1/inventory/503")
    assert resp.status_code == 200
    assert resp.json()["current_quantity"] == 2000.0

    # PUT inventory
    put_resp = client.put("/api/v1/inventory/503", json={"current_quantity": 750.0})
    assert put_resp.status_code == 200
    assert put_resp.json()["current_quantity"] == 750.0

    # GET inventory signals
    sig_resp = client.get("/api/v1/inventory/503/signals")
    assert sig_resp.status_code == 200
    assert "days_of_supply" in sig_resp.json()
    assert "status" in sig_resp.json()


def test_inventory_multi_tenant_isolation(client, db_session):
    # Store owned by user_id=2 (normal client user_id is 10)
    store = Store(id=504, name="Other User Store", store_type="a", assortment="a", user_id=2, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    # Normal user cannot access store 504
    resp = client.get("/api/v1/inventory/504")
    assert resp.status_code == 403

    resp_put = client.put("/api/v1/inventory/504", json={"current_quantity": 100.0})
    assert resp_put.status_code == 403


# ===========================================================================
# 4. Purchase Order Explain & Notification Tests
# ===========================================================================

def test_po_explain_endpoint(client, db_session):
    store = Store(id=505, name="PO Store", store_type="a", assortment="a", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    po = PurchaseOrder(
        store_id=505,
        recommended_qty=350.0,
        current_inventory=150.0,
        forecasted_demand=400.0,
        status="pending",
        created_at=datetime.utcnow(),
    )
    db_session.add(po)
    db_session.commit()

    with patch("app.services.llm_service.safe_generate_insight", AsyncMock(return_value="PO was recommended because forecasted demand of 400.0 plus 15% safety stock exceeds current inventory of 150.0.")):
        resp = client.post(f"/api/v1/purchase-orders/{po.id}/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["po_id"] == po.id
    assert data["recommended_qty"] == 350.0
    assert "explanation" in data
    assert len(data["explanation"]) > 0


def test_po_approval_triggers_nonblocking_notification(db_session):
    store = Store(id=506, name="PO Approval Store", store_type="a", assortment="a", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    po = PurchaseOrder(
        store_id=506,
        recommended_qty=200.0,
        current_inventory=50.0,
        forecasted_demand=220.0,
        status="pending",
        created_at=datetime.utcnow(),
    )
    db_session.add(po)
    db_session.commit()

    # Test client with procurement manager role
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id=10, email="proc@test.com", role=Role.PROCUREMENT_MANAGER.value
    )
    pm_client = TestClient(app)

    with patch("app.services.email_service.send_po_approval_notification", AsyncMock(return_value={"status": "sent"})):
        resp = pm_client.post(f"/api/v1/purchase-orders/{po.id}/approve", json={"status": "approved"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"


# ===========================================================================
# 5. Store Insights Endpoint Tests
# ===========================================================================

def test_store_insights_endpoint(client, db_session):
    store = Store(id=507, name="Insight Store", store_type="c", assortment="b", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    with patch("app.services.llm_service.safe_generate_insight", AsyncMock(return_value="This store has healthy inventory levels with 14 days of supply and forecast indicating steady demand.")):
        resp = client.get("/api/v1/stores/507/insights")
    assert resp.status_code == 200
    data = resp.json()
    assert data["store_id"] == 507
    assert "signals" in data
    assert "insights" in data
    assert len(data["insights"]) > 0


# ===========================================================================
# 6. Celery Scheduled Workflow Tasks Tests
# ===========================================================================

def test_celery_check_low_stock_alerts_task(db_session):
    from app.workers.notification_tasks import check_low_stock_alerts

    store = Store(id=508, name="Alert Task Store", store_type="a", assortment="a", user_id=10, onboarding_status="ready")
    user = User(id=10, email="owner@example.com", hashed_password="pw", role="store_analyst")
    db_session.add(store)
    db_session.add(user)
    db_session.commit()

    mock_signals = {
        "status": "critical_stockout",
        "current_quantity": 50.0,
        "days_of_supply": 1.2,
        "recommended_reorder_qty": 300.0,
    }

    with patch("app.workers.notification_tasks.SessionLocal", return_value=db_session):
        with patch("app.workers.notification_tasks.analyze_inventory_signals", AsyncMock(return_value=mock_signals)):
            with patch("app.workers.notification_tasks.send_email_sync", return_value={"status": "sent"}):
                result = check_low_stock_alerts()
                assert "checked_stores" in result
                assert result["checked_stores"] >= 1
                assert result["alerts_sent"] >= 1


def test_celery_subscription_reminders_task(db_session):
    from app.workers.notification_tasks import send_subscription_reminders

    user = User(id=99, email="sub_user@example.com", hashed_password="pw", role="store_analyst")
    db_session.add(user)
    db_session.commit()

    with patch("app.workers.notification_tasks.SessionLocal", return_value=db_session):
        with patch("app.workers.notification_tasks.send_email_sync", return_value={"status": "sent"}):
            result = send_subscription_reminders()
            assert "reminders_processed" in result
            assert result["reminders_processed"] >= 1


def test_celery_summary_task(db_session):
    from app.workers.summary_tasks import generate_scheduled_summary

    store = Store(id=509, name="Summary Store", store_type="a", assortment="a", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    with patch("app.workers.summary_tasks.SessionLocal", return_value=db_session):
        with patch("app.workers.summary_tasks.generate_weekly_summary", AsyncMock(return_value="Weekly Executive Summary Mock")):
            with patch("app.workers.summary_tasks.send_email_sync", return_value={"status": "sent"}):
                result = generate_scheduled_summary()
                assert result["status"] == "success"
                assert "summary_preview" in result


# ===========================================================================
# 7. Admin Workflows & Deep Health Check Tests
# ===========================================================================

def test_admin_overview_endpoint(db_session):
    c = TestClient(app)

    # 1. Non-admin gets 403
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    resp_forbidden = c.get("/api/v1/admin/overview")
    assert resp_forbidden.status_code == 403

    # 2. Admin gets full overview
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    resp = c.get("/api/v1/admin/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_stores" in data
    assert "total_users" in data
    assert "total_purchase_orders" in data
    assert "pending_purchase_orders" in data
    assert "critical_stockouts" in data
    assert "onboarding_breakdown" in data
    assert "llm_provider" in data


def test_admin_stores_inspection_endpoint(db_session):
    c = TestClient(app)

    # 1. Non-admin gets 403
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    resp_forbidden = c.get("/api/v1/admin/stores")
    assert resp_forbidden.status_code == 403

    # 2. Admin gets stores list
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    resp = c.get("/api/v1/admin/stores")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_admin_email_test_endpoint(db_session):
    c = TestClient(app)

    # 1. Non-admin gets 403
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    resp_forbidden = c.post("/api/v1/admin/email/test", json={"to_email": "ops@example.com"})
    assert resp_forbidden.status_code == 403

    # 2. Admin dispatches test
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    with patch("app.services.email_service.send_email", AsyncMock(return_value={"status": "sent", "message_id": "test-1"})):
        resp = c.post(
            "/api/v1/admin/email/test",
            json={"to_email": "ops@example.com", "subject": "Test", "content": "Checking"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "dispatched"


def test_admin_deep_health_endpoint(db_session):
    c = TestClient(app)

    # 1. Non-admin gets 403
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    resp_forbidden = c.get("/api/v1/admin/health/deep")
    assert resp_forbidden.status_code == 403

    # 2. Admin gets deep health report
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    with patch("redis.Redis.from_url") as mock_redis:
        mock_instance = MagicMock()
        mock_instance.ping.return_value = True
        mock_redis.return_value = mock_instance

        resp = c.get("/api/v1/admin/health/deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "database" in data["components"]
        assert "redis" in data["components"]
