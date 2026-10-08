"""
Phase 7 Production Hardening, Security & Observability Test Suite.

Validates:
  1. Request correlation ID middleware (X-Request-ID propagation)
  2. Database & internal exception sanitization (prevents leakage)
  3. Purchase Order multi-tenant object-level authorization (prevent horizontal privilege escalation)
  4. Sliding-window Redis rate limiter fail-open reliability
  5. Store inventory input validation (rejecting non-numeric/negative quantities)
  6. File upload chunked size limit enforcement
  7. Optimized admin stores matrix query (outer join performance)
  8. Deep health check exception sanitization
  9. Redis-backed alert notification task retry deduplication
"""

import io
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.core.config import settings
from app.core.dependencies import get_current_user, CurrentUser, Role, rate_limiter
from app.db.session import Base, get_db
from app.main import app
from app.models.inventory import Inventory
from app.models.purchase_order import PurchaseOrder
from app.models.store import Store
from app.models.user import User
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
def client():
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    return TestClient(app)


@pytest.fixture
def admin_client():
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    return TestClient(app)


# ===========================================================================
# 1. Observability & Correlation ID Middleware Tests
# ===========================================================================

def test_request_id_generated_and_returned_in_header(client):
    """Verifies that an incoming request without X-Request-ID gets an assigned UUID header."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


def test_custom_request_id_propagated_in_header(client):
    """Verifies that an explicit X-Request-ID is preserved and echoed back."""
    custom_id = "trace-corp-sec-998877"
    resp = client.get("/health", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id") == custom_id


# ===========================================================================
# 2. Security: Error Sanitization Tests
# ===========================================================================

def test_database_exception_sanitization():
    """Verifies that internal database errors do not expose SQL statements or schema to clients."""
    with patch("app.main.readiness_check", side_effect=OperationalError("SELECT secret FROM pass", {}, None)):
        test_client = TestClient(app, raise_server_exceptions=False)
        # Even if a handler or internal query raises an unexpected SQLAlchemyError:
        from fastapi import APIRouter
        err_router = APIRouter()
        @err_router.get("/test-sql-leak")
        def leak_route():
            raise OperationalError("SELECT * FROM sensitive_users_table WHERE secret_col=1", {}, Exception("internal"))
        
        app.include_router(err_router)
        resp = test_client.get("/test-sql-leak")
        assert resp.status_code == 500
        data = resp.json()
        assert "detail" in data
        assert "sensitive_users_table" not in data["detail"]
        assert "SELECT" not in data["detail"]
        assert data["detail"] == "A database error occurred while processing your request."


# ===========================================================================
# 3. Object-Level Authorization (Horizontal Privilege Escalation)
# ===========================================================================

def test_po_approval_denies_unauthorized_user(db_session):
    """User cannot approve purchase order belonging to another tenant's store."""
    # Store owned by user 42
    store = Store(id=701, name="Tenant A Store", user_id=42, onboarding_status="ready")
    db_session.add(store)
    po = PurchaseOrder(
        id=801,
        store_id=701,
        recommended_qty=100.0,
        current_inventory=20.0,
        forecasted_demand=120.0,
        status="pending",
        created_at=datetime.utcnow(),
    )
    db_session.add(po)
    db_session.commit()

    # User 10 (Procurement Manager) attempts to approve PO for Store 701 (owned by user 42)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id=10, email="proc_user10@test.com", role=Role.PROCUREMENT_MANAGER.value
    )
    pm_client = TestClient(app)

    resp = pm_client.post(f"/api/v1/purchase-orders/{po.id}/approve", json={"status": "approved"})
    assert resp.status_code == 403
    assert "permission" in resp.json()["detail"].lower()


def test_po_approval_allowed_for_store_owner_or_admin(db_session):
    """Owner or Admin can approve purchase order."""
    store = Store(id=702, name="Owner Store", user_id=10, onboarding_status="ready")
    db_session.add(store)
    po = PurchaseOrder(
        id=802,
        store_id=702,
        recommended_qty=150.0,
        current_inventory=50.0,
        forecasted_demand=200.0,
        status="pending",
        created_at=datetime.utcnow(),
    )
    db_session.add(po)
    db_session.commit()

    # User 10 (Procurement Manager & Owner) approves
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id=10, email="proc_owner@test.com", role=Role.PROCUREMENT_MANAGER.value
    )
    pm_client = TestClient(app)
    with patch("app.services.email_service.send_po_approval_notification", AsyncMock(return_value={"status": "sent"})):
        resp = pm_client.post(f"/api/v1/purchase-orders/{po.id}/approve", json={"status": "approved"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"


def test_po_explain_denies_unauthorized_user(db_session):
    """User cannot request LLM explanation for another tenant's store PO."""
    store = Store(id=703, name="Tenant B Store", user_id=42, onboarding_status="ready")
    db_session.add(store)
    po = PurchaseOrder(
        id=803,
        store_id=703,
        recommended_qty=250.0,
        current_inventory=30.0,
        forecasted_demand=280.0,
        status="pending",
        created_at=datetime.utcnow(),
    )
    db_session.add(po)
    db_session.commit()

    # User 10 tries to explain order of store 703
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id=10, email="user10@test.com", role=Role.STORE_ANALYST.value
    )
    client = TestClient(app)
    resp = client.post(f"/api/v1/purchase-orders/{po.id}/explain")
    assert resp.status_code == 403


# ===========================================================================
# 4. Rate Limiter Fail-Open Resiliency
# ===========================================================================

@pytest.mark.asyncio
async def test_rate_limiter_fails_open_on_redis_error():
    """Rate limiter allows requests to proceed when Redis is unavailable or raises an exception."""
    limiter_dependency = rate_limiter(max_requests=5, window_seconds=60)
    mock_request = MagicMock()
    mock_request.url.path = "/api/v1/test-endpoint"
    mock_user = CurrentUser(user_id=1, email="test@test.com", role="admin")

    with patch("app.core.dependencies.get_redis", side_effect=Exception("Redis connection timed out")):
        # Must not raise HTTPException(500) or crash; fails open cleanly
        await limiter_dependency(mock_request, mock_user)


# ===========================================================================
# 5. Input Validation & Upload Limits
# ===========================================================================

def test_store_inventory_rejects_invalid_quantity(client, db_session):
    """Updating inventory with non-numeric or negative values returns HTTP 422."""
    store = Store(id=704, name="Validation Store", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    # Non-numeric quantity
    resp1 = client.put(f"/api/v1/stores/{store.id}/inventory", json={"current_quantity": "invalid_number"})
    assert resp1.status_code == 422

    # Negative quantity
    resp2 = client.put(f"/api/v1/stores/{store.id}/inventory", json={"current_quantity": -50.0})
    assert resp2.status_code == 422


def test_sales_upload_rejects_oversized_file(client, db_session):
    """Uploading sales CSV larger than 10MB limit is rejected with HTTP 400."""
    store = Store(id=705, name="Upload Test Store", user_id=10, onboarding_status="ready")
    db_session.add(store)
    db_session.commit()

    # Create mock 11MB payload
    oversized_data = b"date,sales,promo,school_holiday,state_holiday\n" + (b"0" * (11 * 1024 * 1024))
    files = {"file": ("large_sales.csv", oversized_data, "text/csv")}

    resp = client.post(f"/api/v1/stores/{store.id}/upload-sales", files=files)
    assert resp.status_code == 400
    assert "File too large" in resp.json()["detail"]


# ===========================================================================
# 6. Admin Matrix Optimization & Health Check Diagnostics
# ===========================================================================

def test_admin_stores_matrix_with_outerjoin(admin_client, db_session):
    """Admin store matrix returns correctly combined store, user, and inventory data."""
    user = User(id=77, email="owner77@test.com", hashed_password="pw", role="store_analyst")
    store = Store(id=706, name="Admin Query Store", user_id=77, onboarding_status="ready")
    inv = Inventory(store_id=706, current_quantity=845.0)
    db_session.add_all([user, store, inv])
    db_session.commit()

    resp = admin_client.get("/api/v1/admin/stores")
    assert resp.status_code == 200
    data = resp.json()
    matched = [s for s in data if s["id"] == 706]
    assert len(matched) == 1
    assert matched[0]["owner_email"] == "owner77@test.com"
    assert matched[0]["current_stock"] == 845.0


def test_deep_health_check_sanitizes_errors(admin_client):
    """Deep health check reports 'unhealthy' without leaking raw internal stack or connection traces."""
    with patch("sqlalchemy.orm.Session.execute", side_effect=Exception("FATAL: database password leaked")):
        with patch("redis.Redis.from_url") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis.ping.side_effect = Exception("AUTH failed on redis://secret@internal:6379")
            mock_redis_cls.return_value = mock_redis

            resp = admin_client.get("/api/v1/admin/health/deep")
            assert resp.status_code == 200
            data = resp.json()
            assert data["components"]["database"] == "unhealthy"
            assert data["components"]["redis"] == "unhealthy"
            assert "password" not in str(data)
            assert "secret" not in str(data)


# ===========================================================================
# 7. Celery Task Retry Deduplication
# ===========================================================================

def test_check_low_stock_alerts_deduplicates_on_retry(db_session):
    """Notification task avoids re-sending alerts for already processed stores within the same task ID."""
    from app.workers.notification_tasks import check_low_stock_alerts

    store = Store(id=707, name="Retry Test Store", user_id=10, onboarding_status="ready")
    user = User(id=10, email="dedup_owner@test.com", hashed_password="pw", role="store_analyst")
    db_session.add_all([store, user])
    db_session.commit()

    mock_signals = {
        "status": "low_stock",
        "current_quantity": 40.0,
        "days_of_supply": 2.0,
        "recommended_reorder_qty": 200.0,
    }

    mock_redis = MagicMock()
    # First time key does not exist; after set, key exists
    mock_redis.exists.return_value = False

    with patch("app.workers.notification_tasks.SessionLocal", return_value=db_session):
        with patch("app.workers.notification_tasks.analyze_inventory_signals", AsyncMock(return_value=mock_signals)):
            with patch("app.workers.notification_tasks.send_email_sync", return_value={"status": "sent"}) as mock_email:
                with patch("redis.Redis.from_url", return_value=mock_redis):
                    # Run 1
                    res1 = check_low_stock_alerts()
                    assert res1["alerts_sent"] == 1
                    assert mock_email.call_count == 1

                    # Run 2 (Simulate retry where redis now returns True for key existence)
                    mock_redis.exists.return_value = True
                    res2 = check_low_stock_alerts()
                    assert res2["alerts_sent"] == 0
                    assert mock_email.call_count == 1  # Not called again
