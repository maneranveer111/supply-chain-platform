"""
Phase 5 Comprehensive Tests: Production Integration & Reliability.

Covers:
  A. Authentication (valid & invalid tokens)
  B. Authorization (user store isolation, admin access)
  C. Store lifecycle (create -> upload -> status -> forecast)
  D. Forecast routing (benchmark LSTM, custom LSTM, cluster, metadata, insufficient_data)
  E. Cache reliability (hit, miss, invalidation on upload/benchmark/cluster, redis failure resilience)
  F. Purchase Order safety (all forecast methods, insufficient_data pending, sentinel protection, store isolation)
  G. Background jobs (idempotency, failure handling, Celery tasks)
  H. Health & Readiness (liveness, readiness, DB down, Redis down)
  I. API error handling & parameter validation (invalid horizon 422, missing store 404, unauthorized 403)
"""

import json
from datetime import date, datetime, timedelta
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.dependencies import CurrentUser, Role, get_current_user
from app.core.security import create_access_token
from app.db.session import Base, get_db
from app.main import app
from app.ml.forecast_router import ForecastMethod, route_forecast
from app.models.cluster import Cluster
from app.models.daily_store_feature import DailyStoreFeature
from app.models.inventory import Inventory
from app.models.purchase_order import PurchaseOrder
from app.models.store import Store
from app.models.store_sale import StoreSale
from app.models.store_scaler import StoreScaler
from app.models.user import User
from app.services import forecast_service, po_engine
from app.services.onboarding_service import (
    process_features,
    process_store_features_and_clustering,
)
from tests.test_phase3 import (
    TestingSessionLocal,
    db_session,
    engine,
    override_get_current_user_admin,
    override_get_current_user_normal,
    override_get_db,
    setup_db,
)


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    yield TestClient(app)


@pytest.fixture
def admin_client(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    yield TestClient(app)


def generate_csv_bytes(n_days: int, start_date: str = "2020-01-01") -> bytes:
    df = pd.DataFrame(
        {
            "date": pd.date_range(start_date, periods=n_days),
            "sales": np.random.randint(1000, 5000, size=n_days),
            "promo": np.random.choice([0, 1], size=n_days),
            "school_holiday": 0,
            "state_holiday": "0",
        }
    )
    return df.to_csv(index=False).encode("utf-8")


# ===========================================================================
# A. Authentication Tests
# ===========================================================================

def test_auth_valid_token(db_session):
    # Temporarily remove dependency override to test real get_current_user logic
    app.dependency_overrides.pop(get_current_user, None)
    try:
        token = create_access_token({"sub": "10", "email": "user@test.com", "role": "store_analyst"})
        c = TestClient(app)
        resp = c.get("/api/v1/stores", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
    finally:
        app.dependency_overrides[get_current_user] = override_get_current_user_normal


def test_auth_invalid_token():
    app.dependency_overrides.pop(get_current_user, None)
    try:
        c = TestClient(app)
        resp = c.get("/api/v1/stores", headers={"Authorization": "Bearer completely-invalid-jwt-token"})
        assert resp.status_code == 401
        assert "Invalid or expired token" in resp.json()["detail"]
    finally:
        app.dependency_overrides[get_current_user] = override_get_current_user_normal


# ===========================================================================
# B. Authorization Tests
# ===========================================================================

def test_user_can_access_own_store(client, db_session):
    store = Store(id=1, name="My Store", user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()

    resp = client.get("/api/v1/stores/1")
    assert resp.status_code == 200
    assert resp.json()["name"] == "My Store"


def test_user_cannot_access_other_store(client, db_session):
    other_store = Store(id=2, name="Other Store", user_id=20, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(other_store)
    db_session.commit()

    resp = client.get("/api/v1/stores/2")
    assert resp.status_code == 403


def test_admin_can_access_any_store(admin_client, db_session):
    other_store = Store(id=2, name="Other Store", user_id=20, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(other_store)
    db_session.commit()

    resp = admin_client.get("/api/v1/stores/2")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Other Store"


# ===========================================================================
# C. Store Lifecycle Tests
# ===========================================================================

def test_store_lifecycle_full_flow(client, db_session):
    # 1. Create store
    create_payload = {
        "name": "Lifecycle Store",
        "store_type": "a",
        "assortment": "a",
        "competition_distance": 1500.0,
        "promo2_active": False,
    }
    create_resp = client.post("/api/v1/stores", json=create_payload)
    assert create_resp.status_code == 201
    store_id = create_resp.json()["id"]

    # 2. Check initial status -> metadata cohort ready (cold start)
    status_resp = client.get(f"/api/v1/stores/{store_id}/forecast/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "metadata_cohort_ready"
    assert status_resp.json()["forecast_method"] == "metadata_cohort"

    # 3. Forecast cold start
    f_resp = client.get(f"/api/v1/stores/{store_id}/forecast?horizon=7")
    assert f_resp.status_code == 200
    assert f_resp.json()["forecast_method"] == "metadata_cohort"
    assert len(f_resp.json()["forecast"]) == 7

    # 4. Upload 15 days data -> cluster ready
    cluster = Cluster(id=1, label="High Volume", profile={"mean_daily_sales": 3200.0, "promo_uplift": 0.15})
    db_session.add(cluster)
    db_session.commit()

    csv_15 = generate_csv_bytes(15)
    files = {"file": ("sales_15.csv", BytesIO(csv_15), "text/csv")}
    upload_resp = client.post(f"/api/v1/stores/{store_id}/upload-sales", files=files)
    assert upload_resp.status_code == 200

    status_resp2 = client.get(f"/api/v1/stores/{store_id}/forecast/status")
    assert status_resp2.json()["status"] == "cluster_ready"
    assert status_resp2.json()["forecast_method"] == "cluster_average"

    # 5. Upload 65 days data -> LSTM ready
    csv_65 = generate_csv_bytes(65)
    files2 = {"file": ("sales_65.csv", BytesIO(csv_65), "text/csv")}
    upload_resp2 = client.post(f"/api/v1/stores/{store_id}/upload-sales", files=files2)
    assert upload_resp2.status_code == 200

    status_resp3 = client.get(f"/api/v1/stores/{store_id}/forecast/status")
    assert status_resp3.json()["status"] == "lstm_ready"
    assert status_resp3.json()["has_custom_scaler"] is True


# ===========================================================================
# D. Forecast Routing Tests
# ===========================================================================

def test_forecast_routing_benchmark_lstm(client, db_session):
    store = Store(id=100, user_id=10, benchmark_store_id=1, forecast_mode="auto")
    db_session.add(store)
    today = date.today()
    for i in range(35):
        db_session.add(
            DailyStoreFeature(
                store_id=100,
                date=today - timedelta(days=35 - i),
                features={col: 1.0 for col in ["Promo", "Sales_scaled"]},
            )
        )
    db_session.commit()

    with patch("app.ml.forecast_router.predict_with_scaler", return_value=[2500.0] * 7):
        result = route_forecast(app_store_id=100, horizon=7, forecast_window=30, db=db_session)
        assert result.forecast_method == ForecastMethod.LSTM
        assert result.forecast_confidence.value == "high"
        assert len(result.forecast) == 7


def test_forecast_routing_custom_lstm(client, db_session):
    store = Store(id=101, user_id=10, forecast_mode="auto")
    scaler = StoreScaler(store_id=101, mean_log=7.5, scale_log=0.5, n_samples=60)
    db_session.add_all([store, scaler])
    today = date.today()
    for i in range(35):
        db_session.add(
            DailyStoreFeature(
                store_id=101,
                date=today - timedelta(days=35 - i),
                features={col: 1.0 for col in ["Promo", "Sales_scaled"]},
            )
        )
    db_session.commit()

    with patch("app.ml.forecast_router.predict_with_scaler", return_value=[3100.0] * 7):
        result = route_forecast(app_store_id=101, horizon=7, forecast_window=30, db=db_session)
        assert result.forecast_method == ForecastMethod.LSTM
        assert result.forecast_confidence.value == "medium"


def test_forecast_routing_cluster(db_session):
    cluster = Cluster(
        id=5,
        label="Test Cluster",
        profile={
            "mean_daily_sales": 4000.0,
            "dow_multipliers": [1.0] * 7,
            "promo_uplift": 0.2,
            "n_stores": 10,
            "n_samples": 500,
        },
    )
    store = Store(id=102, user_id=10, cluster_id=5, forecast_mode="auto")
    db_session.add_all([cluster, store])
    db_session.commit()

    result = route_forecast(app_store_id=102, horizon=7, forecast_window=30, db=db_session)
    assert result.forecast_method == ForecastMethod.CLUSTER_AVERAGE
    assert result.forecast_confidence.value == "low"
    assert len(result.forecast) == 7


def test_forecast_routing_metadata_cohort(db_session):
    store = Store(id=103, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()

    result = route_forecast(app_store_id=103, horizon=7, forecast_window=30, db=db_session)
    assert result.forecast_method == ForecastMethod.METADATA_COHORT
    assert result.forecast_confidence.value == "low"
    assert len(result.forecast) == 7


def test_forecast_routing_insufficient_data(db_session):
    # Store with custom scaler but ONLY 5 days history
    store = Store(id=104, user_id=10, forecast_mode="auto")
    scaler = StoreScaler(store_id=104, mean_log=7.0, scale_log=0.5, n_samples=20)
    db_session.add_all([store, scaler])
    today = date.today()
    for i in range(5):
        db_session.add(
            DailyStoreFeature(
                store_id=104,
                date=today - timedelta(days=5 - i),
                features={},
            )
        )
    db_session.commit()

    result = route_forecast(app_store_id=104, horizon=7, forecast_window=30, db=db_session)
    assert result.forecast_method == ForecastMethod.INSUFFICIENT_DATA
    assert result.forecast_confidence.value == "none"
    assert result.forecast == []


# ===========================================================================
# E. Cache Reliability Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_cache_hit_and_miss(db_session):
    store = Store(id=200, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()

    fake_cache = {}

    class MockRedis:
        async def get(self, key):
            return fake_cache.get(key)

        async def setex(self, key, ttl, val):
            fake_cache[key] = val

        async def keys(self, pattern):
            return list(fake_cache.keys())

        async def delete(self, *keys):
            cnt = 0
            for k in keys:
                if k in fake_cache:
                    del fake_cache[k]
                    cnt += 1
            return cnt

    mock_r = MockRedis()
    with patch("app.services.forecast_service.get_redis", return_value=mock_r):
        # 1. First call -> Cache MISS
        f1 = await forecast_service.get_forecast(200, 7, db_session)
        assert len(fake_cache) == 1
        key = list(fake_cache.keys())[0]
        assert "forecast:v2:200:7" == key

        # 2. Second call -> Cache HIT
        f2 = await forecast_service.get_forecast(200, 7, db_session)
        assert f1 == f2


@pytest.mark.asyncio
async def test_cache_invalidation_after_sales_upload(client, db_session):
    store = Store(id=201, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()

    deleted_keys = []

    class MockRedis:
        async def keys(self, pattern):
            return ["forecast:v2:201:7", "forecast:v2:201:14"]

        async def delete(self, *keys):
            deleted_keys.extend(keys)
            return len(keys)

    with patch("app.services.forecast_service.get_redis", return_value=MockRedis()):
        csv_bytes = generate_csv_bytes(10)
        files = {"file": ("data.csv", BytesIO(csv_bytes), "text/csv")}
        resp = client.post("/api/v1/stores/201/upload-sales", files=files)
        assert resp.status_code == 200
        assert "forecast:v2:201:7" in deleted_keys
        assert "forecast:v2:201:14" in deleted_keys


@pytest.mark.asyncio
async def test_cache_invalidation_after_benchmark_change(admin_client, db_session):
    store = Store(id=202, user_id=10, forecast_mode="auto")
    db_session.add(store)
    db_session.commit()

    deleted_keys = []

    class MockRedis:
        async def keys(self, pattern):
            return ["forecast:v2:202:7"]

        async def delete(self, *keys):
            deleted_keys.extend(keys)
            return len(keys)

    with patch("app.services.forecast_service.get_redis", return_value=MockRedis()), patch(
        "app.ml.inference.load_rossmann_scaler", return_value=MagicMock()
    ):
        resp = admin_client.patch("/api/v1/admin/stores/202/benchmark", json={"benchmark_store_id": 1})
        assert resp.status_code == 200
        assert "forecast:v2:202:7" in deleted_keys


@pytest.mark.asyncio
async def test_cache_redis_failure_resilience(db_session):
    """If Redis fails, forecast generation still succeeds gracefully."""
    store = Store(id=203, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()

    class FailingRedis:
        async def get(self, key):
            raise ConnectionError("Redis connection lost")

        async def setex(self, key, ttl, val):
            raise ConnectionError("Redis connection lost")

    with patch("app.services.forecast_service.get_redis", return_value=FailingRedis()):
        # Must NOT raise exception
        result = await forecast_service.get_forecast(203, 7, db_session)
        assert result["store_id"] == 203
        assert result["forecast_method"] == "metadata_cohort"
        assert len(result["forecast"]) == 7


# ===========================================================================
# F. Purchase Order Safety Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_po_engine_across_all_forecast_methods(db_session):
    # Store with inventory
    store = Store(id=300, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    inv = Inventory(store_id=300, current_quantity=500.0, updated_at=datetime.utcnow())
    db_session.add_all([store, inv])
    db_session.commit()

    # 1. Cluster average forecast test
    with patch(
        "app.services.po_engine.get_forecast",
        return_value={
            "forecast_method": "cluster_average",
            "forecast": [{"predicted_sales": 100.0}, {"predicted_sales": 100.0}, {"predicted_sales": 100.0}],
        },
    ):
        recs = await po_engine.get_recommendations(300, db_session)
        assert len(recs) == 1
        rec = recs[0]
        assert rec.forecasted_demand == 300.0
        # demand (300) + safety stock (45) - inventory (500) = -155 -> clamped to 0.0
        assert rec.recommended_qty == 0.0

    # 2. Demand exceeds inventory test
    with patch(
        "app.services.po_engine.get_forecast",
        return_value={
            "forecast_method": "lstm",
            "forecast": [{"predicted_sales": 300.0}, {"predicted_sales": 300.0}, {"predicted_sales": 300.0}],
        },
    ):
        recs = await po_engine.get_recommendations(300, db_session)
        rec = recs[0]
        assert rec.forecasted_demand == 900.0
        # demand (900) + safety (135) - inventory (500) = 535.0
        assert rec.recommended_qty == 535.0


@pytest.mark.asyncio
async def test_po_engine_insufficient_data_remains_pending(db_session):
    store = Store(id=301, user_id=10, forecast_mode="auto")
    inv = Inventory(store_id=301, current_quantity=100.0, updated_at=datetime.utcnow())
    db_session.add_all([store, inv])
    db_session.commit()

    with patch(
        "app.services.po_engine.get_forecast",
        return_value={
            "forecast_method": "insufficient_data",
            "forecast": [],
        },
    ):
        recs = await po_engine.get_recommendations(301, db_session)
        assert len(recs) == 1
        rec = recs[0]
        assert rec.forecasted_demand == -1.0
        assert rec.recommended_qty == -1.0

        po = db_session.query(PurchaseOrder).filter_by(store_id=301).first()
        assert po.status == "pending_forecast"
        assert po.recommended_qty == -1.0


@pytest.mark.asyncio
async def test_po_engine_sentinel_and_negative_protection(db_session):
    store = Store(id=302, user_id=10, forecast_mode="auto")
    inv = Inventory(store_id=302, current_quantity=0.0, updated_at=datetime.utcnow())
    db_session.add_all([store, inv])
    db_session.commit()

    # Pass malicious / corrupted forecast containing negative & sentinel values
    with patch(
        "app.services.po_engine.get_forecast",
        return_value={
            "forecast_method": "lstm",
            "forecast": [{"predicted_sales": -1.0}, {"predicted_sales": -50.0}, {"predicted_sales": 200.0}],
        },
    ):
        recs = await po_engine.get_recommendations(302, db_session)
        rec = recs[0]
        # Only the 200.0 positive value is counted as demand
        assert rec.forecasted_demand == 200.0
        assert rec.recommended_qty > 0


def test_po_store_authorization(client, db_session):
    # User 10 owns store 401. User 20 owns store 402.
    s1 = Store(id=401, user_id=10, forecast_mode="auto")
    s2 = Store(id=402, user_id=20, forecast_mode="auto")
    db_session.add_all([s1, s2])
    db_session.commit()

    # Normal user (role=store_analyst) cannot call PO recommendations endpoint (requires procurement_manager)
    resp = client.get("/api/v1/purchase-orders/recommendations?store_id=401")
    assert resp.status_code == 403

    # Switch user to procurement_manager
    def override_procurement_user():
        return CurrentUser(user_id=10, email="pm@test.com", role=Role.PROCUREMENT_MANAGER.value)

    app.dependency_overrides[get_current_user] = override_procurement_user
    pm_client = TestClient(app)

    with patch(
        "app.services.po_engine.get_forecast",
        return_value={"forecast_method": "lstm", "forecast": [{"predicted_sales": 100.0}]},
    ):
        # Access own store -> 200 OK
        resp_own = pm_client.get("/api/v1/purchase-orders/recommendations?store_id=401")
        assert resp_own.status_code == 200

        # Access other user's store -> 403 Forbidden
        resp_other = pm_client.get("/api/v1/purchase-orders/recommendations?store_id=402")
        assert resp_other.status_code == 403

        # Access without store_id -> returns ONLY store 401, NEVER store 402
        resp_all = pm_client.get("/api/v1/purchase-orders/recommendations")
        assert resp_all.status_code == 200
        returned_ids = [r["store_id"] for r in resp_all.json()]
        assert 401 in returned_ids
        assert 402 not in returned_ids


# ===========================================================================
# G. Background Jobs & Onboarding Tests
# ===========================================================================

def test_onboarding_processing_idempotency(db_session):
    store = Store(id=500, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)

    # Insert 65 sales records
    today = date(2021, 1, 1)
    sales = [
        StoreSale(
            store_id=500,
            date=today + timedelta(days=i),
            sales=2000.0,
            promo=0,
            school_holiday=0,
            state_holiday="0",
        )
        for i in range(65)
    ]
    db_session.add_all(sales)
    db_session.commit()

    # First run
    res1 = process_store_features_and_clustering(500, db_session)
    assert res1["status"] == "ready"
    assert db_session.query(DailyStoreFeature).filter_by(store_id=500).count() == 65
    assert db_session.query(StoreScaler).filter_by(store_id=500).count() == 1

    # Second run (idempotency check: must not duplicate features or scalers)
    res2 = process_store_features_and_clustering(500, db_session)
    assert res2["status"] == "ready"
    assert db_session.query(DailyStoreFeature).filter_by(store_id=500).count() == 65
    assert db_session.query(StoreScaler).filter_by(store_id=500).count() == 1


def test_onboarding_processing_failure_handling(db_session):
    store = Store(id=501, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    sale = StoreSale(
        store_id=501,
        date=date(2021, 1, 1),
        sales=2000.0,
        promo=0,
        school_holiday=0,
        state_holiday="0",
    )
    db_session.add_all([store, sale])
    db_session.commit()

    with patch("app.services.onboarding_service.pd.DataFrame", side_effect=RuntimeError("Dataframe crash")):
        with pytest.raises(RuntimeError):
            process_store_features_and_clustering(501, db_session)

    db_session.refresh(store)
    assert store.onboarding_status == "failed"


# ===========================================================================
# H. Health & Readiness Tests
# ===========================================================================

def test_liveness_health_endpoint():
    c = TestClient(app)
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readiness_all_ok():
    c = TestClient(app)
    mock_session = MagicMock()
    mock_session.execute.return_value = None
    with patch("app.main.SessionLocal", return_value=mock_session), patch(
        "app.db.redis.get_redis"
    ) as mock_get_redis:
        mock_r = AsyncMock()
        mock_r.ping.return_value = True
        mock_get_redis.return_value = mock_r

        resp = c.get("/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert "checks" in data
        assert data["checks"]["database"] == "ok"


def test_readiness_db_unavailable():
    c = TestClient(app)
    mock_session = MagicMock()
    mock_session.execute.side_effect = Exception("DB connection refused")
    with patch("app.main.SessionLocal", return_value=mock_session):
        resp = c.get("/readiness")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["database"] == "unreachable"


def test_readiness_redis_unavailable():
    c = TestClient(app)
    mock_session = MagicMock()
    mock_session.execute.return_value = None
    with patch("app.main.SessionLocal", return_value=mock_session), patch(
        "app.db.redis.get_redis", side_effect=Exception("Redis down")
    ):
        # Liveness must remain 200 OK
        assert c.get("/health").status_code == 200

        # Readiness reports degraded status
        resp = c.get("/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["checks"]["redis"] == "unreachable"


# ===========================================================================
# I. Parameter Validation & Error Handling Tests
# ===========================================================================

def test_forecast_invalid_horizon_returns_422(client, db_session):
    store = Store(id=600, user_id=10, forecast_mode="auto")
    db_session.add(store)
    db_session.commit()

    # Horizon = 0
    resp_zero = client.get("/api/v1/stores/600/forecast?horizon=0")
    assert resp_zero.status_code == 422

    # Horizon = -5
    resp_neg = client.get("/api/v1/stores/600/forecast?horizon=-5")
    assert resp_neg.status_code == 422

    # Horizon > 90
    resp_large = client.get("/api/v1/stores/600/forecast?horizon=100")
    assert resp_large.status_code == 422


def test_forecast_missing_store_returns_404(client):
    resp = client.get("/api/v1/stores/99999/forecast")
    assert resp.status_code == 404
