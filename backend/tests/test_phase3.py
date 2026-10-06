"""
Phase 3 Tests: Authorization, Admin Mapping, and Cold-Start Routing.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.session import Base, get_db
from app.core.dependencies import get_current_user, CurrentUser, Role
from app.models.store import Store
from app.models.cluster import Cluster
# Import all models to ensure Base.metadata.create_all creates all tables
from app.models import (
    cluster, daily_store_feature, forecast, inventory, purchase_order, store, store_sale, store_scaler, user
)
from app.ml.forecast_router import route_forecast, ForecastMethod, ForecastConfidence
from app.ml.inference import MIN_HISTORY_ROWS

from sqlalchemy.pool import StaticPool

# --- DB Setup ---
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Auth Mocks ---
def override_get_current_user_normal():
    return CurrentUser(user_id=10, email="user@test.com", role=Role.STORE_ANALYST.value)

def override_get_current_user_admin():
    return CurrentUser(user_id=99, email="admin@test.com", role=Role.ADMIN.value)


# --- 1. Admin Benchmark Mapping Tests ---

def test_admin_can_map_benchmark(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    client = TestClient(app)
    
    # Create store
    store = Store(id=1, forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    # Needs valid rossmann ID. 1 is valid in pkl.
    response = client.patch("/api/v1/admin/stores/1/benchmark", json={"benchmark_store_id": 1})
    assert response.status_code == 200
    assert response.json()["benchmark_store_id"] == 1
    
    # Verify in DB
    db_session.refresh(store)
    assert store.benchmark_store_id == 1

def test_admin_invalid_benchmark_rejected(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    client = TestClient(app)
    store = Store(id=1, forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    # 9999 is not in the mock scaler pkl
    response = client.patch("/api/v1/admin/stores/1/benchmark", json={"benchmark_store_id": 9999})
    assert response.status_code == 400

def test_non_admin_cannot_map_benchmark(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    client = TestClient(app)
    store = Store(id=1, forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    response = client.patch("/api/v1/admin/stores/1/benchmark", json={"benchmark_store_id": 1})
    assert response.status_code == 403


# --- 2. Authorization Tests ---

def test_user_can_access_own_store(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    client = TestClient(app)
    
    store = Store(id=1, user_id=10, forecast_mode="auto") # User 10 owns it
    db_session.add(store)
    db_session.commit()
    
    response = client.get("/api/v1/stores/1/forecast/status")
    assert response.status_code == 200

def test_user_cannot_access_other_store(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    client = TestClient(app)
    
    store = Store(id=2, user_id=11, forecast_mode="auto") # User 11 owns it, user is 10
    db_session.add(store)
    db_session.commit()
    
    response = client.get("/api/v1/stores/2/forecast/status")
    assert response.status_code == 403

def test_user_cannot_access_unowned_demo_store(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    client = TestClient(app)
    
    store = Store(id=3, user_id=None, forecast_mode="auto") # Demo store
    db_session.add(store)
    db_session.commit()
    
    response = client.get("/api/v1/stores/3/forecast/status")
    assert response.status_code == 403

def test_admin_can_access_unowned_demo_store(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    client = TestClient(app)
    
    store = Store(id=4, user_id=None, forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    response = client.get("/api/v1/stores/4/forecast/status")
    assert response.status_code == 200


# --- 3. Cold-Start Routing Tests ---

def test_cluster_routing(db_session):
    # Setup cluster with profile
    cluster = Cluster(id=1, label="test", profile={
        "mean_daily_sales": 1000.0,
        "dow_multipliers": [1.0] * 7,
        "promo_uplift": 0.0
    })
    db_session.add(cluster)
    
    store = Store(id=1, cluster_id=1, forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    result = route_forecast(1, 7, MIN_HISTORY_ROWS, db_session)
    
    assert result.forecast_method == ForecastMethod.CLUSTER_AVERAGE
    assert result.forecast_confidence == ForecastConfidence.LOW
    assert len(result.forecast) == 7
    assert result.forecast[0]["predicted_sales"] == 1000.0

def test_metadata_cohort_routing(db_session):
    store = Store(id=2, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    # We mock the cohort builder to return a profile
    from unittest.mock import patch
    with patch("app.ml.forecast_router.get_metadata_cohort_profile", return_value={
        "mean_daily_sales": 500.0,
        "dow_multipliers": [1.0] * 7,
        "promo_uplift": 0.0
    }):
        result = route_forecast(2, 7, MIN_HISTORY_ROWS, db_session)
        
    assert result.forecast_method == ForecastMethod.METADATA_COHORT
    assert result.forecast_confidence == ForecastConfidence.LOW
    assert result.forecast[0]["predicted_sales"] == 500.0
