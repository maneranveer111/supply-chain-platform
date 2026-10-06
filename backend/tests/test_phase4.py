import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import pandas as pd
import numpy as np
from datetime import date
from io import BytesIO
from unittest.mock import patch, MagicMock

from app.main import app
from app.db.session import Base, get_db
from app.core.dependencies import get_current_user, CurrentUser, Role
from app.models.store import Store
from app.models.store_sale import StoreSale
from app.models.store_scaler import StoreScaler
from app.models.daily_store_feature import DailyStoreFeature
from app.models.cluster import Cluster
from app.ml.forecast_router import route_forecast, ForecastMethod, ForecastConfidence
from app.ml.inference import MIN_HISTORY_ROWS, load_feature_cols

from tests.test_phase3 import (
    engine,
    TestingSessionLocal,
    override_get_db,
    setup_db,
    db_session,
    override_get_current_user_normal,
    override_get_current_user_admin,
)

@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    yield TestClient(app)

@pytest.fixture
def admin_client(db_session):
    app.dependency_overrides[get_current_user] = override_get_current_user_admin
    yield TestClient(app)

def generate_csv_bytes(n_days, start_date="2020-01-01"):
    df = pd.DataFrame({
        "date": pd.date_range(start_date, periods=n_days),
        "sales": np.random.randint(1000, 5000, size=n_days),
        "promo": np.random.choice([0, 1], size=n_days),
        "school_holiday": 0,
        "state_holiday": "0"
    })
    return df.to_csv(index=False).encode('utf-8')

# ===========================================================================
# 1. Store Management Tests
# ===========================================================================

def test_create_store(client):
    req = {
        "name": "Test Store",
        "store_type": "a",
        "assortment": "a",
        "competition_distance": 1000,
        "promo2_active": False,
        "user_id": 999,  # Attempts to spoof user_id
        "benchmark_store_id": 1  # Attempts to set benchmark
    }
    resp = client.post("/api/v1/stores", json=req)
    assert resp.status_code == 201
    data = resp.json()
    assert data["user_id"] == 10  # Must be current user, not spoofed 999
    assert data["benchmark_store_id"] is None  # Normal user cannot set benchmark

def test_list_stores(client, db_session):
    store1 = Store(id=1, name="s1", user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    store2 = Store(id=2, name="s2", user_id=11, store_type="b", assortment="b", forecast_mode="auto")
    demo_store = Store(id=3, name="demo", user_id=None, store_type="c", assortment="c", forecast_mode="auto")
    db_session.add_all([store1, store2, demo_store])
    db_session.commit()
    
    resp = client.get("/api/v1/stores")
    assert resp.status_code == 200
    stores = resp.json()
    assert len(stores) == 1
    assert stores[0]["id"] == 1

def test_admin_can_list_all_stores(admin_client, db_session):
    store1 = Store(id=1, name="s1", user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    store2 = Store(id=2, name="s2", user_id=11, store_type="b", assortment="b", forecast_mode="auto")
    demo_store = Store(id=3, name="demo", user_id=None, store_type="c", assortment="c", forecast_mode="auto")
    db_session.add_all([store1, store2, demo_store])
    db_session.commit()
    
    resp = admin_client.get("/api/v1/stores")
    assert resp.status_code == 200
    assert len(resp.json()) == 3

def test_get_store_detail(client, db_session):
    store = Store(id=1, name="s1", user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    resp = client.get("/api/v1/stores/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 1
    assert data["name"] == "s1"
    assert data["forecast_status"] == "cold_start"

def test_get_store_unauthorized_and_not_found(client, db_session):
    store = Store(id=2, name="other", user_id=11, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    # Other user store
    resp = client.get("/api/v1/stores/2")
    assert resp.status_code == 403
    
    # Not found
    resp = client.get("/api/v1/stores/999")
    assert resp.status_code == 404

# ===========================================================================
# 2. CSV Validation Tests
# ===========================================================================

def test_csv_upload_valid_60_days(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    csv_bytes = generate_csv_bytes(65)
    files = {"file": ("data.csv", BytesIO(csv_bytes), "text/csv")}
    
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 200
    
    scaler = db_session.query(StoreScaler).filter_by(store_id=1).first()
    assert scaler is not None
    assert scaler.n_samples == 65
    
    features = db_session.query(DailyStoreFeature).filter_by(store_id=1).all()
    assert len(features) == 65
    assert len(features[0].features.keys()) == 22
    assert db_session.query(StoreSale).filter_by(store_id=1).count() == 65

def test_csv_upload_empty_file(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    files = {"file": ("data.csv", BytesIO(b""), "text/csv")}
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 400

def test_csv_upload_missing_column(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    df = pd.DataFrame({"date": ["2020-01-01"], "sales": [100]})
    files = {"file": ("data.csv", BytesIO(df.to_csv(index=False).encode('utf-8')), "text/csv")}
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 400
    assert "Missing required columns" in resp.json()["detail"]

def test_csv_upload_invalid_date(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    df = pd.DataFrame({
        "date": ["not-a-date"],
        "sales": [100],
        "promo": [0],
        "school_holiday": [0],
        "state_holiday": ["0"]
    })
    files = {"file": ("data.csv", BytesIO(df.to_csv(index=False).encode('utf-8')), "text/csv")}
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 400

def test_csv_upload_negative_sales(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5),
        "sales": [-100, 200, 300, 400, 500],
        "promo": 0,
        "school_holiday": 0,
        "state_holiday": "0"
    })
    files = {"file": ("data.csv", BytesIO(df.to_csv(index=False).encode('utf-8')), "text/csv")}
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 400
    assert "Sales cannot be negative" in resp.json()["detail"]

def test_csv_upload_invalid_promo(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    df = pd.DataFrame({
        "date": ["2020-01-01"],
        "sales": [100],
        "promo": [5],
        "school_holiday": [0],
        "state_holiday": ["0"]
    })
    files = {"file": ("data.csv", BytesIO(df.to_csv(index=False).encode('utf-8')), "text/csv")}
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 400

def test_csv_duplicate_dates(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01"],
        "sales": [1000, 2000],
        "promo": [0, 0],
        "school_holiday": [0, 0],
        "state_holiday": ["0", "0"]
    })
    files = {"file": ("data.csv", BytesIO(df.to_csv(index=False).encode('utf-8')), "text/csv")}
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 400
    assert "Duplicate dates" in resp.json()["detail"]

def test_csv_non_contiguous_dates(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-03"],  # Missing Jan 2
        "sales": [1000, 2000],
        "promo": [0, 0],
        "school_holiday": [0, 0],
        "state_holiday": ["0", "0"]
    })
    files = {"file": ("data.csv", BytesIO(df.to_csv(index=False).encode('utf-8')), "text/csv")}
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 400
    assert "not contiguous" in resp.json()["detail"]

def test_csv_non_csv_extension(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    files = {"file": ("data.txt", BytesIO(b"abc"), "text/plain")}
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 400

def test_csv_upload_unauthorized_store(client, db_session):
    store = Store(id=2, user_id=11, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    csv_bytes = generate_csv_bytes(10)
    files = {"file": ("data.csv", BytesIO(csv_bytes), "text/csv")}
    resp = client.post("/api/v1/stores/2/upload-sales", files=files)
    assert resp.status_code == 403

# ===========================================================================
# 3. Custom Scaler Mathematical Accuracy Tests
# ===========================================================================

def test_custom_scaler_matches_sklearn_standard_scaler(client, db_session):
    from sklearn.preprocessing import StandardScaler

    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    n_days = 65
    raw_sales = np.array([1000 + i * 50 for i in range(n_days)])
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n_days),
        "sales": raw_sales,
        "promo": 0,
        "school_holiday": 0,
        "state_holiday": "0"
    })
    files = {"file": ("data.csv", BytesIO(df.to_csv(index=False).encode('utf-8')), "text/csv")}
    
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 200
    
    scaler_row = db_session.query(StoreScaler).filter_by(store_id=1).first()
    assert scaler_row is not None
    
    sk_scaler = StandardScaler()
    sk_scaler.fit(np.log1p(raw_sales).reshape(-1, 1))
    
    assert np.isclose(scaler_row.mean_log, sk_scaler.mean_[0], atol=1e-5)
    assert np.isclose(scaler_row.scale_log, sk_scaler.scale_[0], atol=1e-5)
    assert scaler_row.n_samples == n_days

# ===========================================================================
# 4. Cluster Assignment (14-59 days) & Routing Tests
# ===========================================================================

def test_csv_upload_14_days_cluster(client, db_session):
    cluster = Cluster(id=1, label="test", profile={"mean_daily_sales": 3000.0, "promo_uplift": 0.1})
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add_all([cluster, store])
    db_session.commit()
    
    csv_bytes = generate_csv_bytes(20)
    files = {"file": ("data.csv", BytesIO(csv_bytes), "text/csv")}
    
    resp = client.post("/api/v1/stores/1/upload-sales", files=files)
    assert resp.status_code == 200
    
    db_session.refresh(store)
    assert store.cluster_id == 1
    assert db_session.query(StoreScaler).filter_by(store_id=1).first() is None

def test_repeated_upload_replaces_and_updates(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    db_session.add(store)
    db_session.commit()
    
    csv1 = generate_csv_bytes(20)
    files1 = {"file": ("data.csv", BytesIO(csv1), "text/csv")}
    resp1 = client.post("/api/v1/stores/1/upload-sales", files=files1)
    assert resp1.status_code == 200
    assert db_session.query(StoreSale).filter_by(store_id=1).count() == 20
    
    # Re-upload with 65 days
    csv2 = generate_csv_bytes(65)
    files2 = {"file": ("data.csv", BytesIO(csv2), "text/csv")}
    resp2 = client.post("/api/v1/stores/1/upload-sales", files=files2)
    assert resp2.status_code == 200
    assert db_session.query(StoreSale).filter_by(store_id=1).count() == 65
    assert db_session.query(DailyStoreFeature).filter_by(store_id=1).count() == 65
    assert db_session.query(StoreScaler).filter_by(store_id=1).count() == 1

def test_get_store_status_lstm_ready(client, db_session):
    store = Store(id=1, user_id=10, store_type="a", assortment="a", forecast_mode="auto")
    scaler = StoreScaler(store_id=1, mean_log=5.0, scale_log=1.0, n_samples=60)
    db_session.add_all([store, scaler])
    
    for i in range(30):
        db_session.add(DailyStoreFeature(store_id=1, date=date(2020, 1, i+1), features={}))
        
    db_session.commit()
    
    resp = client.get("/api/v1/stores/1")
    assert resp.status_code == 200
    assert resp.json()["forecast_status"] == "lstm_ready"
    assert resp.json()["forecast_method"] == "lstm"
    assert resp.json()["has_custom_scaler"] is True
