"""
Phase 2 forecast routing tests.

Tests all 7 required scenarios from the Phase 2 spec:

TEST 1: id=1, benchmark=NULL, no custom scaler, no history  → insufficient_data
         Critical: must NOT use Rossmann scaler #1.

TEST 2: id=1, benchmark=NULL, custom scaler exists, >=30 rows → lstm
         Custom scaler must come from DB, NOT store_scalers.pkl[1].

TEST 3: id=5000, benchmark=NULL, no custom scaler, no history → insufficient_data
         No ValueError, no HTTP 500, no scaler #5000 lookup.

TEST 4: id=9999, benchmark=valid Rossmann ID, >=30 rows → lstm
         Scaler must come from pkl[benchmark_store_id], NOT pkl[9999].

TEST 5: id=1, benchmark=NULL, no custom scaler, 0-29 rows → insufficient_data
         model.predict() must NOT be called.

TEST 6: id=ANY, benchmark=invalid ID (not in pkl) → insufficient_data
         No unhandled BenchmarkScalerNotFoundError.

TEST 7: PO engine receives insufficient_data → does not crash.

All tests use in-memory SQLite and mock the ML artifacts so no trained
model or pkl files are required to run the test suite.
"""

import types
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import Base

# Import models so Base.metadata is populated
from app.models import (  # noqa: F401
    cluster,
    daily_store_feature,
    forecast,
    inventory,
    purchase_order,
    store,
    store_sale,
    store_scaler,
    user,
)
from app.models.daily_store_feature import DailyStoreFeature
from app.models.store import Store
from app.models.store_scaler import StoreScaler
from app.ml.forecast_router import (
    ForecastConfidence,
    ForecastMethod,
    StoreNotFoundError,
    route_forecast,
)
from app.ml.inference import MIN_HISTORY_ROWS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    """In-memory SQLite session — does not touch the live Supabase DB."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


def _make_store(db, app_id: int, benchmark_store_id: int | None = None) -> Store:
    """Creates a Store with a specific application id."""
    s = Store(
        id=app_id,
        benchmark_store_id=benchmark_store_id,
        forecast_mode="auto",
    )
    db.add(s)
    db.flush()
    return s


def _add_feature_rows(db, app_store_id: int, n_rows: int) -> None:
    """Adds `n_rows` daily_store_features rows for a store."""
    today = date.today()
    for i in range(n_rows):
        day = today - timedelta(days=n_rows - i)
        db.add(DailyStoreFeature(
            store_id=app_store_id,
            date=day,
            features={"Sales_scaled": 0.5, "Promo": 0.0},
        ))
    db.flush()


def _make_fake_rossmann_scaler():
    """Minimal sklearn-compatible StandardScaler duck-type."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.mean_ = np.array([8.0])
    sc.scale_ = np.array([0.5])
    sc.var_ = np.array([0.25])
    sc.n_features_in_ = 1
    sc.n_samples_seen_ = 1000
    return sc


def _fake_model_predict(X, verbose=0):
    """Returns a deterministic 7-day flat forecast (scaled)."""
    horizon = 7
    return np.full((1, horizon), 0.1, dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared mock for _load_artifacts so no actual .keras / .pkl files are needed.
# We return:
#   model    — MagicMock with a predict() side effect
#   rossmann_scalers — dict mapping benchmark_id → scaler
#   comp_scaler — unused in tests
#   feature_cols — minimal list
# ---------------------------------------------------------------------------

FAKE_ROSSMANN_SCALERS = {
    1: _make_fake_rossmann_scaler(),
    2: _make_fake_rossmann_scaler(),
    100: _make_fake_rossmann_scaler(),
}

FAKE_FEATURE_COLS = ["Sales_scaled", "Promo"]  # minimal 2-feature set for tests

def _make_fake_artifacts():
    fake_model = MagicMock()
    fake_model.predict.side_effect = _fake_model_predict
    return fake_model, FAKE_ROSSMANN_SCALERS, MagicMock(), FAKE_FEATURE_COLS


LOAD_ARTIFACTS_PATH = "app.ml.inference._load_artifacts"
LOAD_FEATURE_COLS_PATH = "app.ml.inference.load_feature_cols"


# ---------------------------------------------------------------------------
# TEST 1: benchmark=NULL, no custom scaler, no history → insufficient_data
#          Critical: Rossmann scaler #1 must NOT be used.
# ---------------------------------------------------------------------------

def test_1_no_benchmark_no_scaler_no_history_returns_insufficient(db_session):
    """
    Store id=1, benchmark_store_id=NULL, no custom scaler, no feature rows.
    Must return insufficient_data without touching store_scalers.pkl[1].
    """
    _make_store(db_session, app_id=1, benchmark_store_id=None)

    with patch(LOAD_ARTIFACTS_PATH, return_value=_make_fake_artifacts()) as mock_artifacts, \
         patch(LOAD_FEATURE_COLS_PATH, return_value=FAKE_FEATURE_COLS):

        result = route_forecast(
            app_store_id=1,
            horizon=7,
            forecast_window=MIN_HISTORY_ROWS,
            db=db_session,
        )

    assert result.forecast_method == ForecastMethod.INSUFFICIENT_DATA
    assert result.forecast_confidence == ForecastConfidence.NONE
    assert result.forecast == []

    # _load_artifacts was NOT called (no scaler lookup, no model.predict)
    mock_artifacts.assert_not_called()


# ---------------------------------------------------------------------------
# TEST 2: benchmark=NULL, custom DB scaler, >=30 rows → lstm
#          Scaler source must be DB, NOT pkl[1].
# ---------------------------------------------------------------------------

def test_2_custom_db_scaler_with_sufficient_history_returns_lstm(db_session):
    """
    Store id=1, benchmark_store_id=NULL, custom DB scaler, >=30 rows.
    Must use the DB scaler, NOT store_scalers.pkl[1].
    """
    _make_store(db_session, app_id=1, benchmark_store_id=None)
    _add_feature_rows(db_session, app_store_id=1, n_rows=MIN_HISTORY_ROWS)

    # Insert a custom scaler row for this application store
    db_session.add(StoreScaler(
        store_id=1,
        mean_log=8.0,
        scale_log=0.5,
        n_samples=500,
        updated_at=__import__("datetime").datetime.utcnow(),
    ))
    db_session.flush()

    with patch(LOAD_ARTIFACTS_PATH, return_value=_make_fake_artifacts()), \
         patch(LOAD_FEATURE_COLS_PATH, return_value=FAKE_FEATURE_COLS):

        result = route_forecast(
            app_store_id=1,
            horizon=7,
            forecast_window=MIN_HISTORY_ROWS,
            db=db_session,
        )

    assert result.forecast_method == ForecastMethod.LSTM
    assert result.forecast_confidence == ForecastConfidence.MEDIUM
    assert len(result.forecast) == 7
    for day in result.forecast:
        assert day["predicted_sales"] > 0


# ---------------------------------------------------------------------------
# TEST 3: id=5000, benchmark=NULL, no scaler, no history → insufficient_data
#          No ValueError, no HTTP 500, no scaler #5000 lookup.
# ---------------------------------------------------------------------------

def test_3_high_app_id_no_benchmark_returns_insufficient(db_session):
    """
    Store id=5000, no benchmark, no scaler. Must NOT attempt pkl[5000].
    """
    _make_store(db_session, app_id=5000, benchmark_store_id=None)

    with patch(LOAD_ARTIFACTS_PATH, return_value=_make_fake_artifacts()) as mock_artifacts, \
         patch(LOAD_FEATURE_COLS_PATH, return_value=FAKE_FEATURE_COLS):

        result = route_forecast(
            app_store_id=5000,
            horizon=7,
            forecast_window=MIN_HISTORY_ROWS,
            db=db_session,
        )

    assert result.forecast_method == ForecastMethod.INSUFFICIENT_DATA
    assert result.forecast == []
    # No artifact loading should have happened
    mock_artifacts.assert_not_called()


# ---------------------------------------------------------------------------
# TEST 4: id=9999, benchmark=valid Rossmann ID (100), >=30 rows → lstm
#          Scaler from pkl[100], NOT pkl[9999].
# ---------------------------------------------------------------------------

def test_4_benchmark_set_uses_benchmark_scaler_not_app_id(db_session):
    """
    Store id=9999, benchmark_store_id=100, sufficient history.
    LSTM must use scaler from pkl[100], NOT from pkl[9999].
    """
    _make_store(db_session, app_id=9999, benchmark_store_id=100)
    _add_feature_rows(db_session, app_store_id=9999, n_rows=MIN_HISTORY_ROWS)

    fake_model, fake_scalers, fake_comp, fake_cols = _make_fake_artifacts()

    # Track which scaler key was accessed
    accessed_keys = []

    class TrackingDict(dict):
        def __contains__(self, key):
            return super().__contains__(key)
        def __getitem__(self, key):
            accessed_keys.append(key)
            return super().__getitem__(key)

    tracking_scalers = TrackingDict(fake_scalers)

    with patch(LOAD_ARTIFACTS_PATH, return_value=(fake_model, tracking_scalers, fake_comp, fake_cols)), \
         patch(LOAD_FEATURE_COLS_PATH, return_value=FAKE_FEATURE_COLS):

        result = route_forecast(
            app_store_id=9999,
            horizon=7,
            forecast_window=MIN_HISTORY_ROWS,
            db=db_session,
        )

    assert result.forecast_method == ForecastMethod.LSTM
    assert result.forecast_confidence == ForecastConfidence.HIGH
    assert len(result.forecast) == 7

    # The scaler must have been accessed by benchmark ID 100, not app ID 9999
    assert 9999 not in accessed_keys, (
        f"Rossmann scaler was accessed by application id 9999! accessed_keys={accessed_keys}"
    )
    assert 100 in accessed_keys, (
        f"Expected Rossmann scaler for benchmark_id=100 to be accessed. got={accessed_keys}"
    )


# ---------------------------------------------------------------------------
# TEST 5: id=1, benchmark=NULL, no scaler, insufficient history → insufficient_data
#          model.predict() must NOT be called.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_rows", [0, 1, MIN_HISTORY_ROWS - 1])
def test_5_insufficient_history_never_calls_model(db_session, n_rows):
    """
    With fewer than MIN_HISTORY_ROWS, model.predict must never be called.
    """
    _make_store(db_session, app_id=1, benchmark_store_id=None)
    if n_rows > 0:
        _add_feature_rows(db_session, app_store_id=1, n_rows=n_rows)

    fake_model, fake_scalers, fake_comp, fake_cols = _make_fake_artifacts()

    with patch(LOAD_ARTIFACTS_PATH, return_value=(fake_model, fake_scalers, fake_comp, fake_cols)), \
         patch(LOAD_FEATURE_COLS_PATH, return_value=FAKE_FEATURE_COLS):

        result = route_forecast(
            app_store_id=1,
            horizon=7,
            forecast_window=MIN_HISTORY_ROWS,
            db=db_session,
        )

    assert result.forecast_method == ForecastMethod.INSUFFICIENT_DATA
    fake_model.predict.assert_not_called()


# ---------------------------------------------------------------------------
# TEST 6: Invalid benchmark_store_id (not in pkl) → insufficient_data
#          No unhandled BenchmarkScalerNotFoundError.
# ---------------------------------------------------------------------------

def test_6_invalid_benchmark_id_returns_insufficient_not_exception(db_session):
    """
    Store has benchmark_store_id=9876 (not in store_scalers.pkl).
    Must NOT raise BenchmarkScalerNotFoundError; must return insufficient_data.
    """
    _make_store(db_session, app_id=1, benchmark_store_id=9876)
    _add_feature_rows(db_session, app_store_id=1, n_rows=MIN_HISTORY_ROWS)

    # fake_scalers only has keys 1, 2, 100 — 9876 is absent
    fake_model, fake_scalers, fake_comp, fake_cols = _make_fake_artifacts()

    with patch(LOAD_ARTIFACTS_PATH, return_value=(fake_model, fake_scalers, fake_comp, fake_cols)), \
         patch(LOAD_FEATURE_COLS_PATH, return_value=FAKE_FEATURE_COLS):

        result = route_forecast(
            app_store_id=1,
            horizon=7,
            forecast_window=MIN_HISTORY_ROWS,
            db=db_session,
        )

    assert result.forecast_method == ForecastMethod.INSUFFICIENT_DATA
    assert result.forecast == []
    fake_model.predict.assert_not_called()


# ---------------------------------------------------------------------------
# TEST 7: PO engine with insufficient_data forecast → no crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_7_po_engine_handles_insufficient_data_without_crash(db_session):
    """
    When get_forecast returns insufficient_data, the PO engine must not crash.
    Must return a PurchaseOrderRecommendation with forecasted_demand=-1.
    """
    from app.models.inventory import Inventory
    from app.models.purchase_order import PurchaseOrder as POModel
    from app.services.po_engine import get_recommendations, FORECAST_UNAVAILABLE_SENTINEL

    # Create a store with no benchmark/scaler
    _make_store(db_session, app_id=42, benchmark_store_id=None)
    db_session.add(Inventory(
        store_id=42,
        current_quantity=1000.0,
        updated_at=__import__("datetime").datetime.utcnow(),
    ))
    db_session.flush()

    # Mock get_forecast to return insufficient_data without hitting Redis/DB
    insufficient_forecast = {
        "store_id": 42,
        "horizon": 3,
        "window_used": 30,
        "forecast": [],
        "forecast_method": "insufficient_data",
        "forecast_confidence": "none",
        "generated_at": "2026-10-07T00:00:00",
    }

    with patch("app.services.po_engine.get_forecast", return_value=insufficient_forecast):
        results = await get_recommendations(store_id=42, db=db_session)

    assert len(results) == 1
    rec = results[0]
    assert rec.store_id == 42
    assert rec.forecasted_demand == FORECAST_UNAVAILABLE_SENTINEL
    assert rec.recommended_qty == FORECAST_UNAVAILABLE_SENTINEL
    assert "unavailable" in rec.reason.lower()

    # Verify a PO row was written with pending_forecast status
    po_row = db_session.query(POModel).filter(POModel.store_id == 42).first()
    assert po_row is not None
    assert po_row.status == "pending_forecast"


# ---------------------------------------------------------------------------
# BONUS: store not found raises StoreNotFoundError (not a 500)
# ---------------------------------------------------------------------------

def test_store_not_found_raises_specific_error(db_session):
    """
    Requesting a forecast for a non-existent application store must raise
    StoreNotFoundError, not a generic ValueError or AttributeError.
    """
    with pytest.raises(StoreNotFoundError):
        route_forecast(
            app_store_id=99999,
            horizon=7,
            forecast_window=MIN_HISTORY_ROWS,
            db=db_session,
        )


# ---------------------------------------------------------------------------
# BONUS: benchmark store returns insufficient_data when history is missing
#         even if a valid Rossmann scaler exists for that benchmark_store_id
# ---------------------------------------------------------------------------

def test_benchmark_with_insufficient_history_returns_insufficient(db_session):
    """
    Store has a valid benchmark_store_id=1 (scaler exists in pkl) but only
    15 rows of history.  Must return insufficient_data, not run LSTM.
    """
    _make_store(db_session, app_id=7, benchmark_store_id=1)
    _add_feature_rows(db_session, app_store_id=7, n_rows=15)  # < MIN_HISTORY_ROWS

    fake_model, fake_scalers, fake_comp, fake_cols = _make_fake_artifacts()

    with patch(LOAD_ARTIFACTS_PATH, return_value=(fake_model, fake_scalers, fake_comp, fake_cols)), \
         patch(LOAD_FEATURE_COLS_PATH, return_value=FAKE_FEATURE_COLS):

        result = route_forecast(
            app_store_id=7,
            horizon=7,
            forecast_window=MIN_HISTORY_ROWS,
            db=db_session,
        )

    assert result.forecast_method == ForecastMethod.INSUFFICIENT_DATA
    fake_model.predict.assert_not_called()
