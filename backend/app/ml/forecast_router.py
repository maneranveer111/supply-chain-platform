"""
Forecast Router: resolves the correct forecast method and scaler for any
application store, implementing the Phase 2 routing logic.

Identity rules enforced here:
  - Store.id  is the application database PK.  It is NEVER used to index
    into store_scalers.pkl.
  - Store.benchmark_store_id is the ONLY field that may reference a Rossmann
    training store ID.  It must be non-None AND confirmed to exist in the
    pkl before the Rossmann scaler is used.
  - The store_scalers DB table (custom scalers) is checked when
    benchmark_store_id is None.

Routing decision tree:
  1. benchmark_store_id IS SET
       → load Rossmann scaler from pkl[benchmark_store_id]
       → verify >= MIN_HISTORY_ROWS feature rows exist
       → method="lstm", confidence="high"
       → if benchmark invalid or insufficient history: return insufficient_data

  2. benchmark_store_id IS NULL + custom DB scaler exists + >= MIN_HISTORY_ROWS
       → reconstruct scaler from mean_log/scale_log
       → method="lstm", confidence="high"

  3. benchmark_store_id IS NULL + no usable scaler or insufficient history
       → method="insufficient_data", confidence="none"
       → do NOT call the LSTM

No other outcomes are produced in Phase 2.
Cluster-based fallback is reserved for Phase 3.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sqlalchemy.orm import Session

from app.ml.cluster_forecast import get_metadata_cohort_profile, predict_from_profile
from app.ml.inference import (
    BenchmarkScalerNotFoundError,
    InsufficientHistoryError,
    MIN_HISTORY_ROWS,
    ModelNotLoadedError,
    build_forecast_dates,
    load_feature_cols,
    load_rossmann_scaler,
    predict_with_scaler,
)
from app.ml.profile_builder import resolve_store_cohort_key
from app.models.cluster import Cluster
from app.models.daily_store_feature import DailyStoreFeature
from app.models.store import Store
from app.models.store_scaler import StoreScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums — prevents typo-driven routing bugs
# ---------------------------------------------------------------------------

class ForecastMethod(str, Enum):
    LSTM = "lstm"
    CLUSTER_AVERAGE = "cluster_average"
    METADATA_COHORT = "metadata_cohort"
    INSUFFICIENT_DATA = "insufficient_data"


class ForecastConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


# ---------------------------------------------------------------------------
# Result dataclass — passed back to forecast_service
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    store_id: int
    horizon: int
    window_used: int
    forecast: list[dict] = field(default_factory=list)
    forecast_method: ForecastMethod = ForecastMethod.INSUFFICIENT_DATA
    forecast_confidence: ForecastConfidence = ForecastConfidence.NONE
    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "store_id": self.store_id,
            "horizon": self.horizon,
            "window_used": self.window_used,
            "forecast": self.forecast,
            "forecast_method": self.forecast_method.value,
            "forecast_confidence": self.forecast_confidence.value,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Custom application-level exceptions
# ---------------------------------------------------------------------------

class StoreNotFoundError(Exception):
    """Raised when a requested application store_id does not exist in the DB."""


# ---------------------------------------------------------------------------
# Helper: reconstruct a StandardScaler from DB parameters
# ---------------------------------------------------------------------------

def _reconstruct_custom_scaler(scaler_row: StoreScaler):
    """
    Reconstructs a sklearn StandardScaler from the mean_log and scale_log
    stored in the application store_scalers table.

    This exactly mirrors how StandardScaler.transform works:
        z = (x - mean) / scale
    And inverse_transform:
        x = z * scale + mean

    We do NOT import sklearn here to avoid a hard dependency; instead we
    build a minimal duck-type object that satisfies the inverse_transform
    contract used in predict_with_scaler.
    """
    from sklearn.preprocessing import StandardScaler  # available via scikit-learn requirement

    scaler = StandardScaler()
    scaler.mean_ = np.array([scaler_row.mean_log])
    scaler.scale_ = np.array([scaler_row.scale_log])
    scaler.var_ = np.array([scaler_row.scale_log ** 2])
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = scaler_row.n_samples
    return scaler


# ---------------------------------------------------------------------------
# Helper: load and validate history window for a store
# ---------------------------------------------------------------------------

def _load_history_window(
    app_store_id: int,
    feature_cols: list[str],
    forecast_window: int,
    db: Session,
) -> tuple[np.ndarray | None, int]:
    """
    Loads the most recent `forecast_window` rows of daily_store_features
    for the given application store_id.

    Returns:
        (matrix, row_count) where matrix is shape (window_size, num_features)
        if row_count >= MIN_HISTORY_ROWS, else (None, row_count).

    IMPORTANT: This function does NOT zero-pad for the LSTM path.
    Zero-padding is only acceptable for display/demo purposes, not for
    actual LSTM inference.  Callers must check row_count >= MIN_HISTORY_ROWS
    before using the matrix for LSTM prediction.
    """
    rows = (
        db.query(DailyStoreFeature)
        .filter(DailyStoreFeature.store_id == app_store_id)
        .order_by(DailyStoreFeature.date.desc())
        .limit(forecast_window)
        .all()
    )
    rows = list(reversed(rows))  # chronological order, oldest first
    row_count = len(rows)

    if row_count < MIN_HISTORY_ROWS:
        logger.info(
            "Store app_id=%d has %d feature rows (need >= %d for LSTM).",
            app_store_id, row_count, MIN_HISTORY_ROWS,
        )
        return None, row_count

    # Build the matrix using only the confirmed sufficient rows.
    # If row_count < forecast_window, we use row_count as the actual window.
    # The LSTM accepts variable-length input as long as it's >= 30.
    matrix = np.zeros((row_count, len(feature_cols)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(feature_cols):
            matrix[i, j] = row.features.get(col, 0.0)

    return matrix, row_count


# ---------------------------------------------------------------------------
# Main router function
# ---------------------------------------------------------------------------

def route_forecast(
    app_store_id: int,
    horizon: int,
    forecast_window: int,
    db: Session,
) -> ForecastResult:
    """
    Entry point for all forecast routing.

    Loads the application Store record, determines the correct routing path,
    resolves the scaler, validates history, and either executes the LSTM or
    returns an insufficient_data result.

    Args:
        app_store_id:    The application database Store.id (NOT a Rossmann ID).
        horizon:         Number of forecast days to produce.
        forecast_window: Maximum history rows to load (typically 30).
        db:              SQLAlchemy session.

    Returns:
        ForecastResult with all fields populated.

    Raises:
        StoreNotFoundError: if no Store row exists for app_store_id.
        ModelNotLoadedError: if lstm_model.keras is missing (propagated up
                             so the API can return 503 rather than 500).

    NOTE: This function never raises ValueError for routing conditions.
    All routing failures produce an insufficient_data ForecastResult.
    """
    from datetime import datetime

    generated_at = datetime.utcnow().isoformat()
    dates = build_forecast_dates(horizon)

    # --- Load the application store record ---------------------------------
    store: Store | None = db.query(Store).filter(Store.id == app_store_id).first()
    if store is None:
        raise StoreNotFoundError(
            f"No application store found with id={app_store_id}."
        )

    feature_cols = load_feature_cols()

    # -----------------------------------------------------------------------
    # ROUTING BRANCH A: benchmark_store_id is set
    # -----------------------------------------------------------------------
    if store.benchmark_store_id is not None:
        return _route_benchmark(
            store=store,
            horizon=horizon,
            forecast_window=forecast_window,
            feature_cols=feature_cols,
            dates=dates,
            generated_at=generated_at,
            db=db,
        )

    # -----------------------------------------------------------------------
    # ROUTING BRANCH B/C: benchmark_store_id is NULL
    # -----------------------------------------------------------------------
    return _route_no_benchmark(
        store=store,
        horizon=horizon,
        forecast_window=forecast_window,
        feature_cols=feature_cols,
        dates=dates,
        generated_at=generated_at,
        db=db,
    )


# ---------------------------------------------------------------------------
# Branch A: benchmark_store_id is set
# ---------------------------------------------------------------------------

def _route_benchmark(
    store: Store,
    horizon: int,
    forecast_window: int,
    feature_cols: list[str],
    dates: list[str],
    generated_at: str,
    db: Session,
) -> ForecastResult:
    """Handles stores with an explicit Rossmann benchmark mapping."""
    insufficient = ForecastResult(
        store_id=store.id,
        horizon=horizon,
        window_used=forecast_window,
        forecast=[],
        forecast_method=ForecastMethod.INSUFFICIENT_DATA,
        forecast_confidence=ForecastConfidence.NONE,
        generated_at=generated_at,
    )

    # Step 1: Load the Rossmann scaler for the verified benchmark ID.
    try:
        rossmann_scaler = load_rossmann_scaler(store.benchmark_store_id)
    except BenchmarkScalerNotFoundError:
        logger.warning(
            "Store id=%d has benchmark_store_id=%d but that ID is not in "
            "store_scalers.pkl. Returning insufficient_data.",
            store.id, store.benchmark_store_id,
        )
        return insufficient
    except ModelNotLoadedError:
        # Re-raise: this is a deployment problem, not a routing problem.
        raise

    # Step 2: Load history and verify sufficient rows.
    matrix, row_count = _load_history_window(store.id, feature_cols, forecast_window, db)
    if matrix is None:
        logger.info(
            "Store id=%d (benchmark=%d) has insufficient history (%d rows). "
            "Returning insufficient_data.",
            store.id, store.benchmark_store_id, row_count,
        )
        return insufficient

    # Step 3: Run LSTM with the Rossmann scaler.
    logger.info(
        "Routing store id=%d via LSTM with Rossmann benchmark scaler id=%d.",
        store.id, store.benchmark_store_id,
    )
    predictions = predict_with_scaler(matrix, rossmann_scaler, horizon)

    return ForecastResult(
        store_id=store.id,
        horizon=horizon,
        window_used=row_count,
        forecast=[
            {"day": i + 1, "date": dates[i], "predicted_sales": round(predictions[i], 2)}
            for i in range(horizon)
        ],
        forecast_method=ForecastMethod.LSTM,
        forecast_confidence=ForecastConfidence.HIGH,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# Branch B/C: benchmark_store_id is NULL
# ---------------------------------------------------------------------------

def _route_no_benchmark(
    store: Store,
    horizon: int,
    forecast_window: int,
    feature_cols: list[str],
    dates: list[str],
    generated_at: str,
    db: Session,
) -> ForecastResult:
    """
    Handles stores without a Rossmann benchmark.

    Branch B: custom scaler exists in store_scalers table + sufficient history.
    Branch C: no usable scaler or insufficient history → insufficient_data.
    """
    insufficient = ForecastResult(
        store_id=store.id,
        horizon=horizon,
        window_used=forecast_window,
        forecast=[],
        forecast_method=ForecastMethod.INSUFFICIENT_DATA,
        forecast_confidence=ForecastConfidence.NONE,
        generated_at=generated_at,
    )

    # Check if a custom scaler exists in DB
    custom_scaler_row: StoreScaler | None = (
        db.query(StoreScaler)
        .filter(StoreScaler.store_id == store.id)
        .first()
    )

    # If custom scaler exists and we have sufficient history, run LSTM.
    # Note: Phase 4 will fully implement custom scaler creation/fitting.
    if custom_scaler_row is not None:
        matrix, row_count = _load_history_window(store.id, feature_cols, forecast_window, db)
        if matrix is not None:
            custom_scaler = _reconstruct_custom_scaler(custom_scaler_row)
            logger.info("Routing store id=%d via custom LSTM.", store.id)
            try:
                predictions = predict_with_scaler(matrix, custom_scaler, horizon)
                return ForecastResult(
                    store_id=store.id,
                    horizon=horizon,
                    window_used=row_count,
                    forecast=[{"day": i + 1, "date": dates[i], "predicted_sales": round(predictions[i], 2)} for i in range(horizon)],
                    forecast_method=ForecastMethod.LSTM,
                    forecast_confidence=ForecastConfidence.MEDIUM,
                    generated_at=generated_at,
                )
            except InsufficientHistoryError:
                pass

    # Step 3: Phase 3 CLUSTER FALLBACK
    if store.cluster_id is not None:
        cluster = db.query(Cluster).filter(Cluster.id == store.cluster_id).first()
        if cluster and cluster.profile:
            logger.info("Routing store id=%d via Cluster profile.", store.id)
            # Basic promo assumption for cold start (False) - can be extended later to read future promo plans
            predictions = predict_from_profile(cluster.profile, dates, has_promo=[False]*horizon)
            return ForecastResult(
                store_id=store.id,
                horizon=horizon,
                window_used=0, # Profile based, not history based
                forecast=[{"day": i + 1, "date": dates[i], "predicted_sales": round(predictions[i], 2)} for i in range(horizon)],
                forecast_method=ForecastMethod.CLUSTER_AVERAGE,
                forecast_confidence=ForecastConfidence.LOW,
                generated_at=generated_at,
            )

    # Step 4: Phase 3 METADATA COHORT FALLBACK
    cohort_key = resolve_store_cohort_key(store)
    if cohort_key:
        cohort_profile = get_metadata_cohort_profile(cohort_key)
        if cohort_profile:
            logger.info("Routing store id=%d via Metadata Cohort (%s).", store.id, cohort_key)
            predictions = predict_from_profile(cohort_profile, dates, has_promo=[False]*horizon)
            return ForecastResult(
                store_id=store.id,
                horizon=horizon,
                window_used=0,
                forecast=[{"day": i + 1, "date": dates[i], "predicted_sales": round(predictions[i], 2)} for i in range(horizon)],
                forecast_method=ForecastMethod.METADATA_COHORT,
                forecast_confidence=ForecastConfidence.LOW,
                generated_at=generated_at,
            )

    # Step 5: INSUFFICIENT DATA
    logger.info("Routing store id=%d via Insufficient Data fallback.", store.id)
    return insufficient
