"""
ML inference: loads the Rossmann LSTM artifacts and serves predictions.

Key design invariant (Phase 2):
---------------------------------
The PUBLIC entry point `predict_with_scaler(recent_window, scaler, horizon)`
accepts an ALREADY-RESOLVED scaler object.  This function knows nothing
about application store IDs or Rossmann benchmark IDs.

The PRIVATE helper `load_rossmann_scaler(benchmark_store_id)` exists solely
to reconstruct a StandardScaler from the Rossmann store_scalers.pkl for a
VERIFIED benchmark_store_id.  It is called ONLY by the forecast router
(forecast_service.py), never by the inference layer itself.

This separation guarantees that the application DB store identity (Store.id)
can NEVER accidentally become a lookup key into store_scalers.pkl.

Artifact files expected in saved_models/:
    lstm_model.keras
    store_scalers.pkl
    comp_scaler.pkl
    feature_cols.pkl
"""

import os
import pickle
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from app.core.config import settings

MODEL_PATH = os.path.join(settings.model_dir, "lstm_model.keras")
STORE_SCALERS_PATH = os.path.join(settings.model_dir, "store_scalers.pkl")
COMP_SCALER_PATH = os.path.join(settings.model_dir, "comp_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(settings.model_dir, "feature_cols.pkl")

# Minimum number of timesteps the LSTM requires for a valid forecast.
# This is a hard constraint derived from the training window (30 days).
# Do NOT run the model with fewer rows; use a fallback instead.
MIN_HISTORY_ROWS = 30


# ---------------------------------------------------------------------------
# Custom exceptions — distinguishable from each other and from Python errors
# ---------------------------------------------------------------------------

class ModelNotLoadedError(Exception):
    """Raised when the Keras model file is absent from saved_models/."""


class BenchmarkScalerNotFoundError(Exception):
    """Raised when a requested Rossmann benchmark_store_id does not exist
    in store_scalers.pkl.  Callers should treat this as an invalid mapping,
    not a generic ValueError."""


class InsufficientHistoryError(Exception):
    """Raised when fewer than MIN_HISTORY_ROWS valid feature rows exist.
    The LSTM must never be called in this state."""


# ---------------------------------------------------------------------------
# Fallback feature list — used only when feature_cols.pkl is absent
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_COLS = [
    "Promo", "SchoolHoliday", "IsPromo2Active", "CompetitionDistance_scaled",
    "DayOfWeek_sin", "DayOfWeek_cos", "Sales_scaled",
    "Sales_lag_7", "Sales_roll_mean_7", "Sales_roll_std_7", "Sales_roll_mean_30",
    "StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d",
    "Assortment_a", "Assortment_b", "Assortment_c",
    "StateHoliday_0", "StateHoliday_a",
]


def load_feature_cols() -> list[str]:
    """
    Returns the ordered feature-column list the model expects.
    Reads feature_cols.pkl directly (does not require the Keras model
    itself to be present), so the window-building logic can be tested
    independently of whether a trained model has been deployed.
    """
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, "rb") as f:
            return pickle.load(f)
    return DEFAULT_FEATURE_COLS


# ---------------------------------------------------------------------------
# Artifact loader — cached per process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_artifacts():
    """
    Loads model + all artifact files into memory once per process.
    Raises ModelNotLoadedError if lstm_model.keras is missing.
    """
    if not os.path.exists(MODEL_PATH):
        raise ModelNotLoadedError(
            f"No model found at {MODEL_PATH}. "
            "Train the model (see the notebook) and copy artifacts into saved_models/."
        )

    import tensorflow as tf  # lazy import — app boots without TF if model is absent

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(STORE_SCALERS_PATH, "rb") as f:
        rossmann_scalers = pickle.load(f)
    with open(COMP_SCALER_PATH, "rb") as f:
        comp_scaler = pickle.load(f)
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    return model, rossmann_scalers, comp_scaler, feature_cols


# ---------------------------------------------------------------------------
# Rossmann benchmark scaler lookup
# ---------------------------------------------------------------------------

def load_rossmann_scaler(benchmark_store_id: int):
    """
    Returns the StandardScaler for the given Rossmann training-store ID
    from store_scalers.pkl.

    This is the ONLY function that touches store_scalers.pkl.
    It must ONLY be called when benchmark_store_id is non-None and has
    been explicitly set on a Store record.

    Raises:
        ModelNotLoadedError: if artifacts are not available.
        BenchmarkScalerNotFoundError: if benchmark_store_id is not in the pkl.
    """
    _model, rossmann_scalers, _comp_scaler, _feature_cols = _load_artifacts()

    if benchmark_store_id not in rossmann_scalers:
        raise BenchmarkScalerNotFoundError(
            f"benchmark_store_id={benchmark_store_id} not found in store_scalers.pkl. "
            f"Valid range is approximately 1–1115."
        )

    return rossmann_scalers[benchmark_store_id]


# ---------------------------------------------------------------------------
# Core prediction function — scaler-agnostic
# ---------------------------------------------------------------------------

def predict_with_scaler(
    recent_window: np.ndarray,
    scaler,
    horizon: int = 7,
) -> list[float]:
    """
    Runs the LSTM on a pre-built feature window using an ALREADY-RESOLVED
    scaler.  The caller is responsible for:
      1. Verifying len(recent_window) >= MIN_HISTORY_ROWS.
      2. Resolving the correct scaler (Rossmann benchmark or custom).
      3. Ensuring recent_window has shape (window_size, num_features).

    This function does NOT look up scalers by any store ID.

    Returns: list of `horizon` predicted sales values in REAL units
             (inverse-transformed through scaler + expm1).

    Raises:
        ModelNotLoadedError: if artifacts are absent.
        InsufficientHistoryError: if window has fewer than MIN_HISTORY_ROWS rows
                                   (safety guard — callers should check first).
    """
    if len(recent_window) < MIN_HISTORY_ROWS:
        raise InsufficientHistoryError(
            f"Window has {len(recent_window)} rows; minimum required is {MIN_HISTORY_ROWS}."
        )

    model, _rossmann_scalers, _comp_scaler, _feature_cols = _load_artifacts()

    X = recent_window[np.newaxis, ...]          # (1, window_size, num_features)
    pred_scaled = model.predict(X, verbose=0)[0]  # (horizon,)

    pred_log = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    pred_real = np.expm1(pred_log)

    return pred_real[:horizon].tolist()


# ---------------------------------------------------------------------------
# Utility — kept for backward compatibility with other code paths
# ---------------------------------------------------------------------------

def build_forecast_dates(horizon: int, start_date: datetime | None = None) -> list[str]:
    start = start_date or datetime.utcnow()
    return [(start + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(horizon)]
