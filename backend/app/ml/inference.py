"""
Loads the artifacts produced by the Rossmann LSTM/Prophet notebook
(lstm_model.keras, store_scalers.pkl, comp_scaler.pkl, feature_cols.pkl)
and serves predictions for a single store.

Drop your trained artifacts into `saved_models/` with these exact names
before running the API in a mode that needs real predictions:
    saved_models/lstm_model.keras
    saved_models/store_scalers.pkl
    saved_models/comp_scaler.pkl
    saved_models/feature_cols.pkl
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


class ModelNotLoadedError(Exception):
    pass


# Fallback feature order used only if feature_cols.pkl hasn't been copied
# into saved_models/ yet -- lets the DB-backed window builder run before
# a trained model exists. Once you copy feature_cols.pkl from the
# notebook, that file's order is used instead (it's the real contract
# the model was trained on).
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
    Reads feature_cols.pkl directly (doesn't require the Keras model
    itself to be present), so the window-building logic can run and be
    tested independently of whether a trained model has been deployed.
    """
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, "rb") as f:
            return pickle.load(f)
    return DEFAULT_FEATURE_COLS


@lru_cache(maxsize=1)
def _load_artifacts():
    """
    Cached so the model and scalers are loaded into memory once per
    process, not once per request. lru_cache(maxsize=1) with no args
    acts as a simple singleton loader.
    """
    if not os.path.exists(MODEL_PATH):
        raise ModelNotLoadedError(
            f"No model found at {MODEL_PATH}. "
            "Train the model (see the notebook) and copy artifacts into saved_models/."
        )

    import tensorflow as tf  # imported lazily so the app can boot without TF installed issues

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(STORE_SCALERS_PATH, "rb") as f:
        store_scalers = pickle.load(f)
    with open(COMP_SCALER_PATH, "rb") as f:
        comp_scaler = pickle.load(f)
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    return model, store_scalers, comp_scaler, feature_cols


def predict_for_store(store_id: int, recent_window: np.ndarray, horizon: int = 7) -> list[float]:
    """
    recent_window: shape (window_size, num_features) — the most recent
    `window_size` days of engineered features for this store, in the
    same column order as feature_cols.pkl. In production this comes
    from a feature store or a query against recent daily records; for
    now, the caller (forecast_service) is responsible for building it.

    Returns: list of `horizon` predicted sales values in REAL units
    (already inverse-transformed through this store's scaler + log1p).
    """
    model, store_scalers, _comp_scaler, _feature_cols = _load_artifacts()

    if store_id not in store_scalers:
        raise ValueError(f"No scaler found for store_id={store_id}")

    X = recent_window[np.newaxis, ...]  # add batch dimension -> (1, window, features)
    pred_scaled = model.predict(X, verbose=0)[0]  # shape (horizon,)

    scaler = store_scalers[store_id]
    pred_log = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    pred_real = np.expm1(pred_log)

    return pred_real[:horizon].tolist()


def build_forecast_dates(horizon: int, start_date: datetime | None = None) -> list[str]:
    start = start_date or datetime.utcnow()
    return [(start + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(horizon)]
