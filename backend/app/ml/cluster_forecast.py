"""
Cluster & Cohort Forecasting logic.

This module implements the non-LSTM cold-start paths:
1. cluster_average: Uses a KMeans cluster's persisted profile
2. metadata_cohort: Uses a profile matched on StoreType/Assortment

It calculates predictions using:
  base_daily_sales * dow_multiplier * (1 + promo_uplift)

It does NOT run the LSTM, use scalers, or pad missing history.
"""

import logging
from datetime import datetime
from functools import lru_cache

from app.ml.profile_builder import build_metadata_cohort_profiles

logger = logging.getLogger(__name__)


# Cache cohort profiles globally so they are not rebuilt per forecast request.
# In a real system this could use Redis or refresh periodically.
@lru_cache(maxsize=1)
def _get_metadata_profiles() -> dict:
    logger.info("Loading metadata cohort profiles into memory cache.")
    return build_metadata_cohort_profiles()


def predict_from_profile(
    profile: dict,
    dates: list[str],
    has_promo: list[bool] | None = None,
) -> list[float]:
    """
    Generates a horizon-length list of sales predictions using a
    pre-calculated behavior profile (cluster or cohort).

    Args:
        profile: Profile dictionary (from DB or memory) containing:
                 - mean_daily_sales
                 - dow_multipliers (list of 7 floats, Mon=0..Sun=6)
                 - promo_uplift (fractional)
        dates: List of string dates (YYYY-MM-DD) for the horizon
        has_promo: Optional list of booleans indicating if a promo is active.
                   Defaults to False if None or if too short.

    Returns:
        List of non-negative float predictions, matching len(dates).
    """
    mean_sales = profile.get("mean_daily_sales", 0.0)
    dow_mults = profile.get("dow_multipliers", [1.0] * 7)
    promo_uplift = profile.get("promo_uplift", 0.0)

    # Validate profile to avoid crashes
    if not isinstance(dow_mults, list) or len(dow_mults) != 7:
        logger.warning("Invalid dow_multipliers in profile. Falling back to 1.0.")
        dow_mults = [1.0] * 7
    
    # Ensure minimum zero sales
    mean_sales = max(0.0, mean_sales)

    predictions = []
    for i, date_str in enumerate(dates):
        # Convert date string to python date to get DayOfWeek (0=Mon..6=Sun)
        dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        dow = dt.weekday()
        
        mult = float(dow_mults[dow])
        
        promo_active = False
        if has_promo and i < len(has_promo):
            promo_active = has_promo[i]
            
        uplift = promo_uplift if promo_active else 0.0
        
        # Calculate daily prediction
        pred = mean_sales * mult * (1.0 + uplift)
        predictions.append(max(0.0, float(pred)))
        
    return predictions


def get_metadata_cohort_profile(cohort_key: str) -> dict | None:
    """
    Fetches a cached metadata cohort profile by key (e.g. 'StoreType_a|Assortment_c').
    """
    profiles = _get_metadata_profiles()
    return profiles.get(cohort_key)
