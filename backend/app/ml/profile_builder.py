"""
Profile Builder: constructs cluster and metadata-cohort profiles from
the actual Rossmann training data (processed_df.parquet + store_scalers.pkl).

All numeric values are derived from real data — nothing is invented.

Two profile types:

1. CLUSTER PROFILE
   Built per KMeans cluster after clustering is complete.
   Contains: mean_daily_sales, dow_multipliers, promo_uplift, n_stores, n_samples.

2. METADATA COHORT PROFILE
   Built per (store_type, assortment) combination.
   Same structure as cluster profile.
   Used for zero-history stores that have metadata but no sales data.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

PARQUET_PATH = os.path.join(settings.model_dir, "processed_df.parquet")
STORE_SCALERS_PATH = os.path.join(settings.model_dir, "store_scalers.pkl")


def _load_training_data() -> tuple[pd.DataFrame, dict] | None:
    """Loads parquet + store scalers. Returns None if either is missing."""
    if not os.path.exists(PARQUET_PATH):
        logger.warning("Training parquet not found at %s", PARQUET_PATH)
        return None
    if not os.path.exists(STORE_SCALERS_PATH):
        logger.warning("Store scalers not found at %s", STORE_SCALERS_PATH)
        return None

    df = pd.read_parquet(PARQUET_PATH)
    with open(STORE_SCALERS_PATH, "rb") as f:
        scalers = pickle.load(f)
    return df, scalers


def _reverse_transform_sales(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """
    Adds a 'Sales_real' column by inverse-transforming Sales_scaled
    through each store's scaler: real = expm1(scaler.inverse_transform(scaled)).

    Stores without a scaler in the pkl are dropped.
    """
    result_frames = []
    for store_id, group in df.groupby("Store"):
        if store_id not in scalers:
            continue
        sc = scalers[store_id]
        sales_log = sc.inverse_transform(
            group["Sales_scaled"].values.reshape(-1, 1)
        ).flatten()
        g = group.copy()
        g["Sales_real"] = np.expm1(sales_log)
        result_frames.append(g)

    if not result_frames:
        return pd.DataFrame()
    return pd.concat(result_frames, ignore_index=True)


def _reconstruct_dow(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs integer DayOfWeek (Mon=1..Sun=7) from sin/cos encoding."""
    df = df.copy()
    df["DayOfWeek"] = (
        np.round(
            np.arctan2(df["DayOfWeek_sin"], df["DayOfWeek_cos"]) * 7 / (2 * np.pi)
        )
        % 7
    ).astype(int)
    return df


def _build_profile_from_subset(subset: pd.DataFrame) -> dict:
    """
    Builds a single profile dict from a subset of rows that already have
    Sales_real and DayOfWeek columns.

    Returns:
        {
            "mean_daily_sales": float,
            "dow_multipliers": [float x 7],  # index 0=Mon(DOW=1), index 6=Sun(DOW=0 or 7)
            "promo_uplift": float,
            "n_stores": int,
            "n_samples": int,
        }
    """
    if subset.empty:
        return {
            "mean_daily_sales": 0.0,
            "dow_multipliers": [1.0] * 7,
            "promo_uplift": 0.0,
            "n_stores": 0,
            "n_samples": 0,
        }

    mean_sales = float(subset["Sales_real"].mean())
    n_stores = int(subset["Store"].nunique())
    n_samples = len(subset)

    # Day-of-week multipliers (DOW 0..6 → Mon..Sun mapping)
    dow_means = subset.groupby("DayOfWeek")["Sales_real"].mean()
    dow_multipliers = []
    for d in range(7):
        if d in dow_means.index and mean_sales > 0:
            dow_multipliers.append(round(float(dow_means[d] / mean_sales), 4))
        else:
            dow_multipliers.append(1.0)

    # Promo uplift
    promo_sales = subset.loc[subset["Promo"] == 1, "Sales_real"]
    no_promo_sales = subset.loc[subset["Promo"] == 0, "Sales_real"]
    if len(no_promo_sales) > 0 and no_promo_sales.mean() > 0:
        promo_uplift = float(
            (promo_sales.mean() - no_promo_sales.mean()) / no_promo_sales.mean()
        )
    else:
        promo_uplift = 0.0

    return {
        "mean_daily_sales": round(mean_sales, 2),
        "dow_multipliers": dow_multipliers,
        "promo_uplift": round(promo_uplift, 4),
        "n_stores": n_stores,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Public API: build metadata cohort profiles
# ---------------------------------------------------------------------------

# Store type columns used to partition into cohorts
STORE_TYPE_COLS = [
    "StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d",
]
ASSORTMENT_COLS = [
    "Assortment_a", "Assortment_b", "Assortment_c",
]
COHORT_COLS = STORE_TYPE_COLS + ASSORTMENT_COLS


def build_metadata_cohort_profiles() -> dict[str, dict]:
    """
    Builds forecast profiles keyed by metadata cohort.

    Returns:
        {
            "StoreType_a|Assortment_a": { profile dict },
            "StoreType_a|Assortment_c": { profile dict },
            ...
        }

    Returns empty dict if training data is unavailable.
    """
    loaded = _load_training_data()
    if loaded is None:
        return {}

    df, scalers = loaded
    df = _reverse_transform_sales(df, scalers)
    if df.empty:
        return {}
    df = _reconstruct_dow(df)

    # Determine store types per store
    store_types = df.groupby("Store")[COHORT_COLS].first().reset_index()

    profiles = {}
    combos = store_types[COHORT_COLS].drop_duplicates()
    for _, row in combos.iterrows():
        active_cols = [c for c in COHORT_COLS if row[c] == 1.0]
        if not active_cols:
            continue

        cohort_key = "|".join(sorted(active_cols))

        # Find matching stores
        mask = (store_types[COHORT_COLS] == row).all(axis=1)
        matching_stores = store_types[mask]["Store"].values
        subset = df[df["Store"].isin(matching_stores)]

        profiles[cohort_key] = _build_profile_from_subset(subset)

    logger.info("Built %d metadata cohort profiles.", len(profiles))
    return profiles


def resolve_store_cohort_key(store) -> str | None:
    """
    Determines the metadata cohort key for a Store ORM object
    based on its store_type and assortment fields.

    Returns cohort key string like "StoreType_a|Assortment_c"
    or None if the store has no usable metadata.
    """
    parts = []
    if store.store_type:
        col = f"StoreType_{store.store_type}"
        if col in STORE_TYPE_COLS:
            parts.append(col)
    if store.assortment:
        col = f"Assortment_{store.assortment}"
        if col in ASSORTMENT_COLS:
            parts.append(col)

    if not parts:
        return None
    return "|".join(sorted(parts))


# ---------------------------------------------------------------------------
# Public API: build cluster profiles
# ---------------------------------------------------------------------------

def build_cluster_profiles(
    cluster_assignments: pd.DataFrame,
) -> dict[int, dict]:
    """
    Builds per-cluster forecast profiles.

    Args:
        cluster_assignments: DataFrame with columns [Store, cluster_id]
                             mapping each Rossmann store to its cluster.

    Returns:
        {cluster_id: profile_dict, ...}

    Returns empty dict if training data is unavailable.
    """
    loaded = _load_training_data()
    if loaded is None:
        return {}

    df, scalers = loaded
    df = _reverse_transform_sales(df, scalers)
    if df.empty:
        return {}
    df = _reconstruct_dow(df)

    # Merge cluster assignments
    df = df.merge(cluster_assignments[["Store", "cluster_id"]], on="Store", how="inner")

    profiles = {}
    for cid, group in df.groupby("cluster_id"):
        profiles[int(cid)] = _build_profile_from_subset(group)

    logger.info("Built %d cluster profiles.", len(profiles))
    return profiles
