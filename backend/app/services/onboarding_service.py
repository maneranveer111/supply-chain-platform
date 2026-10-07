"""
Onboarding Service: provides idempotent background and synchronous feature
engineering, custom scaler fitting, and cluster assignment for stores.
"""

import logging
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.clustering import build_store_features
from app.ml.inference import COMP_SCALER_PATH, load_feature_cols
from app.ml.preprocessing import add_cyclical_day_of_week, add_lag_rolling_features
from app.models.cluster import Cluster
from app.models.daily_store_feature import DailyStoreFeature
from app.models.store import Store
from app.models.store_sale import StoreSale
from app.models.store_scaler import StoreScaler

logger = logging.getLogger(__name__)


def process_features(df_sales: pd.DataFrame, store: Store, db: Session) -> None:
    """
    Fits a custom StandardScaler on log1p(Sales) and generates the full 22-column
    feature vector for all historical sales days, replacing any existing features
    for this store idempotently.
    """
    feature_cols = load_feature_cols()

    # Fit scaler on log1p(Sales)
    sales_arr = df_sales["sales"].to_numpy(dtype=float)
    log_sales = np.log1p(sales_arr)
    mean_log = float(np.mean(log_sales))
    scale_log = float(np.std(log_sales))
    if scale_log == 0.0 or np.isnan(scale_log):
        scale_log = 1.0
    n_samples = len(df_sales)

    # Upsert StoreScaler row
    scaler_row = db.query(StoreScaler).filter(StoreScaler.store_id == store.id).first()
    if not scaler_row:
        scaler_row = StoreScaler(store_id=store.id)
        db.add(scaler_row)
    scaler_row.mean_log = mean_log
    scaler_row.scale_log = scale_log
    scaler_row.n_samples = n_samples
    scaler_row.updated_at = datetime.utcnow()
    db.flush()

    df = df_sales.copy()
    df.rename(columns={"sales": "Sales", "date": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Store"] = store.id
    df["Sales_scaled"] = (np.log1p(df["Sales"]) - mean_log) / scale_log

    df["DayOfWeek"] = df["Date"].dt.weekday
    df = add_cyclical_day_of_week(df)
    df = add_lag_rolling_features(df, sales_col="Sales_scaled")

    df["Promo"] = df["promo"]
    df["SchoolHoliday"] = df["school_holiday"]
    df["IsPromo2Active"] = 1 if store.promo2_active else 0

    with open(COMP_SCALER_PATH, "rb") as f:
        comp_scaler = pickle.load(f)
    comp_dist = store.competition_distance if store.competition_distance is not None else 5000.0
    comp_scaled = comp_scaler.transform([[comp_dist]])[0][0]
    df["CompetitionDistance_scaled"] = comp_scaled

    for c in ["a", "b", "c", "d"]:
        df[f"StoreType_{c}"] = 1 if store.store_type == c else 0
    for c in ["a", "b", "c"]:
        df[f"Assortment_{c}"] = 1 if store.assortment == c else 0

    for c in ["0", "a", "b", "c"]:
        df[f"StateHoliday_{c}"] = (df["state_holiday"].astype(str) == c).astype(int)

    # Idempotent delete of existing features before inserting new set
    db.query(DailyStoreFeature).filter(DailyStoreFeature.store_id == store.id).delete()

    features_to_insert = []
    records = df.to_dict("records")
    for row in records:
        feat_dict = {}
        for col in feature_cols:
            if col in row:
                feat_dict[col] = float(row[col])
            else:
                feat_dict[col] = 0.0

        row_date = row["Date"]
        if isinstance(row_date, pd.Timestamp):
            row_date = row_date.date()

        features_to_insert.append(
            DailyStoreFeature(
                store_id=store.id,
                date=row_date,
                features=feat_dict,
            )
        )

    db.add_all(features_to_insert)
    logger.info(
        "Generated %d feature rows and fitted custom scaler for store_id=%d",
        len(features_to_insert),
        store.id,
    )


def process_store_features_and_clustering(store_id: int, db: Session) -> dict:
    """
    Idempotent pipeline for processing a store's sales data:
      - >= 60 distinct days: fits custom scaler + generates 22 daily features
      - 14-59 distinct days: assigns nearest cluster profile
      - < 14 distinct days: resets cluster assignment
    Safely transitions store.onboarding_status through:
      processing -> ready (or failed on error).
    """
    logger.info("Starting onboarding processing for store_id=%d", store_id)
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        logger.error("Cannot process onboarding: store_id=%d does not exist", store_id)
        raise ValueError(f"Store with id={store_id} not found.")

    try:
        store.onboarding_status = "processing"
        db.commit()

        # Query sales ordered by date
        sales_rows = (
            db.query(StoreSale)
            .filter(StoreSale.store_id == store_id)
            .order_by(StoreSale.date.asc())
            .all()
        )

        if not sales_rows:
            logger.info("Store store_id=%d has no sales history. Marked ready.", store_id)
            store.cluster_id = None
            store.onboarding_status = "ready"
            db.commit()
            return {"status": "ready", "history_days": 0, "message": "No sales records"}

        df = pd.DataFrame(
            [
                {
                    "date": s.date,
                    "sales": float(s.sales),
                    "promo": int(s.promo),
                    "school_holiday": int(s.school_holiday),
                    "state_holiday": str(s.state_holiday),
                }
                for s in sales_rows
            ]
        )

        # Unique dates count
        distinct_days = len(df["date"].drop_duplicates())
        logger.info(
            "Store store_id=%d has %d distinct sales days (total rows=%d)",
            store_id,
            distinct_days,
            len(df),
        )

        if distinct_days >= 60:
            process_features(df, store, db)
            store.onboarding_status = "ready"
            db.commit()
            logger.info("Onboarding successful for store_id=%d: LSTM ready", store_id)
            return {
                "status": "ready",
                "history_days": distinct_days,
                "readiness": "lstm_ready",
            }

        elif distinct_days >= 14:
            df_for_clustering = df.copy()
            df_for_clustering.rename(
                columns={"date": "Date", "sales": "Sales", "promo": "Promo"},
                inplace=True,
            )
            df_for_clustering["Store"] = store.id
            agg = build_store_features(df_for_clustering)
            if not agg.empty:
                mean_sales = float(agg.iloc[0]["mean_sales"])
                promo_uplift = float(agg.iloc[0]["promo_uplift"])

                clusters = db.query(Cluster).filter(Cluster.profile.isnot(None)).all()
                best_cluster = None
                best_dist = float("inf")
                for cl in clusters:
                    prof = cl.profile
                    c_mean = prof.get("mean_daily_sales", 0.0)
                    c_uplift = prof.get("promo_uplift", 0.0)
                    dist = ((mean_sales - c_mean) / (c_mean + 1)) ** 2 + (
                        promo_uplift - c_uplift
                    ) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = cl

                if best_cluster:
                    store.cluster_id = best_cluster.id
                    logger.info(
                        "Assigned store_id=%d to cluster_id=%d", store.id, best_cluster.id
                    )
                else:
                    store.cluster_id = None
            else:
                store.cluster_id = None

            # Clean up partial custom scaler / features if history was reduced
            db.query(DailyStoreFeature).filter(DailyStoreFeature.store_id == store.id).delete()
            db.query(StoreScaler).filter(StoreScaler.store_id == store.id).delete()

            store.onboarding_status = "ready"
            db.commit()
            logger.info("Onboarding successful for store_id=%d: cluster ready", store_id)
            return {
                "status": "ready",
                "history_days": distinct_days,
                "cluster_id": store.cluster_id,
                "readiness": "cluster_ready",
            }

        else:
            store.cluster_id = None
            db.query(DailyStoreFeature).filter(DailyStoreFeature.store_id == store.id).delete()
            db.query(StoreScaler).filter(StoreScaler.store_id == store.id).delete()

            store.onboarding_status = "ready"
            db.commit()
            logger.info(
                "Store store_id=%d history below 14 days; marked ready (cold start fallback).",
                store_id,
            )
            return {
                "status": "ready",
                "history_days": distinct_days,
                "readiness": "metadata_cohort_ready",
            }

    except Exception as exc:
        db.rollback()
        logger.error(
            "Error during onboarding processing for store_id=%d: %s",
            store_id,
            exc,
            exc_info=True,
        )
        try:
            store = db.query(Store).filter(Store.id == store_id).first()
            if store:
                store.onboarding_status = "failed"
                db.commit()
        except Exception as inner_exc:
            logger.error(
                "Could not set store_id=%d status to 'failed': %s",
                store_id,
                inner_exc,
            )
            db.rollback()
        raise
