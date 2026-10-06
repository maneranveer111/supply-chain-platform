"""
Populates the database with demo data so the API returns real
(non-mock) forecasts and recommendations out of the box.

Two modes:
  1. If backend/saved_models/processed_df.parquet exists (copied from
     your notebook's checkpoint folder), seeds daily_store_features
     from your ACTUAL engineered Rossmann data.
  2. Otherwise, generates synthetic data for a handful of demo stores
     so the pipeline is still runnable end-to-end.

Usage:
    cd backend
    python -m scripts.seed_data
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db.session import Base, SessionLocal, engine
from app.ml.inference import load_feature_cols
from app.models.daily_store_feature import DailyStoreFeature
from app.models.inventory import Inventory
from app.models.store import Store

PARQUET_PATH = os.path.join(settings.model_dir, "processed_df.parquet")
DEMO_STORE_IDS = [1, 2, 3]


def seed_from_parquet(db, feature_cols: list[str]):
    import pandas as pd

    print(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)

    stores = df["Store"].unique()[:10]  # keep seeding fast; widen if you want more
    print(f"Seeding {len(stores)} stores from real notebook data ...")

    for store_id in stores:
        store_df = df[df["Store"] == store_id].sort_values("Date").tail(settings.forecast_window)
        if not db.query(Store).filter(Store.id == int(store_id)).first():
            db.add(Store(id=int(store_id)))

        for _, row in store_df.iterrows():
            features = {col: float(row[col]) for col in feature_cols if col in row}
            db.add(
                DailyStoreFeature(
                    store_id=int(store_id),
                    date=row["Date"].date(),
                    features=features,
                )
            )

        if not db.query(Inventory).filter(Inventory.store_id == int(store_id)).first():
            db.add(
                Inventory(
                    store_id=int(store_id),
                    current_quantity=float(np.random.randint(500, 5000)),
                    updated_at=datetime.utcnow(),
                )
            )

    db.commit()
    print("Done seeding from real data.")


def seed_synthetic(db, feature_cols: list[str]):
    print(f"No {PARQUET_PATH} found -- seeding synthetic demo data for {DEMO_STORE_IDS}.")
    rng = np.random.default_rng(42)
    today = datetime.utcnow().date()

    for store_id in DEMO_STORE_IDS:
        if not db.query(Store).filter(Store.id == store_id).first():
            db.add(Store(id=store_id, store_type="a", assortment="a"))

        for i in range(settings.forecast_window):
            date = today - timedelta(days=settings.forecast_window - i)
            features = {col: float(rng.normal(0, 1)) for col in feature_cols}
            features["Promo"] = float(rng.integers(0, 2))
            db.add(DailyStoreFeature(store_id=store_id, date=date, features=features))

        if not db.query(Inventory).filter(Inventory.store_id == store_id).first():
            db.add(
                Inventory(
                    store_id=store_id,
                    current_quantity=float(rng.integers(500, 5000)),
                    updated_at=datetime.utcnow(),
                )
            )

    db.commit()
    print("Done seeding synthetic data.")


def main():
    Base.metadata.create_all(bind=engine)  # safety net if migrations haven't been run
    db = SessionLocal()
    feature_cols = load_feature_cols()

    try:
        if os.path.exists(PARQUET_PATH):
            seed_from_parquet(db, feature_cols)
        else:
            seed_synthetic(db, feature_cols)
    finally:
        db.close()


if __name__ == "__main__":
    main()
