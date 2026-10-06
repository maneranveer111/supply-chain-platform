"""
Populates the database with demo data so the API returns real
(non-mock) forecasts and recommendations out of the box.

Two modes:
  1. If backend/saved_models/processed_df.parquet exists (copied from
     your notebook's checkpoint folder), seeds daily_store_features
     from your ACTUAL engineered Rossmann data.
  2. Otherwise, generates synthetic data for a handful of demo stores
     so the pipeline is still runnable end-to-end.

IMPORTANT — Store Identity Separation (post Phase 1)
------------------------------------------------------
Application Store.id is now a database-internal primary key only.
It has NO relationship to Rossmann store numbers.

When seeding from the parquet file (Rossmann data), each demo store
records the Rossmann store number in `benchmark_store_id`.  This is the
ONLY field that links an application store to a Rossmann training scaler.

Synthetic demo stores get benchmark_store_id = None, because they are
NOT Rossmann stores and must NOT accidentally resolve to a Rossmann
training scaler.

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

# Rossmann store numbers to seed from the parquet file.
# These are Rossmann IDs, NOT application store IDs.
ROSSMANN_BENCHMARK_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Names for the synthetic demo stores (no benchmark mapping)
SYNTHETIC_DEMO_NAMES = ["Demo Store Alpha", "Demo Store Beta", "Demo Store Gamma"]


def seed_from_parquet(db, feature_cols: list[str]):
    import pandas as pd

    print(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)

    # Use the configured benchmark IDs; fall back to first N stores in parquet
    available = [s for s in ROSSMANN_BENCHMARK_IDS if s in df["Store"].values]
    print(f"Seeding {len(available)} Rossmann benchmark stores from real notebook data ...")

    for rossmann_id in available:
        store_df = df[df["Store"] == rossmann_id].sort_values("Date").tail(settings.forecast_window)

        # Check by benchmark_store_id, NOT by application Store.id
        existing = db.query(Store).filter(Store.benchmark_store_id == int(rossmann_id)).first()
        if not existing:
            # Let the DB auto-assign id; record Rossmann ID as benchmark_store_id
            new_store = Store(
                benchmark_store_id=int(rossmann_id),
                name=f"Rossmann Demo Store #{rossmann_id}",
                # user_id=None  →  demo/system store, no user ownership
            )
            db.add(new_store)
            db.flush()  # obtain the auto-generated id before foreign-key references
            app_store_id = new_store.id
            print(f"  Created Store(id={app_store_id}, benchmark_store_id={rossmann_id})")
        else:
            app_store_id = existing.id
            print(f"  Store already exists: id={app_store_id}, benchmark_store_id={rossmann_id}")

        for _, row in store_df.iterrows():
            features = {col: float(row[col]) for col in feature_cols if col in row}
            # Use on-conflict-do-nothing semantics via check-before-insert
            existing_feat = (
                db.query(DailyStoreFeature)
                .filter(
                    DailyStoreFeature.store_id == app_store_id,
                    DailyStoreFeature.date == row["Date"].date(),
                )
                .first()
            )
            if not existing_feat:
                db.add(
                    DailyStoreFeature(
                        store_id=app_store_id,
                        date=row["Date"].date(),
                        features=features,
                    )
                )

        if not db.query(Inventory).filter(Inventory.store_id == app_store_id).first():
            db.add(
                Inventory(
                    store_id=app_store_id,
                    current_quantity=float(np.random.randint(500, 5000)),
                    updated_at=datetime.utcnow(),
                )
            )

    db.commit()
    print("Done seeding from real Rossmann data.")


def seed_synthetic(db, feature_cols: list[str]):
    print(f"No {PARQUET_PATH} found -- seeding synthetic demo stores.")
    rng = np.random.default_rng(42)
    today = datetime.utcnow().date()

    for demo_name in SYNTHETIC_DEMO_NAMES:
        # Check by name, NOT by any numeric ID
        existing = db.query(Store).filter(Store.name == demo_name).first()
        if not existing:
            # benchmark_store_id = None — synthetic stores are NOT Rossmann stores
            new_store = Store(
                name=demo_name,
                store_type="a",
                assortment="a",
                benchmark_store_id=None,
                # user_id=None  →  demo/system store
            )
            db.add(new_store)
            db.flush()
            app_store_id = new_store.id
            print(f"  Created synthetic Store(id={app_store_id}, name='{demo_name}')")
        else:
            app_store_id = existing.id
            print(f"  Synthetic store already exists: id={app_store_id}")

        for i in range(settings.forecast_window):
            date = today - timedelta(days=settings.forecast_window - i)
            existing_feat = (
                db.query(DailyStoreFeature)
                .filter(
                    DailyStoreFeature.store_id == app_store_id,
                    DailyStoreFeature.date == date,
                )
                .first()
            )
            if not existing_feat:
                features = {col: float(rng.normal(0, 1)) for col in feature_cols}
                features["Promo"] = float(rng.integers(0, 2))
                db.add(DailyStoreFeature(store_id=app_store_id, date=date, features=features))

        if not db.query(Inventory).filter(Inventory.store_id == app_store_id).first():
            db.add(
                Inventory(
                    store_id=app_store_id,
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
