"""
Clusters stores by sales behaviour. Run periodically via a Celery task
(see workers/cluster_tasks.py), not per-request — store behaviour
doesn't change hour to hour.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLUSTER_LABELS = {
    0: "low-volume stable",
    1: "high-volume stable",
    2: "promo-sensitive",
    3: "seasonal spike",
}


def build_store_features(daily_sales: pd.DataFrame) -> pd.DataFrame:
    """
    daily_sales: columns [Store, Date, Sales, Promo]
    Returns one row per store with aggregate behavioural features.
    """
    agg = daily_sales.groupby("Store").agg(
        mean_sales=("Sales", "mean"),
        std_sales=("Sales", "std"),
        promo_uplift=("Sales", lambda s: _promo_uplift(daily_sales.loc[s.index])),
    )
    return agg.fillna(0).reset_index()


def _promo_uplift(store_df: pd.DataFrame) -> float:
    promo_mean = store_df.loc[store_df["Promo"] == 1, "Sales"].mean()
    non_promo_mean = store_df.loc[store_df["Promo"] == 0, "Sales"].mean()
    if not non_promo_mean or np.isnan(non_promo_mean):
        return 0.0
    return float((promo_mean - non_promo_mean) / non_promo_mean)


def fit_clusters(store_features: pd.DataFrame, n_clusters: int = 4):
    feature_cols = ["mean_sales", "std_sales", "promo_uplift"]
    X = store_features[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    store_features = store_features.copy()
    store_features["cluster_id"] = labels
    store_features["cluster_label"] = store_features["cluster_id"].map(CLUSTER_LABELS)
    return store_features, kmeans, scaler
