"""
Feature engineering shared between training (the notebook) and serving
(this API). Keeping this logic in one place prevents train/serve skew —
the most common source of silent bugs in production ML systems.

NOTE: this mirrors the notebook's Section 5-10 logic. If you change the
feature engineering in the notebook, update this file to match before
retraining, or predictions will use a different feature contract than
what the model was trained on.
"""

import numpy as np
import pandas as pd


def add_cyclical_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    return df


def add_lag_rolling_features(df: pd.DataFrame, sales_col: str = "Sales_scaled") -> pd.DataFrame:
    df = df.sort_values(["Store", "Date"]).copy()
    df["Sales_lag_7"] = df.groupby("Store")[sales_col].shift(7)
    df["Sales_roll_mean_7"] = df.groupby("Store")[sales_col].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["Sales_roll_std_7"] = df.groupby("Store")[sales_col].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).std()
    )
    df["Sales_roll_mean_30"] = df.groupby("Store")[sales_col].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )
    lag_roll_cols = ["Sales_lag_7", "Sales_roll_mean_7", "Sales_roll_std_7", "Sales_roll_mean_30"]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)
    return df


def scale_sales_for_store(sales_log: pd.Series, scaler) -> np.ndarray:
    return scaler.transform(sales_log.values.reshape(-1, 1)).flatten()
