"""
Tests the leakage-safety of rolling features: a rolling feature for
day N must never be able to see day N's own sales value.
"""

import pandas as pd

from app.ml.preprocessing import add_lag_rolling_features


def test_rolling_mean_excludes_current_day():
    df = pd.DataFrame(
        {
            "Store": [1] * 10,
            "Date": pd.date_range("2024-01-01", periods=10),
            "Sales_scaled": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    result = add_lag_rolling_features(df)

    # The row where Sales_scaled=100.0 is a huge outlier (index 5).
    # Its own rolling mean must NOT include that 100.0 value.
    row = result.iloc[5]
    assert row["Sales_roll_mean_7"] < 50, (
        "Rolling mean includes the current day's own value — this is feature leakage."
    )
