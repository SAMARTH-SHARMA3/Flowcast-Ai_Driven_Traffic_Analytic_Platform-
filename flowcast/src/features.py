"""
M3: Feature Engineering
-------------------------
Takes the cleaned, merged table from M2 (flowcast_clean.parquet) and
builds model-ready features: cyclical time encodings, lag features,
rolling window statistics, and categorical encodings.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("features")


def add_time_features(t: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical (sine/cosine) encodings for hour and day_of_week, so
    models correctly treat time as circular (23:00 is close to 00:00,
    Sunday is close to Monday) rather than as a flat numeric scale
    where those boundaries look maximally "far apart."

    No leakage risk here — these are derived purely from each row's
    own timestamp, not from other rows' values.
    """
    t["hour_sin"] = np.sin(2 * np.pi * t["hour"] / 24)
    t["hour_cos"] = np.cos(2 * np.pi * t["hour"] / 24)

    t["dow_sin"] = np.sin(2 * np.pi * t["day_of_week"] / 7)
    t["dow_cos"] = np.cos(2 * np.pi * t["day_of_week"] / 7)

    # A simple, non-cyclical but very useful flag: is this a weekday
    # rush hour? Morning 7-9am, evening 5-7pm, Mon-Fri only.
    is_weekday = t["day_of_week"] < 5  # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    is_morning_rush = t["hour"].between(7, 8)
    is_evening_rush = t["hour"].between(17, 18)
    t["is_rush_hour"] = (is_weekday & (is_morning_rush | is_evening_rush)).astype(int)

    logger.info(f"Added time features: hour_sin/cos, dow_sin/cos, is_rush_hour "
                f"({t['is_rush_hour'].sum()} rush-hour rows, {t['is_rush_hour'].mean()*100:.1f}%)")

    return t


def add_lag_features(t: pd.DataFrame, lags: list[int] = [1, 2, 4, 48]) -> pd.DataFrame:
    """
    Add lagged traffic_volume features: value from N steps ago, PER ROAD.

    lags are in units of 30-min steps: 1 = 30 min ago, 2 = 1 hour ago,
    4 = 2 hours ago, 48 = same time yesterday (24h / 0.5h = 48 steps).

    CRITICAL: must sort by road_id + timestamp first, then group by
    road_id before shifting. Without this, .shift() can pull a value
    from a completely different road's current reading and mislabel
    it as "this road's past" — a serious, silent data leakage bug.
    """
    t = t.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    for lag in lags:
        col_name = f"traffic_volume_lag_{lag}"
        t[col_name] = t.groupby("road_id")["traffic_volume"].shift(lag)
        n_null = t[col_name].isnull().sum()
        logger.info(f"{col_name}: {n_null} nulls (expected at each road's series start, "
                    f"where there's no earlier data to look back to)")

    return t



def add_rolling_features(t: pd.DataFrame, windows: list[int] = [4, 8]) -> pd.DataFrame:
    """
    Add rolling mean/std of traffic_volume over past windows, PER ROAD.

    windows are in 30-min steps: 4 = 2 hours, 8 = 4 hours.

    CRITICAL: shift(1) happens BEFORE rolling() — this excludes the
    current row from its own rolling window. Without the shift, the
    window would include the very value we're trying to predict,
    which is leakage just as serious as the cross-road bug in lags.
    """
    t = t.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    for window in windows:
        mean_col = f"traffic_volume_rolling_mean_{window}"
        std_col = f"traffic_volume_rolling_std_{window}"

        shifted = t.groupby("road_id")["traffic_volume"].shift(1)
        t[mean_col] = shifted.groupby(t["road_id"]).rolling(window).mean().reset_index(drop=True)
        t[std_col] = shifted.groupby(t["road_id"]).rolling(window).std().reset_index(drop=True)

        logger.info(f"{mean_col}/{std_col}: {t[mean_col].isnull().sum()} nulls "
                    f"(expected at each road's series start)")

    return t



# validating the result from above funcitons


if __name__ == "__main__":
    t = pd.read_parquet("data/processed/flowcast_clean.parquet")
    t = add_time_features(t)
    t = add_lag_features(t)
    t = add_rolling_features(t)

    print("Shape:", t.shape)
    print(t[["hour", "hour_sin", "hour_cos", "day_of_week", "dow_sin", "dow_cos", "is_rush_hour"]].head(10))

    print("\nSample lag values for one road:")
    sample = t[t["road_id"] == "NL-007"][["road_id", "timestamp", "traffic_volume", "traffic_volume_lag_1", "traffic_volume_lag_2"]].head(10)
    print(sample)


    print("new section")


    print("\nSample rolling values for one road:")
    sample = t[t["road_id"] == "NL-007"][
        ["timestamp", "traffic_volume", "traffic_volume_rolling_mean_4", "traffic_volume_rolling_std_4"]
    ].head(10)
    print(sample)