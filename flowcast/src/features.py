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


# validating the result from above funcitons


if __name__ == "__main__":
    t = pd.read_parquet("data/processed/flowcast_clean.parquet")
    t = add_time_features(t)

    print("Shape:", t.shape)
    print(t[["hour", "hour_sin", "hour_cos", "day_of_week", "dow_sin", "dow_cos", "is_rush_hour"]].head(10))