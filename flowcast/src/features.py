"""
M3: Feature Engineering
-------------------------
Takes the cleaned, merged table from M2 (flowcast_clean.parquet) and
builds model-ready features per the PRD spec (Section 10.1):
temporal encodings, lag features, rolling statistics, capacity ratios,
weather flags, calendar interactions, and categorical encodings.

LEAKAGE GUARD (per PRD Section 10.1): lag and rolling features are
computed within each road segment in strict time order. No feature
may use information from a future window relative to prediction time.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("features")


def add_time_features(t: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical (sine/cosine) encodings for hour and day_of_week, so models
    treat time as circular (23:00 close to 00:00, Sunday close to
    Monday) rather than as a flat scale. Also adds weekend and
    peak-period (rush hour) flags, per PRD's Temporal feature spec.

    No leakage risk — derived purely from each row's own timestamp.
    """
    t["hour_sin"] = np.sin(2 * np.pi * t["hour"] / 24)
    t["hour_cos"] = np.cos(2 * np.pi * t["hour"] / 24)

    t["dow_sin"] = np.sin(2 * np.pi * t["day_of_week"] / 7)
    t["dow_cos"] = np.cos(2 * np.pi * t["day_of_week"] / 7)

    # Weekend flag: 5=Saturday, 6=Sunday
    t["is_weekend"] = (t["day_of_week"] >= 5).astype(int)

    # Peak-period (rush hour) flag: weekday mornings 7-9am, evenings 5-7pm
    is_weekday = t["day_of_week"] < 5
    is_morning_rush = t["hour"].between(7, 8)
    is_evening_rush = t["hour"].between(17, 18)
    t["is_rush_hour"] = (is_weekday & (is_morning_rush | is_evening_rush)).astype(int)

    logger.info(
        f"Added time features: hour_sin/cos, dow_sin/cos, is_weekend "
        f"({t['is_weekend'].sum()} rows), is_rush_hour ({t['is_rush_hour'].sum()} rows)"
    )
    return t


def add_lag_features(t: pd.DataFrame, col: str, lags: list[int] = [1, 2, 48]) -> pd.DataFrame:
    """
    Add lagged values of `col`, N steps ago, PER ROAD SEGMENT.

    Per PRD spec: t-1, t-2, t-48 (30min/1h/same-window-previous-day),
    applied to both traffic_volume and avg_speed (call this function
    once per column).

    CRITICAL leakage guard: sort by road_id + timestamp, then group by
    road_id before shifting — without this, a value could cross from
    one road's data into another's, silently corrupting the feature.
    """
    t = t.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    for lag in lags:
        col_name = f"{col}_lag_{lag}"
        t[col_name] = t.groupby("road_id")[col].shift(lag)
        n_null = t[col_name].isnull().sum()
        logger.info(f"{col_name}: {n_null} nulls (expected at each road's series start)")

    return t


def add_rolling_features(t: pd.DataFrame, col: str, windows: list[int] = [4, 8]) -> pd.DataFrame:
    """
    Add rolling mean/std of `col` over past windows, PER ROAD SEGMENT.

    Per PRD spec: windows of 4 and 8 steps (2h/4h), applied to both
    traffic_volume and avg_speed.

    CRITICAL leakage guard: shift(1) happens BEFORE rolling() — this
    excludes the current row from its own window. Without the shift,
    the window would include the very value we're trying to predict.
    """
    t = t.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    for window in windows:
        mean_col = f"{col}_rolling_mean_{window}"
        std_col = f"{col}_rolling_std_{window}"

        shifted = t.groupby("road_id")[col].shift(1)
        t[mean_col] = shifted.groupby(t["road_id"]).rolling(window).mean().reset_index(drop=True)
        t[std_col] = shifted.groupby(t["road_id"]).rolling(window).std().reset_index(drop=True)

        logger.info(f"{mean_col}/{std_col}: {t[mean_col].isnull().sum()} nulls (expected at series start)")

    return t


def add_capacity_features(t: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-to-capacity ratio and headroom, per PRD's Capacity feature spec.

    v_c_ratio: how full the road is relative to its capacity (this is
    the same ratio used in M2's derive_congestion_level — here it
    becomes an explicit numeric feature for the model, not just a
    banding rule).
    capacity_headroom: how much room is left before hitting capacity —
    negative values mean the road is over its nominal capacity.
    """
    t["v_c_ratio"] = t["traffic_volume"] / (t["road_capacity"] / 2)
    t["capacity_headroom"] = (t["road_capacity"] / 2) - t["traffic_volume"]

    logger.info(
        f"Added capacity features: v_c_ratio (mean={t['v_c_ratio'].mean():.2f}), "
        f"capacity_headroom (mean={t['capacity_headroom'].mean():.1f})"
    )
    return t


def add_weather_flags(t: pd.DataFrame) -> pd.DataFrame:
    """
    Rain flag, low-visibility flag, temperature bands — per PRD's
    Weather feature spec.

    low_visibility threshold: <1000m, the standard meteorological
    "fog" cutoff (confirmed against this dataset's actual visibility
    range of 301-10,000m — 1000m is a meaningful cut, not an arbitrary
    round number).
    Temperature bands based on this dataset's actual range (5.8-29.8°C).
    """
    t["rain_flag"] = (t["weather_condition_clean"] == "Rain").astype(int)
    t["low_visibility_flag"] = (t["visibility"] < 1000).astype(int)

    def temp_band(temp):
        if pd.isnull(temp):
            return np.nan
        elif temp < 10:
            return "Cold"
        elif temp < 25:
            return "Mild"
        else:
            return "Hot"

    t["temperature_band"] = t["temperature"].apply(temp_band)

    logger.info(
        f"rain_flag: {t['rain_flag'].sum()} rows, "
        f"low_visibility_flag: {t['low_visibility_flag'].sum()} rows, "
        f"temperature_band distribution: {t['temperature_band'].value_counts().to_dict()}"
    )
    return t


def add_calendar_interactions(t: pd.DataFrame) -> pd.DataFrame:
    """
    Holiday x peak interaction, and event proximity — per PRD's
    Calendar feature spec.

    holiday_peak_interaction: captures the idea that a holiday during
    what would normally be rush hour behaves differently from an
    ordinary rush hour (usually much lighter traffic) — a plain
    additive model can't capture this without an explicit interaction term.

    days_to_event: distance in days from this row's date to the
    nearest event day (0 = event day itself). Captures "traffic builds
    up approaching an event," not just an on/off event flag.
    """
    t["holiday_peak_interaction"] = t["public_holiday"] * t["is_rush_hour"]

    event_dates = t.loc[t["event_flag"] == 1, "date"].unique()
    if len(event_dates) > 0:
        event_dates = pd.to_datetime(sorted(event_dates))
        row_dates = pd.to_datetime(t["date"])
        # For each row, find the minimum distance (in days) to any event date
        days_to_event = row_dates.apply(
            lambda d: min(abs((d - ed).days) for ed in event_dates)
        )
        t["days_to_event"] = days_to_event
    else:
        t["days_to_event"] = np.nan

    logger.info(
        f"holiday_peak_interaction: {t['holiday_peak_interaction'].sum()} rows, "
        f"days_to_event range: {t['days_to_event'].min()}-{t['days_to_event'].max()}"
    )
    return t


def add_categorical_features(t: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns into numeric form for modeling.

    - weather_condition_clean (4 categories) + temperature_band (3
      categories): one-hot encoded — cheap, avoids implying false order.
    - road_id (25 categories): label-encoded (single integer column) —
      one-hot would add 25 mostly-zero columns; tree-based models
      (XGBoost, Random Forest) handle label-encoded categoricals fine.

    Original string columns are KEPT for readability; drop later only
    if a specific model requires purely numeric input.
    """
    weather_dummies = pd.get_dummies(t["weather_condition_clean"], prefix="weather")
    temp_band_dummies = pd.get_dummies(t["temperature_band"], prefix="temp")
    t = pd.concat([t, weather_dummies, temp_band_dummies], axis=1)
    logger.info(f"One-hot encoded: {list(weather_dummies.columns)}, {list(temp_band_dummies.columns)}")

    unique_roads = sorted(t["road_id"].unique())
    road_map = {road: i for i, road in enumerate(unique_roads)}
    t["road_id_encoded"] = t["road_id"].map(road_map)
    logger.info(f"Label encoded road_id: {len(unique_roads)} unique roads mapped to 0-{len(unique_roads)-1}")

    return t


def add_categorical_features(t: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns into numeric form for modeling.

    - weather_condition_clean (4 categories), temperature_band (3
      categories), congestion_level (4 categories): one-hot encoded —
      cheap, avoids implying false order.
    - road_id (25 categories): label-encoded (single integer column).

    Original string columns are KEPT for readability; drop later only
    if a specific model requires purely numeric input.
    """
    weather_dummies = pd.get_dummies(t["weather_condition_clean"], prefix="weather")
    temp_band_dummies = pd.get_dummies(t["temperature_band"], prefix="temp")
    congestion_dummies = pd.get_dummies(t["congestion_level"], prefix="congestion")  # NEW
    t = pd.concat([t, weather_dummies, temp_band_dummies, congestion_dummies], axis=1)
    logger.info(
        f"One-hot encoded: {list(weather_dummies.columns)}, "
        f"{list(temp_band_dummies.columns)}, {list(congestion_dummies.columns)}"
    )

    unique_roads = sorted(t["road_id"].unique())
    road_map = {road: i for i, road in enumerate(unique_roads)}
    t["road_id_encoded"] = t["road_id"].map(road_map)
    logger.info(f"Label encoded road_id: {len(unique_roads)} unique roads mapped to 0-{len(unique_roads)-1}")

    return t


if __name__ == "__main__":
    import os

    t = pd.read_parquet("data/processed/flowcast_clean.parquet")
    logger.info(f"Loaded flowcast_clean.parquet: {t.shape}")

    # --- Temporal ---
    t = add_time_features(t)

    # --- Lag features: BOTH traffic_volume and avg_speed, per PRD spec ---
    t = add_lag_features(t, "traffic_volume", lags=[1, 2, 48])
    t = add_lag_features(t, "avg_speed", lags=[1, 2, 48])

    # --- Rolling features: BOTH traffic_volume and avg_speed ---
    t = add_rolling_features(t, "traffic_volume", windows=[4, 8])
    t = add_rolling_features(t, "avg_speed", windows=[4, 8])

    # --- Capacity ---
    t = add_capacity_features(t)

    # --- Weather flags ---
    t = add_weather_flags(t)

    # --- Calendar interactions ---
    t = add_calendar_interactions(t)

    # --- Categorical encoding (last, since it depends on temperature_band existing) ---
    t = add_categorical_features(t)

    print("\nFinal shape:", t.shape)
    print("\nSample of new columns:")
    sample_cols = [
        "road_id", "timestamp", "traffic_volume", "traffic_volume_lag_1",
        "avg_speed_lag_1", "v_c_ratio", "rain_flag", "low_visibility_flag",
        "temperature_band", "holiday_peak_interaction", "days_to_event",
    ]
    print(t[sample_cols].head())



    # --- Verification: days_to_event should be 0 on an actual event day ---
    event_dates = t[t["event_flag"] == 1]["date"].unique()
    if len(event_dates) > 0:
        check_date = event_dates[0]
        check_value = t[t["date"] == check_date]["days_to_event"].iloc[0]
        assert check_value == 0, f"Expected days_to_event=0 on event day {check_date}, got {check_value}"
        logger.info(f"Verified: days_to_event=0 on confirmed event day {check_date}")


# --- Save ---
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/flowcast_features.parquet"
t.to_parquet(output_path, index=False)
logger.info(f"Saved feature-engineered dataset: {t.shape} -> {output_path}")