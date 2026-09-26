import pandas as pd
import numpy as np
import logging
import  json

logger = logging.getLogger("clean")


def deduplicate_traffic(t: pd.DataFrame) -> pd.DataFrame:
    """
    Remove key-level duplicate rows (road_id + date + time).

    These come from documented sensor retries — the same physical
    reading logged twice. We keep the first occurrence and drop the
    rest.
    """
    before = len(t)
    t = t.drop_duplicates(subset=["road_id", "date", "time"], keep="first")
    after = len(t)
    logger.info(f"Deduplication: removed {before - after} rows ({before} -> {after})")
    return t


def fix_impossible_values(t: pd.DataFrame) -> pd.DataFrame:
    """
    Null out physically impossible readings: negative traffic_volume,
    avg_speed > 200 km/h, occupancy > 100%.

    We set these to NaN rather than dropping the row, because the rest
    of that row is still usable — only the specific impossible VALUE
    is garbage. NaN lets impute_missing() treat it the same as a
    genuinely missing reading later, so we only need one imputation
    strategy instead of two.
    """
    n_neg_volume = (t["traffic_volume"] < 0).sum()
    n_speed = (t["avg_speed"] > 200).sum()
    n_occ = (t["occupancy"] > 100).sum()

    t.loc[t["traffic_volume"] < 0, "traffic_volume"] = np.nan
    t.loc[t["avg_speed"] > 200, "avg_speed"] = np.nan
    t.loc[t["occupancy"] > 100, "occupancy"] = np.nan

    logger.info(
        f"Nulled impossible values — traffic_volume: {n_neg_volume}, "
        f"avg_speed: {n_speed}, occupancy: {n_occ}"
    )
    return t



def flag_statistical_outliers_grouped(
    t: pd.DataFrame, cols: list[str], group_col: str = "congestion_level"
) -> pd.DataFrame:
    """
    Add a boolean *_outlier column per numeric column, using IQR bounds
    computed SEPARATELY within each group_col category (default:
    congestion_level).

    Global IQR is the wrong tool for a naturally multi-modal variable
    like avg_speed — a legitimately low speed during Severe congestion
    looks like an "outlier" only when compared against free-flow
    speeds too. Grouping first means we're asking "is this unusual for
    THIS traffic condition", not "is this unusual overall".

    NOTE: congestion_level is ~15% blank at this point in the pipeline
    (it's derived later, in step 9). groupby(dropna=True) means those
    blank rows are silently skipped here and left un-flagged. This is
    an accepted, documented gap — those rows get evaluated again once
    congestion_level exists for all rows.
    """
    for col in cols:
        t[f"{col}_is_outlier"] = False  # default; filled in per group below

        for group_value, group_df in t.groupby(group_col, dropna=True):
            q1, q3 = group_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            mask = (t[group_col] == group_value) & ((t[col] < lower) | (t[col] > upper))
            t.loc[mask, f"{col}_is_outlier"] = True

            logger.info(
                f"{col} | {group_col}={group_value}: bounds [{lower:.1f}, {upper:.1f}], "
                f"{mask.sum()} outliers flagged"
            )

        n_total = t[f"{col}_is_outlier"].sum()
        logger.info(f"{col}: {n_total} total outliers flagged across all groups")

    return t



def impute_missing(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Fill missing values (original raw gaps + impossible-value nulls
    from fix_impossible_values) using grouped median imputation: for
    each road_id + hour + day_of_week combination, fill gaps with that
    group's median. This preserves time-of-day/road-specific traffic
    patterns instead of flattening everything to one global average.

    Falls back to a global median if a specific group has no valid
    values at all to compute a median from.

    NOTE: statistical outliers (flagged in the previous step) are
    intentionally NOT touched here — they were kept as real data by
    design, so they were never turned into NaN and have nothing to
    impute.
    """
    t["hour"] = pd.to_datetime(t["time"], format="%H:%M").dt.hour
    t["day_of_week"] = pd.to_datetime(t["date"]).dt.dayofweek  # 0=Monday

    for col in cols:
        n_before = t[col].isnull().sum()

        group_median = t.groupby(["road_id", "hour", "day_of_week"])[col].transform("median")
        t[col] = t[col].fillna(group_median)

        # fallback for any road/hour/day combo with zero valid values
        global_median = t[col].median()
        t[col] = t[col].fillna(global_median)

        n_after = t[col].isnull().sum()
        logger.info(f"{col}: imputed {n_before - n_after} values (grouped median by road/hour/day)")

    return t


def harmonize_weather_labels(w: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize weather_condition to the 4 controlled-vocabulary values
    from the data dictionary: Clear, Cloudy, Rain, Fog.

    Raw data has 13 variants of these 4 categories (case differences,
    trailing whitespace, alternate spellings like 'rainy'/'foggy').
    Overcast is folded into Cloudy since it's not one of the 4
    documented target categories, and the two are close enough
    conceptually for this dataset's purposes.
    """
    mapping = {
        "clear": "Clear", "CLEAR": "Clear", "Clear": "Clear",
        "cloudy": "Cloudy", "Cloudy": "Cloudy", "Overcast": "Cloudy",
        "rain": "Rain", "RAIN": "Rain", "Rain": "Rain",
        "rainy": "Rain", "rain ": "Rain",
        "fog": "Fog", "FOG": "Fog", "Fog": "Fog", "foggy": "Fog",
    }

    # Strip whitespace and normalize case FIRST, so the mapping dict
    # itself can be much shorter (no need to enumerate every
    # capitalization) — but keep the explicit mapping too, as a safety
    # net that catches anything the strip/lower pass doesn't expect.
    cleaned = w["weather_condition"].str.strip().str.lower()

    lower_mapping = {
        "clear": "Clear",
        "cloudy": "Cloudy", "overcast": "Cloudy",
        "rain": "Rain", "rainy": "Rain",
        "fog": "Fog", "foggy": "Fog",
    }

    w["weather_condition_clean"] = cleaned.map(lower_mapping)

    # Anything that didn't match the map becomes NaN — catch and log it
    # rather than letting it silently disappear.
    unmapped = w[w["weather_condition_clean"].isnull()]["weather_condition"].unique()
    if len(unmapped) > 0:
        logger.warning(f"weather_condition values not covered by mapping: {unmapped}")

    n_changed = (w["weather_condition"] != w["weather_condition_clean"]).sum()
    logger.info(f"Harmonized weather_condition: {n_changed} values normalized")
    logger.info(f"Final categories: {sorted(w['weather_condition_clean'].dropna().unique())}")

    return w




def parse_vehicle_dist(t: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the vehicle_type_dist JSON string column into 4 separate
    numeric columns: pct_2w, pct_car, pct_lcv, pct_hcv.

    Keeping these as separate float columns (rather than one JSON
    string) is necessary before they can be used as model features —
    most ML libraries can't consume a nested JSON string directly.
    """
    def safe_parse(val):
        if pd.isnull(val):
            return {}
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}

    parsed = t["vehicle_type_dist"].apply(safe_parse)

    t["pct_2w"] = parsed.apply(lambda d: d.get("2W", np.nan))
    t["pct_car"] = parsed.apply(lambda d: d.get("Car", np.nan))
    t["pct_lcv"] = parsed.apply(lambda d: d.get("LCV", np.nan))
    t["pct_hcv"] = parsed.apply(lambda d: d.get("HCV", np.nan))

    n_failed = parsed.apply(lambda d: len(d) == 0).sum()
    logger.info(f"Parsed vehicle_type_dist: {len(t) - n_failed} succeeded, {n_failed} failed/empty")

    return t


def align_timestamps(t: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build matching join keys between traffic (30-min grain) and weather
    (hourly grain). Each 30-min traffic window is floored to its
    containing hour, so both the :00 and :30 windows in an hour map to
    the same weather reading, per the data dictionary's rule.
    """
    # Traffic: build full timestamp, then floor to the hour for joining
    t["timestamp"] = pd.to_datetime(t["date"].astype(str) + " " + t["time"])
    t["weather_join_hour"] = t["timestamp"].dt.floor("h")

    # Weather: build full timestamp (already hourly, so this IS the join key)
    w["timestamp"] = pd.to_datetime(w["date"].astype(str) + " " + w["time"])
    w["weather_join_hour"] = w["timestamp"]

    logger.info(f"Traffic timestamp range: {t['timestamp'].min()} to {t['timestamp'].max()}")
    logger.info(f"Weather timestamp range: {w['timestamp'].min()} to {w['timestamp'].max()}")

    return t, w



def merge_all(t: pd.DataFrame, w: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather and calendar data onto the traffic grain.

    Traffic stays the base table throughout — left joins only, so no
    traffic row is ever lost or duplicated, regardless of whether a
    matching weather/calendar row exists.
    """
    before = len(t)

    # Weather: join on station + aligned hour
    merged = t.merge(
        w[["station_id", "weather_join_hour", "weather_condition_clean",
           "temperature", "rainfall", "visibility"]],
        left_on=["weather_station_id", "weather_join_hour"],
        right_on=["station_id", "weather_join_hour"],
        how="left",
    )

    # Calendar: join on date only
    merged = merged.merge(
        c[["date", "public_holiday", "holiday_name", "event_flag", "event_name", "roadwork_flag"]],
        on="date",
        how="left",
    )

    after = len(merged)
    logger.info(f"Merge: {before} traffic rows -> {after} merged rows (should be equal)")

    n_no_weather = merged["weather_condition_clean"].isnull().sum()
    n_no_calendar = merged["public_holiday"].isnull().sum()
    logger.info(f"Rows with no matching weather: {n_no_weather}")
    logger.info(f"Rows with no matching calendar: {n_no_calendar}")

    return merged





"""checking all the functions are working and printing their results"""


if __name__ == "__main__":
    from ingest import load_traffic, load_weather, load_calendar

    # --- Load ---
    t = load_traffic("data/raw/traffic_sensor_log.csv")
    w = load_weather("data/raw/weather_observations.csv")
    cal = load_calendar("data/raw/calendar_events.csv")

    # --- Clean traffic ---
    t = deduplicate_traffic(t)
    t = fix_impossible_values(t)
    t = flag_statistical_outliers_grouped(t, ["traffic_volume", "avg_speed", "occupancy"])
    t = impute_missing(t, ["traffic_volume", "avg_speed", "occupancy"])
    t = parse_vehicle_dist(t)

    # --- Clean weather ---
    w = harmonize_weather_labels(w)

    # --- Align timestamps (needs both t and w) ---
    t, w = align_timestamps(t, w)

    # --- Merge everything ---
    merged = merge_all(t, w, cal)
    print("\nFinal merged shape:", merged.shape)

    # --- Diagnostics (kept from earlier steps, for reference) ---
    print("\navg_speed outliers by congestion_level (grouped IQR):")
    print(t[t["avg_speed_is_outlier"]]["congestion_level"].value_counts())

    print("\nOutlier flag counts:")
    print(t[[col for col in t.columns if "_is_outlier" in col]].sum())

    print("\nRemaining nulls (should be 0):")
    print(t[["traffic_volume", "avg_speed", "occupancy"]].isnull().sum())

    print("\nWeather categories after harmonization:")
    print(w["weather_condition_clean"].value_counts())

    print("\nVehicle distribution row sums (should be ~1.0):")
    print(t[["pct_2w", "pct_car", "pct_lcv", "pct_hcv"]].sum(axis=1).describe())

    