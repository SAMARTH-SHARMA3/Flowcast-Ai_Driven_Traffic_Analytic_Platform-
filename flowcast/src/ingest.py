"""
M1: Ingestion & Validation
---------------------------
Loads the three raw source tables (calendar, weather, traffic) and
validates them against the schema described in the FlowCast Data
Dictionary. This stage only VALIDATES and FLAGS problems — it never
fixes, drops, or imputes anything. That's the job of clean.py (M2).

Keeping validation and fixing in separate files means:
  1. We always have a record of exactly what was wrong with the raw
     data, independent of whatever fix we later decide to apply.
  2. If something breaks downstream, we know immediately whether the
     raw data itself was bad (this file's fault) or a fix introduced
     a bug (clean.py's fault).
"""

import pandas as pd
import logging

# Configure logging once, at module level, so every function below
# can just call logger.info/warning and it's captured consistently.
# (Using logging instead of print() because print statements vanish
# once the pipeline grows — logging can be redirected to a file later
# and gives severity levels, which print() can't.)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")


def load_calendar(path: str) -> pd.DataFrame:
    """
    Load calendar_events.csv.

    Grain: one row per calendar date.
    Known defects: none documented in the data dictionary — but we
    still validate schema/types rather than assuming, since "no known
    defects" isn't the same as "guaranteed correct."
    """
    c = pd.read_csv(path)

    # --- Schema check ---
    # Fail loudly and immediately if the file doesn't have the columns
    # we expect. A missing column means the FILE itself is wrong
    # (e.g. wrong file, corrupted export) — this is unrecoverable,
    # so we raise instead of just logging a warning.
    expected_cols = {
        "date", "public_holiday", "holiday_name",
        "event_flag", "event_name", "roadwork_flag",
    }
    missing_cols = expected_cols - set(c.columns)
    if missing_cols:
        raise ValueError(f"calendar_events.csv missing expected columns: {missing_cols}")

    # --- Type check: parse date ---
    # format="%Y-%m-%d" is explicit on purpose. Letting pandas GUESS
    # the date format is exactly what causes silent DD/MM vs MM/DD
    # bugs — being explicit means a format mismatch raises an error
    # immediately instead of quietly parsing dates wrong.
    c["date"] = pd.to_datetime(c["date"], format="%Y-%m-%d")

    # --- Flag sanity check ---
    # These three columns should only ever contain 0 or 1. If they
    # don't, something is malformed upstream — worth knowing about,
    # but not fatal on its own, so we warn rather than raise.
    for flag_col in ["public_holiday", "event_flag", "roadwork_flag"]:
        bad_values = c[~c[flag_col].isin([0, 1])]
        if len(bad_values) > 0:
            logger.warning(f"{flag_col}: {len(bad_values)} rows with values other than 0/1")

    logger.info(
        f"Loaded calendar_events.csv: {len(c)} rows, "
        f"{c['date'].min().date()} to {c['date'].max().date()}"
    )
    return c


def load_weather(path: str) -> pd.DataFrame:
    """
    Load weather_observations.csv.

    Grain: one row per weather station per clock hour.
    Known defects (per data dictionary):
      - Date format is DD/MM/YYYY (traffic table uses YYYY-MM-DD).
      - weather_condition labels are inconsistently cased/spelled.
      - Small fraction of missing temperature/visibility.
    We only VALIDATE these here — label harmonisation happens in
    clean.py, not here.
    """
    w = pd.read_csv(path)

    # --- Schema check ---
    expected_cols = {
        "station_id", "date", "time", "weather_condition",
        "temperature", "rainfall", "visibility",
    }
    missing_cols = expected_cols - set(w.columns)
    if missing_cols:
        raise ValueError(f"weather_observations.csv missing expected columns: {missing_cols}")

    # --- Type check: parse date ---
    # NOTE: different format string from calendar/traffic on purpose —
    # this table genuinely uses DD/MM/YYYY, confirmed in the data dictionary.
    w["date"] = pd.to_datetime(w["date"], format="%d/%m/%Y")

    # --- Missingness check ---
    # Just counting here, not filling — imputation is clean.py's job.
    n_missing_temp = w["temperature"].isnull().sum()
    n_missing_vis = w["visibility"].isnull().sum()
    logger.info(
        f"weather_observations.csv: {n_missing_temp} missing temperature, "
        f"{n_missing_vis} missing visibility"
    )

    # --- Label consistency check ---
    # We don't fix the labels here — just report how messy they are,
    # so clean.py's normalisation map can be checked against this count.
    raw_conditions = w["weather_condition"].unique()
    logger.info(
        f"weather_condition has {len(raw_conditions)} distinct raw values "
        f"(needs harmonising in clean.py): {sorted(raw_conditions)}"
    )

    logger.info(
        f"Loaded weather_observations.csv: {len(w)} rows, "
        f"stations: {sorted(w['station_id'].unique())}"
    )
    return w


def load_traffic(path: str) -> pd.DataFrame:
    """
    Load traffic_sensor_log.csv — the primary table and modelling grain.

    Grain: one row per road segment per 30-minute window.
    Known defects (per data dictionary):
      - Duplicate rows from detector retries.
      - Missing traffic_volume / avg_speed / occupancy.
      - Impossible values: negative volume, speed > 200, occupancy > 100.
      - ~15% blank congestion_level (derived later, in clean.py).
    Same rule as above: we FLAG these, we do not fix or drop them here.
    """
    t = pd.read_csv(path)

    # --- Schema check ---
    expected_cols = {
        "road_id", "road_name", "latitude", "longitude", "weather_station_id",
        "date", "time", "traffic_volume", "vehicle_count", "vehicle_type_dist",
        "avg_speed", "occupancy", "congestion_level", "travel_time",
        "accident_count", "signal_timing", "road_capacity",
    }
    missing_cols = expected_cols - set(t.columns)
    if missing_cols:
        raise ValueError(f"traffic_sensor_log.csv missing expected columns: {missing_cols}")

    # --- Type check: parse date ---
    # Traffic uses YYYY-MM-DD — different from weather's DD/MM/YYYY.
    # This mismatch is exactly what clean.py has to reconcile before merging.
    t["date"] = pd.to_datetime(t["date"], format="%Y-%m-%d")

    # --- Duplicate check ---
    # Keying on road_id + date + time (not the whole row), because the
    # dictionary says duplicates come from "detector retries" — the same
    # sensor reporting the same window twice, possibly with slightly
    # different values. A whole-row duplicate check would miss retries
    # that differ by a decimal point.
    n_dupes = t.duplicated(subset=["road_id", "date", "time"]).sum()
    logger.warning(f"traffic_sensor_log.csv: {n_dupes} key-level duplicate rows (road_id+date+time)")

    # --- Rule-based (impossible) outlier check ---
    # These are PHYSICAL impossibilities, not statistical outliers —
    # a negative vehicle count or >200km/h average speed can't be real
    # no matter what the rest of the distribution looks like.
    # Statistical (IQR/z-score) outlier detection happens in clean.py,
    # on values that pass this basic sanity check first.
    n_negative_volume = (t["traffic_volume"] < 0).sum()
    n_speed_over_200 = (t["avg_speed"] > 200).sum()
    n_occupancy_over_100 = (t["occupancy"] > 100).sum()
    logger.warning(
        f"Rule-based outliers — negative volume: {n_negative_volume}, "
        f"speed>200: {n_speed_over_200}, occupancy>100: {n_occupancy_over_100}"
    )

    # --- Missingness check ---
    # congestion_level is included here even though ~15% blank is EXPECTED
    # (per the dictionary, it's meant to be derived later) — still worth
    # logging the actual count so the derivation step in clean.py can be
    # checked against it.
    for col in ["traffic_volume", "avg_speed", "occupancy", "congestion_level"]:
        n_missing = t[col].isnull().sum()
        logger.warning(f"{col}: {n_missing} missing ({n_missing / len(t) * 100:.1f}%)")

    logger.info(
        f"Loaded traffic_sensor_log.csv: {len(t)} rows, "
        f"{t['road_id'].nunique()} road segments"
    )
    return t


# Only runs when this file is executed directly (`python src/ingest.py`),
# NOT when it's imported by clean.py or a notebook. This lets us use it
# both as a standalone sanity-check script and as a reusable module.
if __name__ == "__main__":
    cal = load_calendar("data/raw/calendar_events.csv")
    wx = load_weather("data/raw/weather_observations.csv")
    tr = load_traffic("data/raw/traffic_sensor_log.csv")
    logger.info(
        f"Ingestion complete — calendar={len(cal)} rows, "
        f"weather={len(wx)} rows, traffic={len(tr)} rows"
    )