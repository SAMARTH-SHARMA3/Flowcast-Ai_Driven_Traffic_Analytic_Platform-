import pandas as pd
import numpy as np
import logging

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



# to test this

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingest import load_traffic

    t = load_traffic("data/raw/traffic_sensor_log.csv")
    t = deduplicate_traffic(t)
    print("Final shape:", t.shape)



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



# to test the second block

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingest import load_traffic

    t = load_traffic("data/raw/traffic_sensor_log.csv")
    t = deduplicate_traffic(t)
    t = fix_impossible_values(t)
    print("Final shape:", t.shape)
    print(t[["traffic_volume", "avg_speed", "occupancy"]].isnull().sum())