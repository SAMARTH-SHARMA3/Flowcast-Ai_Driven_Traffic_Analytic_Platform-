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