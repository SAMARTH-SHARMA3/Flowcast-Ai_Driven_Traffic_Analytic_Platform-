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


# to check the statistical outlier section


if __name__ == "__main__":
    from ingest import load_traffic

    t = load_traffic("data/raw/traffic_sensor_log.csv")
    t = deduplicate_traffic(t)
    t = fix_impossible_values(t)
    t = flag_statistical_outliers_grouped(t, ["traffic_volume", "avg_speed", "occupancy"])
    t = impute_missing(t, ["traffic_volume", "avg_speed", "occupancy"])

    # Check the distribution BEFORE deciding whether to null anything —
    # this is the diagnostic that caught the global-IQR bias earlier.
    flagged = t[t["avg_speed_is_outlier"]]
    print("\navg_speed outliers by congestion_level (grouped IQR):")
    print(flagged["congestion_level"].value_counts())

    print("\nFinal shape:", t.shape)
    print(t[[c for c in t.columns if "_is_outlier" in c]].sum())
    print(t[["traffic_volume", "avg_speed", "occupancy"]].isnull().sum())


