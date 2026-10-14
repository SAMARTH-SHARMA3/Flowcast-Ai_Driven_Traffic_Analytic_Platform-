"""
M5: Classical ML Engine
-------------------------
Trains, tunes, and compares classical ML models (Linear Regression,
Decision Tree, Random Forest, XGBoost, SVM) on 4 forecasting targets:
traffic volume, travel time (regression), congestion level, accident
risk (classification). Per PRD Section 11.

All targets are FORWARD-LOOKING ("next window") — this is forecasting,
not same-window estimation. Section 11.1's framing: "predict next-
window volume... from lag, weather and calendar features" means the
target must be the FUTURE reading, built via shift(-1), mirroring the
backward shift() used for lag features in M3.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_models")


def add_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 4 forward-looking ("next window") targets per PRD 11.1:
    - target_volume_next: next-window traffic_volume (regression)
    - target_travel_time_next: next-window travel_time (regression)
    - target_congestion_next: next-window congestion_level (4-class)
    - target_accident_next: next-window binary accident flag

    Uses shift(-1), grouped by road_id — the mirror image of M3's
    lag features (shift(+N) looks backward; shift(-1) looks forward).
    Same per-road discipline applies: must never let a value cross
    from one road's series into another's.

    The LAST row of each road's time series will get NaN for every
    target (there is no "next window" to look forward to) — this is
    expected, not a bug; these rows must be dropped before training.
    """
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    # Binary accident flag FIRST (current window), then shift forward.
    # Per M4 Section 5.4: accident_count is 99%+ zero — binary framing
    # (any incident vs none) is more tractable than 3-class.
    df["accident_flag"] = (df["accident_count"] > 0).astype(int)

    target_specs = {
        "target_volume_next": "traffic_volume",
        "target_travel_time_next": "travel_time",
        "target_congestion_next": "congestion_level",
        "target_accident_next": "accident_flag",
    }

    for target_col, source_col in target_specs.items():
        df[target_col] = df.groupby("road_id")[source_col].shift(-1)
        n_null = df[target_col].isnull().sum()
        logger.info(f"{target_col}: {n_null} nulls (expected — one per road, the series' last row)")

    return df




def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into train / validation / test, per PRD 11.3:
    "earliest windows for training, latest for testing — never randomly."

    train_frac=0.70, val_frac=0.15 leaves 0.15 for test (5 months of
    data: ~3.5 months train, ~3 weeks val, ~3 weeks test).

    CRITICAL: this splits by TIME, not by row count per road — every
    road contributes to all three sets, but the actual calendar cutoff
    dates are shared across all roads. This matters because the model
    must be evaluated on "the future, corridor-wide," not on a
    road-by-road shuffle.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    unique_timestamps = df["timestamp"].sort_values().unique()
    n = len(unique_timestamps)

    train_cutoff = unique_timestamps[int(n * train_frac)]
    val_cutoff = unique_timestamps[int(n * (train_frac + val_frac))]

    train = df[df["timestamp"] < train_cutoff]
    val = df[(df["timestamp"] >= train_cutoff) & (df["timestamp"] < val_cutoff)]
    test = df[df["timestamp"] >= val_cutoff]

    logger.info(f"Train: {len(train)} rows, {train['timestamp'].min()} to {train['timestamp'].max()}")
    logger.info(f"Val:   {len(val)} rows, {val['timestamp'].min()} to {val['timestamp'].max()}")
    logger.info(f"Test:  {len(test)} rows, {test['timestamp'].min()} to {test['timestamp'].max()}")

    # Sanity assertion: no overlap, no gaps, chronological order preserved
    assert train["timestamp"].max() < val["timestamp"].min(), "Train/val overlap detected!"
    assert val["timestamp"].max() < test["timestamp"].min(), "Val/test overlap detected!"

    return train, val, test




if __name__ == "__main__":
    df = pd.read_parquet("data/processed/flowcast_features.parquet")
    logger.info(f"Loaded flowcast_features.parquet: {df.shape}")

    df = add_forecast_targets(df)

    print("\nFinal shape:", df.shape)

    # --- Verification: pick one road, check the target genuinely
    # matches the NEXT row's actual value, not the current row's. ---
    sample = df[df["road_id"] == "NL-007"][
        ["timestamp", "traffic_volume", "target_volume_next", "congestion_level", "target_congestion_next"]
    ].head(6)
    print("\nVerification sample (NL-007):")
    print(sample)

    # --- Verification: the LAST row of a road's series should have
    # NaN targets (nothing to look forward to). ---
    last_row = df[df["road_id"] == "NL-007"].tail(1)
    print("\nLast row of NL-007's series (targets should be NaN):")
    print(last_row[["timestamp", "target_volume_next", "target_congestion_next"]])

    print("\nNull counts per target:")
    for col in ["target_volume_next", "target_travel_time_next", "target_congestion_next", "target_accident_next"]:
        print(f"  {col}: {df[col].isnull().sum()}")




    print("new Section")

    # Drop rows with no forecasting target (each road's last row — can't
    # evaluate a prediction with no ground truth to compare against)
    df_modeling = df.dropna(subset=["target_volume_next", "target_congestion_next"])
    logger.info(f"Dropped {len(df) - len(df_modeling)} rows with no next-window target "
                f"(expected: 25, one per road)")

    train, val, test = time_based_split(df_modeling)

    print("\nSplit sizes:")
    print(f"  Train: {len(train)} ({len(train)/len(df_modeling)*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/len(df_modeling)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(df_modeling)*100:.1f}%)")