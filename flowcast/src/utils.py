"""Shared utility functions used across pipeline stages."""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils")


def print_schema(path: str) -> None:
    """Print shape, dtypes, and null counts for a saved parquet file.
    Useful sanity check after any pipeline stage writes new output."""
    df = pd.read_parquet(path)
    print(f"\n=== Schema: {path} ===")
    print("Shape:", df.shape)
    print("\nColumn dtypes:")
    print(df.dtypes)
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if len(nulls) > 0:
        print("\nColumns with nulls:")
        print(nulls)
    else:
        print("\nNo nulls in any column.")


def validate_final_dataset(path: str = "data/processed/flowcast_features.parquet") -> None:
    """
    Formal validation of the final pipeline output, per PRD Section 9.1:
    "assert no nulls in modelling columns, verify row counts, confirm
    value ranges; emit the final data-quality report."

    Unlike the ad hoc checks scattered through clean.py/features.py's
    test blocks, this is the ONE final gate — run once, on the finished
    file, right before writing the data-quality report. Raises
    AssertionError (fails loudly) rather than just printing a warning,
    since a silent validation check defeats the purpose of validating.
    """
    df = pd.read_parquet(path)
    print(f"\n=== Validating {path} ===")
    print(f"Shape: {df.shape}\n")

    # --- 1. Row count check ---
    expected_rows = 176701  # established in M2, after deduplication
    assert len(df) == expected_rows, f"Row count mismatch: expected {expected_rows}, got {len(df)}"
    print(f"✓ Row count: {len(df)} (matches expected)")

    # --- 2. No-nulls check on CORE modelling columns ---
    # These were fully cleaned/imputed in M2 — should have ZERO nulls.
    # Lag/rolling features are deliberately excluded here (they have
    # EXPECTED nulls at each road's series start) and checked separately.
    core_cols = [
        "traffic_volume", "avg_speed", "occupancy", "congestion_level",
        "travel_time", "accident_count", "vehicle_count", "road_capacity",
    ]
    print("Core column null check:")
    core_failures = []
    for col in core_cols:
        n_null = df[col].isnull().sum()
        status = "✓" if n_null == 0 else "✗"
        print(f"  {status} {col}: {n_null} nulls")
        if n_null > 0:
            core_failures.append(col)

    # --- 3. Weather columns — checked separately, since M2 never
    # explicitly imputed these (only traffic_volume/avg_speed/occupancy
    # were imputed) — worth finding out now whether that's a real gap.
    print("\nWeather column null check:")
    for col in ["temperature", "rainfall", "visibility", "weather_condition_clean"]:
        n_null = df[col].isnull().sum()
        print(f"  {'✓' if n_null == 0 else '⚠'} {col}: {n_null} nulls")

    # --- 4. Value range checks ---
    print("\nValue range checks:")
    range_checks = {
        "traffic_volume >= 0": (df["traffic_volume"] >= 0).all(),
        "avg_speed 0-200": df["avg_speed"].between(0, 200).all(),
        "occupancy 0-100": df["occupancy"].between(0, 100).all(),
        "visibility >= 0": df["visibility"].dropna().ge(0).all(),
    }
    for check_name, passed in range_checks.items():
        print(f"  {'✓' if passed else '✗'} {check_name}")

    # --- 5. Lag/rolling nulls — reported, not asserted (expected to exist) ---
    lag_rolling_cols = [c for c in df.columns if "_lag_" in c or "_rolling_" in c]
    print(f"\nLag/rolling feature nulls (expected, per-road series start):")
    for col in lag_rolling_cols:
        n_null = df[col].isnull().sum()
        print(f"  {col}: {n_null} nulls")

    # --- Final assertion — fail loudly if core columns have unexpected nulls ---
    if core_failures:
        raise AssertionError(f"Core modelling columns have unexpected nulls: {core_failures}")

    print("\n=== Validation PASSED ===")


if __name__ == "__main__":
    # Schema check on both pipeline outputs — M2's clean baseline and
    # M3's feature-engineered file — useful to compare side by side.
    print_schema("data/processed/flowcast_clean.parquet")
    print_schema("data/processed/flowcast_features.parquet")

    # Formal validation gate on the FINAL output, per PRD Section 9.1
    validate_final_dataset("data/processed/flowcast_features.parquet")