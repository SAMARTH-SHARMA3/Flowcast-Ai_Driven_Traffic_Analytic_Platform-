"""Shared utility functions used across pipeline stages."""

import pandas as pd
import logging

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


if __name__ == "__main__":
    print_schema("data/processed/flowcast_clean.parquet")