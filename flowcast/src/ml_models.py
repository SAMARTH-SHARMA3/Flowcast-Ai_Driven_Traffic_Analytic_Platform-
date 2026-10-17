"""
M5: Classical ML Engine — Shared Library & Data Prep
-------------------------------------------------------
Contains the functions reused across all per-model training scripts
(train_decision_tree.py, train_random_forest.py, etc.), and a __main__
block that runs ONLY the expensive one-time data prep (targets, split,
features) and the two fast Linear Regression models, then saves the
split DataFrames to a pickle for reuse — avoiding re-running the full
pipeline every time a slower model (Random Forest, XGBoost) is tuned.

IMPORTANT: if you change anything upstream (M3 features, target
construction, split ratios), rerun this file FIRST to regenerate the
pickle before running any train_*.py script — they load from the
pickle, not from flowcast_features.parquet directly.
"""

import pandas as pd
import numpy as np
import logging
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_models")


# ============================================================
# Forward-looking ("next window") forecasting targets
# ============================================================
def add_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 4 forward-looking ("next window") targets per PRD 11.1.
    Uses shift(-1), grouped by road_id — the mirror image of M3's
    backward shift() for lag features. The LAST row of each road's
    series gets NaN for every target (no "next window" to look to).
    """
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

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
        logger.info(f"{target_col}: {n_null} nulls (expected — one per road, series' last row)")

    return df


# ============================================================
# Time-based train/val/test split
# ============================================================
def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically, per PRD 11.3: "earliest for training,
    latest for testing — never randomly."
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

    assert train["timestamp"].max() < val["timestamp"].min(), "Train/val overlap detected!"
    assert val["timestamp"].max() < test["timestamp"].min(), "Val/test overlap detected!"

    return train, val, test


# ============================================================
# Per-target feature selection
# ============================================================
def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """
    Valid feature columns for a given target. Excludes IDs/strings,
    all 4 target columns, and (for target_volume_next specifically)
    v_c_ratio/capacity_headroom/vehicle_count — near-duplicates of
    CURRENT traffic_volume (M4 Section 5.2). congestion_level's
    one-hot columns ARE included for every target (legitimate current-
    window signal, not leakage).
    """
    always_exclude = {
        "road_id", "road_name", "weather_station_id", "date", "time",
        "timestamp", "weather_join_hour", "weather_condition_clean",
        "temperature_band", "holiday_name", "event_name",
        "target_volume_next", "target_travel_time_next",
        "target_congestion_next", "target_accident_next",
    }

    exclude = set(always_exclude)
    if target == "target_volume_next":
        exclude |= {"v_c_ratio", "capacity_headroom", "vehicle_count"}

    feature_cols = [c for c in df.columns if c not in exclude and c != "congestion_level"]
    return feature_cols


# ============================================================
# Scaling + data prep (fit scaler on TRAIN only, per PRD 11.3)
# ============================================================
def prepare_model_data(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    feature_cols: list[str], target_col: str,
):
    """
    Scaled X/y arrays for train/val/test. Drops rows with NaN in
    features/target (lag/rolling series-start nulls). StandardScaler
    fit ONLY on train — val/test are transformed, never re-fit
    (fitting on the full dataset would leak test-distribution info
    into scaling).
    """
    def clean(df):
        before = len(df)
        df_clean = df.dropna(subset=feature_cols + [target_col])
        logger.info(f"  Dropped {before - len(df_clean)} rows with NaN ({before} -> {len(df_clean)})")
        return df_clean

    train_c, val_c, test_c = clean(train), clean(val), clean(test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_c[feature_cols])
    X_val = scaler.transform(val_c[feature_cols])
    X_test = scaler.transform(test_c[feature_cols])

    y_train = train_c[target_col].values
    y_val = val_c[target_col].values
    y_test = test_c[target_col].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ============================================================
# Regression metrics (PRD 13.2: RMSE primary, MAE/MAPE/R² supporting)
# ============================================================
def regression_metrics(y_true, y_pred, label: str = "") -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    logger.info(f"{label} — RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# ============================================================
# Overfitting check (train vs. test comparison)
# ============================================================
def check_overfitting(model, X_train, y_train, X_test, y_test, label: str = "") -> None:
    """Large train-vs-test gap signals overfitting — especially
    relevant for unconstrained Decision Trees, which can perfectly
    memorize training data."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f"\n--- {label}: Train vs. Test ---")
    regression_metrics(y_train, y_pred_train, label="Train")
    regression_metrics(y_test, y_pred_test, label="Test")


# ============================================================
# Linear Regression from scratch (NumPy), per Build-it-once rule
# ============================================================
class LinearRegressionScratch:
    """
    Batch gradient descent, implemented directly with NumPy.
    Model: ŷ = Xw + b | Cost (MSE): J = mean((ŷ-y)^2)
    Gradients: dJ/dw = (2/n)X^T(ŷ-y), dJ/db = (2/n)sum(ŷ-y)
    """
    def __init__(self, learning_rate: float = 0.03, n_iterations: int = 3000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            cost = np.mean((y_pred - y) ** 2)

            # Divergence guard — stop if too-high a learning rate
            # caused gradient descent to overshoot and blow up.
            if not np.isfinite(cost):
                logger.warning(f"Iteration {i}: cost diverged — learning_rate too high. Stopping.")
                break

            self.cost_history.append(cost)
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 500 == 0:
                logger.info(f"  Iteration {i}: cost = {cost:.2f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


if __name__ == "__main__":
    # --- Load + build targets (one-time, relatively cheap) ---
    df = pd.read_parquet("data/processed/flowcast_features.parquet")
    logger.info(f"Loaded flowcast_features.parquet: {df.shape}")
    df = add_forecast_targets(df)

    df_modeling = df.dropna(subset=["target_volume_next", "target_congestion_next"])
    logger.info(f"Dropped {len(df) - len(df_modeling)} rows with no next-window target")

    train, val, test = time_based_split(df_modeling)

    target = "target_volume_next"
    feature_cols = get_feature_columns(df_modeling, target)
    logger.info(f"Feature count for {target}: {len(feature_cols)}")

    # --- Fast models: both Linear Regression versions ---
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
        train, val, test, feature_cols, target
    )

    print("\n=== Linear Regression (sklearn) ===")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    metrics_lr = regression_metrics(y_test, lr.predict(X_test), label="Test")

    print("\n=== Linear Regression (from-scratch NumPy) ===")
    lr_scratch = LinearRegressionScratch(learning_rate=0.03, n_iterations=3000)
    lr_scratch.fit(X_train, y_train)
    metrics_scratch = regression_metrics(y_test, lr_scratch.predict(X_test), label="Test")

    # --- Save split DataFrames for reuse by per-model training scripts ---
    with open("data/processed/model_data_volume.pkl", "wb") as f:
        pickle.dump({
            "train": train, "val": val, "test": test,
            "feature_cols": feature_cols, "target": target,
        }, f)
    logger.info("Saved train/val/test splits -> data/processed/model_data_volume.pkl")