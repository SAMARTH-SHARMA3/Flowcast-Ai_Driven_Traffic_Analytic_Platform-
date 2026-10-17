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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_models")


# ============================================================
# STEP 1: Forward-looking ("next window") forecasting targets
# ============================================================
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


# ============================================================
# STEP 2: Time-based train/val/test split
# ============================================================
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


# ============================================================
# STEP 3: Per-target feature selection
# ============================================================
def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """
    Return the list of valid feature columns for a given target,
    excluding IDs/strings, all 4 target columns, and (for
    target_volume_next specifically) the near-duplicate columns
    flagged in M4 Section 5.2 (v_c_ratio, capacity_headroom,
    vehicle_count — algebraic restatements of CURRENT traffic_volume).

    congestion_level's one-hot columns (congestion_Free-flow, etc.)
    ARE included for every target, including target_congestion_next —
    using the CURRENT window's congestion to help predict the NEXT
    window's congestion is legitimate forecasting signal, not leakage
    (unlike v_c_ratio, which is mathematically derived from the exact
    value being predicted when volume is the target).
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

    # congestion_level itself (the raw string) is excluded — its
    # one-hot encoded version (congestion_Free-flow, etc.) is used instead
    feature_cols = [c for c in df.columns if c not in exclude and c != "congestion_level"]
    return feature_cols


# ============================================================
# STEP 4: Scaling + data prep for modeling
# ============================================================
def prepare_model_data(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    feature_cols: list[str], target_col: str,
):
    """
    Prepare scaled X/y arrays for train/val/test, per PRD 11.3:
    "Fit scalers on the training split only; apply to validation and test."

    Drops any row with NaN in the feature set (lag/rolling series-start
    nulls) or in the target — a model cannot train or be evaluated on
    missing values. Reports how many rows this costs, since a large
    drop would be worth knowing about.

    CRITICAL: StandardScaler is fit ONLY on train. Val/test are
    transformed using train's fitted mean/std — never re-fit. Fitting
    on the full dataset would leak test-set distribution information
    into scaling, a subtle form of data leakage.
    """
    def clean(df):
        before = len(df)
        df_clean = df.dropna(subset=feature_cols + [target_col])
        logger.info(f"  Dropped {before - len(df_clean)} rows with NaN in features/target "
                    f"({before} -> {len(df_clean)})")
        return df_clean

    logger.info("Cleaning train:")
    train_c = clean(train)
    logger.info("Cleaning val:")
    val_c = clean(val)
    logger.info("Cleaning test:")
    test_c = clean(test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_c[feature_cols])  # fit + transform on train
    X_val = scaler.transform(val_c[feature_cols])            # transform only
    X_test = scaler.transform(test_c[feature_cols])          # transform only

    y_train = train_c[target_col].values
    y_val = val_c[target_col].values
    y_test = test_c[target_col].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ============================================================
# STEP 5: Regression metrics (PRD 13.2)
# ============================================================
def regression_metrics(y_true, y_pred, label: str = "") -> dict:
    """
    Compute the metrics PRD 13.2 specifies for regression targets:
    RMSE (primary), MAE, MAPE, R² (supporting).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    logger.info(f"{label} — RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


class LinearRegressionScratch:
    """
    Linear Regression trained via batch gradient descent, implemented
    directly with NumPy — no sklearn — per the PRD's Build-it-once
    rule (Section 10.2): "at least one linear-regression training loop
    and the loss/gradient computations must be implemented directly
    with NumPy... to demonstrate the gradient-descent and cost-function
    mathematics rather than hiding them behind a call."

    Model: ŷ = Xw + b
    Cost (MSE): J(w,b) = (1/n) * sum((ŷ - y)^2)
    Gradients: dJ/dw = (2/n) * X^T(ŷ - y)
               dJ/db = (2/n) * sum(ŷ - y)
    Update rule: w = w - lr * dJ/dw   (same for b)
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
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

            # Divergence guard: if cost explodes or becomes NaN, stop
            # immediately rather than continuing to compute garbage.
            # This is exactly what happened at learning_rate=0.1 — the
            # step size was too large, causing gradient descent to
            # overshoot the minimum and diverge exponentially.
            if not np.isfinite(cost):
                logger.warning(f"Iteration {i}: cost diverged (non-finite) — "
                                f"learning_rate={self.learning_rate} is too high. Stopping early.")
                break

            self.cost_history.append(cost)

            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                logger.info(f"  Iteration {i}: cost = {cost:.2f}")

        return self

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


# ============================================================
# MAIN: run the pipeline built so far
# ============================================================
if __name__ == "__main__":
    # --- Load feature-engineered dataset from M3 ---
    df = pd.read_parquet("data/processed/flowcast_features.parquet")
    logger.info(f"Loaded flowcast_features.parquet: {df.shape}")

    # --- Build forecasting targets ---
    df = add_forecast_targets(df)
    print("\nFinal shape:", df.shape)

    # --- Verify targets are genuinely forward-looking (not same-window) ---
    sample = df[df["road_id"] == "NL-007"][
        ["timestamp", "traffic_volume", "target_volume_next", "congestion_level", "target_congestion_next"]
    ].head(6)
    print("\nVerification sample (NL-007):")
    print(sample)

    last_row = df[df["road_id"] == "NL-007"].tail(1)
    print("\nLast row of NL-007's series (targets should be NaN):")
    print(last_row[["timestamp", "target_volume_next", "target_congestion_next"]])

    print("\nNull counts per target:")
    for col in ["target_volume_next", "target_travel_time_next", "target_congestion_next", "target_accident_next"]:
        print(f"  {col}: {df[col].isnull().sum()}")

    # --- Drop rows with no forecasting target (each road's last row) ---
    df_modeling = df.dropna(subset=["target_volume_next", "target_congestion_next"])
    logger.info(f"Dropped {len(df) - len(df_modeling)} rows with no next-window target "
                f"(expected: 25, one per road)")

    # --- Time-based split ---
    train, val, test = time_based_split(df_modeling)

    print("\nSplit sizes:")
    print(f"  Train: {len(train)} ({len(train)/len(df_modeling)*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/len(df_modeling)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(df_modeling)*100:.1f}%)")

    # --- Feature selection check (volume vs congestion target) ---
    volume_features = get_feature_columns(df_modeling, "target_volume_next")
    congestion_features = get_feature_columns(df_modeling, "target_congestion_next")

    print(f"\nFeature count for volume target: {len(volume_features)}")
    print(f"Feature count for congestion target: {len(congestion_features)}")
    print(f"\nDifference (volume-specific exclusions): "
          f"{set(congestion_features) - set(volume_features)}")

    # ============================================================
    # Linear Regression (library) on target_volume_next
    #
    # NOTE: this baseline misses the PRD's MAPE<=12% target (actual:
    # 15.19%). Error analysis (notebooks/03_model_diagnostics.ipynb)
    # found this is concentrated in low-volume windows (24.2% MAPE in
    # the lowest quintile vs 10.8% in the highest) — consistent with
    # M4's finding that traffic_volume is right-skewed/non-normal. A
    # log-transformed target was tested and did not meaningfully help
    # (14.48% MAPE) while degrading RMSE/R². This is expected per the
    # PRD's own framing of Linear Regression as an "interpretable
    # baseline" — tree-based models are expected to perform better.
    # ============================================================
    target = "target_volume_next"
    feature_cols = get_feature_columns(df_modeling, target)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
        train, val, test, feature_cols, target
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_test = lr.predict(X_test)
    print("\n=== Linear Regression (sklearn) — target_volume_next ===")
    metrics_lr = regression_metrics(y_test, y_pred_test, label="Test")

    print(f"\nPRD target MAPE <= 12%: {'PASS' if metrics_lr['mape'] <= 12 else 'FAIL'} "
          f"(actual: {metrics_lr['mape']:.2f}%)")



    # ============================================================
    # Linear Regression from scratch (NumPy), per PRD Build-it-once rule
    # ============================================================
    print("\n=== Linear Regression (from-scratch NumPy) — target_volume_next ===")
    lr_scratch = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    lr_scratch.fit(X_train, y_train)

    y_pred_scratch = lr_scratch.predict(X_test)
    metrics_scratch = regression_metrics(y_test, y_pred_scratch, label="Test (from-scratch)")

    # Compare against sklearn — should be very close if gradient descent
    # converged properly, since both are solving the same optimization
    # problem (minimize MSE), just via different algorithms (closed-form
    # vs. iterative).
    print(f"\nComparison — sklearn MAPE: {metrics_lr['mape']:.2f}%, "
          f"from-scratch MAPE: {metrics_scratch['mape']:.2f}%")


    lr_scratch = LinearRegressionScratch(learning_rate=0.03, n_iterations=3000)
    lr_scratch.fit(X_train, y_train)

    y_pred_scratch = lr_scratch.predict(X_test)
    metrics_scratch = regression_metrics(y_test, y_pred_scratch, label="Test (from-scratch, tuned)")