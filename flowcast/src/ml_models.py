"""
M5: Classical ML Engine — Shared Library & Data Prep
-------------------------------------------------------
Shared functions reused by every per-model training script
(train_decision_tree.py, train_random_forest.py, train_xgboost.py), plus
a __main__ block that runs the one-time data prep:

  - Regression targets (volume, travel time): fast baselines (both
    Linear Regression versions and a naive persistence baseline), and a
    pickle of the train/val/test splits per target.
  - Classification targets (congestion, accident risk): class
    distribution per split, a pickle per target, and the naive
    baselines for congestion (majority class and persistence).

The pickles let the slower models (Random Forest, XGBoost) skip
re-running the full pipeline on every experiment.

IMPORTANT: if anything upstream changes (M3 features, target
construction, split ratios), rerun this file FIRST to regenerate the
pickles — the train_*.py scripts load from the pickles, not from
flowcast_features.parquet. Pickles are gitignored (they can exceed
GitHub's 100MB limit and are regeneratable).
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
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
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

    # Binary accident flag (per M4 Section 5.4: accident_count is 99%+
    # zero, so "any incident vs none" is more tractable than 3-class)
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
    Split chronologically, per PRD 11.3: "earliest windows for
    training, latest for testing — never randomly."
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
    Valid feature columns for a given target. Excludes IDs/strings and
    all 4 target columns. For target_volume_next ONLY, also excludes
    v_c_ratio / capacity_headroom / vehicle_count — near-duplicates of
    CURRENT traffic_volume (M4 Section 5.2); for other targets they
    remain valid. congestion_level's one-hot columns ARE included for
    every target (legitimate current-window signal, not leakage).
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

    # The raw congestion_level string is excluded; its one-hot columns
    # (congestion_Free-flow, etc.) are used instead.
    return [c for c in df.columns if c not in exclude and c != "congestion_level"]


# ============================================================
# Scaling + data prep (scaler fit on TRAIN only, per PRD 11.3)
# ============================================================
def prepare_model_data(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    feature_cols: list[str], target_col: str,
):
    """
    Scaled X/y arrays for train/val/test. Drops rows with NaN in the
    features or target (lag/rolling series-start nulls). StandardScaler
    is fit ONLY on train — val/test are transformed, never re-fit
    (fitting on the full dataset would leak test-distribution
    information into scaling).
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
# Regression metrics (PRD 13.2: RMSE primary; MAE, MAPE, R² supporting)
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
    """A large train-vs-test gap signals overfitting — especially
    relevant for unconstrained Decision Trees, which can perfectly
    memorize their training data."""
    print(f"\n--- {label}: Train vs. Test ---")
    regression_metrics(y_train, model.predict(X_train), label="Train")
    regression_metrics(y_test, model.predict(X_test), label="Test")


# ============================================================
# Classification metrics (PRD 13.2)
# ============================================================
# Ordered Free-flow -> Severe so confusion matrices read in severity order.
CONGESTION_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]


def classification_metrics(y_true, y_pred, labels: list, label: str = "") -> dict:
    """
    Metrics PRD 13.2 specifies for the 4-class congestion target:
    macro-F1 (primary); per-class precision/recall, confusion matrix and
    accuracy (supporting). zero_division=0 means a class that is never
    predicted scores 0 instead of raising a warning.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info(f"{label} — macro-F1: {macro_f1:.4f}, accuracy: {acc:.4f}")
    per_class = pd.DataFrame(
        {"precision": prec, "recall": rec, "f1": f1, "support": support}, index=labels
    )
    return {"macro_f1": macro_f1, "accuracy": acc, "per_class": per_class, "confusion_matrix": cm}


def show_classification_report(result: dict, labels: list) -> None:
    """Print per-class metrics and the confusion matrix (rows = true class,
    columns = predicted class)."""
    print(result["per_class"].round(4))
    cm = pd.DataFrame(
        result["confusion_matrix"],
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm)


# ============================================================
# Linear Regression from scratch (NumPy) — PRD Build-it-once rule
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

            # Divergence guard — stop if too high a learning rate
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


# ============================================================
# MAIN: one-time data prep, fast baselines, pickles
# ============================================================
if __name__ == "__main__":
    # --- Load + build targets (one-time, shared across all targets) ---
    df = pd.read_parquet("data/processed/flowcast_features.parquet")
    logger.info(f"Loaded flowcast_features.parquet: {df.shape}")
    df = add_forecast_targets(df)

    df_modeling = df.dropna(subset=["target_volume_next", "target_congestion_next"])
    logger.info(f"Dropped {len(df) - len(df_modeling)} rows with no next-window target")

    train, val, test = time_based_split(df_modeling)

    # ============================================================
    # REGRESSION TARGETS (FR-07): fast baselines + a pickle per target
    # ============================================================
    for target in ["target_volume_next", "target_travel_time_next"]:
        print(f"\n{'='*60}\nTARGET: {target}\n{'='*60}")

        feature_cols = get_feature_columns(df_modeling, target)
        logger.info(f"Feature count for {target}: {len(feature_cols)}")

        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
            train, val, test, feature_cols, target
        )

        print(f"\n=== Linear Regression (sklearn) — {target} ===")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        metrics_lr = regression_metrics(y_test, lr.predict(X_test), label="Test")

        print(f"\n=== Linear Regression (from-scratch NumPy) — {target} ===")
        lr_scratch = LinearRegressionScratch(learning_rate=0.03, n_iterations=3000)
        lr_scratch.fit(X_train, y_train)
        metrics_scratch = regression_metrics(y_test, lr_scratch.predict(X_test), label="Test")

        # Save pickle, named per target — model_data_volume.pkl,
        # model_data_travel_time.pkl
        pickle_name = target.replace("target_", "model_data_").replace("_next", "")
        pickle_path = f"data/processed/{pickle_name}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump({
                "train": train, "val": val, "test": test,
                "feature_cols": feature_cols, "target": target,
            }, f)
        logger.info(f"Saved train/val/test splits -> {pickle_path}")

        # Persistence baseline: predict next window = current window
        current_col = "traffic_volume" if target == "target_volume_next" else "travel_time"
        test_c = test.dropna(subset=feature_cols + [target])
        print(f"\n=== Persistence baseline — {target} ===")
        regression_metrics(test_c[target].values, test_c[current_col].values, label="Test (persistence)")

    # ============================================================
    # CLASSIFICATION TARGETS (FR-08): congestion (4-class), accident (binary)
    # ============================================================
    for target, name in [("target_congestion_next", "congestion"),
                         ("target_accident_next", "accident")]:
        print(f"\n{'='*60}\nTARGET: {target}\n{'='*60}")
        feature_cols = get_feature_columns(df_modeling, target)
        logger.info(f"Feature count for {target}: {len(feature_cols)}")

        # Class balance in each chronological split. A time-based split can
        # shift class frequencies, and a rare class with few test examples
        # makes its metrics noisy — worth knowing before modelling.
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            counts = split_df[target].value_counts()
            pct = (counts / counts.sum() * 100).round(2)
            print(f"\n{split_name} class distribution:")
            print(pd.DataFrame({"count": counts, "pct": pct}))

        pickle_path = f"data/processed/model_data_{name}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump({
                "train": train, "val": val, "test": test,
                "feature_cols": feature_cols, "target": target,
            }, f)
        logger.info(f"Saved train/val/test splits -> {pickle_path}")

    # ============================================================
    # Congestion baselines, scored on the same test rows the models will use
    # ============================================================
    target = "target_congestion_next"
    feature_cols = get_feature_columns(df_modeling, target)
    train_c = train.dropna(subset=feature_cols + [target])
    test_c = test.dropna(subset=feature_cols + [target])

    # Majority class, taken from TRAIN only (never from test)
    majority_class = train_c[target].mode()[0]
    print(f"\n=== Baseline: always predict '{majority_class}' (train majority) ===")
    res = classification_metrics(
        test_c[target], [majority_class] * len(test_c), CONGESTION_ORDER, label="Test (majority class)"
    )
    show_classification_report(res, CONGESTION_ORDER)

    # Persistence: next window's congestion = current window's congestion
    print("\n=== Baseline: persistence (next = current congestion level) ===")
    res = classification_metrics(
        test_c[target], test_c["congestion_level"], CONGESTION_ORDER, label="Test (persistence)"
    )
    show_classification_report(res, CONGESTION_ORDER)


    # ============================================================
    # verified how the congestion labels relate to the dictionary’s volume-to-capacity rule
    # ============================================================

    df = pd.read_parquet("data/processed/flowcast_features.parquet")

def band(r):
    # Same bands as M2's derive_congestion_level
    if r < 0.5: return "Free-flow"
    if r < 0.8: return "Moderate"
    if r < 1.0: return "Heavy"
    return "Severe"

derived = df["v_c_ratio"].apply(band)
match = (derived == df["congestion_level"])
m = match.mean()
print(f"Labels equal to the V/C band: {m*100:.2f}% ({match.sum()} of {len(df)})")
print(f"Implied match rate on the ORIGINAL (non-derived) labels: {(m - 0.1507) / 0.8493 * 100:.1f}%")
print(pd.crosstab(df["congestion_level"], derived, rownames=["label"], colnames=["V/C band"]))