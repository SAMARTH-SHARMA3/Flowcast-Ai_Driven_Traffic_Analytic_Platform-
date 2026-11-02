"""
Random Forest training/tuning for a regression target.

Usage:
    python src/train_random_forest.py                     # target_volume_next (default)
    python src/train_random_forest.py --target travel_time

Loads pre-split data from data/processed/model_data_<target>.pkl
(produced by ml_models.py).
"""

import argparse
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml_models import prepare_model_data, regression_metrics, logger

# ============================================================
# Which target to train on (default: volume, so the original
# command keeps working unchanged)
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--target", default="volume", choices=["volume", "travel_time"],
                    help="Which regression target to train on")
args = parser.parse_args()

with open(f"data/processed/model_data_{args.target}.pkl", "rb") as f:
    data = pickle.load(f)

train, val, test = data["train"], data["val"], data["test"]
feature_cols, target = data["feature_cols"], data["target"]
print(f"Training Random Forest on: {target}")

X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
    train, val, test, feature_cols, target
)

# ============================================================
# Model-selection rule (judgment call, stated explicitly):
# - volume: select on validation MAPE — it is this target's PRD
#   success criterion (<=12%), and it keeps the previously
#   documented volume results reproducible.
# - travel_time: select on validation RMSE — the PRD's primary
#   regression metric, and this target has no MAPE threshold.
# Selection ALWAYS uses the validation set, never test.
# ============================================================
select_metric = "mape" if args.target == "volume" else "rmse"
print(f"Model selection metric (validation): {select_metric.upper()}")


# Depth grid is target-specific: for travel_time the first run showed
# validation RMSE rising monotonically with depth (best at the lower
# edge, depth 10), so shallower depths are added for that target.
# Volume keeps its original grid so its documented results stay
# reproducible.
depth_grid = [10, 15, None] if args.target == "volume" else [5, 8, 10, 15, None]

# ============================================================
# Grid search over n_estimators x max_depth, evaluated on VAL
# (n_jobs=-1 parallelizes tree training across CPU cores)
# ============================================================
print("\n=== Random Forest — hyperparameter search (validation set) ===")
rf_results = []
for n_est in [50, 100, 200]:
    for depth in depth_grid:
        rf = RandomForestRegressor(
            n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        m = regression_metrics(
            y_val, rf.predict(X_val), label=f"n_est={n_est}, depth={depth}, Val"
        )
        rf_results.append({"n_estimators": n_est, "max_depth": depth, **m})

# Display table (a DataFrame turns None into NaN, so it is used for
# DISPLAY ONLY — selection below uses the original list so that
# max_depth=None stays a real None when passed back to the model).
print(f"\nAll configurations, sorted by validation {select_metric.upper()}:")
print(pd.DataFrame(rf_results).sort_values(select_metric))

best = min(rf_results, key=lambda r: r[select_metric])
best_n, best_depth = best["n_estimators"], best["max_depth"]
print(f"\nSelected: n_estimators={best_n}, max_depth={best_depth} "
      f"(validation {select_metric.upper()}: {best[select_metric]:.4g})")

# If the winning depth is the shallowest in the grid, a shallower
# forest might do better — worth widening the grid and rerunning.
if best_depth == depth_grid[0]:
    print(f"NOTE: best depth ({best_depth}) is at the lower edge of the grid — consider trying shallower depths.")

# ============================================================
# Final model at the selected config — evaluated on TEST
# ============================================================
print(f"\n=== Random Forest (n_est={best_n}, depth={best_depth}, final) ===")
rf_final = RandomForestRegressor(
    n_estimators=best_n, max_depth=best_depth, random_state=42, n_jobs=-1
)
rf_final.fit(X_train, y_train)
metrics_rf = regression_metrics(y_test, rf_final.predict(X_test), label="Test (final)")

# The 12% MAPE criterion applies to VOLUME forecast accuracy only.
if args.target == "volume":
    print(f"PRD MAPE <= 12%: {'PASS' if metrics_rf['mape'] <= 12 else 'FAIL'} ({metrics_rf['mape']:.2f}%)")

importances_rf = pd.Series(rf_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances_rf.head(10))

# ============================================================
# Comparison against earlier models (test set). These reference
# numbers are copied from earlier runs, so update them if those
# models are retrained.
# ============================================================
print("\nComparison across models so far (test set):")
if args.target == "volume":
    print("  Linear Regression (sklearn):  MAPE 15.19%")
    print("  Linear Regression (scratch):  MAPE 15.35%")
    print("  Decision Tree (depth=10):     MAPE 11.50%")
    print(f"  Random Forest:                MAPE {metrics_rf['mape']:.2f}%")
else:
    print("  Linear Regression (sklearn):  RMSE 1.38, R² 0.7233")
    print("  Linear Regression (scratch):  RMSE 1.38, R² 0.7233")
    print(f"  Random Forest:                RMSE {metrics_rf['rmse']:.2f}, R² {metrics_rf['r2']:.4f}")