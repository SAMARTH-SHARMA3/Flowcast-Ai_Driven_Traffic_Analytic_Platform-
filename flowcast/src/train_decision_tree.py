"""
Decision Tree training/tuning for a regression target.

Usage:
    python src/train_decision_tree.py                    # target_volume_next (default)
    python src/train_decision_tree.py --target travel_time

Loads pre-split data from data/processed/model_data_<target>.pkl
(produced by ml_models.py) rather than re-running the full data
pipeline.
"""

import argparse
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ml_models import prepare_model_data, regression_metrics, check_overfitting, logger

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
print(f"Training Decision Tree on: {target}")

X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
    train, val, test, feature_cols, target
)

# ============================================================
# Model-selection rule (judgment call, stated explicitly):
# - volume: select on validation MAPE — it is this target's PRD
#   success criterion (<=12%), and it keeps the documented volume
#   results reproducible.
# - travel_time: select on validation RMSE — the PRD's primary
#   regression metric, and this target has no MAPE threshold.
# Same rule as train_random_forest.py and train_xgboost.py.
# Selection ALWAYS uses the validation set, never test.
# ============================================================
select_metric = "mape" if args.target == "volume" else "rmse"
print(f"Model selection metric (validation): {select_metric.upper()}")

# ============================================================
# Unconstrained tree — demonstrate overfitting
# ============================================================
print("\n=== Decision Tree (unconstrained) ===")
dt_unconstrained = DecisionTreeRegressor(random_state=42)
dt_unconstrained.fit(X_train, y_train)
check_overfitting(dt_unconstrained, X_train, y_train, X_test, y_test, label="Unconstrained Tree")

# ============================================================
# Depth tuning on the VALIDATION set (never test — tuning on test
# would leak the final evaluation data into model selection)
# ============================================================
depth_grid = [3, 5, 8, 10, 15, None]

print("\n=== Decision Tree — depth tuning (validation set) ===")
depth_results = []
for depth in depth_grid:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    m = regression_metrics(y_val, dt.predict(X_val), label=f"max_depth={depth}, Val")
    depth_results.append({"max_depth": depth, **m})

# Selection uses the original list (not a DataFrame), so max_depth=None
# stays a real None. On ties, min() keeps the first (shallowest) depth.
best = min(depth_results, key=lambda r: r[select_metric])
best_depth = best["max_depth"]
print(f"\nSelected max_depth={best_depth} "
      f"(lowest validation {select_metric.upper()}: {best[select_metric]:.4g})")

# Edge-of-grid checks. A winner at the SHALLOW edge means a shallower
# tree might do better, so the grid is worth widening. A winner of None
# (unlimited depth) cannot be widened past — but it deserves a note,
# since an unconstrained tree is the overfitting-prone case.
if best_depth == depth_grid[0]:
    print(f"NOTE: best depth ({best_depth}) is at the lower edge of the grid — "
          "consider trying shallower depths.")
if best_depth is None:
    print("NOTE: unconstrained depth won on validation — check the train/test "
          "gap above before trusting it.")

# ============================================================
# Final model at the selected depth — evaluated on TEST
# ============================================================
print(f"\n=== Decision Tree (max_depth={best_depth}, final) ===")
dt_final = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
dt_final.fit(X_train, y_train)
metrics_dt = regression_metrics(y_test, dt_final.predict(X_test), label="Test (final)")

# The 12% MAPE criterion is a PRD success criterion for VOLUME forecast
# accuracy only — travel time has no numeric threshold (RMSE is its
# primary metric), so don't print pass/fail for it.
if args.target == "volume":
    print(f"PRD MAPE <= 12%: {'PASS' if metrics_dt['mape'] <= 12 else 'FAIL'} ({metrics_dt['mape']:.2f}%)")

importances = pd.Series(dt_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances.head(10))

# ============================================================
# Volume-only diagnostic: performance without traffic_volume
# (the redundancy check from M5 — showed the 90% importance on one
# feature reflects multicollinearity, not fragility)
# ============================================================
if args.target == "volume":
    feature_cols_nv = [c for c in feature_cols if c != "traffic_volume"]
    X_train_nv, X_val_nv, X_test_nv, y_train_nv, y_val_nv, y_test_nv, _ = prepare_model_data(
        train, val, test, feature_cols_nv, target
    )

    print(f"\n=== Decision Tree (max_depth={best_depth}, WITHOUT traffic_volume) ===")
    dt_nv = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    dt_nv.fit(X_train_nv, y_train_nv)
    metrics_nv = regression_metrics(y_test_nv, dt_nv.predict(X_test_nv), label="Test (no traffic_volume)")

    print(f"\nComparison: WITH={metrics_dt['mape']:.2f}% MAPE, WITHOUT={metrics_nv['mape']:.2f}% MAPE")