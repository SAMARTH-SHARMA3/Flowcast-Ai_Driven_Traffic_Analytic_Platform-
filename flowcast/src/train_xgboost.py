"""
XGBoost training/tuning for a regression target.

Usage:
    python src/train_xgboost.py                       # target_volume_next (default)
    python src/train_xgboost.py --target travel_time

Loads pre-split data from data/processed/model_data_<target>.pkl
(produced by ml_models.py).
"""

import argparse
import pickle
import pandas as pd
import xgboost as xgb

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
print(f"Training XGBoost on: {target}")

X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
    train, val, test, feature_cols, target
)

# ============================================================
# Model-selection rule (judgment call, stated explicitly):
# - volume: validation MAPE (this target's PRD success criterion,
#   and consistent with the Random Forest script)
# - travel_time: validation RMSE (the PRD's primary regression
#   metric; this target has no MAPE threshold)
# Selection ALWAYS uses the validation set, never test.
# ============================================================
select_metric = "mape" if args.target == "volume" else "rmse"
print(f"Model selection metric (validation): {select_metric.upper()}")

# Depth grid is target-specific: for travel_time the best depth sat at
# the lower edge (4), so shallower depths are added for that target.
# Volume keeps its original grid so its documented results reproduce.
depth_grid = [4, 6, 8] if args.target == "volume" else [2, 3, 4, 6, 8]

# ============================================================
# Boosting setup. Early stopping halts training once validation
# RMSE has not improved for EARLY_STOP consecutive rounds, so the
# model's size is chosen by the data. BUT if the best round lands
# within the last EARLY_STOP rounds of the ceiling, early stopping
# never had room to trigger — the model was cut off while still
# improving. Such configs are refit with a larger ceiling so every
# configuration gets a fair chance to converge before comparison.
# ============================================================
EARLY_STOP = 20
BASE_CEILING = 1000
EXTENDED_CEILING = 3000


def fit_xgb(max_depth: int, learning_rate: float, ceiling: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=ceiling,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        early_stopping_rounds=EARLY_STOP,
        eval_metric="rmse",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def hit_ceiling(model: xgb.XGBRegressor, ceiling: int) -> bool:
    """True if early stopping could not have triggered before the ceiling."""
    return model.best_iteration >= ceiling - EARLY_STOP


# ============================================================
# Grid search over max_depth x learning_rate, evaluated on VAL
# ============================================================
print("\n=== XGBoost — hyperparameter search (validation set, early stopping) ===")

results = []
models = {}  # keep each fitted model so the winner needs no retraining
for max_depth in depth_grid:
    for learning_rate in [0.01, 0.05, 0.1]:
        ceiling = BASE_CEILING
        model = fit_xgb(max_depth, learning_rate, ceiling)

        if hit_ceiling(model, ceiling):
            print(f"  depth={max_depth}, lr={learning_rate}: still improving at round "
                  f"{model.best_iteration} (ceiling {ceiling}) — refitting with ceiling {EXTENDED_CEILING}")
            ceiling = EXTENDED_CEILING
            model = fit_xgb(max_depth, learning_rate, ceiling)
            if hit_ceiling(model, ceiling):
                print(f"  NOTE: depth={max_depth}, lr={learning_rate} still had not "
                      f"converged at ceiling {ceiling}")

        m = regression_metrics(
            y_val, model.predict(X_val),
            label=f"depth={max_depth}, lr={learning_rate}, best_iter={model.best_iteration}, Val"
        )
        results.append({
            "max_depth": max_depth, "learning_rate": learning_rate,
            "ceiling": ceiling, "best_iteration": model.best_iteration, **m,
        })
        models[(max_depth, learning_rate)] = model

print(f"\nAll configurations, sorted by validation {select_metric.upper()}:")
print(pd.DataFrame(results).sort_values(select_metric))

best = min(results, key=lambda r: r[select_metric])
best_depth, best_lr = best["max_depth"], best["learning_rate"]
print(f"\nSelected: max_depth={best_depth}, learning_rate={best_lr}, "
      f"{best['best_iteration']} rounds (validation {select_metric.upper()}: {best[select_metric]:.4g})")

# If the winner sits on an edge of the grid, the true optimum may lie
# beyond it — worth widening the grid and rerunning. The depth check
# follows depth_grid, so it stays correct for both targets.
edge_notes = []
if best_depth in (depth_grid[0], depth_grid[-1]):
    edge_notes.append(
        f"max_depth={best_depth} is at an edge of the grid "
        f"({depth_grid[0]}-{depth_grid[-1]})"
    )
if best_lr in (0.01, 0.1):
    edge_notes.append(f"learning_rate={best_lr} is at an edge of the grid (0.01-0.1)")
for note in edge_notes:
    print(f"NOTE: {note} — consider widening the grid.")

# ============================================================
# Final model = the selected config's already-fitted model,
# evaluated on TEST (random_state is fixed, so refitting would
# give an identical model — reusing it just saves the time)
# ============================================================
print(f"\n=== XGBoost (depth={best_depth}, lr={best_lr}, final) ===")
xgb_final = models[(best_depth, best_lr)]
print(f"Final model used {xgb_final.best_iteration} boosting rounds")

metrics_xgb = regression_metrics(y_test, xgb_final.predict(X_test), label="Test (final)")

# The 12% MAPE criterion applies to VOLUME forecast accuracy only.
if args.target == "volume":
    print(f"PRD MAPE <= 12%: {'PASS' if metrics_xgb['mape'] <= 12 else 'FAIL'} ({metrics_xgb['mape']:.2f}%)")

importances_xgb = pd.Series(xgb_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances_xgb.head(10))

# ============================================================
# Comparison against earlier models (test set). Reference numbers
# are copied from earlier runs — update them if those models are
# retrained.
# ============================================================
print("\nComparison across models so far (test set):")
if args.target == "volume":
    print("  Linear Regression (sklearn):  MAPE 15.19%")
    print("  Linear Regression (scratch):  MAPE 15.35%")
    print("  Decision Tree (depth=10):     MAPE 11.50%")
    print("  Random Forest (200, None):    MAPE 10.19%")
    print(f"  XGBoost:                      MAPE {metrics_xgb['mape']:.2f}%")
else:
    print("  Linear Regression (sklearn):  RMSE 1.38, R² 0.7233")
    print("  Linear Regression (scratch):  RMSE 1.38, R² 0.7233")
    print("  Decision Tree (depth=8):      RMSE 1.28, R² 0.7611")
    print("  Random Forest (200, depth=10): RMSE 1.13, R² 0.8142")
    print(f"  XGBoost:                      RMSE {metrics_xgb['rmse']:.2f}, R² {metrics_xgb['r2']:.4f}")