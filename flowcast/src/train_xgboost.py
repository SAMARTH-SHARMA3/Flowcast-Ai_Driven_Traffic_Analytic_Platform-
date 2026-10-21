"""
XGBoost training/tuning for target_volume_next.
Loads pre-split data from model_data_volume.pkl.
"""

import pickle
import pandas as pd
import xgboost as xgb

from ml_models import prepare_model_data, regression_metrics, logger

with open("data/processed/model_data_volume.pkl", "rb") as f:
    data = pickle.load(f)

train, val, test = data["train"], data["val"], data["test"]
feature_cols, target = data["feature_cols"], data["target"]

X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
    train, val, test, feature_cols, target
)

# ============================================================
# XGBoost with early stopping: train with a generous max number
# of boosting rounds, but stop as soon as validation performance
# stops improving for `early_stopping_rounds` consecutive rounds.
# This directly prevents the "boosted too long" overfitting risk —
# the model's actual final size is chosen by the data, not a guess.
# ============================================================
print("=== XGBoost — hyperparameter search (validation set, early stopping) ===")

xgb_results = []
for max_depth in [4, 6, 8]:
    for learning_rate in [0.01, 0.05, 0.1]:
        model = xgb.XGBRegressor(
            n_estimators=1000,          # generous ceiling — early stopping finds the real number
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            early_stopping_rounds=20,   # stop if val metric hasn't improved in 20 rounds
            eval_metric="rmse",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_metrics = regression_metrics(
            y_val, model.predict(X_val),
            label=f"depth={max_depth}, lr={learning_rate}, best_iter={model.best_iteration}, Val"
        )
        xgb_results.append({
            "max_depth": max_depth, "learning_rate": learning_rate,
            "best_iteration": model.best_iteration, **val_metrics,
        })

xgb_results_df = pd.DataFrame(xgb_results).sort_values("mape")
print("\nAll configurations, sorted by MAPE:")
print(xgb_results_df)



# ============================================================
# Follow-up: did lr=0.01 configs actually converge, or were
# they artificially capped at n_estimators=1000?
# ============================================================
print("\n=== Follow-up: lr=0.01 with higher ceiling ===")
for max_depth in [4, 6]:
    model = xgb.XGBRegressor(
        n_estimators=3000, max_depth=max_depth, learning_rate=0.01,
        random_state=42, early_stopping_rounds=20, eval_metric="rmse",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    regression_metrics(
        y_val, model.predict(X_val),
        label=f"depth={max_depth}, lr=0.01, best_iter={model.best_iteration} (ceiling=3000), Val"
    )




    # ============================================================
# Final model: max_depth=8, learning_rate=0.1 — evaluated on TEST
# ============================================================
print("\n=== XGBoost (depth=8, lr=0.1, final) ===")
xgb_final = xgb.XGBRegressor(
    n_estimators=1000, max_depth=8, learning_rate=0.1,
    random_state=42, early_stopping_rounds=20, eval_metric="rmse",
)
xgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Final model used {xgb_final.best_iteration} boosting rounds")

metrics_xgb = regression_metrics(y_test, xgb_final.predict(X_test), label="Test (final)")
print(f"PRD MAPE <= 12%: {'PASS' if metrics_xgb['mape'] <= 12 else 'FAIL'} ({metrics_xgb['mape']:.2f}%)")

importances_xgb = pd.Series(xgb_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances_xgb.head(10))

print(f"\nComparison across all models (test MAPE):")
print(f"  Linear Regression (sklearn):  15.19%")
print(f"  Linear Regression (scratch):  15.35%")
print(f"  Decision Tree (depth=10):     11.50%")
print(f"  Random Forest (200, None):    10.19%")
print(f"  XGBoost (depth=8, lr=0.1):    {metrics_xgb['mape']:.2f}%")