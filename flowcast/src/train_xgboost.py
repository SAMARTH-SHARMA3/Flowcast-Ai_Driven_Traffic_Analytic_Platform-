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