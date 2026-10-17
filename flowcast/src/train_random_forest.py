"""
Random Forest training/tuning for target_volume_next.
Loads pre-split data from model_data_volume.pkl.
"""

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml_models import prepare_model_data, regression_metrics, logger

with open("data/processed/model_data_volume.pkl", "rb") as f:
    data = pickle.load(f)

train, val, test = data["train"], data["val"], data["test"]
feature_cols, target = data["feature_cols"], data["target"]

X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
    train, val, test, feature_cols, target
)

# ============================================================
# Grid search over n_estimators x max_depth, evaluated on VAL
# ============================================================
print("=== Random Forest — hyperparameter search (validation set) ===")
rf_results = []
for n_est in [50, 100, 200]:
    for depth in [10, 15, None]:
        rf = RandomForestRegressor(
            n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        val_metrics = regression_metrics(
            y_val, rf.predict(X_val), label=f"n_est={n_est}, depth={depth}, Val"
        )
        rf_results.append({"n_estimators": n_est, "max_depth": depth, **val_metrics})

rf_results_df = pd.DataFrame(rf_results).sort_values("mape")
print("\nAll configurations, sorted by MAPE:")
print(rf_results_df)



# ============================================================
# Final model: n_estimators=200, max_depth=None — evaluated on TEST
# ============================================================
print("\n=== Random Forest (n_est=200, depth=None, final) ===")
rf_final = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf_final.fit(X_train, y_train)

metrics_rf = regression_metrics(y_test, rf_final.predict(X_test), label="Test (final)")
print(f"PRD MAPE <= 12%: {'PASS' if metrics_rf['mape'] <= 12 else 'FAIL'} ({metrics_rf['mape']:.2f}%)")

importances_rf = pd.Series(rf_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances_rf.head(10))

print(f"\nComparison across models so far (test MAPE):")
print(f"  Linear Regression (sklearn):  15.19%")
print(f"  Linear Regression (scratch):  15.35%")
print(f"  Decision Tree (depth=10):     11.50%")
print(f"  Random Forest (200, None):    {metrics_rf['mape']:.2f}%")