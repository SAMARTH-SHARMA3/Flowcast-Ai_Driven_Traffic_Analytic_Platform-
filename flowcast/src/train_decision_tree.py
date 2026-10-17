"""
Decision Tree training/tuning for target_volume_next.
Loads pre-split data from model_data_volume.pkl (produced by
ml_models.py) rather than re-running the full data pipeline.
"""

import pickle
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ml_models import prepare_model_data, regression_metrics, check_overfitting, logger

with open("data/processed/model_data_volume.pkl", "rb") as f:
    data = pickle.load(f)

train, val, test = data["train"], data["val"], data["test"]
feature_cols, target = data["feature_cols"], data["target"]

X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
    train, val, test, feature_cols, target
)

# ============================================================
# Unconstrained tree — demonstrate overfitting
# ============================================================
print("=== Decision Tree (unconstrained) ===")
dt_unconstrained = DecisionTreeRegressor(random_state=42)
dt_unconstrained.fit(X_train, y_train)
check_overfitting(dt_unconstrained, X_train, y_train, X_test, y_test, label="Unconstrained Tree")

# ============================================================
# Depth tuning on VALIDATION set
# ============================================================
print("\n=== Decision Tree — depth tuning (validation set) ===")
for depth in [3, 5, 8, 10, 15, None]:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    regression_metrics(y_val, dt.predict(X_val), label=f"max_depth={depth}, Val")

# ============================================================
# Final model (max_depth=10) — evaluated on TEST
# ============================================================
print("\n=== Decision Tree (max_depth=10, final) ===")
dt_final = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_final.fit(X_train, y_train)
metrics_dt = regression_metrics(y_test, dt_final.predict(X_test), label="Test (final)")
print(f"PRD MAPE <= 12%: {'PASS' if metrics_dt['mape'] <= 12 else 'FAIL'} ({metrics_dt['mape']:.2f}%)")

importances = pd.Series(dt_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances.head(10))

# ============================================================
# Diagnostic: performance without traffic_volume (redundancy check)
# ============================================================
feature_cols_nv = [c for c in feature_cols if c != "traffic_volume"]
X_train_nv, X_val_nv, X_test_nv, y_train_nv, y_val_nv, y_test_nv, _ = prepare_model_data(
    train, val, test, feature_cols_nv, target
)

print("\n=== Decision Tree (max_depth=10, WITHOUT traffic_volume) ===")
dt_nv = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_nv.fit(X_train_nv, y_train_nv)
metrics_nv = regression_metrics(y_test_nv, dt_nv.predict(X_test_nv), label="Test (no traffic_volume)")

print(f"\nComparison: WITH={metrics_dt['mape']:.2f}% MAPE, WITHOUT={metrics_nv['mape']:.2f}% MAPE")