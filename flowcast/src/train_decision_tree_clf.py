"""
Decision Tree classifier for target_congestion_next (4-class).

Usage:
    python src/train_decision_tree_clf.py

Loads pre-split data from data/processed/model_data_congestion.pkl
(produced by ml_models.py). Accident risk (binary, ROC-AUC) is added
in a later step.
"""

import pickle
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from ml_models import (
    prepare_model_data, classification_metrics, show_classification_report,
    CONGESTION_ORDER, logger,
)

with open("data/processed/model_data_congestion.pkl", "rb") as f:
    data = pickle.load(f)

train, val, test = data["train"], data["val"], data["test"]
feature_cols, target = data["feature_cols"], data["target"]
print(f"Training Decision Tree classifier on: {target}")

# Features are scaled for consistency with the other models (and because
# the SVM needs it); scaling is harmless for trees.
X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(
    train, val, test, feature_cols, target
)

# ============================================================
# Model-selection rule (stated in advance): validation MACRO-F1 —
# the PRD's headline metric for congestion (13.2). Selection uses
# the validation set only, never test.
# ============================================================
print("Model selection metric (validation): MACRO-F1")

depth_grid = [3, 5, 8, 10, 15, None]
weight_grid = [None, "balanced"]   # "balanced" = weight classes inversely to their frequency

print("\n=== Decision Tree classifier — grid search (validation set) ===")
results = []
for depth in depth_grid:
    for cw in weight_grid:
        clf = DecisionTreeClassifier(max_depth=depth, class_weight=cw, random_state=42)
        clf.fit(X_train, y_train)

        # Train macro-F1 is logged next to validation so overfitting is visible
        train_f1 = f1_score(
            y_train, clf.predict(X_train), labels=CONGESTION_ORDER,
            average="macro", zero_division=0,
        )
        val_res = classification_metrics(
            y_val, clf.predict(X_val), CONGESTION_ORDER,
            label=f"depth={depth}, class_weight={cw}, Val",
        )
        results.append({
            "max_depth": depth, "class_weight": cw,
            "train_macro_f1": train_f1,
            "val_macro_f1": val_res["macro_f1"],
            "val_accuracy": val_res["accuracy"],
        })

# Table is for DISPLAY only (a DataFrame turns None depths into NaN);
# selection uses the original list so max_depth=None stays a real None.
print("\nAll configurations, sorted by validation macro-F1:")
print(pd.DataFrame(results).sort_values("val_macro_f1", ascending=False).round(4).to_string(index=False))

# max() keeps the first of any tied configs, i.e. the shallowest,
# unweighted one
best = max(results, key=lambda r: r["val_macro_f1"])
best_depth, best_cw = best["max_depth"], best["class_weight"]
print(f"\nSelected: max_depth={best_depth}, class_weight={best_cw} "
      f"(validation macro-F1: {best['val_macro_f1']:.4f})")

if best_depth == depth_grid[0]:
    print(f"NOTE: best depth ({best_depth}) is at the lower edge of the grid — "
          "consider trying shallower depths.")
if best_depth is None:
    print("NOTE: unconstrained depth won on validation — compare train vs val "
          "macro-F1 in the table above before trusting it.")

# ============================================================
# Final model at the selected config — evaluated once on TEST
# ============================================================
print(f"\n=== Decision Tree classifier (depth={best_depth}, class_weight={best_cw}, final) ===")
clf_final = DecisionTreeClassifier(max_depth=best_depth, class_weight=best_cw, random_state=42)
clf_final.fit(X_train, y_train)

res = classification_metrics(
    y_test, clf_final.predict(X_test), CONGESTION_ORDER, label="Test (final)"
)
show_classification_report(res, CONGESTION_ORDER)

# Reference numbers are copied from the ml_models.py baseline run
# (test set); update them if the baselines are rerun.
print(f"\nReference (test): persistence macro-F1 0.6111; majority-class 0.1876; "
      f"PRD target macro-F1 >= 0.80")
print(f"This model:       macro-F1 {res['macro_f1']:.4f}")

importances = pd.Series(clf_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances.head(10))