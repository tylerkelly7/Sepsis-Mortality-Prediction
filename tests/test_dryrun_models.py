import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.models import repeated_cv_with_mixed_search, get_classifiers, get_param_distributions

# 1. Create a tiny synthetic dataset
X, y = make_classification(
    n_samples=200, n_features=10, n_informative=5,
    n_redundant=2, random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Restrict to one classifier for speed (Logistic Regression)
classifiers = {"LogisticRegression": get_classifiers()["LogisticRegression"]}
param_spaces = {"LogisticRegression": get_param_distributions()["LogisticRegression"]}

# 3. Run dry test
results, summary = repeated_cv_with_mixed_search(
    X_train, y_train, X_test, y_test,
    classifiers=classifiers,
    param_spaces=param_spaces,
    n_splits=2, n_repeats=1,  # very light CV
    n_iter_random=2,
    save_prefix="results/models/dryrun/",
    mode="dryrun",
    log_mlflow=True   # check MLflow logging too
)

print("\n🔎 Dry run summary:")
print(summary)

import os

print("\n📂 Artifacts created in dry run:")
for f in os.listdir("results/models/dryrun/"):
    print(" -", f)

import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

# Pick the best non-SMOTE model from the dry run
model_path = "results/models/dryrun/dryrun_20250929_2323_best_model.pkl"  # <-- update timestamp

with open(model_path, "rb") as f:
    best_model = pickle.load(f)

# Predict on the original dry run test set
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"✅ Reloaded model from {model_path}")
print(f"Holdout ROC-AUC after reload: {auc:.4f}")
