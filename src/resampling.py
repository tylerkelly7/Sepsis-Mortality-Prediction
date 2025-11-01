# src/resampling.py

"""
Resampling utilities for Masters-Thesis project.
Currently supports SMOTE, extendable to ADASYN or others.
"""

from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import pandas as pd
import os
from src.utils import resolve_path

# --------------------------------------------------
# 1. Resampling
# --------------------------------------------------
def resample_training_data(X_train, y_train, method="smote", random_state=42, **kwargs):
    """
    Resample ONLY the training set.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training labels.
        method (str): Resampling strategy ("smote", "adasyn", or None).
        random_state (int): Seed for reproducibility.
        **kwargs: Extra parameters for the sampler.

    Returns:
        tuple: (X_train_res, y_train_res)
    """
    if method is None:
        print("⚠️ No resampling applied — returning original training data.")
        return X_train, y_train

    if method.lower() == "smote":
        sampler = SMOTE(random_state=random_state, **kwargs)
    elif method.lower() == "adasyn":
        sampler = ADASYN(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unsupported resampling method: {method}")

    print(f"🔁 Applying {method.upper()} to training data ...")
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"✅ Resampled training set shape: {X_res.shape}")
    print(f"   Class balance after resampling: {Counter(y_res)}")

    return X_res, y_res


# --------------------------------------------------
# 2. Class Balance Printer (Sanity Check)
# --------------------------------------------------
def print_class_balance(y, title="Dataset"):
    counter = Counter(y)
    print(f"{title} class balance: {dict(counter)}")


# --------------------------------------------------
# 3. Save imbalanced datasets
# --------------------------------------------------
def save_resampled_dataset(X_train, y_train, filename: str,
                           base_dir: str = "data/processed/smote_datasets") -> str:
    out_dir = resolve_path(base_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    pd.DataFrame(X_train).assign(target=y_train).to_csv(out_path, index=False)
    print(f"✅ Resampled dataset saved to {out_path}")
    return out_path
