# src/models.py

"""
Model utilities for Masters-Thesis project.

All scaling and resampling are performed upstream
in 03_feature_engineering and 04_model_training.

Includes:
- Classifier registry
- Hyperparameter distributions
- Mixed search with CV (Grid vs Randomized) for both original and SMOTE-balanced training sets.
- Save/load functions
"""

import os
import json
import glob
import pickle
import datetime
import traceback
import time

from fastapi import logger
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from zipfile import ZipFile  # <-- if you ever use zipfile itself, not Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score
)
from sklearn.metrics import roc_auc_score, classification_report, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

# from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import mlflow
import mlflow.sklearn

from src.utils import resolve_path

# ==================================================
# 1. Classifier Registry
# ==================================================

def get_classifiers(random_state=42):
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        "LGBM": LGBMClassifier(random_state=random_state, verbose=-1),
        "CatBoost": CatBoostClassifier(task_type="CPU", verbose=0, random_state=random_state, allow_writing_files=False),
        "SVC": SVC(probability=False, shrinking=True, max_iter=-1, cache_size=2000, random_state=random_state),
        "MLP": MLPClassifier(max_iter=2000, random_state=random_state),
        "NaiveBayes": GaussianNB()
    }


# ==================================================
# 2. Param Distributions
# ==================================================

def get_param_distributions():
    from scipy.stats import randint, uniform

    return {
        "RandomForest": {
            "clf__n_estimators": randint(200, 1001),
            "clf__max_depth": [None, 10, 20, 50],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__min_samples_split": randint(2, 11),
            "clf__min_samples_leaf": randint(1, 5),
            "clf__bootstrap": [True, False]
        },
        "GradientBoosting": {
            "clf__n_estimators": randint(100, 501),
            "clf__learning_rate": uniform(0.01, 0.3),
            "clf__max_depth": [3, 5, 10],
            "clf__min_samples_split": randint(2, 11),
            "clf__min_samples_leaf": randint(1, 5),
            "clf__subsample": uniform(0.7, 0.3),
            "clf__max_features": ["sqrt", "log2", None]
        },
        "XGB": {
            "clf__n_estimators": randint(200, 1001),
            "clf__learning_rate": uniform(0.01, 0.3),
            "clf__max_depth": randint(3, 11),
            "clf__subsample": uniform(0.7, 0.3),
            "clf__colsample_bytree": uniform(0.7, 0.3),
            "clf__gamma": uniform(0, 0.5),
            "clf__min_child_weight": randint(1, 7)
        },
        "LGBM": {
            "clf__n_estimators": randint(200, 1001),
            "clf__learning_rate": uniform(0.01, 0.3),
            "clf__max_depth": [-1, 10, 20, 50],
            "clf__num_leaves": randint(31, 256),
            "clf__subsample": uniform(0.7, 0.3),
            "clf__colsample_bytree": uniform(0.7, 0.3),
            "clf__min_child_samples": randint(10, 101)
        },
        "SVC": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__kernel": ["linear"],
            "clf__gamma": ["scale", "auto"],
            "clf__shrinking": [True, False]
        },
        "MLP": {
            "clf__hidden_layer_sizes": [(64,), (64,32)],
            "clf__activation": ["relu"],
            "clf__solver": ["adam"],                  # drop 'sgd' for stability
            "clf__alpha": [1e-4, 1e-3],
            "clf__learning_rate_init": [1e-3],
            "clf__early_stopping": [True],
            "clf__n_iter_no_change": [10]
        },
        "DecisionTree": {
            "clf__max_depth": [None, 5, 10, 20, 50],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4, 10],
            "clf__criterion": ["gini", "entropy", "log_loss"]
        },
       "LogisticRegression": [
            # Case 1: lbfgs (only supports l2)
            {
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs", "newton-cg", "sag"],
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__max_iter": [1000]
            },
            # Case 2: saga with l1 or elasticnet
            {
                "clf__penalty": ["l1", "elasticnet"],
                "clf__solver": ["saga"],
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__l1_ratio": [0, 0.5, 1],   # only used if penalty=elasticnet
                "clf__max_iter": [1000]
            },
            # Case 3: liblinear (supports l1 and l2 only)
            {
                "clf__penalty": ["l1", "l2"],
                "clf__solver": ["liblinear"],
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__max_iter": [1000]
            }
        ],
        "NaiveBayes": {
            # NB has fewer tunable parameters — depends on variant
            "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },
        "CatBoost": {
            "clf__iterations": randint(200, 1001),
            "clf__depth": randint(4, 10),
            "clf__learning_rate": uniform(0.01, 0.3),
            "clf__l2_leaf_reg": randint(1, 10)
        }
    }

def get_n_iter_random_per_clf():
    """
    Return dictionary of number of random iterations per classifier for RandomizedSearchCV.
    """
    return {
        "RandomForest": 75,
        "GradientBoosting": 75,
        "XGB": 75,
        "LGBM": 75,
        "CatBoost": 75
    }





# ==================================================
# 3. Helper functions
# ==================================================

# Helper to switch between sampler (SMOTE) + class weights in param space
# ⚠️ Currently unused: retained for future sensitivity analyses

def _with_sampler_and_weights(name, base_space, y_train, random_state=42):
    """
    Given a base param grid (dict or list of dicts) for 'clf__*',
    return a list of dicts that toggles between:
      - sampler='passthrough' + class weighting (where supported)
      - sampler=SMOTE(...)   + no class weighting (or neutral pos weights)
    """
    if not isinstance(base_space, list):
        grids = [base_space]
    else:
        grids = base_space

    n_pos = int(np.sum(y_train))
    n_neg = int(len(y_train) - n_pos)
    # avoid div-by-zero
    spw = float(n_neg / n_pos) if n_pos > 0 else 1.0
    catboost_weights = [[1.0, spw]]  # [w0, w1]

    final = []
    for g in grids:
        # --- branch A: no sampler + (maybe) class weights
        g_passthrough = dict(g)
        if name in {"LogisticRegression", "DecisionTree", "RandomForest", "SVC"}:
            # ensure we search both with/without weighting on the non-SMOTE branch
            g_passthrough.setdefault("clf__class_weight", ["balanced", None])
        elif name == "XGB":
            # imbalance handled via scale_pos_weight
            g_passthrough.setdefault("clf__scale_pos_weight", [spw])
        elif name == "LGBM":
            # pick one approach; scale_pos_weight is simple
            g_passthrough.setdefault("clf__scale_pos_weight", [spw])
        elif name == "CatBoost":
            g_passthrough.setdefault("clf__class_weights", catboost_weights)
        # (GradientBoosting, MLP, NaiveBayes: nothing to add)
        g_passthrough["sampler"] = ["passthrough"]
        final.append(g_passthrough)

        # --- branch B: SMOTE + neutral weights
        g_smote = dict(g)
        if name in {"LogisticRegression", "DecisionTree", "RandomForest", "SVC"}:
            g_smote["clf__class_weight"] = [None]
        elif name == "XGB":
            g_smote["clf__scale_pos_weight"] = [1.0]
        elif name == "LGBM":
            g_smote["clf__scale_pos_weight"] = [1.0]
        elif name == "CatBoost":
            g_smote["clf__class_weights"] = [[1.0, 1.0]]
        # others: leave as-is
        g_smote["sampler"] = [SMOTE(random_state=random_state)]
        final.append(g_smote)

    return final

def _decision_or_proba(estimator, X):
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        return s if s.ndim == 1 else s[:, 1]
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.predict(X)  # last resort

def auc_scorer(estimator, X, y):
    """Return AUROC or NaN safely inside GridSearchCV."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        # get fitted estimator (GridSearchCV will call fit before scoring)
        if hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X)
            if scores.ndim > 1:
                scores = scores[:, 1]
        elif hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(X)[:, 1]
        else:
            # fallback: use predictions as pseudo-scores
            scores = estimator.predict(X)

        # if all scores identical, roc_auc_score will fail
        if np.allclose(scores, scores[0]):
            return np.nan
        return roc_auc_score(y, scores)
    except Exception:
        return np.nan

# ==================================================
# 4. Mixed Search with Repeated CV
# ==================================================

def repeated_cv_with_mixed_search(
    X_train, y_train, X_test, y_test,
    classifiers, param_spaces,
    X_train_smote=None, y_train_smote=None,
    n_splits=5, n_repeats=10, scoring=None,
    n_jobs=-1, random_state=42, verbose=1,
    n_iter_random=None, n_iter_random_per_clf=None,
    save_prefix: str = resolve_path("results/models/original/"),   # e.g., results/models/{original|w2v}/
    descriptive_cv=True,
    mode="original",                          # "original" or "w2v"
    log_mlflow=True                           # <-- Flag for flexible logging
):
    """
    Hyperparameters selected on NON-SMOTE data via mixed search (Grid/Random).
    Best params are then retrained on SMOTE-balanced train set (if provided).
    Produces a summary table with both non-SMOTE and SMOTE metrics, and saves:
      - {mode}_{timestamp}_best_model.pkl
      - {mode}_{timestamp}_best_smote_model.pkl
      - {mode}_{timestamp}_full_summary_with_smote.csv
      - {mode}_{timestamp}_full.pkl (results dict)
    Saves models + results locally, and optionally logs all params/metrics/artifacts to MLflow.
    """
    overall_start = time.time()
    # --------------------------------------------------
    # 1. Setup and initialization
    # --------------------------------------------------
    save_prefix = resolve_path(save_prefix)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_prefix = f"{mode}_{timestamp}"

    # Create a dedicated subfolder for this run
    run_dir = os.path.join(save_prefix, run_prefix)
    os.makedirs(run_dir, exist_ok=True)


    results = {}
    summary_rows = []

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    randomized_models = {"RandomForest", "XGB", "LGBM", "GradientBoosting", "CatBoost"}

    # --- MLflow experiment initialization (robust + OS-agnostic) ---
    if log_mlflow:
        from pathlib import Path
        import mlflow

        # Use dedicated tracking directory
        tracking_dir = Path(resolve_path("mlflow_tracking")).absolute()
        mlflow.set_tracking_uri(tracking_dir.as_uri())

        # --- Unified MLflow experiment setup ---
        experiment_name = "Thesis_ModelTraining"  # Single parent experiment
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name)
        else:
            exp_id = exp.experiment_id

        mlflow.set_experiment(experiment_name)
        print(f"✅ MLflow tracking initialized under unified experiment '{experiment_name}'")
        print(f"Tracking URI: {tracking_dir.as_uri()} (Experiment ID: {exp_id})")


        # Safely end any previous run
        if mlflow.active_run() is not None:
            mlflow.end_run()

        # Start new parent run
        parent_run = mlflow.start_run(run_name=f"{mode}_session")
    else:
        parent_run = None

    # Set default scoring if not provided
    if scoring is None:
        scoring = auc_scorer # fallback to direct callable scorer

    # --------------------------------------------------
    # 2. Hyperparameter search on NON-SMOTE training data
    # --------------------------------------------------

    for name, clf in classifiers.items():
        print(f"\n🔹 Running {name}...")
        start_time = time.time()

        # One nested MLflow run per classifier that covers BOTH non-SMOTE and SMOTE
        if log_mlflow:
            mlflow_cm = mlflow.start_run(run_name=f"{mode}_{name}", nested=True)
        else:
            mlflow_cm = None

        if mlflow_cm:
            # --- Add MLflow metadata tags for this run ---
            mlflow.set_tag("group", "Thesis_ModelTraining")
            mlflow.set_tag("mode", mode)
            mlflow.set_tag("classifier", name)
            mlflow.set_tag("baseline", False)  # (Task 19 can set dynamically)

            # Log key CV parameters for traceability
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("n_repeats", n_repeats)
            mlflow.log_param("scoring", scoring)
            mlflow.log_param("n_iter_random", n_iter_random)


        try:
            # IMPORTANT: No SMOTE step in the search pipeline (selection on non-SMOTE data).
            pipe = Pipeline([
                ("clf", clf),
            ])
            
            # Retrieve parameter space normally
            param_space = param_spaces.get(name, {})

            # Get parameter space with sampler + class weights toggling
            # base_space = param_spaces.get(name, {})
            # param_space = _with_sampler_and_weights(name, base_space, y_train, random_state)

            if name in randomized_models:
                n_iter = (n_iter_random_per_clf.get(name)
                            if n_iter_random_per_clf and name in n_iter_random_per_clf
                            else n_iter_random)
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=param_space,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    refit=True,
                    verbose=verbose,
                    random_state=random_state
                )
            else:
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_space,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    refit=True,
                    verbose=verbose
                )

            # Fit CV search on ORIGINAL training data (non-SMOTE)
            import logging
            from joblib import parallel_backend

            logging.getLogger('joblib').setLevel(logging.INFO)
            with parallel_backend('loky', n_jobs=-1):
                search.fit(X_train, y_train)


            # ==================================================
            # Special handling for SVC: refit best model with probability=True
            # ==================================================
            if name == "SVC":
                best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                try:
                    print("   Re-training best SVC with probability=True for calibrated AUROC...")
                    svc_proba = SVC(**best_params, probability=True, tol=1e-3, cache_size=2000, random_state=random_state)
                    svc_proba.fit(X_train, y_train)
                    search.best_estimator_ = svc_proba  # replace tuned estimator with calibrated version

                    if log_mlflow:
                        mlflow.set_tag("svc_refit_with_proba", True)
                except Exception as e:
                    print(f"   ⚠️ Warning: SVC refit with probability=True failed ({e}); "
                        f"falling back to non-probability model.")

            # --------------------------------------------------
            # 3. Holdout evaluation (non-SMOTE best pipeline)
            # --------------------------------------------------
            y_pred = search.predict(X_test)
            y_proba = _decision_or_proba(search.best_estimator_, X_test)

            test_auc = roc_auc_score(y_test, y_proba)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Initialize summary row
            summary_row = {
                "Classifier": name,
                "Best Params": search.best_params_,
                "CV Mean Score": search.best_score_,
                "CV Std Score": search.cv_results_["std_test_score"][search.best_index_],
                "Holdout ROC-AUC": test_auc,
                "Holdout Precision": report["1"]["precision"],
                "Holdout Recall": report["1"]["recall"],
                "Holdout F1": report["1"]["f1-score"],
                "Descriptive CV Mean AUC": None,
                "Descriptive CV Std AUC": None,
                "Holdout ROC-AUC (SMOTE)": None,
                "Descriptive CV Mean AUC (SMOTE)": None,
                "Descriptive CV Std AUC (SMOTE)": None,
                "Final Holdout ROC-AUC (SMOTE)": None,
            }

            # --------------------------------------------------
            # 4. Optional descriptive CV on NON-SMOTE training
            # --------------------------------------------------
            # All data is pre-scaled — directly run cross_val_score without internal scaler
            if descriptive_cv:
                print(f"   Performing descriptive StratifiedKFold CV on original training set for {name}...")
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

                try:
                    cv_scores = []
                    for train_idx, val_idx in kf.split(X_train, y_train):
                        y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
                        if len(np.unique(y_val)) < 2:
                            continue  # skip single-class fold
                        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr = y_train.iloc[train_idx]
                        est_fold = clone(search.best_estimator_)
                        est_fold.fit(X_tr, y_tr)
                        y_val_pred = _decision_or_proba(est_fold, X_val)


                        cv_scores.append(roc_auc_score(y_val, y_val_pred))
                    cv_scores = np.array(cv_scores)

                except ValueError as e:
                    # e.g., a fold has a single class (extreme imbalance)
                    summary_row["Descriptive CV Mean AUC"] = np.nan
                    summary_row["Descriptive CV Std AUC"]  = np.nan
                    print(f"   Skipping descriptive CV (original): {e}")
                else:
                    if cv_scores.size == 0 or np.all(np.isnan(cv_scores)):
                        summary_row["Descriptive CV Mean AUC"] = np.nan
                        summary_row["Descriptive CV Std AUC"]  = np.nan
                        print("   Descriptive CV AUC: nan ± nan (all folds invalid)")
                    else:
                        summary_row["Descriptive CV Mean AUC"] = float(np.nanmean(cv_scores))
                        summary_row["Descriptive CV Std AUC"]  = float(np.nanstd(cv_scores))
                        print(f"   Descriptive CV AUC: "
                            f"{summary_row['Descriptive CV Mean AUC']:.4f} ± "
                            f"{summary_row['Descriptive CV Std AUC']:.4f}")

            # --------------------------------------------------
            # 5. Save classifier CV results + metrics
            # --------------------------------------------------
            results[name] = {
                "best_estimator": search.best_estimator_,   # already fitted during CV
                "best_params": search.best_params_,
                "cv_results": search.cv_results_,
                "test_metrics": {
                    "roc_auc": test_auc,
                    "classification_report": report
                }
            }

            # Always save classifier model locally
            model_path = os.path.join(run_dir, f"{run_prefix}_{name}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(search.best_estimator_, f)
            print(f"💾 Saved {name} model to {model_path}")

            # Log to MLflow on top (optional)
            if log_mlflow:
                mlflow.log_artifact(model_path, artifact_path="models")

            print(f"✅ {name} done. Best params: {search.best_params_}")
            print(f"   CV ROC-AUC: {search.best_score_:.4f} ± {search.cv_results_['std_test_score'][search.best_index_]:.3f}")
            print(f"   Holdout ROC-AUC: {test_auc:.4}")

            # --------------------------------------------------
            # MLflow Logging (non-SMOTE metrics)
            # --------------------------------------------------
            metrics_non_smote = {
                "cv_mean_auc": search.best_score_,
                "cv_std_auc": search.cv_results_["std_test_score"][search.best_index_],
                "holdout_auc": test_auc,
                "holdout_precision": report["1"]["precision"],
                "holdout_recall": report["1"]["recall"],
                "holdout_f1": report["1"]["f1-score"],
            }
            if descriptive_cv:
                metrics_non_smote["descriptive_cv_mean_auc"] = summary_row["Descriptive CV Mean AUC"]
                metrics_non_smote["descriptive_cv_std_auc"] = summary_row["Descriptive CV Std AUC"]

            # Always save metrics JSON locally
            metrics_path = os.path.join(run_dir, f"{run_prefix}_{name}_metrics_non_smote.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_non_smote, f, indent=4)
            print(f"💾 Saved non-SMOTE metrics for {name} to {metrics_path}")

            # --------------------------------------------------
            # 5.5  Retrain/Evaluate this classifier on SMOTE (inside same MLflow run)
            # --------------------------------------------------
            d_mean, d_std = (None, None)  # defaults if not doing descriptive CV on SMOTE
            if X_train_smote is not None and y_train_smote is not None:
                best_pipe_smote = clone(search.best_estimator_)
                best_pipe_smote.fit(X_train_smote, y_train_smote)

                # Holdout predictions (SMOTE-trained)
                y_pred_proba_smote = _decision_or_proba(best_pipe_smote, X_test)

                auc_smote = roc_auc_score(y_test, y_pred_proba_smote)
                print(f"   SMOTE Holdout ROC-AUC: {auc_smote:.4f}")


                # Optional descriptive CV on SMOTE
                if descriptive_cv:
                    print(f"   Performing descriptive StratifiedKFold CV on SMOTE training set for {name}...")
                    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                    cv_scores_smote = []

                    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_smote, y_train_smote), start=1):
                        y_val = y_train_smote.iloc[val_idx] if isinstance(y_train_smote, pd.Series) else y_train_smote[val_idx]
                        # Skip folds that contain only one class
                        if len(np.unique(y_val)) < 2:
                            print(f"      ⚠️  Skipping fold {fold_idx}: single-class validation fold.")
                            continue

                        X_tr, X_val = (
                            X_train_smote.iloc[train_idx],
                            X_train_smote.iloc[val_idx],
                        )
                        y_tr = y_train_smote.iloc[train_idx]

                        try:
                            est_fold_smote = clone(best_pipe_smote)
                            est_fold_smote.fit(X_tr, y_tr)
                            y_val_pred = _decision_or_proba(est_fold_smote, X_val)


                            auc_val = roc_auc_score(y_val, y_val_pred)
                            cv_scores_smote.append(auc_val)
                        except Exception as e:
                            print(f"      ⚠️  Error in fold {fold_idx}: {e}")
                            continue

                    if len(cv_scores_smote) == 0:
                        d_mean, d_std = np.nan, np.nan
                        print("   Descriptive CV AUC (SMOTE): nan ± nan (all folds invalid)")
                    else:
                        d_mean, d_std = float(np.nanmean(cv_scores_smote)), float(np.nanstd(cv_scores_smote))
                        print(f"   Descriptive CV AUC (SMOTE): {d_mean:.4f} ± {d_std:.4f}")
                        print(f"   ({len(cv_scores_smote)} valid folds out of {kf.get_n_splits()})")

                # Update summary_row with SMOTE stats BEFORE appending it
                summary_row["Holdout ROC-AUC (SMOTE)"] = auc_smote
                summary_row["Descriptive CV Mean AUC (SMOTE)"] = d_mean
                summary_row["Descriptive CV Std AUC (SMOTE)"] = d_std

                # Save SMOTE metrics JSON locally
                metrics_smote = {
                    "holdout_auc_smote": auc_smote
                }
                if descriptive_cv:
                    metrics_smote["descriptive_cv_mean_auc_smote"] = d_mean
                    metrics_smote["descriptive_cv_std_auc_smote"] = d_std

                metrics_path_sm = os.path.join(run_dir, f"{run_prefix}_{name}_metrics_smote.json")
                
                # 💾 Save per-classifier SMOTE model
                smote_model_path_individual = os.path.join(run_dir, f"{run_prefix}_{name}_smote_model.pkl")
                with open(smote_model_path_individual, "wb") as f:
                    pickle.dump(best_pipe_smote, f)
                print(f"💾 Saved SMOTE-trained {name} model to {smote_model_path_individual}")
                # Track SMOTE model in results dict
                results[name]["best_estimator_smote"] = best_pipe_smote
                results[name]["smote_metrics"] = metrics_smote

                with open(metrics_path_sm, "w") as f:
                    json.dump(metrics_smote, f, indent=4)
                print(f"💾 Saved SMOTE metrics for {name} to {metrics_path_sm}")

                if log_mlflow:
                    mlflow.log_metrics({
                        "holdout_roc_auc_smote": auc_smote,
                        **({} if not descriptive_cv else {
                            "descriptive_cv_mean_auc_smote": d_mean if d_mean is not None else np.nan,
                            "descriptive_cv_std_auc_smote": d_std if d_std is not None else np.nan
                        })
                    })
                    mlflow.log_artifact(metrics_path_sm, artifact_path="metrics")
            if log_mlflow:
                mlflow.set_tag("smote_cv_auc_summary", f"{d_mean:.4f} ± {d_std:.4f}")
            # Append AFTER SMOTE section so the row includes both phases
            summary_rows.append(summary_row)

            # Log to MLflow on top (optional)
            if log_mlflow:
                mlflow.log_metrics(metrics_non_smote)
                mlflow.log_artifact(metrics_path, artifact_path="metrics")

        except Exception as e:
            # Mark the run as failed and attach traceback when MLflow is on
            if log_mlflow:
                mlflow.set_tag("run_status", "failed")
                mlflow.set_tag("exception", type(e).__name__)
                tb_path = os.path.join(run_dir, f"{run_prefix}_{name}_traceback.txt")
                with open(tb_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                mlflow.log_artifact(tb_path, artifact_path="errors")
            print(f"❌ {name} failed: {e}")
            raise
        finally:
            # Log and print runtime for this classifier
            runtime_minutes = (time.time() - start_time) / 60
            print(f"⏱️  Runtime for {name}: {runtime_minutes:.2f} minutes")

            if log_mlflow:
                mlflow.log_metric("runtime_minutes", runtime_minutes)

            # Close nested MLflow run for this classifier
            if mlflow_cm:
                mlflow.end_run()
                print(f"🏁 MLflow run for '{name}' closed cleanly.")
    
    # --------------------------------------------------
    # 6. Build summary table (non-SMOTE results only)
    # --------------------------------------------------
    summary_df = pd.DataFrame(summary_rows).sort_values(by="Holdout ROC-AUC", ascending=False)

    # --------------------------------------------------
    # 7. Save & evaluate best NON-SMOTE model
    # --------------------------------------------------
    print("\n📌 Evaluating all classifiers on holdout test set using CV-trained pipelines:")
    for name in classifiers.keys():
        best_pipe = results[name]["best_estimator"]
        y_pred_proba = _decision_or_proba(best_pipe, X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"{name}: Holdout ROC-AUC = {auc:.4f}")


    best_idx = summary_df["Holdout ROC-AUC"].idxmax()
    best_model_name = summary_df.loc[best_idx, "Classifier"]

    # ✅ Use the already-fitted model and its stored AUC directly
    best_pipeline = results[best_model_name]["best_estimator"]
    test_auc = summary_df.loc[best_idx, "Holdout ROC-AUC"]

    print(f"\n🏆 Best classifier (no SMOTE) = {best_model_name}, Holdout ROC-AUC = {test_auc:.4f}")

    # Save the same fitted pipeline (no refit)
    best_model_path = os.path.join(run_dir, f"{run_prefix}_best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"💾 Saved best model (no SMOTE) to {best_model_path}")


    # --------------------------------------------------
    # 8. Evaluate and save best SMOTE model
    # --------------------------------------------------
    if (
        "Holdout ROC-AUC (SMOTE)" in summary_df.columns
        and summary_df["Holdout ROC-AUC (SMOTE)"].notna().any()
    ):
        best_smote_idx = summary_df["Holdout ROC-AUC (SMOTE)"].idxmax()
        best_smote_name = summary_df.loc[best_smote_idx, "Classifier"]
        best_smote_auc = summary_df.loc[best_smote_idx, "Holdout ROC-AUC (SMOTE)"]

        print(f"🏆 Best SMOTE-trained classifier by holdout AUC = {best_smote_name}, AUC = {best_smote_auc:.4f}")

        # Final evaluation of best SMOTE-trained model
        final_smote_model = clone(results[best_smote_name]["best_estimator"])
        final_smote_model.fit(X_train_smote, y_train_smote)
        y_test_proba = _decision_or_proba(final_smote_model, X_test)

        final_smote_auc = roc_auc_score(y_test, y_test_proba)
        print(f"🎯 Final evaluation of best SMOTE-trained classifier = {best_smote_name}, ROC-AUC = {final_smote_auc:.4f}")

        summary_df.loc[
            summary_df["Classifier"] == best_smote_name, "Final Holdout ROC-AUC (SMOTE)"
        ] = final_smote_auc

        smote_model_path = os.path.join(run_dir, f"{run_prefix}_best_smote_model.pkl")
        with open(smote_model_path, "wb") as f:
            pickle.dump(final_smote_model, f)
        print(f"💾 Saved best SMOTE-trained model to {smote_model_path}")

        # Save final SMOTE metrics JSON locally
        final_smote_metrics = {
            "best_smote_classifier": best_smote_name,
            "final_holdout_auc_smote": final_smote_auc,
        }
        final_smote_metrics_path = os.path.join(run_dir, f"{run_prefix}_best_smote_metrics.json")
        with open(final_smote_metrics_path, "w") as f:
            json.dump(final_smote_metrics, f, indent=4)
        print(f"💾 Saved final SMOTE metrics to {final_smote_metrics_path}")

        if log_mlflow:
            mlflow.log_artifact(smote_model_path, artifact_path="models")
            mlflow.log_artifact(final_smote_metrics_path, artifact_path="metrics")

    else:
        print("ℹ️ No SMOTE metrics available in summary_df; skipping best-SMOTE selection.")


    # --------------------------------------------------
    # 9. Save full results (summary + dict)
    # --------------------------------------------------
    full_summary_csv_path = os.path.join(run_dir, f"{run_prefix}_full_summary_with_smote.csv")
    summary_df.to_csv(full_summary_csv_path, index=False)
    print(f"💾 Saved full summary including original and SMOTE metrics to {full_summary_csv_path}")

    full_results_path = os.path.join(run_dir, f"{run_prefix}_full.pkl")
    with open(full_results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"💾 Saved full results dict to {full_results_path}")

    if log_mlflow:
        mlflow.log_artifact(full_summary_csv_path, artifact_path="summaries")
        mlflow.log_artifact(full_results_path, artifact_path="results")

    # 💾 Save SMOTE-only subset of results (parallel to full.pkl)
    smote_results = {
        name: {
            "best_estimator_smote": res.get("best_estimator_smote"),
            "smote_metrics": res.get("smote_metrics"),
        }
        for name, res in results.items()
        if "best_estimator_smote" in res
    }

    smote_full_path = os.path.join(run_dir, f"{run_prefix}_smote_full.pkl")
    with open(smote_full_path, "wb") as f:
        pickle.dump(smote_results, f)
    print(f"💾 Saved all best SMOTE-trained classifiers to {smote_full_path}")

    if log_mlflow:
        mlflow.log_artifact(smote_full_path, artifact_path="results")

    # --------------------------------------------------
    # 10. Safely close MLflow parent run
    # --------------------------------------------------
    # Log total runtime for the full mode (e.g., 'original' or 'w2v_radiology')
    total_runtime_minutes = (time.time() - overall_start) / 60
    print(f"⏱️  Total runtime for mode '{mode}': {total_runtime_minutes:.2f} minutes\n\n---\n")

    if log_mlflow:
        mlflow.log_metric("total_runtime_minutes", total_runtime_minutes)

    if log_mlflow and parent_run is not None and mlflow.active_run() is not None:
        mlflow.end_run()
        print(f"🏁 MLflow run for '{mode}' closed cleanly.")

    return results, summary_df

# ==================================================
# 4. Save/Load Utilities
# ==================================================

def save_all_classifiers(results, save_prefix="results/models/original/", mode="original"):
    """
    Save all best_estimator models from results dict to disk with timestamped filenames.

    Args:
        results (dict): Results dict from repeated_cv_with_mixed_search().
        save_prefix (str): Directory to save models under.
        mode (str): Mode (e.g., "original", "w2v_radiology").
    """
    save_prefix = resolve_path(save_prefix)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_prefix = f"{mode}_{timestamp}"

    run_dir = os.path.join(save_prefix, run_prefix)
    os.makedirs(run_dir, exist_ok=True)

    for name, res in results.items():
        if "best_estimator" in res and res["best_estimator"] is not None:
            model_path = os.path.join(run_dir, f"{run_prefix}_{name}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(res["best_estimator"], f)
            print(f"💾 Saved {name} model to {model_path}")


def load_all_classifiers(save_prefix="results/models/original/", timestamp=None, mode="original"):
    """
    Load all classifier models from a specific run directory.

    Args:
        save_prefix (str): Base directory (e.g., "results/models/original/").
        timestamp (str, optional): Timestamp of the run to load (e.g., "20251011_2237").
        mode (str): Mode identifier (e.g., "original", "w2v_radiology").

    Returns:
        dict: {classifier_name: model_object}
    """
    save_prefix = resolve_path(save_prefix)

    if timestamp is None:
        # Load the most recent run folder automatically
        run_dirs = sorted(
            glob.glob(os.path.join(save_prefix, f"{mode}_*")),
            key=os.path.getmtime,
            reverse=True
        )
        if not run_dirs:
            raise FileNotFoundError(f"No saved runs found under {save_prefix}.")
        run_dir = run_dirs[0]
        timestamp = os.path.basename(run_dir).split("_", 1)[1]
    else:
        run_dir = os.path.join(save_prefix, f"{mode}_{timestamp}")

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    pattern = os.path.join(run_dir, f"{mode}_{timestamp}_*_model.pkl")

    loaded_models = {}
    for model_file in glob.glob(pattern):
        clf_name = os.path.basename(model_file).split("_")[-2]
        with open(model_file, "rb") as f:
            loaded_models[clf_name] = pickle.load(f)
        print(f"✅ Loaded {clf_name} from {model_file}")

    return loaded_models


# ==================================================
# 5. Helper to Neutralize Class Weights for SMOTE Retrain
# ==================================================

def _neutralize_weights_for_smote(pipe):
    """
    Return a cloned estimator with class-weighting neutralized for SMOTE retrain.
    Handles LR/DT/RF/SVC, XGB, LGBM, CatBoost.
    Works with sklearn Pipeline or a bare estimator.
    """
    from sklearn.base import clone
    est = clone(pipe)

    # If it's a pipeline, grab the classifier step name
    clf_step_name = None
    if hasattr(est, "named_steps") and "clf" in est.named_steps:
        clf = est.named_steps["clf"]
        clf_step_name = "clf"
    else:
        clf = est

    def _set_param(estimator, param_name, value):
        try:
            estimator.set_params(**{param_name: value})
        except ValueError:
            pass  # param not supported; ignore

    # sklearn family (if supported)
    _set_param(est, "clf__class_weight", None)
    _set_param(est, "class_weight", None)

    # XGBoost / LightGBM
    _set_param(est, "clf__scale_pos_weight", 1.0)
    _set_param(est, "scale_pos_weight", 1.0)

    # CatBoost: expects list [w0, w1]
    _set_param(est, "clf__class_weights", [1.0, 1.0])
    _set_param(est, "class_weights", [1.0, 1.0])

    return est
