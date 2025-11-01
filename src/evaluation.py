# src/evaluation.py
"""
src/evaluation.py
Evaluation utilities for Masters-Thesis project
-----------------------------------------------
Computes metrics (AUROC, Accuracy, Precision, Recall, F1, PR-AUC, Brier score)
and exports standardized summaries for visualization and reporting.

Author: Tyler Kelly
"""

import os
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List
from pathlib import Path
from src.utils import resolve_path

from IPython.display import Image, display

# =========================================================
# 1. Export Summary
# =========================================================

from pathlib import Path
from src.utils import resolve_path

from pathlib import Path
from datetime import datetime
from src.utils import resolve_path

def export_summary(summary_df, mode="original", save_prefix="results/evaluation", include_time=False):
    """
    Export summary DataFrame to results/evaluation/{date}/ as a single CSV per mode.
    
    Example output:
        results/evaluation/20251013/original_baseline_summary.csv
    """
    # Build dated folder path (e.g., results/evaluation/20251013/)
    ts_format = "%Y%m%d_%H%M" if include_time else "%Y%m%d"
    timestamp = datetime.now().strftime(ts_format)
    out_dir = Path(resolve_path(f"{save_prefix}/{timestamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename pattern: {mode}_summary.csv
    out_file = f"{mode}_{timestamp}_summary.csv"
    out_path = out_dir / out_file

    summary_df.to_csv(out_path, index=False)
    print(f"✅ Summary exported to {out_path}")

    return out_path




# =========================================================
# 2. Probability Helper
# =========================================================
def get_proba(model, X):
    """
    Safely obtain probability estimates.
    Falls back to decision_function scaled to [0,1].
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    raise ValueError(f"{model.__class__.__name__} lacks probability output.")


# =========================================================
# 3.  Core Metric Evaluation
# =========================================================
def evaluate_classifier(model, X_test, y_test, clf_name, mode):
    """Compute performance metrics for a single classifier."""
    y_proba = get_proba(model, X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "Classifier": clf_name,
        "Mode": mode,
        "AUROC": roc_auc_score(y_test, y_proba),
        "PR_AUC": average_precision_score(y_test, y_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_test, y_proba)
    }
    return {k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}


def evaluate_all_models(models, X_test, y_test, mode="original",
                        save_prefix="results/evaluation", include_time=False):
    """
    Evaluate all models and save results under results/evaluation/{date}/.

    This version mirrors auc_scorer() logic from models.py
    so AUROC values match training (handles decision_function and predict_proba).
    """

    # Convert to NumPy array to ignore feature names
    X_test_np = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
    results = []

    for name, model in models.items():
        try:
            # --- Consistent AUROC logic ---
            if hasattr(model, "decision_function"):
                y_pred_proba = model.decision_function(X_test_np)
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 1]
            elif hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_np)[:, 1]
            else:
                # fallback if neither available
                y_pred_proba = model.predict(X_test_np)

            # handle constant score edge case
            if np.allclose(y_pred_proba, y_pred_proba[0]):
                auroc = np.nan
            else:
                auroc = roc_auc_score(y_test, y_pred_proba)

            # binary threshold at 0.5 for classification metrics
            y_pred = (y_pred_proba > 0.5).astype(int)

            results.append({
                "Classifier": name,
                "AUROC": auroc,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred)
            })

        except Exception as e:
            print(f"⚠️ Error evaluating {name}: {e}")
            continue

    # Save under single-date folder
    ts_format = "%Y%m%d_%H%M" if include_time else "%Y%m%d"
    timestamp = datetime.now().strftime(ts_format)
    out_dir = Path(resolve_path(f"{save_prefix}/{timestamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = f"{mode}_summary.csv"
    out_path = out_dir / out_file
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    print(f"✅ Evaluation summary saved to {out_path}")
    return df

# ======================================================
#  Unwrap Nested Model Dicts Into Pipelines
# ======================================================

def unwrap_best_estimators_smote(models_dict):
    """
    Unwraps the 'best_estimator_smote' from a dictionary of SMOTE-trained model results.
    Falls back to 'best_estimator' if a SMOTE key is not found.
    Skips non-dict entries safely.

    Parameters
    ----------
    models_dict : dict
        Dictionary containing classifier names as keys and model metadata as values.

    Returns
    -------
    dict
        Dictionary mapping classifier names to fitted estimators (SMOTE or fallback non-SMOTE).
    """
    unwrapped = {}
    for name, obj in models_dict.items():
        if isinstance(obj, dict):
            if "best_estimator_smote" in obj:
                unwrapped[name] = obj["best_estimator_smote"]
            elif "best_estimator" in obj:
                unwrapped[name] = obj["best_estimator"]
            else:
                unwrapped[name] = obj
        else:
            unwrapped[name] = obj
    return unwrapped

def unwrap_best_estimators_non_smote(models_dict):
    """Force extraction of non-SMOTE best estimators."""
    out = {}
    for name, obj in models_dict.items():
        if isinstance(obj, dict) and "best_estimator" in obj:
            out[name] = obj["best_estimator"]
        else:
            out[name] = obj
    return out



# =========================================================
# 3A.  Custom Metrics (F2)
# =========================================================
def f2_score(precision, recall):
    """Compute F2 score (recall-weighted)."""
    return 5 * (precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0.0


# =========================================================
# 3B.  Extended Evaluation (Train + Test)
# =========================================================
def evaluate_classifier_extended(model, X_train, y_train, X_test, y_test,
                                 clf_name, mode="original_baseline", threshold=0.5, verbose=True):
    """
    Evaluate both train and test metrics including F2 score.
    Mirrors evaluate_all_models() scoring logic for consistency
    with 04–05 pipeline (decision_function / predict_proba handling).
    Assumes model is pre-trained.
    """

    def get_scores(estimator, X):
        """Return continuous decision scores, matching auc_scorer logic."""
        if hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X)
            if scores.ndim > 1:
                scores = scores[:, 1]
        elif hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(X)[:, 1]
        else:
            scores = estimator.predict(X)
        return scores

    # convert inputs to numpy arrays to ignore feature names
    X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    X_test_np  = X_test.to_numpy()  if hasattr(X_test, "to_numpy")  else X_test

    # --- Predictions and probabilities ---
    y_scores_train = get_scores(model, X_train_np)
    y_pred_train = (y_scores_train >= threshold).astype(int)

    y_scores_test = get_scores(model, X_test_np)
    y_pred_test = (y_scores_test >= threshold).astype(int)

    # --- Training metrics ---
    prec_tr = precision_score(y_train, y_pred_train, zero_division=0)
    rec_tr  = recall_score(y_train, y_pred_train, zero_division=0)
    auc_tr  = np.nan if np.allclose(y_scores_train, y_scores_train[0]) else roc_auc_score(y_train, y_scores_train)

    metrics_train = {
        "AUC_train": auc_tr,
        "Accuracy_train": accuracy_score(y_train, y_pred_train),
        "F1_train": f1_score(y_train, y_pred_train, zero_division=0),
        "Precision_train": prec_tr,
        "Recall_train": rec_tr,
        "F2_train": f2_score(prec_tr, rec_tr)
    }

    # --- Test metrics ---
    prec_te = precision_score(y_test, y_pred_test, zero_division=0)
    rec_te  = recall_score(y_test, y_pred_test, zero_division=0)
    auc_te  = np.nan if np.allclose(y_scores_test, y_scores_test[0]) else roc_auc_score(y_test, y_scores_test)

    metrics_test = {
        "AUC_test": auc_te,
        "Accuracy_test": accuracy_score(y_test, y_pred_test),
        "F1_test": f1_score(y_test, y_pred_test, zero_division=0),
        "Precision_test": prec_te,
        "Recall_test": rec_te,
        "F2_test": f2_score(prec_te, rec_te)
    }

    # --- Combine and print summary ---
    combined = {"Classifier": clf_name, "Mode": mode}
    combined.update(metrics_train)
    combined.update(metrics_test)

    if verbose:
        print(f"\n📊 {clf_name} ({mode})")
        for k, v in combined.items():
            if k not in ("Classifier", "Mode"):
                print(f"   {k:<16}: {v:.4f}")

    return {
        k: round(v, 4) if isinstance(v, (float, np.floating)) else v
        for k, v in combined.items()
    }

# =========================================================
# 4.  ROC & PR Curves (Per Dataset)
# =========================================================
def plot_roc_curves(models, X_test, y_test, mode):
    """Draw ROC curves for all classifiers in a given dataset mode."""
    plt.figure(figsize=(10, 8))
    for name, mdl in models.items():
        try:
            y_p = get_proba(mdl, X_test)
            fpr, tpr, _ = roc_curve(y_test, y_p)
            auc_val = roc_auc_score(y_test, y_p)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.2f})")
        except Exception as e:
            print(f"[skip] {name}: {e}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves – {mode}")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    path = Path(resolve_path(f"results/figures/{mode}/ROC_all.png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    display(Image(filename=f"{path}"))
    print(f"📈 ROC curves saved → {path}")


def plot_pr(models, X_test, y_test, mode):
    """Precision–Recall curve for all classifiers in one dataset."""
    plt.figure(figsize=(10, 8))
    for name, mdl in models.items():
        try:
            y_p = get_proba(mdl, X_test)
            precision, recall, _ = precision_recall_curve(y_test, y_p)
            pr_auc = average_precision_score(y_test, y_p)
            plt.plot(recall, precision, lw=2, label=f"{name} (AUC={pr_auc:.2f})")
        except Exception as e:
            print(f"[skip] {name}: {e}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curves – {mode}")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    path = Path(resolve_path(f"results/figures/{mode}/PR_all.png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    display(Image(filename=f"{path}"))
    print(f"📊 PR curves saved → {path}")


# =========================================================
# 5.  Multi-Dataset Comparisons
# =========================================================
def plot_roc_across_datasets(models_dicts: List[Dict[str, Any]],
                             dataset_labels: List[str],
                             X_tests: List, y_tests: List):
    """Compare ROC curves across dataset modes."""
    plt.figure(figsize=(12, 10))
    legend_entries = []
    for models, label, X, y in zip(models_dicts, dataset_labels, X_tests, y_tests):
        for name, mdl in models.items():
            y_p = get_proba(mdl, X)
            fpr, tpr, _ = roc_curve(y, y_p)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2)
            legend_entries.append((roc_auc, f"{name} ({label}) AUC={roc_auc:.2f}"))
    legend_entries.sort(key=lambda x: x[0], reverse=True)
    handles, labels = [], []
    for _, lbl in legend_entries:
        handles.append(plt.Line2D([0], [0]))
        labels.append(lbl)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison Across Datasets")
    plt.legend(handles, labels, loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    path = resolve_path("results/figures/ROC_comparison_across_datasets.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"📉 Multi-dataset ROC comparison saved → {path}")


def plot_bar_comparison(
    data=None,
    all_models=None,
    dataset_labels=None,
    X_tests=None,
    y_tests=None,
    metric="roc_auc",
    value_col=None,
    label_col="Classifier",
    title=None,
    ylabel=None,
    save_path=None,
    color="steelblue",
):
    """
    Generalized bar plotting utility for evaluation summaries.
    
    Can operate in two modes:
      (1) Model-based comparison across datasets (uses model dicts + X/y).
      (2) DataFrame-based plotting (e.g., ΔAUROC summary).

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame containing precomputed metric values (e.g., merged summary).
    all_models : list of dicts, optional
        List of model dictionaries (one per dataset variant).
    dataset_labels : list of str, optional
        Labels for each dataset variant (used in legend).
    X_tests, y_tests : list, optional
        Matching test sets for each dataset.
    metric : str
        Metric to compute ("roc_auc" or others).
    value_col : str, optional
        Column name in `data` to plot when using DataFrame mode.
    label_col : str
        Column name containing classifier/model names.
    title, ylabel : str, optional
        Custom plot labels.
    save_path : str, optional
        Path to save figure.
    color : str
        Default color for bars in DataFrame mode.
    """

    # ─────────────── Mode 1: From model dictionaries ───────────────
    if all_models is not None and X_tests is not None and y_tests is not None:
        classifiers = list(all_models[0].keys())
        n_datasets = len(all_models)
        bar_width = 0.8 / n_datasets
        x = np.arange(len(classifiers))
        plt.figure(figsize=(12, 6))
        for i, (models, label, X, y) in enumerate(zip(all_models, dataset_labels, X_tests, y_tests)):
            vals = []
            for name in classifiers:
                mdl = models[name]
                y_p = mdl.predict_proba(X)[:, 1] if hasattr(mdl, "predict_proba") else mdl.decision_function(X)
                vals.append(roc_auc_score(y, y_p) if metric == "roc_auc" else np.nan)
            plt.bar(x + i * bar_width, vals, width=bar_width, label=label)
        plt.xticks(x + bar_width * (n_datasets - 1) / 2, classifiers, rotation=45, ha="right")
        plt.ylabel(ylabel or metric.upper())
        plt.title(title or f"{metric.upper()} Comparison Across Datasets")
        plt.legend()

    # ─────────────── Mode 2: From precomputed DataFrame ───────────────
    elif data is not None and value_col is not None:
        plt.figure(figsize=(10, 5))
        classifiers = data[label_col]
        vals = data[value_col]
        plt.bar(classifiers, vals, color=color, alpha=0.85)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(ylabel or value_col)
        plt.title(title or f"{value_col} Comparison")
    else:
        raise ValueError("Either provide (all_models + X_tests + y_tests) or (data + value_col).")

    # ─────────────── Final Formatting ───────────────
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save and display
    if save_path:
        from src.utils import resolve_path
        path = resolve_path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"📊 Bar chart saved → {path}")
        display(Image(filename=str(path)))
        plt.close()
    else:
        plt.show()

def plot_delta_auroc_bar(summary_df, delta_col="Δ_AUROC_generalization", save_path=None):
    """Plot ΔAUROC bar chart across classifiers."""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 5))
    classifiers = summary_df["Classifier"]
    deltas = summary_df[delta_col]

    bars = plt.bar(classifiers, deltas, color="steelblue", alpha=0.85)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Δ AUROC (Holdout − CV)")
    plt.title("Generalization Gap per Classifier")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 ΔAUROC bar chart saved → {save_path}")
    else:
        plt.show()



# =========================================================
# 6.  Feature Importance (Tree Models)
# =========================================================
def plot_top_features(tree_model_pipe, top_n=15, mode="original"):
    """Horizontal bar of top-N importances from tree-based pipeline."""
    clf = tree_model_pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise TypeError(f"{type(clf).__name__} lacks feature_importances_")
    scaler = tree_model_pipe.named_steps.get("scaler")
    if hasattr(scaler, "feature_names_in_"):
        names = scaler.feature_names_in_
    elif hasattr(clf, "feature_names_in_"):
        names = clf.feature_names_in_
    else:
        names = [f"feature_{i}" for i in range(clf.n_features_in_)]
    df = (pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
          .sort_values("importance", ascending=False).head(top_n))
    plt.figure(figsize=(8, 6))
    plt.barh(df["feature"][::-1], df["importance"][::-1], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Importances – {type(clf).__name__}")
    plt.tight_layout()
    path = resolve_path(f"results/figures/{mode}/TopFeatures_{type(clf).__name__}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"🌳 Feature-importance plot saved → {path}")


# =========================================================
# 7.  SHAP Explainability (Tree Models)
# =========================================================
def plot_shap_summary(tree_model_pipe, X_sample, top_n=15, mode="original"):
    """SHAP summary (bar) for tree-based models."""
    clf = tree_model_pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise TypeError(f"{type(clf).__name__} not supported for SHAP explainability")
    explainer = shap.Explainer(clf, X_sample)
    shap_vals = explainer(X_sample)
    shap.summary_plot(shap_vals, X_sample, max_display=top_n, plot_type="bar", show=False)
    path = resolve_path(f"results/figures/{mode}/SHAP_{type(clf).__name__}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"🧠 SHAP summary saved → {path}")

# =========================================================
# 8. SHAP Dependence Plot Helper
# =========================================================

def plot_shap_dependence(shap_values, X, feature_name, clf_name, mode, save_dir="results/figures/shap/"):
    """
    Generate and save a SHAP dependence plot for a given feature.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values computed from shap.Explainer().
    X : pd.DataFrame
        Input sample data corresponding to shap_values.
    feature_name : str
        The feature to plot dependence for.
    clf_name : str
        Classifier name (e.g., 'XGB', 'LGBM').
    mode : str
        Variant or dataset label (e.g., 'original_baseline').
    save_dir : str, optional
        Directory to save the dependence plot (default: results/figures/shap/).
    """
    save_path = resolve_path(f"{save_dir}/shap_dependence_{mode}_{clf_name}_{feature_name}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shap.dependence_plot(
            feature_name,
            shap_values.values,
            X,
            show=False
        )
        plt.title(f"Dependence: {feature_name} ({clf_name}, {mode})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Saved SHAP dependence plot → {save_path}")
    except Exception as e:
        print(f"⚠️ Could not create SHAP dependence plot for {feature_name}: {e}")

    