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
from datetime import datetime
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

from tableone import TableOne

from IPython.display import Image, display

# =========================================================
# 1. Export Summary
# =========================================================

def export_summary(
    summary_df, mode="original", save_prefix="results/evaluation", include_time=False
):
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


# ======================================================
# 2. Unwrap Nested Model Dicts Into Pipelines
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
# 3. Probability Helper
# =========================================================
def get_scores(model, X):
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
        # normalize to [0,1]
        min_s, max_s = scores.min(), scores.max()
        return (scores - min_s) / (max_s - min_s + 1e-8)
    raise ValueError(f"{model.__class__.__name__} lacks probability output.")
    # last resort
    preds = model.predict(X)
    return np.clip(preds, 0, 1)


# =========================================================
# 4.  Core Metric Evaluation
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
        "Brier": brier_score_loss(y_test, y_proba),
    }
    return {
        k: round(v, 4) if isinstance(v, (float, np.floating)) else v
        for k, v in metrics.items()
    }


def evaluate_all_models(
    models,
    X_test,
    y_test,
    mode="original",
    save_prefix="results/evaluation",
    include_time=False,
    models_meta=None,
):
    """
    Evaluate all models on holdout set.
    Adds CV mean/std from nested metadata dicts if provided,
    and computes Brier score for probabilistic calibration.
    """

    X_test_np = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
    results = []

    for name, model in models.items():
        try:
            # --- Get predictions / probabilities ---
            y_pred_proba = get_scores(model, X_test_np)

            # --- Primary performance metrics ---
            auroc = np.nan if np.allclose(y_pred_proba, y_pred_proba[0]) else roc_auc_score(y_test, y_pred_proba)
            y_pred = (y_pred_proba > 0.5).astype(int)
            brier = brier_score_loss(y_test, y_pred_proba)

            # --- Get CV mean/std from metadata ---
            cv_mean, cv_std = (np.nan, np.nan)
            if models_meta:
                cv_mean, cv_std = extract_cv_stats(models_meta, name, mode)

            results.append(
                {
                    "Classifier": name,
                    "AUROC": auroc,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred),
                    "F1": f1_score(y_test, y_pred),
                    "Brier": brier,
                    "Mean CV AUROC": cv_mean,
                    "SD CV AUROC": cv_std,
                }
            )

        except Exception as e:
            print(f"⚠️ Error evaluating {name}: {e}")
            continue

    # ──────────────────────────────────────────────
    # Save to results/evaluation/<date>/
    # ──────────────────────────────────────────────
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

# =========================================================
# 4A.  Custom Metrics (F2)
# =========================================================
def f2_score(precision, recall):
    """Compute F2 score (recall-weighted)."""
    return (
        5 * (precision * recall) / ((4 * precision) + recall)
        if (precision + recall) > 0
        else 0.0
    )


# =========================================================
# 4B.  Extended Evaluation (Train + Test)
# =========================================================
def evaluate_classifier_extended(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    clf_name,
    mode="original_baseline",
    threshold=0.5,
    verbose=True,
    models_meta=None,
):
    """
    Extended evaluation including train/test metrics,
    CV mean/std from nested metadata dicts (if provided),
    and Brier score for calibration.
    """

    X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    X_test_np = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test

    y_scores_train = get_scores(model, X_train_np)
    y_pred_train = (y_scores_train >= threshold).astype(int)
    y_scores_test = get_scores(model, X_test_np)
    y_pred_test = (y_scores_test >= threshold).astype(int)

    # --- Train metrics ---
    prec_tr = precision_score(y_train, y_pred_train, zero_division=0)
    rec_tr = recall_score(y_train, y_pred_train, zero_division=0)
    auc_tr = np.nan if np.allclose(y_scores_train, y_scores_train[0]) else roc_auc_score(y_train, y_scores_train)
    brier_tr = brier_score_loss(y_train, y_scores_train)

    # --- Test metrics ---
    prec_te = precision_score(y_test, y_pred_test, zero_division=0)
    rec_te = recall_score(y_test, y_pred_test, zero_division=0)
    auc_te = np.nan if np.allclose(y_scores_test, y_scores_test[0]) else roc_auc_score(y_test, y_scores_test)
    brier_te = brier_score_loss(y_test, y_scores_test)

    # --- Pull CV mean/std from metadata ---
    cv_mean, cv_std = (np.nan, np.nan)
    if models_meta:
        cv_mean, cv_std = extract_cv_stats(models_meta, clf_name, mode)

    combined = {
        "Classifier": clf_name,
        "Mode": mode,
        "AUC_train": auc_tr,
        "AUC_test": auc_te,
        "CV_AUROC_mean": cv_mean,
        "CV_AUROC_std": cv_std,
        "Brier_train": brier_tr,
        "Brier_test": brier_te,
        "Accuracy_train": accuracy_score(y_train, y_pred_train),
        "Accuracy_test": accuracy_score(y_test, y_pred_test),
        "Precision_train": prec_tr,
        "Precision_test": prec_te,
        "Recall_train": rec_tr,
        "Recall_test": rec_te,
        "F1_train": f1_score(y_train, y_pred_train, zero_division=0),
        "F1_test": f1_score(y_test, y_pred_test, zero_division=0),
        "F2_train": f2_score(prec_tr, rec_tr),
        "F2_test": f2_score(prec_te, rec_te),
    }

    if verbose:
        print(
            f"\n📊 {clf_name} ({mode}) → "
            f"AUC_test={auc_te:.4f}, Brier_test={brier_te:.4f}, "
            f"CV_mean={cv_mean:.4f}, CV_std={cv_std:.4f}"
        )

    return {k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in combined.items()}


def _safe_get_proba(model, X):
    """Return probabilities or normalized decision scores for pipelines or estimators."""
    # Unwrap pipeline if necessary
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        model = model.named_steps["clf"]

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min())

    raise AttributeError(f"{type(model).__name__} has neither predict_proba nor decision_function")

# ======================================================
# 5. Extract CV Stats from Nested Metadata
# ======================================================

def extract_cv_stats(models_meta, clf_name, mode=None):
    """
    Extract mean/std test scores from nested model metadata dicts.
    Supports both non-SMOTE and SMOTE variants.
    """
    cv_mean, cv_std = np.nan, np.nan
    try:
        meta = models_meta.get(clf_name, {})
        # Case 1: non-SMOTE
        if "cv_results" in meta:
            cv = meta["cv_results"]
            if "mean_test_score" in cv and "std_test_score" in cv:
                cv_mean = float(np.nanmax(cv["mean_test_score"]))
                cv_std = float(cv["std_test_score"][np.nanargmax(cv["mean_test_score"])])
        # Case 2: SMOTE keys if requested
        elif mode and "smote" in mode.lower() and "smote_metrics" in meta:
            cv_mean = float(meta["smote_metrics"].get("descriptive_cv_mean_auc_smote", np.nan))
            cv_std = float(meta["smote_metrics"].get("descriptive_cv_std_auc_smote", np.nan))
    except Exception as e:
        print(f"[warn] Could not extract CV stats for {clf_name}: {e}")
    return cv_mean, cv_std

# =========================================================
# 6.  ROC & PR Curves (Per Dataset)
# =========================================================
def plot_roc_curves(models, X_test, y_test, mode):
    """Draw ROC curves for all classifiers in a given dataset mode."""
    plt.figure(figsize=(10, 8))
    for name, mdl in models.items():
        try:
            y_p = _safe_get_proba(mdl, X_test)
            fpr, tpr, _ = roc_curve(y_test, y_p)
            auc_val = roc_auc_score(y_test, y_p)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.2f})")
        except Exception as e:
            print(f"[skip] {name}: {e}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves – {mode}")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    path = Path(resolve_path(f"results/figures/{mode}/{mode}_ROC_all.png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    display(Image(filename=f"{path}"))
    print(f"📈 ROC curves saved → {path}")


def plot_pr(models, X_test, y_test, mode):
    """Precision–Recall curve for all classifiers in one dataset."""
    plt.figure(figsize=(10, 8))
    for name, mdl in models.items():
        try:
            y_p = _safe_get_proba(mdl, X_test)
            precision, recall, _ = precision_recall_curve(y_test, y_p)
            pr_auc = average_precision_score(y_test, y_p)
            plt.plot(recall, precision, lw=2, label=f"{name} (AUC={pr_auc:.2f})")
        except Exception as e:
            print(f"[skip] {name}: {e}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curves – {mode}")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    path = Path(resolve_path(f"results/figures/{mode}/{mode}_PR_all.png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    display(Image(filename=f"{path}"))
    print(f"📊 PR curves saved → {path}")


# =========================================================
# 7.  Multi-Dataset Comparisons
# =========================================================
def plot_roc_across_datasets(
    models_dicts: List[Dict[str, Any]],
    dataset_labels: List[str],
    X_tests: List,
    y_tests: List,
):
    """Compare ROC curves across dataset modes."""
    plt.figure(figsize=(12, 10))
    legend_entries = []
    for models, label, X, y in zip(models_dicts, dataset_labels, X_tests, y_tests):
        for name, mdl in models.items():
            y_p = _safe_get_proba(mdl, X)
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
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison Across Datasets")
    plt.legend(handles, labels, loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    path = resolve_path("results/figures/ROC_comparison_across_datasets.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
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
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (models, label, X, y) in enumerate(
            zip(all_models, dataset_labels, X_tests, y_tests)
        ):
            vals = []
            for name in classifiers:
                mdl = models[name]
                y_p = _safe_get_proba(mdl, X)
                vals.append(roc_auc_score(y, y_p) if metric == "roc_auc" else np.nan)
            ax.bar(x + i * bar_width, vals, width=bar_width, label=label)

        ax.set_xticks(x + bar_width * (n_datasets - 1) / 2)
        ax.set_xticklabels(classifiers, rotation=45, ha="right")
        ax.set_ylabel(ylabel or metric.upper())
        ax.set_title(title or f"{metric.upper()} Comparison Across Datasets")

    # ─────────────── Mode 2: From precomputed DataFrame ───────────────
    elif data is not None and value_col is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        classifiers = data[label_col]
        vals = data[value_col]
        ax.bar(classifiers, vals, color=color, alpha=0.85)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticklabels(classifiers, rotation=45, ha="right")
        ax.set_ylabel(ylabel or value_col)
        ax.set_title(title or f"{value_col} Comparison")
    else:
        raise ValueError(
            "Either provide (all_models + X_tests + y_tests) or (data + value_col)."
        )

    # ─────────────── Final Formatting ───────────────
    ax.grid(axis="y", alpha=0.3)
    if ax.get_legend() is not None:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            title="Dataset Variant",
        )

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # add space for side legend

    # Save and display
    if save_path:
        from src.utils import resolve_path
        path = resolve_path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"📊 Bar chart saved → {path}")
        display(Image(filename=str(path)))
        plt.close(fig)
    else:
        plt.show()

# =========================================================
# 8.  ΔAUROC Bar Chart
# =========================================================

def plot_delta_auroc_bar(
    summary_df, delta_col="Δ_AUROC_generalization", save_path=None
):
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
# 9.  Feature Importance (Tree Models)
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
    df = (
        pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    plt.figure(figsize=(8, 6))
    plt.barh(df["feature"][::-1], df["importance"][::-1], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Importances – {type(clf).__name__}")
    plt.tight_layout()
    path = resolve_path(f"results/figures/{mode}/TopFeatures_{type(clf).__name__}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🌳 Feature-importance plot saved → {path}")


# =========================================================
# 10.  SHAP Explainability (Tree Models)
# =========================================================
def plot_shap_summary(tree_model_pipe, X_sample, top_n=15, mode="original"):
    """SHAP summary (bar) for tree-based models."""
    clf = tree_model_pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise TypeError(f"{type(clf).__name__} not supported for SHAP explainability")
    explainer = shap.Explainer(clf, X_sample)
    shap_vals = explainer(X_sample)
    shap.summary_plot(
        shap_vals, X_sample, max_display=top_n, plot_type="bar", show=False
    )
    path = resolve_path(f"results/figures/{mode}/SHAP_{type(clf).__name__}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🧠 SHAP summary saved → {path}")


# =========================================================
# 11. SHAP Dependence Plot Helper
# =========================================================


def plot_shap_dependence(
    shap_values, X, feature_name, clf_name, mode, save_dir="results/figures/shap/"
):
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
    save_path = resolve_path(
        f"{save_dir}/shap_dependence_{mode}_{clf_name}_{feature_name}.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shap.dependence_plot(feature_name, shap_values.values, X, show=False)
        plt.title(f"Dependence: {feature_name} ({clf_name}, {mode})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Saved SHAP dependence plot → {save_path}")
    except Exception as e:
        print(f"⚠️ Could not create SHAP dependence plot for {feature_name}: {e}")

# =========================================================
# 12. Generate Table 1
# =========================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
from src.utils import resolve_path


def generate_table1(df: pd.DataFrame, output_name: str = "table1_structured"):

    outcome = "hospital_expire_flag"
    if outcome not in df.columns:
        raise ValueError(f"{outcome} not in dataframe")

    # ============================================================
    # Variable definitions
    # ============================================================

    continuous = [
        "max_age", "los_icu", "sofa_score", "avg_urineoutput",
        "glucose_min", "glucose_max", "glucose_average",
        "sodium_min", "sodium_max", "sodium_average",
        "heart_rate_min", "heart_rate_max", "heart_rate_mean",
        "sbp_min", "sbp_max", "sbp_mean",
        "dbp_min", "dbp_max", "dbp_mean",
        "resp_rate_min", "resp_rate_max", "resp_rate_mean",
        "spo2_min", "spo2_max", "spo2_mean",
        "albumin"
    ]

    # Variables to FORCE into mean (SD) even if Shapiro–Wilk fails
    forced_mean_sd = {
        "max_age",
        "heart_rate_min", "heart_rate_max", "heart_rate_mean",
        "sbp_min", "sbp_max", "sbp_mean",
        "dbp_min", "dbp_max", "dbp_mean",
        "resp_rate_min", "resp_rate_max", "resp_rate_mean",
        "spo2_min", "spo2_max", "spo2_mean",
        "sodium_min", "sodium_max", "sodium_average",
    }


    categorical = [
        "first_hosp_stay", "suspected_infection", "sepsis3", "coma",
        "diabetes_without_cc", "diabetes_with_cc", "severe_liver_disease",
        "aids", "renal_disease",
        "race_Black or African American",
        "race_Hispanic or Latin",
        "race_Others race",
        "race_White",
        "antibiotic_Vancomycin",
        "antibiotic_Vancomycin Antibiotic Lock",
        "antibiotic_Vancomycin Enema",
        "antibiotic_Vancomycin Intrathecal",
        "antibiotic_Vancomycin Oral Liquid",
        "gender_F", "gender_M"
    ]

    label_map = {
        "max_age": "Age (years)",
        "los_icu": "ICU Length of Stay (days)",
        "sofa_score": "SOFA Score",
        "avg_urineoutput": "Average Urine Output (mL/day)",

        "glucose_min": "Glucose (Min)",
        "glucose_max": "Glucose (Max)",
        "glucose_average": "Glucose (Average)",
        "sodium_min": "Sodium (Min)",
        "sodium_max": "Sodium (Max)",
        "sodium_average": "Sodium (Average)",

        "heart_rate_min": "Heart Rate (Min)",
        "heart_rate_max": "Heart Rate (Max)",
        "heart_rate_mean": "Heart Rate (Mean)",
        "sbp_min": "Systolic BP (Min)",
        "sbp_max": "Systolic BP (Max)",
        "sbp_mean": "Systolic BP (Mean)",
        "dbp_min": "Diastolic BP (Min)",
        "dbp_max": "Diastolic BP (Max)",
        "dbp_mean": "Diastolic BP (Mean)",
        "resp_rate_min": "Respiratory Rate (Min)",
        "resp_rate_max": "Respiratory Rate (Max)",
        "resp_rate_mean": "Respiratory Rate (Mean)",
        "spo2_min": "SpO2 (Min)",
        "spo2_max": "SpO2 (Max)",
        "spo2_mean": "SpO2 (Mean)",
        "albumin": "Albumin (g/dL)",

        "first_hosp_stay": "First Hospital Stay",
        "suspected_infection": "Suspected Infection",
        "sepsis3": "Sepsis-3 Criteria Met",
        "coma": "Coma at Admission",
        "diabetes_without_cc": "Diabetes (No Complications)",
        "diabetes_with_cc": "Diabetes (With Complications)",
        "severe_liver_disease": "Severe Liver Disease",
        "aids": "AIDS/HIV",
        "renal_disease": "Renal Disease",

        "gender_F": r"\hspace{1em}Female",
        "gender_M": r"\hspace{1em}Male",

        # Race (with indent)
        "race_Black or African American": r"\hspace{1em}Black or African American",
        "race_Hispanic or Latin": r"\hspace{1em}Hispanic or Latin",
        "race_Others race": r"\hspace{1em}Other Race",
        "race_White": r"\hspace{1em}White",

        # Vancomycin (with indent)
        "antibiotic_Vancomycin": r"\hspace{1em}IV",
        "antibiotic_Vancomycin Antibiotic Lock": r"\hspace{1em}Antibiotic Lock",
        "antibiotic_Vancomycin Enema": r"\hspace{1em}Enema",
        "antibiotic_Vancomycin Intrathecal": r"\hspace{1em}Intrathecal",
        "antibiotic_Vancomycin Oral Liquid": r"\hspace{1em}Oral Liquid",
    }

    # Synthetic parent row for Race
    label_map["RACE_PARENT"] = "Race"
    label_map["SEX_PARENT"] = "Sex"

    # ============================================================
    # Masks
    # ============================================================

    mask_all = df[outcome].notna()
    mask_surv = df[outcome] == 0
    mask_nonsurv = df[outcome] == 1

    n_all = int(mask_all.sum())
    n_surv = int(mask_surv.sum())
    n_nonsurv = int(mask_nonsurv.sum())

    # ============================================================
    # Shapiro normality
    # ============================================================

    normal_map = {}
    for var in continuous:
        x = df.loc[mask_all, var].dropna()
        if len(x) > 5000:
            x = x.sample(5000, random_state=42)
        if len(x) < 3:
            normal_map[var] = False
            continue
        try:
            _, p = shapiro(x)
            normal_map[var] = (p >= 0.05)
        except Exception:
            normal_map[var] = False

    # ============================================================
    # Formatting helpers
    # ============================================================

    def fmt_cont(var, mask):
        s = df.loc[mask, var].dropna()
        if s.empty:
            return ""

        # Forced mean(SD) variables
        if var in forced_mean_sd:
            return f"{s.mean():.1f} ({s.std(ddof=1):.1f})"

        # Default Shapiro–Wilk decision
        if normal_map[var]:
            return f"{s.mean():.1f} ({s.std(ddof=1):.1f})"

        # Median [IQR]
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


    def fmt_bin(var, mask):
        s = df.loc[mask, var]
        yes = int((s == 1).sum())
        n = int(mask.sum())
        pct = 100 * yes / n
        return f"{yes}, {pct:.1f}\\%"


    def p_cont(var):
        x0 = df.loc[mask_surv, var].dropna()
        x1 = df.loc[mask_nonsurv, var].dropna()
        if len(x0) < 3 or len(x1) < 3:
            return ""
        if normal_map[var] and len(x0) >= 30 and len(x1) >= 30:
            _, p = ttest_ind(x0, x1, equal_var=False)
        else:
            _, p = mannwhitneyu(x0, x1)
        return "$<0.001$" if p < 0.001 else f"{p:.3f}"

    def p_bin(var):
        a = int(((df[outcome] == 0) & (df[var] == 1)).sum())
        b = int(((df[outcome] == 0) & (df[var] == 0)).sum())
        c = int(((df[outcome] == 1) & (df[var] == 1)).sum())
        d = int(((df[outcome] == 1) & (df[var] == 0)).sum())
        tab = np.array([[a, b], [c, d]])
        if tab.sum() == 0:
            return ""
        expected = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
        if (expected < 5).any():
            _, p = fisher_exact(tab)
        else:
            _, p, _, _ = chi2_contingency(tab)
        return "$<0.001$" if p < 0.001 else f"{p:.3f}"

    # ============================================================
    # Build ordered rows (with section markers)
    # ============================================================

    rows = []

    # First row: n
    rows.append({
        "Variable": "n",
        "Overall": str(n_all),
        "Survivors": str(n_surv),
        "Non-Survivors": str(n_nonsurv),
        "P-Value": ""
    })

    SECTIONS = [
        ("Demographics", [
            "max_age",
            "SEX_PARENT",
            "gender_F", "gender_M",
            "RACE_PARENT",
            "race_Black or African American",
            "race_Hispanic or Latin",
            "race_Others race",
            "race_White",
        ]),
        ("ICU Characteristics", [
            "los_icu", "coma", "first_hosp_stay", "suspected_infection", "sepsis3"
        ]),
        ("Vitals", [
            "heart_rate_min", "heart_rate_max", "heart_rate_mean",
            "sbp_min", "sbp_max", "sbp_mean",
            "dbp_min", "dbp_max", "dbp_mean",
            "resp_rate_min", "resp_rate_max", "resp_rate_mean",
            "spo2_min", "spo2_max", "spo2_mean",
            "avg_urineoutput"
        ]),
        ("Laboratory Measurements", [
            "glucose_min", "glucose_max", "glucose_average",
            "sodium_min", "sodium_max", "sodium_average",
            "albumin"
        ]),
        ("Comorbidities", [
            "diabetes_without_cc", "diabetes_with_cc",
            "severe_liver_disease", "aids", "renal_disease"
        ]),
        ("Vancomycin Administration", [
            "antibiotic_Vancomycin",
            "antibiotic_Vancomycin Antibiotic Lock",
            "antibiotic_Vancomycin Enema",
            "antibiotic_Vancomycin Intrathecal",
            "antibiotic_Vancomycin Oral Liquid"
        ])
    ]

    for sec_name, vars_in_sec in SECTIONS:
        # section marker row
        rows.append({
            "Variable": f"SECTION:{sec_name}",
            "IsSectionStart": sec_name == SECTIONS[0][0],  # mark first section
            "Overall": "",
            "Survivors": "",
            "Non-Survivors": "",
            "P-Value": ""
        })

        for var in vars_in_sec:
            label = label_map.get(var, var)

            if var == "SEX_PARENT":
                # Sex parent row – skip (no stats)
                rows.append({
                    "Variable": label,
                    "Overall": "",
                    "Survivors": "",
                    "Non-Survivors": "",
                    "P-Value": ""
                })
                continue

            if var == "RACE_PARENT":
                # Race parent row – label only, no stats
                rows.append({
                    "Variable": label,
                    "Overall": "",
                    "Survivors": "",
                    "Non-Survivors": "",
                    "P-Value": ""
                })
                continue

            if var in continuous:
                rows.append({
                    "Variable": label,
                    "Overall": fmt_cont(var, mask_all),
                    "Survivors": fmt_cont(var, mask_surv),
                    "Non-Survivors": fmt_cont(var, mask_nonsurv),
                    "P-Value": p_cont(var)
                })
            else:
                rows.append({
                    "Variable": label,
                    "Overall": fmt_bin(var, mask_all),
                    "Survivors": fmt_bin(var, mask_surv),
                    "Non-Survivors": fmt_bin(var, mask_nonsurv),
                    "P-Value": p_bin(var)
                })

    df_out = pd.DataFrame(rows)

    # ============================================================
    # Write LaTeX longtable (standalone)
    # ============================================================

    output_dir = resolve_path("reports/tables")
    os.makedirs(output_dir, exist_ok=True)

    tex_path = output_dir / f"{output_name}.tex"
    csv_path = output_dir / f"{output_name}.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = []
    lines.append(r"\setlength{\tabcolsep}{3pt}  % default is 6pt")
    lines.append(r"\begin{longtable}{p{3.5cm}cccc}")
    lines.append(r"\caption{Descriptive Statistics for Structured Features Included in this Study.}")
    lines.append(r"\label{tab:table1} \\")

    lines.append(r"\toprule")
    lines.append(r"Variable & Overall & Survivors & Non-Survivors & P-Value \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    lines.append(r"\toprule")
    lines.append(r"Variable & Overall & Survivors & Non-Survivors & P-Value \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{r}{Continued on next page} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfoot")

    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, row in df_out.iterrows():
        var = str(row["Variable"])
        if row["Variable"].startswith("SECTION:"):
            sec_name = row["Variable"].replace("SECTION:", "")
            if row.get("IsSectionStart", False):
                # first section: NO top midrule
                lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{sec_name}}}}} \\[-3pt]")
                lines.append(r"\midrule")
            else:
                # later sections: midrule ABOVE, clean look
                lines.append(r"\midrule")
                lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{sec_name}}}}} \\[-3pt]")
                lines.append(r"\midrule")
            continue

        lines.append(
            f"{var} & {row['Overall']} & "
            f"{row['Survivors']} & {row['Non-Survivors']} & {row['P-Value']} \\\\"
        )

    lines.append(r"\end{longtable}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Table 1 saved:\n - {csv_path}\n - {tex_path}")

    return df_out
