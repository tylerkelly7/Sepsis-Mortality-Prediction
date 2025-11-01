# src/stat_tests.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union, Optional
from sklearn.metrics import roc_auc_score

ArrayLike = Union[np.ndarray, Sequence[float], pd.Series, List[float]]

# -----------------------------
# Utilities
# -----------------------------
def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.values
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1-D.")
    return x

def _check_binary_labels(y: np.ndarray):
    vals = np.unique(y)
    if not set(vals).issubset({0, 1}):
        raise ValueError("y_true must be binary with values {0,1}.")

import numpy as np
from scipy.stats import norm

# ------------------------------------------------------------
# Midrank helper (unchanged)
# ------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(n, dtype=float)
    T2[J] = T
    return T2


# ------------------------------------------------------------
# Correct Fast DeLong
# ------------------------------------------------------------
def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    pos_preds = predictions_sorted_transposed[:, :m]
    neg_preds = predictions_sorted_transposed[:, m:]

    tx = np.empty((predictions_sorted_transposed.shape[0], m))
    ty = np.empty((predictions_sorted_transposed.shape[0], n))
    for r in range(predictions_sorted_transposed.shape[0]):
        tx[r] = _compute_midrank(pos_preds[r])
        ty[r] = _compute_midrank(neg_preds[r])

    tz = np.hstack((pos_preds, neg_preds))
    T = np.empty((predictions_sorted_transposed.shape[0], m + n))
    for r in range(predictions_sorted_transposed.shape[0]):
        T[r] = _compute_midrank(tz[r])

    aucs = T[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (T[:, :m] - tx) / n
    v10 = 1.0 - (T[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    S = sx / m + sy / n
    return aucs, S


# ------------------------------------------------------------
# Paired DeLong test (direct SMT replica)
# ------------------------------------------------------------
def delong_roc_test(y_true, y_pred1, y_pred2):
    """Paired DeLong test comparing two ROC AUCs."""
    y_true = np.asarray(y_true).astype(int)
    y_pred1 = np.asarray(y_pred1, dtype=float)
    y_pred2 = np.asarray(y_pred2, dtype=float)

    # ---- Sort by labels (positives first) ----
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    preds = np.vstack((y_pred1, y_pred2))[:, order]

    label_1_count = int(np.sum(y_true_sorted))
    aucs, S = _fast_delong(preds, label_1_count)

    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = np.sqrt(var)
    z = diff / se
    p = 2 * norm.sf(abs(z))

    return {
        "auc1": aucs[0],
        "auc2": aucs[1],
        "delta": diff,
        "se": se,
        "z": z,
        "p": p,
    }


# -----------------------------
# Bootstrap AUROC CI (percentile)
# -----------------------------
def bootstrap_auc_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_boot: int = 2000,
    alpha: float = 0.95,
    random_state: Optional[int] = 42,
    stratified: bool = True,
) -> Dict[str, float]:
    """
    Percentile bootstrap CI for AUROC on a single model.
    Returns dict with: auc, mean_boot, se, ci_low, ci_high.
    """
    y_true = _to_numpy(y_true); y_pred = _to_numpy(y_pred)
    _check_binary_labels(y_true)

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    aucs = []
    if stratified:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos = len(pos_idx); n_neg = len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Both classes needed for AUROC.")
        for _ in range(n_boot):
            s_pos = rng.choice(pos_idx, size=n_pos, replace=True)
            s_neg = rng.choice(neg_idx, size=n_neg, replace=True)
            idx = np.concatenate([s_pos, s_neg])
            aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    else:
        for _ in range(n_boot):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))

    aucs = np.asarray(aucs)
    base_auc = roc_auc_score(y_true, y_pred)
    mean_boot = float(np.mean(aucs))
    se = float(np.std(aucs, ddof=1))
    lo = float(np.quantile(aucs, (1 - alpha) / 2))
    hi = float(np.quantile(aucs, 1 - (1 - alpha) / 2))
    return dict(auc=base_auc, mean_boot=mean_boot, se=se, ci_low=lo, ci_high=hi)

# -----------------------------
# Holm–Bonferroni FWER control
# -----------------------------
def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, Union[List[float], List[bool]]]:
    """
    Holm–Bonferroni step-down procedure. Returns adjusted p-values and rejections.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    rejections = np.zeros(m, dtype=bool)
    prev = 0.0
    for k, idx in enumerate(order):
        adj_val = (m - k) * p[idx]
        adj[idx] = max(prev, adj_val)
        prev = adj[idx]
    adj = np.minimum(adj, 1.0)
    # decisions step-down
    sorted_adj = adj[order]
    for k, val in enumerate(sorted_adj):
        if val <= alpha:
            rejections[order[k]] = True
        else:
            rejections[order[k:]] = False
            break
    return {"p_adjusted": adj.tolist(), "reject": rejections.tolist()}

# -----------------------------
# Batch comparison helper
# -----------------------------
def compare_aurocs(
    model_probs: Dict[str, np.ndarray],
    y_true: ArrayLike,
    pairs: Sequence[Tuple[str, str]],
    method: str = "delong",
    alpha: float = 0.95,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """
    Compare AUROCs across model pairs on the SAME subjects.
    model_probs maps model_name -> probability array (aligned to y_true).
    pairs: list of (model_a, model_b) to test (delta = auc_a - auc_b).
    method: "delong" or "bootstrap" (bootstrap here returns CIs for each model, no paired p-value).
    """
    y = _to_numpy(y_true)
    rows = []
    for a, b in pairs:
        pa = _to_numpy(model_probs[a]); pb = _to_numpy(model_probs[b])
        if len(pa) != len(y) or len(pb) != len(y):
            raise ValueError(f"Probability length mismatch for pair ({a}, {b}).")
        auc_a = roc_auc_score(y, pa)
        auc_b = roc_auc_score(y, pb)
        delta = auc_a - auc_b

        if method.lower() == "delong":
            res = delong_roc_test(y, pa, pb)
            rows.append({
                "model_a": a, "model_b": b,
                "auc_a": auc_a, "auc_b": auc_b, "delta": delta,
                "se_delta": res["se"], "z": res["z"], "p_value": res["p"],
                "ci_low": np.nan, "ci_high": np.nan, "method": "delong"
            })
        elif method.lower() == "bootstrap":
            # Per-model CIs; no paired p-value here
            ca = bootstrap_auc_ci(y, pa, n_boot=n_boot, alpha=alpha)
            cb = bootstrap_auc_ci(y, pb, n_boot=n_boot, alpha=alpha)
            rows.append({
                "model_a": a, "model_b": b,
                "auc_a": auc_a, "auc_b": auc_b, "delta": delta,
                "se_delta": np.nan, "z": np.nan, "p_value": np.nan,
                "ci_low": ca["ci_low"], "ci_high": ca["ci_high"], "method": "bootstrap_a"
            })
            rows.append({
                "model_a": b, "model_b": a,
                "auc_a": auc_b, "auc_b": auc_a, "delta": -delta,
                "se_delta": np.nan, "z": np.nan, "p_value": np.nan,
                "ci_low": cb["ci_low"], "ci_high": cb["ci_high"], "method": "bootstrap_b"
            })
        else:
            raise ValueError("method must be 'delong' or 'bootstrap'.")

    return pd.DataFrame(rows)
