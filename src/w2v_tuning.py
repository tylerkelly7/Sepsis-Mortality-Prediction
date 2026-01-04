"""
src/w2v_tuning.py
=========================================
Handles Word2Vec hyperparameter tuning, intrinsic evaluation,
and export of tuned embeddings for the Masters-Thesis project
on sepsis mortality prediction.

Outputs:
- Intermediate models: embedding_cache/w2v/variants/{note_type}/
- Tuned models: embedding_cache/w2v/tuned/{note_type}/
- Intrinsic metrics: results/embeddings/{note_type}/
- Visualization: reports/figures/w2v_intrinsic_{note_type}.png

This module is designed for use in notebooks such as
`08_w2v_hyperparam_search.ipynb`.
"""

import os
import time
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import resolve_path
from matplotlib.patches import Patch


# =====================================================
# 1. Parameter Grid Definition
# =====================================================


def define_w2v_param_grid():
    """Return parameter grid for coarse-to-moderate hyperparameter search."""
    return {
        "vector_size": [100, 200],  # moderate sizes; 300 = +40% time, marginal gain
        "window": [5, 10],  # local (5) vs broader (10) context
        "min_count": [2, 5],  # include rare tokens vs prune noise
        "sg": [0, 1],  # CBOW vs Skip-Gram
        "negative": [5, 10],  # sampling depth trade-off
        "epochs": [15, 25],
    }


# =====================================================
# 2. Training Utilities
# =====================================================


def train_w2v_variant(sentences, params, tag, note_type):
    """
    Train a Word2Vec variant on tokenized sentences and save under variants folder.

    Args:
        sentences: tokenized list of sentences for the corpus.
        params: dict of Word2Vec hyperparameters.
        tag: short tag for the run (e.g., 'cfg3').
        note_type: e.g., 'Radiology', 'Discharge'.

    Returns:
        model, runtime_seconds, model_path
    """
    start = time.time()
    model = Word2Vec(sentences, workers=12, seed=42, **params)

    save_dir = resolve_path(f"embedding_cache/w2v/variants/{note_type}")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{tag}.model")

    model.save(model_path)
    runtime = round(time.time() - start, 2)
    return model, runtime, model_path


# =====================================================
# 3. Intrinsic Evaluation (Cosine Similarity)
# =====================================================


def intrinsic_similarity(model, note_type, sample_size=200):
    """
    Compute mean cosine similarity across random subset of vocab.
    Saves matrix optionally for inspection.

    Args:
        model: trained Word2Vec model.
        note_type: note category name.
        sample_size: number of tokens to sample.

    Returns:
        mean_cosine_similarity (float)
    """
    vocab = list(model.wv.key_to_index.keys())
    if len(vocab) < sample_size:
        sample_size = len(vocab)
    sample_words = vocab[:sample_size]
    vectors = np.array([model.wv[w] for w in sample_words])
    sims = cosine_similarity(vectors)

    # Save full matrix to results for inspection if desired
    save_dir = resolve_path(f"results/embeddings/{note_type}")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "cosine_similarity_matrix.npy"), sims)

    mean_sim = np.mean(sims[np.triu_indices_from(sims, 1)])
    return mean_sim


# =====================================================
# 4. Grid Search Loop
# =====================================================


def run_w2v_grid_search(sentences, note_type="Radiology", limit=None):
    """
    Iterate over parameter combinations and record intrinsic metrics.

    Args:
        sentences: tokenized corpus
        note_type: note group (Radiology / Discharge / Combined)
        limit: optional cap on number of configurations

    Returns:
        DataFrame of results
    """
    grid = define_w2v_param_grid()
    param_combos = list(itertools.product(*grid.values()))

    # --- NEW LOGGING ---
    import os, time

    total_configs = len(param_combos)
    cpu_count = os.cpu_count()
    print(
        f"🔧 Preparing to train {total_configs} Word2Vec configurations for {note_type} "
        f"on {cpu_count} available cores..."
    )

    if limit:
        print(
            f"⚠️ Limiting grid search to first {limit} configurations for demonstration.\n"
        )
        param_combos = param_combos[:limit]

    start_all = time.time()
    records = []

    for i, combo in enumerate(param_combos, start=1):
        params = dict(zip(grid.keys(), combo))
        print(f"[{note_type}] Training config {i}/{len(param_combos)}: {params}")
        model, runtime, _ = train_w2v_variant(
            sentences, params, f"{note_type}_cfg{i}", note_type
        )
        mean_sim = intrinsic_similarity(model, note_type)
        records.append({**params, "runtime_s": runtime, "mean_cosine_sim": mean_sim})

    elapsed = round(time.time() - start_all, 2)
    print(
        f"✅ Completed {len(param_combos)} configs for {note_type} in {elapsed/60:.2f} minutes.\n"
    )

    df_results = pd.DataFrame(records)
    save_dir = resolve_path(f"results/embeddings/{note_type}")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "w2v_search_results.csv")
    df_results.to_csv(csv_path, index=False)

    return df_results


# =====================================================
# 5. Select Best Configuration and Retrain
# =====================================================


def select_and_save_best(sentences, results_df, note_type):
    """
    Retrain the best configuration and save to tuned/{note_type}/ folder.

    Args:
        sentences: tokenized corpus
        results_df: DataFrame of previous intrinsic results
        note_type: Radiology / Discharge / Combined

    Returns:
        model, best_params
    """
    from src.utils import resolve_path
    import os, json, time

    # keep only legitimate Word2Vec parameters
    valid_keys = ["vector_size", "window", "min_count", "sg", "negative", "epochs"]
    best_params = {
        k: v
        for k, v in results_df.sort_values("mean_cosine_sim", ascending=False)
        .iloc[0]
        .to_dict()
        .items()
        if k in valid_keys
    }

    # Cast all numeric parameters to int
    for k in best_params:
        best_params[k] = int(best_params[k])

    # ✅ Define and create tuned directory before saving
    tuned_dir = resolve_path(f"embedding_cache/w2v/tuned/{note_type}")
    os.makedirs(tuned_dir, exist_ok=True)

    # --- NEW: status print before retraining ---
    print(f"🔄 Retraining {note_type} Word2Vec model with best parameters...")
    print(f"   Parameters: {best_params}")

    start_time = time.time()

    # Retrain with best parameters
    model, runtime, _ = train_w2v_variant(
        sentences, best_params, f"best_{note_type}", note_type
    )

    elapsed = round((time.time() - start_time) / 60, 2)
    print(f"✅ Finished retraining {note_type} model in {elapsed} minutes.")

    # Save model to tuned directory
    best_model_path = os.path.join(tuned_dir, f"best_{note_type}_w2v.model")
    model.save(best_model_path)

    # Save metadata (best parameters)
    meta_dir = resolve_path(f"results/embeddings/{note_type}")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"✅ [{note_type}] Best model saved to {tuned_dir}")
    print(f"   Parameters: {best_params}")

    return model, best_params


# =====================================================
# 6. Visualization Utility
# =====================================================


def plot_intrinsic_results(note_type: str):
    """
    Plot intrinsic Word2Vec evaluation results for a given note type.
    - Two discrete colors for vector_size (e.g., 100 vs 200)
    - Two brightness/shade levels for window (e.g., 5 vs 10)
    - sg = 0 (CBOW) and sg = 1 (Skip-gram) separated visually
    Saves figure to reports/figures/w2v_intrinsic_{note_type}.png
    """
    # ------------------------------------------------------------
    # Load results
    # ------------------------------------------------------------
    results_path = resolve_path(f"results/embeddings/{note_type}/w2v_search_results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at {results_path}")

    df_results = pd.read_csv(results_path)
    required_cols = {"sg", "mean_cosine_sim", "vector_size", "window"}
    if not required_cols.issubset(df_results.columns):
        raise ValueError(f"Expected columns {required_cols} in {results_path}")

    # ------------------------------------------------------------
    # Sort and group
    # ------------------------------------------------------------
    df_results = df_results.sort_values(["sg", "vector_size", "window"]).reset_index(drop=True)
    df_cbow = df_results[df_results["sg"] == 0].copy()
    df_skip = df_results[df_results["sg"] == 1].copy()

    # ------------------------------------------------------------
    # Custom X-axis labels: show (min_count, negative)
    # ------------------------------------------------------------
    if {"min_count", "negative"}.issubset(df_results.columns):
        df_results["config_label"] = [
            f"({int(mc)}, {int(neg)})" for mc, neg in zip(df_results["min_count"], df_results["negative"])
        ]
        x_labels = df_results["config_label"].tolist()
    else:
        x_labels = list(range(1, len(df_results) + 1))


    # ------------------------------------------------------------
    # Discrete color/shade assignment
    # ------------------------------------------------------------
    # Two colors for vector_size (purple, orange)
    vec_palette = {
        100: np.array([106/255, 81/255, 163/255]),   # purple
        200: np.array([230/255, 85/255, 13/255])     # orange
    }
    # Shades for window (5 = darker, 10 = lighter)
    window_shades = {5: 0.6, 10: 1.0}

    colors = []
    for _, row in df_results.iterrows():
        base_rgb = vec_palette.get(row["vector_size"], np.array([0.5, 0.5, 0.5]))
        brightness = window_shades.get(row["window"], 1.0)
        color = np.clip(base_rgb * brightness, 0, 1)
        colors.append(tuple(np.append(color, 1.0)))  # alpha = 1

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(df_results)), df_results["mean_cosine_sim"], color=colors, edgecolor="black")

    ax.set_xlabel("Configuration ID")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(f"Word2Vec Intrinsic Evaluation ({note_type})")

    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_xlabel("(min_count, negative)")


    # Divider + sg labels
    cbow_count = len(df_cbow)
    ax.axvline(x=cbow_count - 0.5, color="gray", linestyle="--", alpha=0.6)
    ax.text(cbow_count / 2, ax.get_ylim()[1] * 0.95, "sg = 0 (CBOW)", ha="center", fontsize=10)
    ax.text(cbow_count + (len(df_skip) / 2), ax.get_ylim()[1] * 0.95,
            "sg = 1 (Skip-gram)", ha="center", fontsize=10)

    # ------------------------------------------------------------
    # Legend for discrete vector_size × window combinations
    # ------------------------------------------------------------
    legend_elems = []
    for vsize, base_rgb in vec_palette.items():
        for win, shade in window_shades.items():
            face = tuple(np.append(np.clip(base_rgb * shade, 0, 1), 1))
            legend_elems.append(
                Patch(facecolor=face, edgecolor='black', label=f"Vector {vsize}, Window {win}")
            )

    ax.legend(
        handles=legend_elems,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.68), 
        frameon=False,
        fontsize=9,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4
    )

    fig.tight_layout()

    # ------------------------------------------------------------
    # Save and show
    # ------------------------------------------------------------
    save_path = resolve_path(f"reports/figures/w2v_intrinsic_{note_type}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)

    print(f"✅ Saved intrinsic evaluation plot to {save_path}")

# =====================================================
# 7. Full Orchestrator
# =====================================================


def optimize_word2vec(sentences, note_type, limit=None, visualize=True):
    """
    Full tuning pipeline:
    - Runs parameter grid search
    - Computes intrinsic cosine similarities
    - Selects and retrains best model
    - Saves best params + plots

    Args:
        sentences: list of tokenized sentences
        note_type: Radiology / Discharge / Combined
        limit: optionally limit number of configs for lightweight run
        visualize: if True, produces barplot in reports/figures/
    """
    print(f"🔹 Starting W2V tuning for {note_type}...")
    df_results = run_w2v_grid_search(sentences, note_type, limit=limit)
    if visualize:
        plot_intrinsic_results(df_results, note_type)
    model, best_params = select_and_save_best(sentences, df_results, note_type)
    print(f"✅ [{note_type}] Best model saved to embedding_cache/w2v/tuned/{note_type}/")
    print(
        f"   Params: vector_size={best_params['vector_size']}, window={best_params['window']}, sg={best_params['sg']}"
    )
    return model, best_params
