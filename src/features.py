"""
src/features.py

Feature engineering functions for Masters-Thesis project.
Includes:
- Word2Vec training/loading/extraction
- Scaling of embeddings
- BERT embedding preparation (no training/evaluation)
- Optional dimensionality reduction
"""

import os
from sys import prefix
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
from src.utils import resolve_path

# ==================================================
# 1. Structured Feature Scaling (Train/Test)
# ==================================================


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    prefix: str = "original",
    id_col: str = "subject_id",
    target_col: str = "hospital_expire_flag",
    save_dir: str = "data/processed",
):
    """
    Fit a StandardScaler on the training set, transform both train and test sets,
    and save them to data/processed/{prefix}/ with descriptive filenames.

    Args:
        X_train, X_test (pd.DataFrame): Train/test feature matrices (unscaled).
        y_train, y_test (pd.Series): Corresponding labels.
        prefix (str): Dataset variant ("original" or "w2v").
        id_col (str): ID column (excluded from scaling).
        target_col (str): Target column name.
        save_dir (str): Base directory for saving.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    os.makedirs(os.path.join(save_dir, prefix), exist_ok=True)

    # Identify numeric features (exclude ID + target)
    feature_cols = [c for c in X_train.columns if c not in [id_col, target_col]]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    # Return scaled DataFrames only — saving handled in Step 9 via save_feature_dataset()
    print(f"✅ Scaled {prefix} features prepared (not saved — handled downstream)")

    return X_train_scaled, X_test_scaled, y_train, y_test


# ==================================================
# 2. Word2Vec Utilities
# ==================================================


def train_word2vec(corpus_path: str, model_out: str, baseline: bool = False, **params):
    """
    Train a Word2Vec model from a corpus of notes.

    Args:
        corpus_path (str): Path to training corpus (txt file, one doc per line).
        model_out (str): Path to save the trained model (.model).
        baseline (bool): If True, appends 'baseline/' to the save path.
        **params: Hyperparameters for gensim Word2Vec.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    # Resolve paths relative to project root
    corpus_path = resolve_path(corpus_path)

    # Optionally append "baseline" directory before saving
    if baseline:
        model_out = resolve_path(
            os.path.join(
                os.path.dirname(model_out), "baseline", os.path.basename(model_out)
            )
        )
    else:
        model_out = resolve_path(model_out)

    # Load Corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        documents = [line.strip().split() for line in f if line.strip()]

    model = Word2Vec(sentences=documents, **params)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    model.save(str(model_out))
    status = "baseline" if baseline else "custom"
    print(f"✅ Word2Vec model ({status}) trained and saved to {model_out}")

    return model


def get_w2v_params(note_type: str = "radiology") -> dict:
    """
    Return default Word2Vec hyperparameters for a given note type.
    Allows central management of parameters.

    Args:
        note_type (str): One of {"radiology", "discharge", "combined"}.
                         Used to allow future customization if needed.

    Returns:
        dict: Dictionary of Word2Vec hyperparameters.
    """
    base_params = {
        "vector_size": 100,
        "window": 5,
        "sg": 0,  # 0 = CBOW, 1 = Skip-gram
        "epochs": 10,
        "min_count": 2,  # ignore rare words
        "workers": 4,  # parallelism
    }

    # Example customization (future expansion)
    if note_type == "discharge":
        base_params["window"] = 10
    elif note_type == "combined":
        base_params["epochs"] = 15

    return base_params


def load_word2vec(model_path: str, baseline: bool | None = None) -> Word2Vec:
    """Load a saved Word2Vec model."""
    # Resolve and auto-detect baseline vs tuned model
    model_path = resolve_path(model_path)

    # Determine baseline and non-baseline paths
    base_dir = os.path.dirname(model_path)
    basename = os.path.basename(model_path)
    baseline_path = os.path.join(base_dir, "baseline", basename)

    # If user explicitly requested baseline
    if baseline is True:
        candidate_path = baseline_path
    # If user explicitly requested tuned/custom
    elif baseline is False:
        candidate_path = model_path
    # If not specified, auto-detect whichever exists
    else:
        candidate_path = (
            baseline_path if os.path.exists(resolve_path(baseline_path)) else model_path
        )

    # Load model
    candidate_path = resolve_path(candidate_path)
    model = Word2Vec.load(str(candidate_path))

    print(f"✅ Loaded Word2Vec model from {candidate_path}")
    return model


def get_word_embeddings(model: Word2Vec) -> pd.DataFrame:
    """
    Extract word embeddings into a DataFrame.

    Args:
        model (gensim.models.Word2Vec): Trained Word2Vec model.

    Returns:
        pd.DataFrame: Word embeddings with words as index.
    """
    words = list(model.wv.index_to_key)
    vectors = model.wv[words]
    return pd.DataFrame(vectors, index=words)


def scale_w2v_embeddings(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    prefix: str = "w2v",
    id_col: str = "subject_id",
    save_dir: str = "embedding_cache/w2v/baseline",
):
    """
    Fit a StandardScaler on training W2V embeddings and apply to both train/test.

    Args:
        X_train, X_test (pd.DataFrame): Datasets containing Word2Vec columns.
        prefix (str): Dataset variant ("w2v").
        id_col (str): Identifier column.
        save_dir (str): Directory to save scaled embeddings.

    Returns:
        tuple: (X_train_scaled, X_test_scaled)
    """
    os.makedirs(os.path.join(save_dir, prefix), exist_ok=True)

    w2v_cols = [
        col
        for col in X_train.columns
        if col.startswith("w2v_")
        or col.startswith("rad_w2v_")
        or col.startswith("w2v_dis_")
        or col.startswith("w2v_comb_")
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[w2v_cols] = scaler.fit_transform(X_train[w2v_cols])
    X_test_scaled[w2v_cols] = scaler.transform(X_test[w2v_cols])

    # Save scaled embeddings only (for archival and reuse)
    embed_dir = resolve_path(os.path.join(save_dir, prefix))
    os.makedirs(embed_dir, exist_ok=True)

    X_train_scaled.to_csv(
        os.path.join(embed_dir, f"scaled_{prefix}_embeddings_train.csv"), index=False
    )
    X_test_scaled.to_csv(
        os.path.join(embed_dir, f"scaled_{prefix}_embeddings_test.csv"), index=False
    )

    print(
        f"✅ Scaled {prefix} embeddings saved to {embed_dir} (embeddings only, not merged)"
    )

    return X_train_scaled, X_test_scaled


# Word2Vec Subject-Level Embeddings


def get_subject_embedding(doc: str, model: Word2Vec) -> np.ndarray:
    """
    Compute averaged Word2Vec embedding for a single document.

    Args:
        doc (str): Raw text (radiology or discharge notes).
        model (Word2Vec): Trained Word2Vec model.

    Returns:
        np.ndarray: Averaged embedding vector for the document.
    """
    words = doc.lower().split()
    valid_words = [w for w in words if w in model.wv]

    if len(valid_words) == 0:
        return np.zeros(model.vector_size)

    vectors = [model.wv[w] for w in valid_words]
    return np.mean(vectors, axis=0)


def apply_embeddings_to_subjects(
    df: pd.DataFrame, text_col: str, model: Word2Vec, prefix: str = "w2v_rad_"
) -> pd.DataFrame:
    """
    Generate averaged embeddings for each subject's notes and return a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing subject_id and note text column.
        text_col (str): Column with note text (e.g., "Radiology_notes").
        model (Word2Vec): Trained Word2Vec model.
        prefix (str): Prefix for embedding columns.

    Returns:
        pd.DataFrame: DataFrame with subject_id + embedding columns.
    """
    embeddings = np.vstack(
        [get_subject_embedding(str(doc), model) for doc in df[text_col]]
    )

    embed_df = pd.DataFrame(
        embeddings, columns=[f"{prefix}{i+1}" for i in range(model.vector_size)]
    )
    embed_df["subject_id"] = df["subject_id"].values

    return embed_df


def merge_embeddings_with_features(
    X_train_features: pd.DataFrame,
    X_test_features: pd.DataFrame,
    X_train_embed: pd.DataFrame,
    X_test_embed: pd.DataFrame,
    id_col: str = "subject_id",
    prefix: str = "w2v",
    save_dir: str = "data/processed",
):
    """
    Merge structured features and embeddings for both train and test sets.

    Args:
        X_train_features, X_test_features: Scaled structured feature DataFrames.
        X_train_embed, X_test_embed: Scaled embedding DataFrames.
        id_col (str): Column used to merge.
        prefix (str): Dataset variant ("w2v").
        save_dir (str): Directory to save merged outputs.

    Returns:
        tuple: (X_train_merged, X_test_merged)
    """
    out_dir = resolve_path(os.path.join(save_dir, prefix))
    os.makedirs(out_dir, exist_ok=True)

    X_train_merged = X_train_features.merge(X_train_embed, on=id_col, how="left")
    X_test_merged = X_test_features.merge(X_test_embed, on=id_col, how="left")

    X_train_merged.to_csv(
        os.path.join(out_dir, f"data_{prefix}_xtrain.csv"), index=False
    )
    X_test_merged.to_csv(
        os.path.join(out_dir, f"data_{prefix}_xtest.csv"), index=False
    )

    print(f"✅ Merged {prefix} train/test sets saved under {out_dir}")
    return X_train_merged, X_test_merged


# ==================================================
# 3. BERT Embedding Preparation (no modeling)
# ==================================================


def get_bert_embeddings(
    texts, model_name="emilyalsentzer/Bio_ClinicalBERT", batch_size=16, device=None
):
    """
    Generate BERT embeddings for a list of texts (prep only).

    Args:
        texts (list[str]): List of clinical note strings.
        model_name (str): Pretrained HuggingFace model to load.
        batch_size (int): Batch size for encoding.
        device (str): "cuda" or "cpu". Defaults to GPU if available.

    Returns:
        torch.Tensor: Tensor of embeddings (one per text).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


def bert_to_dataframe(
    embeddings: torch.Tensor, ids: list, prefix: str = "bert_"
) -> pd.DataFrame:
    """
    Convert BERT embeddings into a DataFrame with subject IDs.

    Args:
        embeddings (torch.Tensor): Tensor of shape (n_samples, hidden_dim).
        ids (list): List of subject IDs aligned with embeddings.
        prefix (str): Prefix for embedding columns.

    Returns:
        pd.DataFrame: DataFrame with subject_id + embedding columns.
    """
    df = pd.DataFrame(embeddings.numpy(), index=ids)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "subject_id"}, inplace=True)
    df.columns = ["subject_id"] + [f"{prefix}{i+1}" for i in range(df.shape[1] - 1)]

    return df


# ==================================================
# 4. Optional Dimensionality Reduction
# ==================================================

from sklearn.decomposition import PCA


def reduce_dimensions(
    df: pd.DataFrame,
    embedding_prefix: str,
    n_components: int = 50,
    id_col: str = "subject_id",
) -> pd.DataFrame:
    """
    Reduce dimensionality of embedding columns using PCA.

    Args:
        df (pd.DataFrame): DataFrame with embedding columns.
        embedding_prefix (str): Prefix of embedding columns (e.g., "w2v_" or "bert_").
        n_components (int): Number of dimensions to retain.

    Returns:
        pd.DataFrame: Reduced-dimension embeddings merged with IDs.
    """
    embed_cols = [c for c in df.columns if c.startswith(embedding_prefix)]
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df[embed_cols])

    reduced_df = pd.DataFrame(
        reduced, columns=[f"{embedding_prefix}pc{i+1}" for i in range(n_components)]
    )
    reduced_df[id_col] = df[id_col].values
    return reduced_df


# ==================================================
# 5. Save/Load Feature-Engineered Datasets
# ==================================================


def save_feature_dataset(
    df: pd.DataFrame, filename: str, base_dir: str = "data/processed"
) -> str:
    """
    Save a feature-engineered dataset (train/test or merged).

    Automatically resolves the full path relative to the repo root
    and creates parent directories if needed.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): File name only (e.g., 'data_w2v_radiology_xtrain.csv').
        base_dir (str): Base directory (default: 'data/processed').

    Returns:
        str: Absolute path of the saved dataset.
    """
    out_dir = resolve_path(base_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)

    print(f"✅ Saved feature dataset → {out_path}")
    return out_path


def load_feature_dataset(
    filename: str, base_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Load a previously saved feature-engineered dataset.

    Args:
        filename (str): Name of the dataset file (e.g., 'data_w2v_radiology.csv').
        base_dir (str): Base directory where dataset is stored (default: 'data/processed').

    Returns:
        pd.DataFrame: Loaded dataset with features + embeddings.
    """

    path = resolve_path(os.path.join(base_dir, filename))
    df = pd.read_csv(path, index=False)
    print(f"✅ Loaded feature dataset from {path} (shape: {df.shape})")
    return df


def save_w2v_embeddings(
    embed_df: pd.DataFrame,
    filename: str,
    base_dir: str = "data/processed/embeddings/w2v",
) -> str:
    """
    Save subject-level Word2Vec embeddings to CSV.

    Args:
        embed_df (pd.DataFrame): DataFrame with subject_id + embeddings.
        filename (str): Name of the output file.
        base_dir (str): Directory to save (default: 'data/processed').

    Returns:
        str: Full path to saved embeddings.
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, filename)
    embed_df.to_csv(out_path, index=False)
    print(f"✅ Word2Vec embeddings saved to {out_path}")
    return out_path


def save_bert_embeddings(
    embed_df: pd.DataFrame,
    filename: str,
    base_dir: str = "data/processed/embeddings/bert",
) -> str:
    """
    Save BERT embeddings DataFrame to CSV.

    Args:
        embed_df (pd.DataFrame): DataFrame with subject_id + BERT embeddings.
        filename (str): Name of the output file.
        base_dir (str): Directory to save (default: 'data/processed').

    Returns:
        str: Full path to saved embeddings.
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, filename)
    embed_df.to_csv(out_path, index=False)
    print(f"✅ BERT embeddings saved to {out_path}")
    return out_path


def save_reduced_embeddings(
    embed_df: pd.DataFrame,
    filename: str,
    base_dir: str = "data/processed/embeddings/pca",
) -> str:
    """
    Save PCA-reduced embeddings DataFrame to CSV.

    Args:
        embed_df (pd.DataFrame): DataFrame with subject_id + reduced embeddings.
        filename (str): Name of the output file.
        base_dir (str): Directory to save (default: 'data/processed').

    Returns:
        str: Full path to saved embeddings.
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, filename)
    embed_df.to_csv(out_path, index=False)
    print(f"✅ Reduced embeddings saved to {out_path}")
    return out_path


# ==================================================
# 6. Dataset Validation Utility
# ==================================================

import pandas as pd
import os
from src.utils import resolve_path


def validate_saved_datasets(
    prefixes=("original", "w2v_radiology", "w2v_discharge", "w2v_combined"),
    check_alignment=False,
):
    """
    Validate that all expected processed datasets exist and match expected shapes.

    Args:
        prefixes (tuple[str]): Dataset variants to validate.
        check_alignment (bool): If True, also checks that subject_id alignment between
                                X_ and y_ files is consistent.

    Returns:
        pd.DataFrame: Summary table of existence, shapes, and (optionally) alignment results.
    """
    summary_rows = []

    for prefix in prefixes:
        base_dir = resolve_path(f"data/processed/{prefix}")

        expected_files = {
            "X_train": f"data_{prefix}_xtrain.csv",
            "X_test": f"data_{prefix}_xtest.csv",
            "y_train": f"data_{prefix}_ytrain.csv",
            "y_test": f"data_{prefix}_ytest.csv",
        }

        for split, fname in expected_files.items():
            fpath = os.path.join(base_dir, fname)

            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                row = {
                    "Variant": prefix,
                    "Split": split,
                    "File": fname,
                    "Exists": True,
                    "Rows": df.shape[0],
                    "Columns": df.shape[1],
                }

                # Optional subject_id alignment check
                if check_alignment and split.startswith("X_"):
                    yfile = (
                        expected_files["y_train"]
                        if "train" in split
                        else expected_files["y_test"]
                    )
                    ypath = os.path.join(base_dir, yfile)
                    if os.path.exists(ypath):
                        y_df = pd.read_csv(ypath)

                        # Only check alignment if both DataFrames contain subject_id
                        if "subject_id" in df.columns and "subject_id" in y_df.columns:
                            common_ids = set(df["subject_id"]).intersection(
                                set(y_df["subject_id"])
                            )
                            aligned = len(common_ids) == len(df)
                            row["Aligned"] = aligned
                        else:
                            row["Aligned"] = "n/a"
                    else:
                        row["Aligned"] = None

                summary_rows.append(row)
            else:
                summary_rows.append(
                    {
                        "Variant": prefix,
                        "Split": split,
                        "File": fname,
                        "Exists": False,
                        "Rows": None,
                        "Columns": None,
                    }
                )

    return pd.DataFrame(summary_rows)


###
###
### Consider moving this to new home if it doesn't belong here
###
###

import itertools
import random
from datetime import datetime


def random_search_word2vec(
    corpus_path: str,
    param_grid: dict,
    n_iter: int = 10,
    out_dir: str = "results/word2vec_random",
):
    """
    Perform random search over Word2Vec hyperparameters.
    Trains models, saves them, and returns the paths.

    Args:
        corpus_path (str): Path to training corpus (txt file, one doc per line).
        param_grid (dict): Dict of params with lists of candidate values.
            Example:
            {
                "sg": [0, 1],
                "vector_size": [100, 200, 300],
                "window": [5, 10, 15],
                "min_count": [1, 3, 5],
                "negative": [5, 10],
                "epochs": [10, 20]
            }
        n_iter (int): Number of random configurations to sample.
        out_dir (str): Directory to save trained models.

    Returns:
        list of dicts: Each dict contains {"params": config, "model_path": path}.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Expand search space
    keys, values = zip(*param_grid.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Randomly sample n_iter configs
    configs = random.sample(all_configs, min(n_iter, len(all_configs)))

    results = []
    for i, config in enumerate(configs, 1):
        print(f"Training {i}/{len(configs)} with params: {config}")

        # Train model
        model = train_word2vec(
            corpus_path=corpus_path,
            model_out=None,  # save manually
            vector_size=config["vector_size"],
            window=config["window"],
            min_count=config["min_count"],
            sg=config["sg"],
            epochs=config["epochs"],
        )

        # Save with descriptive filename
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (
            f"w2v_sg{config['sg']}_dim{config['vector_size']}_"
            f"win{config['window']}_min{config['min_count']}_"
            f"neg{config['negative']}_iter{config['epochs']}_{stamp}.model"
        )
        model_path = os.path.join(out_dir, model_name)

        model.save(model_path)

        results.append({"params": config, "model_path": model_path})

    return results

# ==================================================
# 7. LLM Standard Scaling
# ==================================================

def scale_llm_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model: str,
    prefix: str,
    id_col: str = "subject_id",
    feature_prefix: str | None = None,
    save_train_name: str = "scaled_llm_train.csv",
    save_test_name: str = "scaled_llm_test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a StandardScaler on training LLM features and apply to both train/test.

    Saving convention:
        base = resolve_path(DEFAULT_CACHE_DIR) / cfg.model / SCHEMA_VERSION

    Args:
        X_train, X_test: DataFrames that include `id_col` and LLM feature columns.
        model: LLM model name (e.g., "gpt-4o-mini").
        prefix: feature prefix used in columns (e.g., "llmB_" or "llmA_").
        id_col: identifier column (default "subject_id").
        feature_prefix: optional override for column selection; if None, uses `prefix`.
        save_train_name/save_test_name: output filenames.

    Returns:
        (X_train_scaled, X_test_scaled)
    """
    from src.llm_structured_features import SCHEMA_VERSION, DEFAULT_CACHE_DIR

    # If feature_prefix is provided, use that for matching columns; else use prefix
    col_prefix = feature_prefix or prefix

    if id_col not in X_train.columns or id_col not in X_test.columns:
        raise ValueError(f"Missing id_col='{id_col}' in train/test.")

    # Select LLM feature columns by prefix
    llm_cols = [c for c in X_train.columns if c.startswith(col_prefix) and c != id_col]
    if not llm_cols:
        raise ValueError(
            f"No LLM feature columns found starting with '{col_prefix}'. "
            f"Example columns: {X_train.columns.tolist()[:20]}"
        )

    # Defensive: ensure test has same columns
    missing_in_test = [c for c in llm_cols if c not in X_test.columns]
    if missing_in_test:
        raise ValueError(f"Test set missing LLM columns: {missing_in_test[:10]} ...")

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[llm_cols] = scaler.fit_transform(X_train[llm_cols])
    X_test_scaled[llm_cols] = scaler.transform(X_test[llm_cols])

    # Save scaled outputs to the requested base path
    base = resolve_path(DEFAULT_CACHE_DIR) / f"{model}" / f"{SCHEMA_VERSION}"
    base.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_csv(base / save_train_name, index=False)
    X_test_scaled.to_csv(base / save_test_name, index=False)

    print(f"✅ Scaled LLM features saved to: {base}")
    print(f"   - {save_train_name}")
    print(f"   - {save_test_name}")

    return X_train_scaled, X_test_scaled