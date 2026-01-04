"""
src/llm_embeddings.py

LLM embedding utilities for Masters-Thesis extension.

Purpose
-------
Generate patient-level dense embeddings from aggregated clinical note text
(using an API-based embedding model) as a drop-in replacement for Word2Vec-derived
embedding vectors.

This module is designed to integrate with the existing repo structure and data flow:
- Input: nlp_ready_df with one aggregated concatenated note string per subject_id
- Output: a tidy embedding table keyed by subject_id, plus a persistent cache for reuse

Key design choices
-------------------------------------
- Embedding model: text-embedding-3-small
- Embedding dimensionality: 256
- One embedding vector per patient (one aggregated note string per subject_id)
- Caching keyed by: (subject_id, note_hash, model, dimensions)

Notes
-----
- This module does NOT merge embeddings into X_train/X_test.
- This module does NOT perform scaling. Scaling should be handled in your existing
  scaling pipeline (train-only fit; apply to test).
- API calls are isolated to a single function _call_embedding_provider() for minimal churn.
"""

from __future__ import annotations

# ==================================================
# 0. Imports
# ==================================================

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils import resolve_path

from src.llm_cache_utils import build_input_signature_hash

from src.llm_long_context_summary import (
    LongContextSummaryConfig,
    prepare_long_context_summary_inputs,
    run_long_context_summary_pipeline,
)


# ==================================================
# 1. Configuration and Constants
# ==================================================


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for embedding generation.

    Args:
        model (str): Embedding model name (provider-specific).
        dimensions (int): Target embedding dimensionality.
        batch_size (int): Number of texts per request batch.
        max_retries (int): Retry attempts for transient API failures.
        retry_backoff_sec (float): Base delay for exponential backoff.
        cache_dir (str): Relative cache directory under project root.
    """

    model: str = "text-embedding-3-small"
    dimensions: int = 256
    batch_size: int = 64
    max_retries: int = 6
    retry_backoff_sec: float = 1.0
    cache_dir: str = "embedding_cache/llm_embeddings"
    use_batch_api: bool = True
    batch_completion_window: str = "24h"


CACHE_KEY_COLS = ["subject_id", "note_hash", "model", "dimensions"]
DEFAULT_EMB_ARTIFACT_PATH = "data/processed/llm_A/embeddings_all.parquet"
DEFAULT_LC_SUMMARY_ARTIFACT_PATH = "data/processed/llm_A/long_context_summaries.parquet"
DEFAULT_MANIFEST_NAME = "embeddings_manifest.json"




def get_default_embeddings_artifact_path() -> Path:
    return resolve_path(DEFAULT_EMB_ARTIFACT_PATH)


# ==================================================
# 2. Input preparation (nlp_ready_df -> embedding inputs)
# ==================================================


def prepare_embedding_inputs(
    nlp_ready_df: pd.DataFrame,
    text_col: str,
    id_col: str = "subject_id",
) -> pd.DataFrame:
    """
    Prepare the embedding input table from nlp_ready_df.

    Args:
        nlp_ready_df (pd.DataFrame): Must contain subject_id and aggregated note text column.
        text_col (str): Column name holding aggregated concatenated note string per patient.
        id_col (str): Column name for patient identifier (default: subject_id).

    Returns:
        pd.DataFrame: columns: [subject_id, note_text, note_hash]
    """
    if id_col not in nlp_ready_df.columns:
        raise ValueError(f"Missing id_col='{id_col}' in nlp_ready_df.")
    if text_col not in nlp_ready_df.columns:
        raise ValueError(f"Missing text_col='{text_col}' in nlp_ready_df.")

    df = nlp_ready_df[[id_col, text_col]].copy()
    df = df.rename(columns={id_col: "subject_id", text_col: "note_text"})
    df["note_text"] = df["note_text"].fillna("").astype(str)

    # Deterministic text hash: invalidates cache when text changes
    df["note_hash"] = df["note_text"].map(_sha256_text)

    # Keep stable ordering
    df = df.sort_values("subject_id").reset_index(drop=True)
    return df


def _sha256_text(s: str) -> str:
    s = "" if s is None else str(s)
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()



# ==================================================
# 3. Cache I/O
# ==================================================


def get_cache_paths(cfg: EmbeddingConfig) -> Tuple[Path, Path]:
    """
    Compute cache data and metadata paths for the given configuration.

    Returns:
        (cache_table_path, cache_meta_path)
    """
    base = resolve_path(cfg.cache_dir)
    cache_table = base / f"cache_{cfg.model}_dim{cfg.dimensions}.parquet"
    cache_meta = base / f"cache_{cfg.model}_dim{cfg.dimensions}_meta.json"
    return cache_table, cache_meta


def load_embedding_cache(cfg: EmbeddingConfig) -> pd.DataFrame:
    """
    Load cache table if present.

    Returns:
        pd.DataFrame: empty if not found.
    """
    cache_table, _ = get_cache_paths(cfg)
    if cache_table.exists():
        return pd.read_parquet(cache_table)
    return pd.DataFrame()


def save_embedding_cache(cfg: EmbeddingConfig, df_cache: pd.DataFrame) -> None:
    """
    Save cache table and metadata.

    Expected cache schema:
        subject_id, note_hash, model, dimensions, emb_0001..emb_D

    Args:
        cfg (EmbeddingConfig): config
        df_cache (pd.DataFrame): cache table
    """
    cache_table, cache_meta = get_cache_paths(cfg)
    cache_table.parent.mkdir(parents=True, exist_ok=True)
    
    df_cache = df_cache.drop_duplicates(subset=CACHE_KEY_COLS, keep="last").reset_index(drop=True)
    df_cache.to_parquet(cache_table, index=False)

    meta = {
        "model": cfg.model,
        "dimensions": cfg.dimensions,
        "n_rows": int(df_cache.shape[0]),
        "columns": list(df_cache.columns),
        "created_by": "src/llm_embeddings.py",
        "schema_version": "llm_embeddings_cache_v1"
    }
    cache_meta.write_text(json.dumps(meta, indent=2))

def validate_cache_schema(df_cache: pd.DataFrame, cfg: EmbeddingConfig) -> None:
    if df_cache.empty:
        return

    # Required key cols
    for c in CACHE_KEY_COLS:
        if c not in df_cache.columns:
            raise ValueError(f"Cache missing required column: {c}")

    # Required embedding columns for this cfg.dimensions
    emb_cols = [f"emb_{j:04d}" for j in range(1, cfg.dimensions + 1)]
    missing_emb = [c for c in emb_cols if c not in df_cache.columns]
    if missing_emb:
        raise ValueError(
            f"Cache missing embedding columns for dim={cfg.dimensions}. "
            f"Missing count={len(missing_emb)} (example: {missing_emb[:3]})"
        )
    
def load_embeddings_artifact(path: str = DEFAULT_EMB_ARTIFACT_PATH) -> pd.DataFrame:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings artifact not found: {p}")
    return pd.read_parquet(p)


def load_embeddings_manifest(artifact_path: str = DEFAULT_EMB_ARTIFACT_PATH) -> dict:
    a = resolve_path(artifact_path)
    m = a.parent / DEFAULT_MANIFEST_NAME
    if not m.exists():
        raise FileNotFoundError(f"Embeddings manifest not found: {m}")
    return json.loads(m.read_text())

def embeddings_artifact_is_valid(
    cfg: EmbeddingConfig,
    df_inputs: pd.DataFrame,
    artifact_path: str = DEFAULT_EMB_ARTIFACT_PATH,
) -> bool:
    a = resolve_path(artifact_path)
    m = a.parent / DEFAULT_MANIFEST_NAME


    # Must exist
    if (not a.exists()) or (not m.exists()):
        return False

    # Load manifest
    try:
        manifest = json.loads(m.read_text())
    except Exception:
        return False
    
    required_fields = {"model", "dimensions", "source_path", "source_mtime", "n_rows", "created_at_utc", "input_signature_hash"}
    if not required_fields.issubset(manifest.keys()):
        return False


    # Check manifest signature fields
    if manifest.get("model") != cfg.model:
        return False
    if int(manifest.get("dimensions", -1)) != int(cfg.dimensions):
        return False

    # Check input signature matches
    sig = build_input_signature_hash(
        df_inputs,
        key_cols=["subject_id", "note_hash"],
    )

    if manifest.get("input_signature_hash") != sig:
        return False

    # Check row count + embedding columns
    try:
        df = pd.read_parquet(a)
    except Exception:
        return False

    if df.shape[0] != df_inputs.shape[0]:
        return False
    
    if int(manifest.get("n_rows", -1)) != int(df_inputs.shape[0]):
        return False


    emb_cols = [f"emb_{j:04d}" for j in range(1, cfg.dimensions + 1)]
    if any(c not in df.columns for c in emb_cols):
        return False

    if "subject_id" not in df.columns or "note_hash" not in df.columns:
        return False
    
    src_p = resolve_path(manifest.get("source_path", ""))
    src_mtime_now = src_p.stat().st_mtime if src_p.exists() else None
    if manifest.get("source_mtime") != src_mtime_now:
        return False

    return True





# ==================================================
# 4. Embedding generation (cache-aware)
# ==================================================


def build_llm_embeddings(
    df_inputs: pd.DataFrame,
    cfg: EmbeddingConfig = EmbeddingConfig(),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build embeddings for a set of (subject_id, note_text) rows with caching.

    Args:
        df_inputs (pd.DataFrame): must contain columns:
            - subject_id
            - note_text
            - note_hash
        cfg (EmbeddingConfig): embedding generation config
        verbose (bool): print progress and cache stats

    Returns:
        pd.DataFrame: columns:
            subject_id, note_hash, model, dimensions, emb_0001..emb_D
        (aligned to df_inputs rows by subject_id/note_hash)
    """
    required = {"subject_id", "note_text", "note_hash"}
    missing = required - set(df_inputs.columns)
    if missing:
        raise ValueError(f"df_inputs missing required columns: {sorted(missing)}")

    df_cache = load_embedding_cache(cfg)
    validate_cache_schema(df_cache, cfg)

    if not df_cache.empty:   
        df_cache = df_cache.loc[
            (df_cache["model"] == cfg.model) & (df_cache["dimensions"] == cfg.dimensions)
        ].copy()


    if verbose:
        n_cached = 0 if df_cache.empty else df_cache.shape[0]
        print(f"🔎 LLM Embeddings cache loaded: {n_cached} rows (model={cfg.model}, dim={cfg.dimensions})")

    # Identify missing embeddings by (subject_id, note_hash)
    join_keys = ["subject_id", "note_hash"]
    if df_cache.empty:
        df_missing = df_inputs.copy()
        df_missing["_missing"] = True
    else:
        df_merge = df_inputs[join_keys].merge(
            df_cache[join_keys],
            on=join_keys,
            how="left",
            indicator=True,
        )
        missing_mask = df_merge["_merge"].ne("both")
        df_missing = df_inputs.loc[missing_mask].copy()
        df_missing["_missing"] = True

    if verbose:
        print(f"🧩 Embeddings needed: {int(df_missing.shape[0])} / {int(df_inputs.shape[0])}")

    # Fetch new embeddings (batched) for missing rows
    if not df_missing.empty:
        if cfg.use_batch_api:
            df_new = _fetch_embeddings_batch_api(df_missing=df_missing, cfg=cfg, verbose=verbose)
        else:
            df_new = _fetch_embeddings_batched(
                subject_ids=df_missing["subject_id"].tolist(),
                note_hashes=df_missing["note_hash"].tolist(),
                texts=df_missing["note_text"].tolist(),
                cfg=cfg,
                verbose=verbose,
            )

        validate_cache_schema(df_new, cfg)

        # Update cache
        if df_cache.empty:
            df_cache = df_new
        else:
            df_cache = pd.concat([df_cache, df_new], ignore_index=True)

        # Deduplicate defensively
        df_cache = df_cache.drop_duplicates(
            subset=CACHE_KEY_COLS,
            keep="last",
        ).reset_index(drop=True)

        save_embedding_cache(cfg, df_cache)

        if verbose:
            print(f"✅ Cache updated: now {int(df_cache.shape[0])} rows")

    # Return embeddings aligned to df_inputs
    df_out = df_inputs[["subject_id", "note_hash"]].merge(
        df_cache,
        on=["subject_id", "note_hash"],
        how="left",
    )

    # Sanity check
    emb_cols = [c for c in df_out.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("No embedding columns found in output. Cache may be malformed.")

    # Rows missing all emb values indicates a cache miss / API failure
    miss_rows = df_out[emb_cols].isna().all(axis=1)
    if miss_rows.any():
        n_miss = int(miss_rows.sum())
        raise RuntimeError(
            f"Missing embeddings for {n_miss} rows after cache+fetch. "
            f"Check API failures or cache schema."
        )

    return df_out


def _fetch_embeddings_batched(
    subject_ids: List[int],
    note_hashes: List[str],
    texts: List[str],
    cfg: EmbeddingConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch embeddings for the provided texts in batches, with retries/backoff.

    Returns:
        pd.DataFrame: subject_id, note_hash, model, dimensions, emb_0001..emb_D
    """
    if not (len(subject_ids) == len(note_hashes) == len(texts)):
        raise ValueError("Input lists must be the same length.")

    all_batches = []
    n = len(texts)
    emb_cols = [f"emb_{j:04d}" for j in range(1, cfg.dimensions + 1)]

    for start in range(0, n, cfg.batch_size):
        end = min(start + cfg.batch_size, n)
        batch_texts = texts[start:end]
        batch_subjects = subject_ids[start:end]
        batch_hashes = note_hashes[start:end]

        if verbose:
            print(f"➡️  Embedding batch {start}:{end} (size={len(batch_texts)})")

        emb = _retry_call_embedding_provider(batch_texts, cfg=cfg)

        if not isinstance(emb, np.ndarray):
            emb = np.asarray(emb)

        if emb.shape != (len(batch_texts), cfg.dimensions):
            raise ValueError(
                f"Embedding shape mismatch: got {emb.shape}, expected {(len(batch_texts), cfg.dimensions)}"
            )

        df_batch = pd.DataFrame(emb, columns=emb_cols)
        df_batch.insert(0, "dimensions", cfg.dimensions)
        df_batch.insert(0, "model", cfg.model)
        df_batch.insert(0, "note_hash", batch_hashes)
        df_batch.insert(0, "subject_id", batch_subjects)

        all_batches.append(df_batch)

    return pd.concat(all_batches, ignore_index=True)

def _fetch_embeddings_batch_api(
    df_missing: pd.DataFrame,
    cfg: EmbeddingConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch embeddings for missing rows using OpenAI Batch API.

    Expects df_missing columns: subject_id, note_hash, note_text
    Returns: subject_id, note_hash, model, dimensions, emb_0001..emb_D
    """
    # Paths
    batch_dir = resolve_path("embedding_cache/llm_embeddings/batch_runs")
    batch_dir.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    input_jsonl = batch_dir / f"embeddings_input_{cfg.model}_{cfg.dimensions}_{run_tag}.jsonl"
    output_jsonl = batch_dir / f"embeddings_output_{cfg.model}_{cfg.dimensions}_{run_tag}.jsonl"

    # 1) Write JSONL
    write_embeddings_batch_jsonl(df_missing=df_missing, cfg=cfg, out_jsonl=input_jsonl)

    # 2) Upload
    file_id = upload_batch_input_file(input_jsonl, verbose=verbose)

    # 3) Create batch job
    batch_id = create_embeddings_batch_job(
        input_file_id=file_id,
        completion_window=getattr(cfg, "batch_completion_window", "24h"),
        verbose=verbose,
    )

    # 4) Poll
    batch_obj = poll_batch_job(batch_id, poll_interval_sec=30, verbose=verbose)
    status = batch_obj.get("status")

    if status != "completed":
        raise RuntimeError(f"Batch job did not complete successfully. status={status} batch_id={batch_id}")

    output_file_id = batch_obj.get("output_file_id")
    if not output_file_id:
        raise RuntimeError(f"Batch completed but output_file_id missing. batch_id={batch_id}")

    # 5) Download output
    download_batch_output_file(output_file_id, out_path=output_jsonl, verbose=verbose)

    # 6) Parse output to df_new
    df_new = parse_embeddings_batch_output(output_jsonl_path=output_jsonl, cfg=cfg)

    return df_new



def _retry_call_embedding_provider(texts: List[str], cfg: EmbeddingConfig) -> np.ndarray:
    """
    Call the embedding provider with retry and exponential backoff.

    Args:
        texts (List[str]): batch texts
        cfg (EmbeddingConfig): config

    Returns:
        np.ndarray: (len(texts), cfg.dimensions)
    """
    attempt = 0
    while True:
        try:
            return _call_embedding_provider(texts, cfg)
        except Exception as e:
            attempt += 1
            if attempt > cfg.max_retries:
                raise RuntimeError(
                    f"Embedding provider failed after {cfg.max_retries} retries. Last error: {e}"
                ) from e

            sleep_s = cfg.retry_backoff_sec * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 30.0)  # cap
            time.sleep(sleep_s)


# ==================================================
# 5. Provider API call (wire-in point)
# ==================================================


def _call_embedding_provider(texts: List[str], cfg: EmbeddingConfig) -> np.ndarray:
    """
    Official OpenAI Python client implementation.

    Requires env var:
        OPENAI_API_KEY

    Returns:
        np.ndarray shape (len(texts), cfg.dimensions)
    """
    # Local import keeps the module importable even if openai isn't installed
    from openai import OpenAI

    # Validate inputs: OpenAI embeddings endpoint does not accept empty strings
    safe_texts = []
    for t in texts:
        t = "" if t is None else str(t)
        t = t.strip()
        safe_texts.append(t if t else " ")  # single space to avoid empty-string error

    client = OpenAI()  # reads OPENAI_API_KEY from environment

    resp = client.embeddings.create(
        model=cfg.model,
        input=safe_texts,
        dimensions=cfg.dimensions,  # supported for text-embedding-3-* models
    )

    # resp.data is a list of objects with .embedding
    vectors = [row.embedding for row in resp.data]  # list[list[float]]

    return np.asarray(vectors, dtype=np.float32)



# ==================================================
# 6. Convenience I/O helpers
# ==================================================

def canonicalize_embeddings_table(df: pd.DataFrame, cfg: EmbeddingConfig) -> pd.DataFrame:
    emb_cols = [f"emb_{j:04d}" for j in range(1, cfg.dimensions + 1)]
    cols = ["subject_id", "note_hash", "model", "dimensions"] + emb_cols
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Embeddings table missing required columns: {missing[:5]}")
    return df[cols].copy()


def save_embeddings_table(
    df_embeddings: pd.DataFrame,
    out_path: str,
    cfg: EmbeddingConfig,
    verbose: bool = True,
) -> Path:
    """
    Save an embeddings table to a project-relative path using resolve_path().

    Args:
        df_embeddings (pd.DataFrame): output from build_llm_embeddings()
        out_path (str): project-relative path
        cfg (EmbeddingConfig): embedding configuration (for canonicalization)
        verbose (bool): print path info

    Returns:
        Path: resolved path on disk
    """
    p = resolve_path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df_out = canonicalize_embeddings_table(df_embeddings, cfg)

    if p.suffix.lower() == ".parquet":
        df_out.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        df_out.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported extension for embeddings output: {p.suffix}")

    if verbose:
        print(f"💾 Embeddings saved (canonicalized): {p}")
    return p


def write_embeddings_manifest(
    artifact_path: Path,
    cfg: EmbeddingConfig,
    source_path: str,
    n_rows: int,
    input_signature_hash: str,
) -> Path:
    manifest_path = artifact_path.parent / DEFAULT_MANIFEST_NAME

    src_p = resolve_path(source_path)
    src_mtime = src_p.stat().st_mtime if src_p.exists() else None

    manifest = {
        "model": cfg.model,
        "dimensions": cfg.dimensions,
        "source_path": source_path,
        "source_mtime": src_mtime,
        "n_rows": int(n_rows),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_signature_hash": input_signature_hash,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path

# =================================================
# 7. Orchestrator (guard → build → normalize → save)
# =================================================

def run_llm_embeddings_pipeline(
    df_inputs: pd.DataFrame,
    cfg: EmbeddingConfig,
    artifact_path: str = DEFAULT_EMB_ARTIFACT_PATH,
    source_path: str = "data/interim/data_nlp_ready.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Orchestrate embedding generation with guard + artifact + manifest.

    Returns canonicalized embeddings table loaded from disk or freshly generated.
    """
    if embeddings_artifact_is_valid(cfg=cfg, df_inputs=df_inputs, artifact_path=artifact_path):
        if verbose:
            print("✅ embeddings artifact valid — loading (no API calls).")
        return load_embeddings_artifact(artifact_path)

    if verbose:
        print("⚙️ embeddings artifact missing/invalid — generating embeddings.")

    df_emb = build_llm_embeddings(df_inputs, cfg=cfg, verbose=verbose)

    saved_path = save_embeddings_table(df_emb, artifact_path, cfg=cfg, verbose=verbose)

    sig = build_input_signature_hash(df_inputs, key_cols=["subject_id", "note_hash"])

    write_embeddings_manifest(
        artifact_path=saved_path,
        cfg=cfg,
        source_path=source_path,
        n_rows=int(df_emb.shape[0]),
        input_signature_hash=sig,
    )

    if verbose:
        print("🧾 Manifest written next to embeddings artifact.")

    return load_embeddings_artifact(artifact_path)

def run_long_context_embeddings_pipeline(
    nlp_ready_df: pd.DataFrame,
    raw_text_col: str,
    id_col: str = "subject_id",
    summary_cfg: LongContextSummaryConfig = LongContextSummaryConfig(),
    embedding_cfg: EmbeddingConfig = EmbeddingConfig(),
    summary_artifact_path: str = "data/processed/llm_A/long_context_summaries.parquet",
    embeddings_artifact_path: str = DEFAULT_EMB_ARTIFACT_PATH,
    source_path: str = "data/interim/data_nlp_ready.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Path A end-to-end:
      raw unordered radiology text -> long-context bilateral summary -> embeddings

    Returns:
      embeddings artifact DataFrame (canonicalized)
    """
    # 1) Build summary inputs (subject_id + raw_text + raw_hash)
    df_sum_inputs = prepare_long_context_summary_inputs(
        nlp_ready_df=nlp_ready_df,
        text_col=raw_text_col,
        id_col=id_col,
    )

    # 2) Run long-context summary pipeline (cached + guarded)
    df_sum = run_long_context_summary_pipeline(
        df_inputs=df_sum_inputs,
        cfg=summary_cfg,
        artifact_path=summary_artifact_path,
        source_path=source_path,
        verbose=verbose,
    )

    # 3) Prepare embedding inputs from summary_text
    df_for_embeddings = df_sum[["subject_id", "summary_text"]].rename(columns={"summary_text": raw_text_col})
    df_emb_inputs = prepare_embedding_inputs(
        nlp_ready_df=df_for_embeddings,
        text_col=raw_text_col,
        id_col="subject_id",
    )

    # 4) Run embeddings pipeline (cached + guarded)
    df_emb = run_llm_embeddings_pipeline(
        df_inputs=df_emb_inputs,
        cfg=embedding_cfg,
        artifact_path=embeddings_artifact_path,
        source_path=source_path,
        verbose=verbose,
    )

    return df_emb



# ==================================================
# 8. Batch API JSONL calls (optional)
# ==================================================

def write_embeddings_batch_jsonl(
    df_missing: pd.DataFrame,
    cfg: EmbeddingConfig,
    out_jsonl: Path,
) -> Path:
    """
    Create a Batch API JSONL input file for /v1/embeddings.

    Each line is a request input object:
      {"custom_id": "...", "method": "POST", "url": "/v1/embeddings", "body": {...}}

    Notes:
    - custom_id must be unique per request within the batch.
    - body parameters match the embeddings endpoint parameters.
    """

    required = {"subject_id", "note_hash", "note_text"}
    missing = required - set(df_missing.columns)
    if missing:
        raise ValueError(f"df_missing missing required columns for batch jsonl: {sorted(missing)}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for sid, h, text in zip(df_missing["subject_id"], df_missing["note_hash"], df_missing["note_text"]):
            custom_id = f"{int(sid)}::{h}"  # stable + unique (subject_id + note_hash)
            obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": cfg.model,
                    "input": (" " if (text is None or str(text).strip() == "") else str(text)),
                    "dimensions": cfg.dimensions,
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return out_jsonl

def upload_batch_input_file(jsonl_path: Path, verbose: bool = True) -> str:
    from openai import OpenAI
    """
    Upload a JSONL file for Batch API processing.

    Returns:
        str: uploaded file_id (e.g., "file_...")

    Note:
        Batch input files must be uploaded with purpose="batch".
    """

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Batch JSONL not found: {jsonl_path}")

    client = OpenAI()
    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    file_id = uploaded.id
    if verbose:
        print(f"⬆️ Batch input uploaded: {jsonl_path.name} → {file_id}")

    return file_id

def create_embeddings_batch_job(
    input_file_id: str,
    completion_window: str = "24h",
    verbose: bool = True,
) -> str:
    from openai import OpenAI
    """
    Create a Batch job for /v1/embeddings using a previously uploaded input file.

    Returns:
        str: batch_id (e.g., "batch_...")
    """

    client = OpenAI()

    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/embeddings",
        completion_window=completion_window,
    )

    batch_id = batch.id
    if verbose:
        print(f"🚀 Batch job created: {batch_id} (window={completion_window})")

    return batch_id

def poll_batch_job(
    batch_id: str,
    poll_interval_sec: int = 30,
    verbose: bool = True,
) -> dict:
    from openai import OpenAI
    """
    Poll a Batch job until completion, failure, or expiration.

    Returns:
        dict: final batch object (as dict)
    """

    client = OpenAI()

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status

        if verbose:
            print(f"⏳ Batch {batch_id} status: {status}")

        if status in {"completed", "failed", "expired", "canceled"}:
            return batch.model_dump() if hasattr(batch, "model_dump") else batch

        time.sleep(poll_interval_sec)

def download_batch_output_file(file_id: str, out_path: Path, verbose: bool = True) -> Path:
    from openai import OpenAI
    """
    Download a Batch output file to disk.
    """

    client = OpenAI()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = client.files.content(file_id).read()
    out_path.write_bytes(content)

    if verbose:
        print(f"⬇️ Batch output downloaded: {file_id} → {out_path}")

    return out_path


def parse_embeddings_batch_output(
    output_jsonl_path: Path,
    cfg: EmbeddingConfig,
) -> pd.DataFrame:
    """
    Parse Batch output JSONL into the standard cache DataFrame schema.

    Expected custom_id format:
        "{subject_id}::{note_hash}"
    """

    if not output_jsonl_path.exists():
        raise FileNotFoundError(f"Batch output JSONL not found: {output_jsonl_path}")

    rows = []
    with output_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            custom_id = obj.get("custom_id")
            if not custom_id or "::" not in custom_id:
                continue

            sid_str, note_hash = custom_id.split("::", 1)
            subject_id = int(sid_str)

            # Batch output includes the API response under "response"
            resp = obj.get("response", {})
            body = resp.get("body", {})

            data = body.get("data", [])
            if not data:
                raise RuntimeError(f"Missing embedding data for custom_id={custom_id}")

            emb = data[0].get("embedding")
            if emb is None:
                raise RuntimeError(f"Missing embedding vector for custom_id={custom_id}")

            rec = {
                "subject_id": subject_id,
                "note_hash": note_hash,
                "model": cfg.model,
                "dimensions": cfg.dimensions,
            }
            for j, v in enumerate(emb, start=1):
                rec[f"emb_{j:04d}"] = float(v)

            rows.append(rec)

    df_new = pd.DataFrame(rows)
    return df_new