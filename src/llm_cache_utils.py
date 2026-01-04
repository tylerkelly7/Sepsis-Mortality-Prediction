"""
src/llm_cache_utils.py

Shared cache + artifact + manifest utilities for LLM-based extensions.

Purpose
-------
Provide generic, reusable helpers for:
- text hashing
- input signature hashing (to detect input changes deterministically)
- cache load/save (parquet)
- manifest write/read
- artifact validity checks (no-regenerate guard)

Notes
-----
- This module is intentionally generic (no embeddings-specific column naming).
- Domain-specific modules (llm_embeddings.py, llm_structured_features.py) should:
  * define their cache keys + artifact paths
  * define expected columns
  * call these helpers to enforce consistent behavior
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.utils import resolve_path


# ==================================================
# 1. Hashing
# ==================================================


def hash_text_sha256(text: Optional[str]) -> str:
    t = "" if text is None else str(text)
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def build_input_signature_hash(
    df_inputs: pd.DataFrame,
    key_cols: Sequence[str],
    include_text_col: Optional[str] = None,
) -> str:
    """
    Build a deterministic signature hash of inputs.

    Typical usage:
      - key_cols: ["subject_id", "note_hash"]
      - include_text_col: None (recommended once note_hash exists)

    If include_text_col is provided, it is included verbatim in the signature
    (can be large; usually avoid and rely on note_hash instead).

    Returns:
        str: sha256 hex digest
    """
    missing = [c for c in key_cols if c not in df_inputs.columns]
    if missing:
        raise ValueError(f"df_inputs missing required key cols for signature: {missing}")

    cols = list(key_cols)
    if include_text_col:
        if include_text_col not in df_inputs.columns:
            raise ValueError(f"df_inputs missing include_text_col={include_text_col}")
        cols.append(include_text_col)

    s = df_inputs[cols].copy()
    s = s.sort_values(by=list(key_cols)).reset_index(drop=True)
    payload = s.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ==================================================
# 2. Cache I/O
# ==================================================


def load_cache_table(cache_path: str | Path) -> pd.DataFrame:
    p = resolve_path(str(cache_path)) if not isinstance(cache_path, Path) else cache_path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def save_cache_table(df_cache: pd.DataFrame, cache_path: str | Path, key_cols: Sequence[str]) -> Path:
    p = resolve_path(str(cache_path)) if not isinstance(cache_path, Path) else cache_path
    p.parent.mkdir(parents=True, exist_ok=True)

    if df_cache.empty:
        # Write empty cache as a file for transparency if desired; keep consistent behavior
        df_cache.to_parquet(p, index=False)
        return p

    df_cache = df_cache.drop_duplicates(subset=list(key_cols), keep="last").reset_index(drop=True)
    df_cache.to_parquet(p, index=False)
    return p


# ==================================================
# 3. Manifest
# ==================================================


def write_manifest(
    artifact_path: str | Path,
    manifest_name: str,
    manifest: Dict[str, Any],
) -> Path:
    a = resolve_path(str(artifact_path)) if not isinstance(artifact_path, Path) else artifact_path
    m = a.parent / manifest_name
    m.write_text(json.dumps(manifest, indent=2))
    return m


def read_manifest(artifact_path: str | Path, manifest_name: str) -> Optional[Dict[str, Any]]:
    a = resolve_path(str(artifact_path)) if not isinstance(artifact_path, Path) else artifact_path
    m = a.parent / manifest_name
    if not m.exists():
        return None
    try:
        return json.loads(m.read_text())
    except Exception:
        return None


def make_standard_manifest(
    *,
    model: str,
    signature: str,
    source_path: str,
    n_rows: int,
    manifest_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    src_p = resolve_path(source_path)
    src_mtime = src_p.stat().st_mtime if src_p.exists() else None

    manifest = {
        "model": model,
        "source_path": str(src_p),
        "source_mtime": src_mtime,
        "n_rows": int(n_rows),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_signature_hash": signature,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    return manifest


# ==================================================
# 4. Artifact validity (no-regenerate guard)
# ==================================================


def artifact_is_valid(
    *,
    artifact_path: str | Path,
    manifest_name: str,
    df_inputs: pd.DataFrame,
    signature_key_cols: Sequence[str],
    expected_manifest_fields: Sequence[str],
    expected_manifest_kv: Dict[str, Any],
    expected_required_cols: Sequence[str],
) -> bool:
    """
    Generic guard:
      - artifact + manifest exist
      - manifest has required fields
      - manifest matches expected key/value fields (e.g., model, schema_version, dimensions)
      - input signature hash matches current df_inputs signature
      - source mtime matches current source file mtime
      - artifact has expected row count and required columns
    """
    a = resolve_path(str(artifact_path)) if not isinstance(artifact_path, Path) else artifact_path
    m = a.parent / manifest_name

    if (not a.exists()) or (not m.exists()):
        return False

    manifest = read_manifest(a, manifest_name)
    if not manifest:
        return False

    # Required manifest fields
    if not set(expected_manifest_fields).issubset(set(manifest.keys())):
        return False

    # Key/value matches
    for k, v in expected_manifest_kv.items():
        if manifest.get(k) != v:
            return False

    # Signature match
    sig_now = build_input_signature_hash(df_inputs, key_cols=signature_key_cols)
    if manifest.get("input_signature_hash") != sig_now:
        return False

    # Source mtime match
    src_p = resolve_path(manifest.get("source_path", ""))
    src_mtime_now = src_p.stat().st_mtime if src_p.exists() else None
    if manifest.get("source_mtime") != src_mtime_now:
        return False

    # Artifact read + shape/cols check
    try:
        df = pd.read_parquet(a)
    except Exception:
        return False

    if int(df.shape[0]) != int(df_inputs.shape[0]):
        return False

    for c in expected_required_cols:
        if c not in df.columns:
            return False

    return True

