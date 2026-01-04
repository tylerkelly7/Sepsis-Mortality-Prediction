"""
src/llm_structured_features.py

LLM structured feature extraction utilities for Masters-Thesis extension.

Purpose
-------
Extract patient-level, interpretable clinical NLP features from aggregated
radiology note text using an API-based LLM with strict Structured Outputs.

This module is designed to integrate with the existing repo structure and data flow:
- Input: nlp_ready_df with one aggregated concatenated note string per subject_id
- Output: a tidy feature table keyed by subject_id, produced from a fixed JSON schema

Key design choices
------------------
- LLM model: gpt-4o-mini (schema version includes model name)
- Output format: Structured Outputs (JSON Schema), strict=True
- One JSON record per patient (one aggregated note string per subject_id)
- Conservative defaults on uncertainty:
    * categoricals -> "unknown"
    * binary flags -> 0
    * severity -> "unknown"

Notes
-----
- This module defines the locked schema and extraction call only.
- Caching and persistence are handled in the next subtask (12b.3.3+).
- API calls are isolated to _call_structured_extractor() for minimal churn.
"""
from __future__ import annotations
# ==================================================
# 0. Imports
# ==================================================


import json
import logging
import time
from datetime import datetime, timezone
import re
import hashlib

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Set, Union

import pandas as pd

from src.utils import resolve_path

from src.llm_cache_utils import hash_text_sha256, build_input_signature_hash, load_cache_table, save_cache_table

# NOTE: OpenAI client import is intentionally local within functions
# to keep import-time failures (missing package, missing key) from
# affecting non-LLM workflows.

# ==================================================
# 1. Sepsis Mortality Feature Schema, Module Constants
# ==================================================

# Schema version includes model name to avoid ambiguity across LLMs
SCHEMA_VERSION = "sepsis_mortality_gpt_4o_mini_v1"

INFECTION_SOURCE_ENUM = [
    "respiratory",
    "urinary",
    "intra_abdominal",
    "skin_soft_tissue",
    "line_catheter",
    "cns",
    "other",
    "unknown",
]

PATHOGEN_ENUM = ["bacterial", "viral", "fungal", "mixed", "unknown"]

IMAGING_SEVERITY_ENUM = ["mild", "moderate", "severe", "unknown"]

# Strict JSON Schema definition used by Structured Outputs
# (no additional keys allowed; all keys required)
SEPSIS_MORTALITY_SCHEMA: Dict[str, Any] = {
    "name": SCHEMA_VERSION,
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Infection context
            "suspected_infection_source": {"type": "string", "enum": INFECTION_SOURCE_ENUM},
            "suspected_pathogen_type": {"type": "string", "enum": PATHOGEN_ENUM},
            # Respiratory (SOFA respiratory)
            "resp_failure_present": {"type": "integer", "enum": [0, 1]},
            "mechanical_ventilation_present": {"type": "integer", "enum": [0, 1]},
            "hypoxemia_present": {"type": "integer", "enum": [0, 1]},
            # Cardiovascular (SOFA CV)
            "hypotension_present": {"type": "integer", "enum": [0, 1]},
            "vasopressor_use_present": {"type": "integer", "enum": [0, 1]},
            # Renal (SOFA renal)
            "aki_present": {"type": "integer", "enum": [0, 1]},
            "oliguria_anuria_present": {"type": "integer", "enum": [0, 1]},
            "dialysis_present": {"type": "integer", "enum": [0, 1]},
            # Coagulation (SOFA coag)
            "thrombocytopenia_present": {"type": "integer", "enum": [0, 1]},
            # Liver (SOFA liver)
            "hyperbilirubinemia_present": {"type": "integer", "enum": [0, 1]},
            # CNS (SOFA CNS)
            "altered_mentation_present": {"type": "integer", "enum": [0, 1]},
            # Mortality modifiers
            "high_risk_course_language_present": {"type": "integer", "enum": [0, 1]},
            "limitation_of_care_present": {"type": "integer", "enum": [0, 1]},
            # Imaging-level severity
            "imaging_severity_impression": {"type": "string", "enum": IMAGING_SEVERITY_ENUM},
        },
        "required": [
            "suspected_infection_source",
            "suspected_pathogen_type",
            "resp_failure_present",
            "mechanical_ventilation_present",
            "hypoxemia_present",
            "hypotension_present",
            "vasopressor_use_present",
            "aki_present",
            "oliguria_anuria_present",
            "dialysis_present",
            "thrombocytopenia_present",
            "hyperbilirubinemia_present",
            "altered_mentation_present",
            "high_risk_course_language_present",
            "limitation_of_care_present",
            "imaging_severity_impression",
        ],
    },
}

# ==================================================
# Model-level schema metadata (for normalization)
# ==================================================

SEPSIS_MORTALITY_MODEL_SCHEMA: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,

    "binary": [
        "resp_failure_present",
        "mechanical_ventilation_present",
        "hypoxemia_present",
        "hypotension_present",
        "vasopressor_use_present",
        "aki_present",
        "oliguria_anuria_present",
        "dialysis_present",
        "thrombocytopenia_present",
        "hyperbilirubinemia_present",
        "altered_mentation_present",
        "high_risk_course_language_present",
        "limitation_of_care_present",
    ],

    "categorical": {
        "suspected_infection_source": INFECTION_SOURCE_ENUM,
        "suspected_pathogen_type": PATHOGEN_ENUM,
    },

    "ordinal": {
        "imaging_severity_impression": {
            "mild": 0,
            "moderate": 1,
            "severe": 2,
            "unknown": 3,
        }
    },
}

CACHE_KEY_COLS = ["subject_id", "note_hash", "model", "schema_version"]

DEFAULT_FEATURES_ARTIFACT_PATH = "data/processed/llm_B/features_all.parquet"
DEFAULT_MANIFEST_NAME = "features_manifest.json"
DEFAULT_CACHE_DIR = "embedding_cache/llm_structured_features"
DEFAULT_PROGRESS_MANIFEST_NAME = "features_manifest_progress.json"

DEFAULT_MAX_INPUT_TOKENS = 125_000

# ==================================================
# 2. Configuration
# ==================================================


@dataclass(frozen=True)
class ExtractionConfig:
    """
    Configuration for structured feature extraction.

    Args:
        model (str): LLM name for structured extraction (chat completion model).
        max_retries (int): Retry attempts for transient failures.
        retry_backoff_sec (float): Base delay for exponential backoff.
        timeout_sec (Optional[float]): Optional request timeout (if supported).
    """

    model: str = "gpt-4o-mini"
    max_retries: int = 6
    retry_backoff_sec: float = 1.0
    timeout_sec: Optional[float] = None


# ==================================================
# 3. Prompts
# ==================================================

SYSTEM_PROMPT = (
    "Extract only the requested sepsis mortality feature fields from the provided radiology note text. "
    "Return values that match the schema exactly. "
    "If uncertain, use conservative defaults: categoricals='unknown', binary flags=0, and severity='unknown'. "
    "Do not include any extra keys."
)

# ==================================================
# 4. Core extraction (Structured Outputs)
# ==================================================


def _call_structured_extractor(note_text: Optional[str], cfg: ExtractionConfig) -> Dict[str, Any]:
    """
    Call the LLM with Structured Outputs (JSON Schema) and return parsed JSON dict.

    Args:
        note_text (Optional[str]): aggregated radiology note text (may be None/NA).
        cfg (ExtractionConfig): extraction config.

    Returns:
        dict: parsed JSON with keys exactly matching SEPSIS_MORTALITY_SCHEMA.
    """
    from openai import OpenAI

    client = OpenAI()

    user_text = "" if note_text is None else str(note_text)

    # Chat Completions with response_format=json_schema
    # Schema is strict; output should conform without extra keys.
    kwargs = {}
    if cfg.timeout_sec is not None:
        # Some clients support request timeout via client-level config or per-call;
        # keep this as a placeholder knob without hard dependency.
        kwargs["timeout"] = cfg.timeout_sec

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": SEPSIS_MORTALITY_SCHEMA,
        },
        **kwargs,
    )

    content = resp.choices[0].message.content
    return json.loads(content)


def extract_features_with_retries(note_text: Optional[str], cfg: ExtractionConfig) -> Dict[str, Any]:
    """
    Retry wrapper for structured extraction with exponential backoff.

    Args:
        note_text (Optional[str]): aggregated radiology note text (may be None/NA).
        cfg (ExtractionConfig): extraction config.

    Returns:
        dict: extracted features matching the schema.
    """
    last_err: Optional[Exception] = None

    for attempt in range(cfg.max_retries):
        try:
            return _call_structured_extractor(note_text=note_text, cfg=cfg)
        except Exception as e:
            last_err = e
            sleep_s = cfg.retry_backoff_sec * (2**attempt)
            time.sleep(sleep_s)

    raise RuntimeError(f"Structured extraction failed after {cfg.max_retries} retries: {last_err}")

def _clip_text_to_max_tokens(text: str, max_tokens: int = 120000, model: str = "gpt-4o-mini") -> str:
    """
    Deterministically clip text to stay under max_tokens.
    Uses head+tail strategy to preserve both early and recent content.
    """
    text = "" if text is None else str(text)

    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback: rough estimate ~4 chars/token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]

    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text

    half = max_tokens // 2
    clipped = enc.decode(toks[:half]) + "\n...\n" + enc.decode(toks[-half:])
    return clipped


# ==================================================
# 5. Simple smoke test helper (optional)
# ==================================================


def smoke_test_structured_extraction() -> None:
    """
    Minimal smoke test to validate:
    - OpenAI API key is present
    - LLM returns schema-conforming JSON
    """
    cfg = ExtractionConfig(model="gpt-4o-mini")

    test_text = "Patient is critically ill with multiorgan failure. DNR/DNI. Worsening respiratory status."
    out = extract_features_with_retries(test_text, cfg)
    assert isinstance(out, dict) and len(out) > 0, "Smoke test: extractor returned empty/non-dict output"

    # Quick sanity checks
    required = set(SEPSIS_MORTALITY_SCHEMA["schema"]["required"])
    missing = required - set(out.keys())
    if missing:
        raise RuntimeError(f"Smoke test failed: missing keys: {sorted(missing)}")

    extra = set(out.keys()) - required
    if extra:
        raise RuntimeError(f"Smoke test failed: extra keys present: {sorted(extra)}")

    print("✅ Structured extraction smoke test passed.")
    print(out)


# ==================================================
# 6. Cache I/O
# ==================================================

def get_cache_paths(cfg: ExtractionConfig) -> tuple[Path, Path]:
    base = resolve_path(DEFAULT_CACHE_DIR) / f"{cfg.model}" / f"{SCHEMA_VERSION}"
    cache_table = base / "features_cache.parquet"
    cache_meta = base / "features_cache_meta.json"
    return cache_table, cache_meta

def prepare_feature_inputs(df: pd.DataFrame, text_col: str, id_col: str = "subject_id") -> pd.DataFrame:
    out = df[[id_col, text_col]].copy()
    out = out.rename(columns={id_col: "subject_id", text_col: "note_text"})
    out["note_hash"] = out["note_text"].apply(hash_text_sha256)
    return out

def load_feature_cache(cfg: ExtractionConfig) -> pd.DataFrame:
    cache_table, _ = get_cache_paths(cfg)
    return load_cache_table(cache_table)


def save_feature_cache(cfg: ExtractionConfig, df_cache: pd.DataFrame) -> None:
    cache_table, cache_meta = get_cache_paths(cfg)

    # Save canonical cache table (dedup handled in shared util)
    save_cache_table(df_cache, cache_table, key_cols=CACHE_KEY_COLS)

    # Write small cache meta next to it (keeps transparency)
    df_cache2 = df_cache.drop_duplicates(subset=CACHE_KEY_COLS, keep="last").reset_index(drop=True)
    meta = {
        "model": cfg.model,
        "schema_version": SCHEMA_VERSION,
        "n_rows": int(df_cache2.shape[0]),
        "columns": list(df_cache2.columns),
        "created_by": "src/llm_structured_features.py",
    }
    cache_meta.parent.mkdir(parents=True, exist_ok=True)
    cache_meta.write_text(json.dumps(meta, indent=2))

# ==================================================
# 7. Main extraction workflow
# ==================================================

def build_llm_structured_features(df_inputs: pd.DataFrame, cfg: ExtractionConfig, verbose: bool = True) -> pd.DataFrame:
    required = {"subject_id", "note_text", "note_hash"}
    missing = required - set(df_inputs.columns)
    if missing:
        raise ValueError(f"df_inputs missing required columns: {sorted(missing)}")

    df_cache = load_feature_cache(cfg)

    if not df_cache.empty:
        missing_keys = [c for c in ["model", "schema_version", "subject_id", "note_hash"] if c not in df_cache.columns]
        if missing_keys:
            raise ValueError(f"Feature cache missing required columns: {missing_keys}")

        df_cache = df_cache.loc[
            (df_cache["model"] == cfg.model) & (df_cache["schema_version"] == SCHEMA_VERSION)
        ].copy()


    # identify missing by (subject_id, note_hash)
    join_keys = ["subject_id", "note_hash"]
    if df_cache.empty:
        df_missing = df_inputs.copy()
    else:
        df_merge = df_inputs[join_keys].merge(df_cache[join_keys], on=join_keys, how="left", indicator=True)
        df_missing = df_inputs.loc[df_merge["_merge"].ne("both")].copy()

    if verbose:
        print(f"🧩 Extractions needed: {int(df_missing.shape[0])} / {int(df_inputs.shape[0])}")

    if not df_missing.empty:
        records = []
        for _, r in df_missing.iterrows():
            feat = extract_features_with_retries(r["note_text"], cfg)
            records.append(
                {
                    "subject_id": int(r["subject_id"]),
                    "note_hash": r["note_hash"],
                    "model": cfg.model,
                    "schema_version": SCHEMA_VERSION,
                    **feat,
                }
            )
        df_new = pd.DataFrame(records)

        df_cache = df_new if df_cache.empty else pd.concat([df_cache, df_new], ignore_index=True)
        df_cache = df_cache.drop_duplicates(subset=CACHE_KEY_COLS, keep="last").reset_index(drop=True)
        save_feature_cache(cfg, df_cache)

    df_out = df_inputs[["subject_id", "note_hash"]].merge(df_cache, on=["subject_id", "note_hash"], how="left")
    return df_out



def save_features_table(df_features: pd.DataFrame, out_path: str, verbose: bool = True) -> Path:
    p = resolve_path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(p, index=False)
    if verbose:
        print(f"💾 Structured features saved: {p}")
    return p

#==================================================
# 8. Normalization Structured LLM Features
#==================================================

def normalize_structured_llm_features(
    df_cache: pd.DataFrame,
    model_schema: Dict[str, Any],
    model: str,
    prefix: str = "llmB_",
) -> pd.DataFrame:
    """
    Normalize cached structured LLM features into a model-ready table.

    Args:
        df_cache (pd.DataFrame): structured feature cache
        model_schema (dict): SEPSIS_MORTALITY_MODEL_SCHEMA
        model (str): expected LLM model
        prefix (str): feature prefix

    Returns:
        pd.DataFrame: one row per subject_id, numeric features only
    """

    schema_version = model_schema["schema_version"]

    # --------------------------------------------------
    # 1. Filter cache
    # --------------------------------------------------
    required_cols = {"subject_id", "model", "schema_version"}
    missing = required_cols - set(df_cache.columns)
    if missing:
        raise ValueError(f"Cache missing required columns: {sorted(missing)}")
    
    df = df_cache.loc[
        (df_cache["model"] == model) &
        (df_cache["schema_version"] == schema_version)
    ].copy()


    # --------------------------------------------------
    # 2. Column groups from schema
    # --------------------------------------------------
    binary_cols = model_schema["binary"]
    categorical_cols = list(model_schema["categorical"].keys())
    ordinal_cols = list(model_schema["ordinal"].keys())

    keep_cols = (
        ["subject_id"]
        + binary_cols
        + categorical_cols
        + ordinal_cols
    )

    df = df[keep_cols]

    # --------------------------------------------------
    # 3. Binary validation
    # --------------------------------------------------
    for c in binary_cols:
        df[c] = df[c].fillna(0).astype(int)
        if not set(df[c].unique()).issubset({0, 1}):
            raise ValueError(f"Invalid binary values in {c}")

    # --------------------------------------------------
    # 4. Encode categorical variables
    # --------------------------------------------------
    for col, levels in model_schema["categorical"].items():
        df[col] = (
            df[col]
            .fillna("unknown")
            .where(df[col].isin(levels), "unknown")
        )

        df = pd.get_dummies(
            df,
            columns=[col],
            prefix=f"{prefix}{col}",
            dtype=int,
        )

        expected = [f"{prefix}{col}_{lvl}" for lvl in levels]
        df = df.reindex(columns=df.columns.tolist() + [c for c in expected if c not in df.columns], fill_value=0)


    # --------------------------------------------------
    # 5. Encode ordinal variables
    # --------------------------------------------------
    for col, mapping in model_schema["ordinal"].items():
        df[col] = (
            df[col]
            .fillna("unknown")
            .map(mapping)
            .fillna(mapping["unknown"])
            .astype(int)
        )

        df.rename(columns={col: f"{prefix}{col}"}, inplace=True)

    # --------------------------------------------------
    # 6. Prefix binary columns
    # --------------------------------------------------
    for c in binary_cols:
        df.rename(columns={c: f"{prefix}{c}"}, inplace=True)

    # --------------------------------------------------
    # 7. Final checks
    # --------------------------------------------------
    if not df["subject_id"].is_unique:
        raise ValueError("Duplicate subject_id after normalization")

    if df.drop(columns=["subject_id"]).isna().any().any():
        raise ValueError("NaNs present after normalization")

    df = df[["subject_id"] + sorted([c for c in df.columns if c != "subject_id"])]

    return df


# ==================================================
# 9. Artifact validation (no-regenerate guard)
# ==================================================

def features_artifact_is_valid(
    df_inputs: pd.DataFrame,
    model_schema: Dict[str, Any],
    model: str,
    prefix: str = "llmB_",
    artifact_path: str = DEFAULT_FEATURES_ARTIFACT_PATH,
) -> bool:
    """
    Validate whether the structured-features artifact can be reused.

    Conditions:
    - artifact + manifest exist
    - model + schema_version match
    - input signature hash matches
    - row count matches df_inputs
    - required columns present
    """
    artifact = resolve_path(artifact_path)
    manifest_path = artifact.parent / DEFAULT_MANIFEST_NAME

    if (not artifact.exists()) or (not manifest_path.exists()):
        return False

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False

    if manifest.get("model") != model:
        return False
    if manifest.get("schema_version") != model_schema["schema_version"]:
        return False

    # Input signature guard
    sig_now = build_input_signature_hash(df_inputs, key_cols=["subject_id", "note_hash"])
    if manifest.get("input_signature_hash") != sig_now:
        return False

    # Load artifact
    try:
        df = pd.read_parquet(artifact)
    except Exception:
        return False

    if df.shape[0] != df_inputs.shape[0]:
        return False

    # Required columns
    required = (
        ["subject_id"]
        + [f"{prefix}{c}" for c in model_schema["binary"]]
        + [
            f"{prefix}{col}_{lvl}"
            for col, lvls in model_schema["categorical"].items()
            for lvl in lvls
        ]
        + [f"{prefix}{c}" for c in model_schema["ordinal"].keys()]
    )

    for c in required:
        if c not in df.columns:
            return False

    return True

# ==================================================
# 10. Save canonical structured-features artifact
# ==================================================

def save_structured_features_artifact(
    df_features: pd.DataFrame,
    df_inputs: pd.DataFrame,
    model_schema: Dict[str, Any],
    model: str,
    out_path: str = DEFAULT_FEATURES_ARTIFACT_PATH,
    source_path: Optional[str] = None,
) -> Path:
    """
    Save structured features artifact and write manifest.

    Args:
        df_features: normalized features (one row per subject_id)
        df_inputs: original inputs used for extraction (for signature)
        model_schema: SEPSIS_MORTALITY_MODEL_SCHEMA
        model: LLM model name
        out_path: artifact path
        source_path: optional path to input CSV for mtime tracking

    Returns:
        Path to saved artifact
    """
    p = resolve_path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Save artifact
    df_features.to_parquet(p, index=False)

    # Manifest
    manifest = {
        "model": model,
        "schema_version": model_schema["schema_version"],
        "n_rows": int(df_features.shape[0]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_signature_hash": build_input_signature_hash(df_inputs, key_cols=["subject_id", "note_hash"]),

    }

    if source_path:
        src = resolve_path(source_path)
        manifest["source_path"] = str(src)
        manifest["source_mtime"] = src.stat().st_mtime if src.exists() else None

    manifest_path = p.parent / DEFAULT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return p

# ==================================================
# 11. Orchestrator (guard → build → cache every 100 -> normalize → save)
# ==================================================

def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_progress_manifest(
    manifest_path: Path,
    payload: Dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(manifest_path)


def run_llm_structured_features_pipeline(
    df_inputs: pd.DataFrame,
    cfg: ExtractionConfig,
    model_schema: Dict[str, Any],
    artifact_path: str = DEFAULT_FEATURES_ARTIFACT_PATH,
    prefix: str = "llmB_",
    source_path: Optional[str] = None,
    verbose: bool = True,
    batch_size: int = 100,
    checkpoint_every: int = 100,
    resume: bool = True,
    retry_errors: bool = False,
    max_patients: Optional[int] = None,
    truncation_policy: Optional[TruncationPolicy] = None,
) -> pd.DataFrame:
    """
    Orchestrate structured feature extraction with:
      - Guard: reuse artifact if valid
      - Batch cache-building with periodic cache saves + progress manifest updates
      - Single normalize + artifact save at end

    df_inputs must include: subject_id, note_text, note_hash
    """

    # --------------------------------------------------
    # 0) Guard: reuse artifact if valid (unless we're explicitly retrying errors)
    # --------------------------------------------------
    if (not retry_errors) and features_artifact_is_valid(
        df_inputs=df_inputs,
        model_schema=model_schema,
        model=cfg.model,
        prefix=prefix,
        artifact_path=artifact_path,
    ):
        if verbose:
            print("✅ Structured features artifact valid — loading from disk")
        return pd.read_parquet(resolve_path(artifact_path))

    if verbose:
        print("♻️  Building structured features (artifact missing/invalid)")

    # --------------------------------------------------
    # 1) Prep inputs
    # --------------------------------------------------
    required_cols = {"subject_id", "note_text", "note_hash"}
    missing_cols = required_cols - set(df_inputs.columns)
    if missing_cols:
        raise ValueError(f"df_inputs missing required columns: {sorted(missing_cols)}")

    df_work = df_inputs.copy()
    if max_patients is not None:
        df_work = df_work.iloc[:max_patients].copy()

    total = len(df_work)

    artifact_p = resolve_path(artifact_path)
    manifest_p = artifact_p.parent / DEFAULT_PROGRESS_MANIFEST_NAME

    # --------------------------------------------------
    # 2) Load existing cache (for resume)
    # --------------------------------------------------
    df_cache = load_feature_cache(cfg)
    if df_cache is None or df_cache.empty:
        df_cache = pd.DataFrame()

    if not df_cache.empty:
        # Keep only current model + schema_version
        df_cache = df_cache.loc[
            (df_cache.get("model") == cfg.model) &
            (df_cache.get("schema_version") == SCHEMA_VERSION)
        ].copy()

    # Build "already done" set using cache keys
    completed_keys = set()
    if not df_cache.empty and all(c in df_cache.columns for c in CACHE_KEY_COLS):
        if "__error" in df_cache.columns:
            ok_mask = df_cache["__error"].isna()
        else:
            ok_mask = pd.Series(True, index=df_cache.index)

        completed_keys = set(map(tuple, df_cache.loc[ok_mask, CACHE_KEY_COLS].values.tolist()))



    # Determine which rows still need extraction (by (subject_id, note_hash, model, schema_version))
    # This matches your cache contract and prevents re-calling the same patient when resuming.
    def _row_key(row: pd.Series) -> tuple:
        return (int(row["subject_id"]), row["note_hash"], cfg.model, SCHEMA_VERSION)

    keys_needed: List[tuple] = []
    recs_needed: List[Dict[str, Any]] = []

    for rec in df_work[["subject_id", "note_text", "note_hash"]].to_dict(orient="records"):
        key = (int(rec["subject_id"]), rec["note_hash"], cfg.model, SCHEMA_VERSION)
        if key not in completed_keys:
            keys_needed.append(key)
            recs_needed.append(rec)

    needed = len(recs_needed)
    already = total - needed

    if verbose:
        print(f"🧩 Extractions needed: {needed} / {total}")

    # Initialize progress manifest (running)
    run_t0 = time.time()
    progress = {
        "status": "running",
        "started_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "model": cfg.model,
        "schema_version": SCHEMA_VERSION,
        "prefix": prefix,
        "source_path": source_path,
        "artifact_path": str(artifact_p),
        "total_rows": total,
        "completed_rows": already,
        "needed_rows": needed,
        "ok": 0,
        "fail": 0,
        "last_subject_id": None,
        "note": "Cache is being built; normalized artifact will be written at end of run.",
    }
    _write_progress_manifest(manifest_p, progress)

    # --------------------------------------------------
    # 3) Batch LLM extraction into cache
    # --------------------------------------------------
    if needed == 0:
        if verbose:
            print("✅ Cache already complete for current inputs; proceeding to normalization and save.")
    else:
        new_cache_rows: List[Dict[str, Any]] = []
        ok_total = 0
        fail_total = 0

        for start in range(0, needed, batch_size):
            end = min(start + batch_size, needed)
            batch = recs_needed[start:end]
            batch_idx = (start // batch_size) + 1

            if verbose:
                print(f"[batch {batch_idx:03d}] subjects {start+1}–{end} ({len(batch)}) starting…")

            batch_t0 = time.time()
            ok = 0
            fail = 0

            for rec in batch:
                sid = int(rec["subject_id"])
                try:
                    note_text = rec["note_text"]

                    llm_text, trunc_meta = prepare_note_text_for_llm(
                        subject_id=sid,
                        note_text=note_text,
                        policy=truncation_policy,
                    )

                    feats = extract_features_with_retries(llm_text, cfg)


                    # cache row: metadata + raw schema keys (NO prefixing here)
                    cache_row = {
                        "subject_id": sid,
                        "note_hash": rec["note_hash"],
                        "model": cfg.model,
                        "schema_version": SCHEMA_VERSION,
                        "llm_token_est_before": trunc_meta["token_est_before"],
                        "llm_token_est_after": trunc_meta["token_est_after"],
                        "llm_was_truncated": trunc_meta["was_truncated"],
                        "llm_trunc_saved_path": trunc_meta["trunc_saved_path"],
                        **feats,
                    }
                    new_cache_rows.append(cache_row)
                    ok += 1
                except Exception as e:
                    # Conservative defaults so normalization never fails
                    defaults = {}

                    # binary defaults
                    for k in model_schema["binary"]:
                        defaults[k] = 0

                    # categorical defaults
                    for k in model_schema["categorical"].keys():
                        defaults[k] = "unknown"

                    # ordinal defaults (store string; normalization maps it)
                    for k in model_schema["ordinal"].keys():
                        defaults[k] = "unknown"

                    cache_row = {
                        "subject_id": sid,
                        "note_hash": rec["note_hash"],
                        "model": cfg.model,
                        "schema_version": SCHEMA_VERSION,
                        "__error": str(e),
                        **defaults,
                    }
                    new_cache_rows.append(cache_row)
                    fail += 1


                progress["last_subject_id"] = sid

            ok_total += ok
            fail_total += fail

            batch_elapsed = time.time() - batch_t0
            completed_now = already + ok_total + fail_total

            if verbose:
                avg = batch_elapsed / max(len(batch), 1)
                run_elapsed_min = (time.time() - run_t0) / 60.0
                print(
                    f"[batch {batch_idx:03d}] done: ok={ok} fail={fail} | "
                    f"completed={completed_now}/{total} | "
                    f"batch_elapsed={batch_elapsed:.1f}s | avg={avg:.2f}s/patient | run_elapsed={run_elapsed_min:.1f}m"
                )

            # checkpoint: save cache + update progress manifest every checkpoint_every rows processed
            processed_this_run = ok_total + fail_total
            should_checkpoint = (processed_this_run % checkpoint_every == 0) or (end == needed)

            if should_checkpoint:
                df_new = pd.DataFrame(new_cache_rows)
                new_cache_rows = []

                if not df_new.empty:
                    if df_cache.empty:
                        df_cache = df_new
                    else:
                        df_cache = pd.concat([df_cache, df_new], ignore_index=True)

                    # Deduplicate by cache keys (keep last)
                    if all(c in df_cache.columns for c in CACHE_KEY_COLS):
                        df_cache = df_cache.drop_duplicates(subset=CACHE_KEY_COLS, keep="last").reset_index(drop=True)

                    save_feature_cache(cfg, df_cache)

                progress.update(
                    {
                        "updated_at": _utc_now_iso(),
                        "completed_rows": completed_now,
                        "needed_rows": max(total - completed_now, 0),
                        "ok": ok_total,
                        "fail": fail_total,
                    }
                )
                _write_progress_manifest(manifest_p, progress)

                if verbose:
                    print(f"💾 checkpoint cache saved (rows in cache={len(df_cache)})")
                    print(f"🧾 progress manifest updated -> {manifest_p} (completed={completed_now}/{total})")

    # --------------------------------------------------
    # 4) Normalize ONCE at end + write artifact ONCE
    # --------------------------------------------------
    if verbose:
        print("🧪 Normalizing structured features from cache (final step)…")

    # Reload cache to be safe (optional), or just use df_cache in memory
    # df_cache = load_feature_cache(cfg)

    df_feat = normalize_structured_llm_features(
        df_cache=df_cache,
        model_schema=model_schema,
        model=cfg.model,
        prefix=prefix,
    )

    save_structured_features_artifact(
        df_features=df_feat,
        df_inputs=df_inputs,
        model_schema=model_schema,
        model=cfg.model,
        out_path=artifact_path,
        source_path=source_path,
    )

    if verbose:
        print(f"💾 Saved structured features artifact: {resolve_path(artifact_path)}")

    # Mark complete in manifest (note: save_structured_features_artifact also writes manifest;
    # this overwrites again with a completion marker that keeps progress stats.)
    progress.update(
        {
            "status": "complete",
            "updated_at": _utc_now_iso(),
            "completed_rows": total,
            "needed_rows": 0,
            "note": "Cache build complete; normalized artifact written.",
        }
    )
    _write_progress_manifest(manifest_p, progress)

    return df_feat

# =================================================
# 12. Edge Case Handling Utilities
# =================================================

# --- Token utilities (minimal truncation; does not mutate nlp_ready_df) ------------

@dataclass(frozen=True)
class TruncationPolicy:
    """Controls optional minimal truncation for extremely long patient texts.

    Design goals:
      - Only trim when necessary (token estimate > max_tokens).
      - Optionally apply to a specific set of subject_ids (for targeted fixes).
      - Never overwrite/replace note text inside nlp_ready_df; truncation is applied
        only to the text sent to the LLM in this pipeline run.
      - Persist truncated inputs to disk for auditability & reproducibility.
    """
    max_tokens: int = 125_000
    apply_to_subject_ids: Optional[Set[Union[int, str]]] = None  # None => apply to all
    save_dir: str = "embedding_cache/llm_structured_features/gpt-4o-mini/truncated_inputs"
    head_fraction: float = 0.60  # keep 60% head, 40% tail


def _estimate_tokens(text: str) -> int:
    """Heuristic token estimate (safe offline).

    Uses ~4 characters/token as a rough approximation; sufficient for deciding
    whether to truncate extremely long inputs.
    """
    if not text:
        return 0
    compact = re.sub(r"\s+", " ", text).strip()
    return int(len(compact) / 4) + 1


def _clip_text_to_max_tokens(text: str, max_tokens: int, head_fraction: float = 0.60) -> str:
    """Approximate clip to max_tokens by keeping head+tail (preserves summary + impression)."""
    if not text:
        return text

    est = _estimate_tokens(text)
    if est <= max_tokens:
        return text

    max_chars = max(1, int(max_tokens * 4))
    head_chars = int(max_chars * head_fraction)
    tail_chars = max_chars - head_chars

    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars > 0 else ""

    marker = "\n\n[...TRUNCATED_FOR_TOKEN_LIMIT...]\n\n"
    return head + marker + tail


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def prepare_note_text_for_llm(
    *,
    subject_id: Union[int, str],
    note_text: str,
    policy: Optional[TruncationPolicy],
) -> Tuple[str, Dict[str, Union[int, str, bool, None]]]:
    """Return (text_for_llm, meta) applying minimal truncation when configured."""
    meta: Dict[str, Union[int, str, bool, None]] = {
        "token_est_before": _estimate_tokens(note_text or ""),
        "token_est_after": None,
        "was_truncated": False,
        "trunc_saved_path": None,
    }

    if policy is None:
        return note_text, meta

    if policy.apply_to_subject_ids is not None and subject_id not in policy.apply_to_subject_ids:
        return note_text, meta

    if meta["token_est_before"] <= policy.max_tokens:
        return note_text, meta

    clipped = _clip_text_to_max_tokens(
        note_text,
        max_tokens=policy.max_tokens,
        head_fraction=policy.head_fraction,
    )
    meta["token_est_after"] = _estimate_tokens(clipped)
    meta["was_truncated"] = True

    try:
        save_dir = resolve_path(policy.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        h = _hash_text(note_text)
        out_path = save_dir / f"subject_{subject_id}_max{policy.max_tokens}_{h}.txt"
        out_path.write_text(clipped, encoding="utf-8")
        meta["trunc_saved_path"] = str(out_path)
    except Exception as e:
        logger.warning("Failed to save truncated input for subject_id=%s: %s", subject_id, e)

    return clipped, meta
