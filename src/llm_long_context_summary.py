"""
src/llm_long_context_summary.py

Path A: long-context compression for unordered radiology text.

Goal
----
Create one stable, non-temporal, clinically-salient summary per subject_id that:
- emphasizes global disease burden and seriousness (not progression)
- is robust to arbitrary ordering of concatenated reports
- is safe to embed with standard embedding models (downstream)

Design
------
- Inputs: one aggregated note string per subject_id (unordered)
- Output: one bilateral summary string per subject_id:
    === GLOBAL CLINICAL SUMMARY (UNORDERED TEXT) ===
    [guardrailed, 1500–3000 tokens target]
    === NON-TEMPORAL FINDINGS SNAPSHOT ===
    [sectioned bullets]
- Cache keyed by: (subject_id, raw_hash, model, prompt_version)
- Artifact: parquet + manifest (no-regenerate guard)

Notes
-----
- This module does not infer chronology, progression, or causality.
- It treats input as an unordered set of observations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from src.utils import resolve_path
from src.llm_cache_utils import (
    hash_text_sha256,
    build_input_signature_hash,
    load_cache_table,
    save_cache_table,
    artifact_is_valid,
    make_standard_manifest,
    write_manifest,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROMPT_VERSION = "pathA_long_context_summary_v1"
DEFAULT_SUMMARY_ARTIFACT_PATH = "data/processed/llm_A/long_context_summaries.parquet"
DEFAULT_MANIFEST_NAME = "long_context_summary_manifest.json"

CACHE_DIR = "embedding_cache/llm_long_context_summary"


@dataclass(frozen=True)
class LongContextSummaryConfig:
    """
    Configuration for Path A long-context summarization.

    model: long-context LLM model (generation)
    max_output_tokens: output cap for bilateral summary
    temperature: keep deterministic
    max_retries: transient error retries
    retry_backoff_sec: exponential backoff base
    """
    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 3500
    temperature: float = 0.0
    max_retries: int = 6
    retry_backoff_sec: float = 1.0


CACHE_KEY_COLS = ["subject_id", "raw_hash", "model", "prompt_version"]


def get_cache_paths(cfg: LongContextSummaryConfig) -> Tuple[Path, Path]:
    base = resolve_path(CACHE_DIR)
    cache_table = base / f"cache_{cfg.model}_{PROMPT_VERSION}.parquet"
    cache_meta = base / f"cache_{cfg.model}_{PROMPT_VERSION}_meta.json"
    return cache_table, cache_meta


# --------------------------------------------------
# Prompt (exact, bilateral, guardrailed)
# --------------------------------------------------

SYSTEM_PROMPT = """You are a clinical text summarization engine.

Safety and bias constraints:
- The input contains multiple radiology reports that are UNORDERED and may appear in arbitrary sequence.
- Do NOT infer temporal order, progression, improvement, interval change, or causality.
- Treat the input as an unordered set of observations.
- Avoid narrative phrasing that implies change over time (e.g., worsening, improving, progression, interval increase) unless the exact wording is explicitly present in the text, and even then do not imply chronology.
- Only include findings explicitly stated or clearly implied by the reports. Do not add unsupported clinical interpretation.
- If findings are inconsistent, unclear, or mixed, say "mixed/unclear" instead of resolving ambiguity.
- It is acceptable to state "unknown" when the text does not support a determination.

Output requirements:
- Produce a single text output with TWO sections exactly in this order and with these headers:
  1) "=== GLOBAL CLINICAL SUMMARY (UNORDERED TEXT) ==="
  2) "=== NON-TEMPORAL FINDINGS SNAPSHOT ==="
- The GLOBAL summary should be a compact clinical synthesis emphasizing overall disease burden/seriousness,
  extent of disease, organ involvement, high-risk language, limitation-of-care language if present, and overall clinical seriousness.
  Do not mention dates or attempt sequencing.
- The SNAPSHOT section must be sectioned bullets with no narrative paragraphs.
"""

USER_TEMPLATE = """Summarize the following UNORDERED radiology text as global clinical representation (not time series).

Return exactly:

=== GLOBAL CLINICAL SUMMARY (UNORDERED TEXT) ===
[Target length: 1500–3000 tokens if the source is long enough; shorter is acceptable if content is limited.
No temporal inference. Uncertainty allowed.]

=== NON-TEMPORAL FINDINGS SNAPSHOT ===
Overall disease burden:
- ...

Organ system involvement:
- Pulmonary: ...
- Renal: ...
- Hepatic: ...
- Cardiovascular: ...
- Neurologic: ...
- Hematologic: ...
- Other: ...

High-risk imaging features present:
- Diffuse infiltrates / multifocal pneumonia pattern: yes/no/unknown
- Large pleural effusions: yes/no/unknown
- Intra-abdominal catastrophe (perforation/ischemia/abscess) language: yes/no/unknown
- Extensive metastatic disease / malignancy burden: yes/no/unknown
- Multiorgan involvement mentioned across reports: yes/no/unknown

Explicit severity / risk language observed (quote short phrases if present):
- ...

Limitations / uncertainty noted:
- ...

Radiology text (unordered):
{note_text}
"""


# --------------------------------------------------
# Input preparation
# --------------------------------------------------

def prepare_long_context_summary_inputs(
    nlp_ready_df: pd.DataFrame,
    text_col: str,
    id_col: str = "subject_id",
) -> pd.DataFrame:
    if id_col not in nlp_ready_df.columns:
        raise ValueError(f"Missing id_col='{id_col}' in nlp_ready_df.")
    if text_col not in nlp_ready_df.columns:
        raise ValueError(f"Missing text_col='{text_col}' in nlp_ready_df.")

    df = nlp_ready_df[[id_col, text_col]].copy()
    df = df.rename(columns={id_col: "subject_id", text_col: "raw_text"})
    df["raw_text"] = df["raw_text"].fillna("").astype(str)

    df["raw_hash"] = df["raw_text"].map(hash_text_sha256)

    df = df.sort_values("subject_id").reset_index(drop=True)
    return df


# --------------------------------------------------
# Cache validation
# --------------------------------------------------

def validate_cache_schema(df_cache: pd.DataFrame) -> None:
    if df_cache.empty:
        return
    required = set(CACHE_KEY_COLS) | {"summary_text", "summary_hash", "created_at_utc"}
    missing = required - set(df_cache.columns)
    if missing:
        raise ValueError(f"Long-context summary cache missing columns: {sorted(missing)}")


# --------------------------------------------------
# Provider call
# --------------------------------------------------

def _call_long_context_llm(note_text: str, cfg: LongContextSummaryConfig) -> str:
    from openai import OpenAI
    client = OpenAI()

    user_msg = USER_TEMPLATE.format(note_text=note_text)

    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    # OpenAI python client returns choices[0].message.content
    out = resp.choices[0].message.content
    return "" if out is None else str(out).strip()


def _retry_call_long_context_llm(note_text: str, cfg: LongContextSummaryConfig) -> str:
    attempt = 0
    while True:
        try:
            return _call_long_context_llm(note_text, cfg)
        except Exception as e:
            attempt += 1
            if attempt > cfg.max_retries:
                raise RuntimeError(
                    f"Long-context LLM failed after {cfg.max_retries} retries. Last error: {e}"
                ) from e
            sleep_s = cfg.retry_backoff_sec * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 30.0)
            time.sleep(sleep_s)


# --------------------------------------------------
# Build summaries (cache-aware)
# --------------------------------------------------

def build_long_context_summaries(
    df_inputs: pd.DataFrame,
    cfg: LongContextSummaryConfig = LongContextSummaryConfig(),
    verbose: bool = True,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    required = {"subject_id", "raw_text", "raw_hash"}
    missing = required - set(df_inputs.columns)
    if missing:
        raise ValueError(f"df_inputs missing required columns: {sorted(missing)}")

    cache_table, cache_meta = get_cache_paths(cfg)
    df_cache = load_cache_table(cache_table)
    validate_cache_schema(df_cache)

    if not df_cache.empty:
        df_cache = df_cache.loc[
            (df_cache["model"] == cfg.model) &
            (df_cache["prompt_version"] == PROMPT_VERSION)
        ].copy()

    if verbose:
        n_cached = 0 if df_cache.empty else df_cache.shape[0]
        print(f"🔎 Long-context summary cache loaded: {n_cached} rows (model={cfg.model})")

    join_keys = ["subject_id", "raw_hash"]
    if df_cache.empty:
        df_missing = df_inputs.copy()
    else:
        df_merge = df_inputs[join_keys].merge(df_cache[join_keys], on=join_keys, how="left", indicator=True)
        df_missing = df_inputs.loc[df_merge["_merge"].ne("both")].copy()

    if verbose:
        print(f"🧩 Summaries needed: {int(df_missing.shape[0])} / {int(df_inputs.shape[0])}")

    new_rows = []
    if not df_missing.empty:
        for i, row in df_missing.iterrows():
            sid = int(row["subject_id"])
            raw_text = row["raw_text"]

            if verbose and (len(new_rows) % 25 == 0):
                print(f"➡️  Summarizing: {len(new_rows)} / {int(df_missing.shape[0])}")

            summary = _retry_call_long_context_llm(raw_text, cfg)
            if not summary:
                summary = (
                    "=== GLOBAL CLINICAL SUMMARY (UNORDERED TEXT) ===\n"
                    "unknown\n\n"
                    "=== NON-TEMPORAL FINDINGS SNAPSHOT ===\n"
                    "Overall disease burden:\n- unknown\n"
                )

            rec = {
                "subject_id": sid,
                "raw_hash": row["raw_hash"],
                "model": cfg.model,
                "prompt_version": PROMPT_VERSION,
                "summary_text": summary,
                "summary_hash": hash_text_sha256(summary),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            new_rows.append(rec)

            # checkpoint cache every N new summaries to avoid re-paying on interruption
            if checkpoint_every and (len(new_rows) % checkpoint_every == 0):
                df_new = pd.DataFrame(new_rows)
                new_rows = []

                if df_cache.empty:
                    df_cache = df_new
                else:
                    df_cache = pd.concat([df_cache, df_new], ignore_index=True)

                df_cache = df_cache.drop_duplicates(subset=CACHE_KEY_COLS, keep="last").reset_index(drop=True)
                save_cache_table(df_cache, cache_table, key_cols=CACHE_KEY_COLS)

                meta = {
                    "model": cfg.model,
                    "prompt_version": PROMPT_VERSION,
                    "n_rows": int(df_cache.shape[0]),
                    "columns": list(df_cache.columns),
                    "created_by": "src/llm_long_context_summary.py",
                    "schema_version": "llm_long_context_summary_cache_v1",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                cache_meta.parent.mkdir(parents=True, exist_ok=True)
                cache_meta.write_text(json.dumps(meta, indent=2))

                if verbose:
                    print(f"💾 checkpoint summary cache saved (rows={len(df_cache)})")


        df_new = pd.DataFrame(new_rows)
        validate_cache_schema(df_new)

        # Update cache
        if df_cache.empty:
            df_cache = df_new
        else:
            df_cache = pd.concat([df_cache, df_new], ignore_index=True)

        df_cache = df_cache.drop_duplicates(subset=CACHE_KEY_COLS, keep="last").reset_index(drop=True)
        save_cache_table(df_cache, cache_table, key_cols=CACHE_KEY_COLS)

        meta = {
            "model": cfg.model,
            "prompt_version": PROMPT_VERSION,
            "n_rows": int(df_cache.shape[0]),
            "columns": list(df_cache.columns),
            "created_by": "src/llm_long_context_summary.py",
            "schema_version": "llm_long_context_summary_cache_v1",
        }
        cache_meta.parent.mkdir(parents=True, exist_ok=True)
        cache_meta.write_text(json.dumps(meta, indent=2))

        if verbose:
            print(f"✅ Summary cache updated: now {int(df_cache.shape[0])} rows")

    # Align output to inputs
    df_out = df_inputs[["subject_id", "raw_hash"]].merge(
        df_cache,
        on=["subject_id", "raw_hash"],
        how="left",
    )

    if df_out["summary_text"].isna().any():
        raise RuntimeError("Missing summaries after cache+fetch. Check API failures or cache schema.")

    # Keep stable ordering and minimal cols
    keep = ["subject_id", "raw_hash", "summary_text", "summary_hash", "model", "prompt_version"]
    df_out = df_out[keep].sort_values("subject_id").reset_index(drop=True)
    return df_out


# --------------------------------------------------
# Artifact + manifest orchestrator
# --------------------------------------------------

def run_long_context_summary_pipeline(
    df_inputs: pd.DataFrame,
    cfg: LongContextSummaryConfig,
    artifact_path: str = DEFAULT_SUMMARY_ARTIFACT_PATH,
    source_path: str = "data/interim/data_nlp_ready.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Guard -> build -> save artifact + manifest.
    """
    expected_manifest_fields = ["model", "source_path", "source_mtime", "n_rows", "created_at_utc", "input_signature_hash", "prompt_version"]
    expected_manifest_kv = {"model": cfg.model, "prompt_version": PROMPT_VERSION}
    expected_required_cols = ["subject_id", "raw_hash", "summary_text", "summary_hash", "model", "prompt_version"]

    if artifact_is_valid(
        artifact_path=artifact_path,
        manifest_name=DEFAULT_MANIFEST_NAME,
        df_inputs=df_inputs,
        signature_key_cols=["subject_id", "raw_hash"],
        expected_manifest_fields=expected_manifest_fields,
        expected_manifest_kv=expected_manifest_kv,
        expected_required_cols=expected_required_cols,
    ):
        if verbose:
            print("✅ long-context summary artifact valid — loading (no API calls).")
        return pd.read_parquet(resolve_path(artifact_path))

    if verbose:
        print("⚙️ long-context summary artifact missing/invalid — generating summaries.")

    df_sum = build_long_context_summaries(df_inputs, cfg=cfg, verbose=verbose)

    p = resolve_path(artifact_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_parquet(p, index=False)

    sig = build_input_signature_hash(df_inputs, key_cols=["subject_id", "raw_hash"])
    manifest = make_standard_manifest(
        model=cfg.model,
        signature=sig,
        source_path=source_path,
        n_rows=int(df_sum.shape[0]),
        manifest_extra={"prompt_version": PROMPT_VERSION},
    )
    write_manifest(p, DEFAULT_MANIFEST_NAME, manifest)

    if verbose:
        print("💾 Long-context summaries saved + manifest written.")

    return pd.read_parquet(p)
