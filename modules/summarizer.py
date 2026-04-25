"""
summarizer.py
=============
Abstractive document summarization using HuggingFace Seq2Seq models.

Supported models
----------------
  BART-large-CNN   — facebook/bart-large-cnn
      Best quality; trained on 300k CNN/DailyMail news articles.
      Output: ~3–5 well-formed sentences.

  DistilBART       — sshleifer/distilbart-cnn-12-6
      6-layer distilled BART; ~2× faster, minor quality drop.

  T5-Small         — t5-small
      Very fast; great for CPU-only setups. Lower quality on long docs.

  T5-Base          — t5-base
      Balanced speed/quality; good for medium-length documents.

Multi-chunk strategy
--------------------
  1. Divide the document into ≤600-word chunks with 60-word overlap.
  2. Summarize each chunk independently (parallel if possible).
  3. Join chunk summaries → "meta-document".
  4. If the meta-document is still long, run one final summarization pass
     ("summary of summaries") for a concise final output.

T5 models require a task prefix "summarize: " prepended to input text.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from modules.chunker import chunk_for_summarization
from modules.utils   import get_cached_pipeline, safe_progress


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

SUMMARIZATION_MODELS: dict[str, str] = {
    "BART-Large CNN  (Best Quality)":  "facebook/bart-large-cnn",
    "DistilBART  (Faster, Good)":      "sshleifer/distilbart-cnn-12-6",
    "T5-Base  (Balanced)":             "t5-base",
    "T5-Small  (Lightest / Fastest)":  "t5-small",
}

# T5 models require this prefix for summarization tasks
_T5_PREFIX = "summarize: "


# ─────────────────────────────────────────────────────────────────────────────
# Single Chunk Summarization
# ─────────────────────────────────────────────────────────────────────────────

def _summarize_chunk(
    text:       str,
    pipeline,
    model_name: str,
    max_length: int,
    min_length: int,
) -> str:
    """
    Summarize a single text chunk using the provided pipeline.

    Args:
        text       : Input text (must fit within model's token window).
        pipeline   : Loaded HuggingFace summarization pipeline.
        model_name : Model identifier (used to detect T5 prefix requirement).
        max_length : Maximum token count for the generated summary.
        min_length : Minimum token count for the generated summary.

    Returns:
        Summary string.
    """
    if "t5" in model_name.lower():
        text = _T5_PREFIX + text

    result = pipeline(
        text,
        max_length  = max_length,
        min_length  = min_length,
        do_sample   = False,   # deterministic beam search
        truncation  = True,
    )
    return result[0]["summary_text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Full Document Summarization
# ─────────────────────────────────────────────────────────────────────────────

def summarize_document(
    text:              str,
    model_key:         str = "BART-Large CNN  (Best Quality)",
    max_length:        int = 150,
    min_length:        int = 40,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Summarize a (potentially long) document using chunk-and-merge.

    Args:
        text              : Full extracted document text.
        model_key         : Key from SUMMARIZATION_MODELS dict.
        max_length        : Max output tokens per chunk summary.
        min_length        : Min output tokens per chunk summary.
        progress_callback : Optional callable(current, total).

    Returns:
        dict:
          - 'final_summary'   : Polished summary string.
          - 'chunk_summaries' : List of per-chunk summary strings.
          - 'chunk_count'     : Number of chunks processed.
          - 'model'           : Model identifier used.
          - 'two_pass'        : True if a second summarization pass was done.
    """
    model_name = SUMMARIZATION_MODELS.get(model_key, "facebook/bart-large-cnn")
    pipeline   = get_cached_pipeline("summarization", model_name)

    # ── 1. Split document into chunks ──────────────────────────────────────
    chunks = chunk_for_summarization(text)
    if not chunks:
        raise ValueError("No text found to summarize.")

    # ── 2. Summarize each chunk ────────────────────────────────────────────
    chunk_summaries: List[str] = []
    for idx, chunk in enumerate(chunks):
        safe_progress(progress_callback, idx + 1, len(chunks) + 1)
        summary = _summarize_chunk(chunk, pipeline, model_name, max_length, min_length)
        chunk_summaries.append(summary)

    # ── 3. Merge chunk summaries ───────────────────────────────────────────
    merged  = " ".join(chunk_summaries)
    two_pass = False

    # ── 4. Final pass if merged text is still long ─────────────────────────
    word_budget = max_length * 2   # heuristic: merged > 2× target → re-summarize
    if len(merged.split()) > word_budget:
        safe_progress(progress_callback, len(chunks) + 1, len(chunks) + 1)
        final = _summarize_chunk(merged, pipeline, model_name, max_length, min_length)
        two_pass = True
    else:
        final = merged

    return {
        "final_summary":   final,
        "chunk_summaries": chunk_summaries,
        "chunk_count":     len(chunks),
        "model":           model_name,
        "two_pass":        two_pass,
    }
