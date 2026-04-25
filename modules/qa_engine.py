"""
qa_engine.py
============
Extractive Question Answering using BERT-based models fine-tuned on SQuAD 2.0.

How it works
------------
  1. The document is split into BERT-safe chunks (≤380 words each).
  2. Every chunk is used as the "context" for the user's question.
  3. The QA pipeline returns an answer span + confidence score per chunk.
  4. Chunks are sorted by confidence score; the best answer is returned.
  5. Optionally, the top-3 candidates are surfaced for transparency.

This approach guarantees answers can be found from any part of the document,
regardless of its length — overcoming BERT's 512-token input limit.

Supported models
----------------
  deepset/roberta-base-squad2    — Best accuracy; recommended default.
  deepset/bert-base-cased-squad2 — Classic BERT; slightly lower accuracy.
  deepset/minilm-uncased-squad2  — Tiny model; fastest inference.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from modules.chunker import chunk_for_qa
from modules.utils   import get_cached_pipeline, safe_progress, confidence_label


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

QA_MODELS: dict[str, str] = {
    "RoBERTa-Base SQuAD2  (Best Accuracy)": "deepset/roberta-base-squad2",
    "BERT-Base Cased SQuAD2  (Classic)":    "deepset/bert-base-cased-squad2",
    "MiniLM-Uncased SQuAD2  (Fastest)":     "deepset/minilm-uncased-squad2",
}


# ─────────────────────────────────────────────────────────────────────────────
# Single-Context QA
# ─────────────────────────────────────────────────────────────────────────────

def _answer_from_context(
    question: str,
    context:  str,
    pipeline,
) -> dict:
    """
    Run QA on a single (question, context) pair.

    The pipeline is configured with ``handle_impossible_answer=True`` which
    enables SQuAD 2.0 "no answer" detection — the model can return an empty
    span if the question cannot be answered from this context.

    Returns:
        dict with keys: answer, score, start, end.
    """
    return pipeline(
        question               = question,
        context                = context,
        max_answer_len         = 200,
        handle_impossible_answer = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Chunk QA
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(
    question:          str,
    document_text:     str,
    model_key:         str = "RoBERTa-Base SQuAD2  (Best Accuracy)",
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Find the best answer to ``question`` anywhere in ``document_text``.

    Args:
        question          : Natural language question.
        document_text     : Full extracted document text.
        model_key         : Key from QA_MODELS dict.
        progress_callback : Optional callable(current, total).

    Returns:
        dict:
          - 'answer'           : Best answer string.
          - 'score'            : Confidence score (0–1).
          - 'confidence_label' : Human-readable confidence string.
          - 'confidence_color' : Hex colour for the label.
          - 'context_snippet'  : Passage that contained the answer (first 600 chars).
          - 'chunk_index'      : Index (0-based) of the winning chunk.
          - 'all_candidates'   : Sorted list of all candidate dicts.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not document_text.strip():
        raise ValueError("Document text cannot be empty.")

    model_name = QA_MODELS.get(model_key, "deepset/roberta-base-squad2")
    pipeline   = get_cached_pipeline("question-answering", model_name)

    # ── Split document ─────────────────────────────────────────────────────
    chunks = chunk_for_qa(document_text)
    if not chunks:
        raise ValueError("Document produced no processable chunks.")

    # ── Query every chunk ──────────────────────────────────────────────────
    candidates: List[dict] = []
    for idx, chunk in enumerate(chunks):
        safe_progress(progress_callback, idx + 1, len(chunks))
        try:
            res = _answer_from_context(question, chunk, pipeline)
            ans = res.get("answer", "").strip()
            # Skip empty / whitespace-only answers
            if ans:
                candidates.append({
                    "answer":          ans,
                    "score":           float(res.get("score", 0.0)),
                    "context_snippet": chunk[:600],
                    "chunk_index":     idx,
                })
        except Exception:
            continue  # Skip chunks that cause inference errors

    # ── Return best result ─────────────────────────────────────────────────
    if not candidates:
        label, color = confidence_label(0.0)
        return {
            "answer":           "No answer found in the document.",
            "score":            0.0,
            "confidence_label": label,
            "confidence_color": color,
            "context_snippet":  "",
            "chunk_index":      -1,
            "all_candidates":   [],
        }

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best  = candidates[0]
    label, color = confidence_label(best["score"])

    return {
        "answer":           best["answer"],
        "score":            best["score"],
        "confidence_label": label,
        "confidence_color": color,
        "context_snippet":  best["context_snippet"],
        "chunk_index":      best["chunk_index"],
        "all_candidates":   candidates,
    }
