"""
chunker.py
==========
Splits long documents into overlapping chunks that fit within
HuggingFace model token windows.

Why chunking?
-------------
Most Transformer models have hard input limits:
  BERT / RoBERTa  →  512 tokens  ≈  380 words
  BART-large      →  1024 tokens ≈  780 words
  T5-small/base   →  512 tokens  ≈  380 words
  MarianMT        →  512 tokens  ≈  380 words

Sentence-aware splitting ensures chunks always end on a sentence
boundary (never mid-sentence), and overlap preserves context across
boundaries — critical for QA where the answer may span the join.

Word count is used as a fast proxy for token count:
  1 English word ≈ 1.3 tokens (BPE tokenizer estimate).
"""

from __future__ import annotations

import re
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Sentence Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> List[str]:
    """
    Heuristically split text into sentences.

    Handles:
      - English '. ! ?'
      - Urdu '۔ ؟'
      - Paragraph breaks (treated as sentence boundaries)
      - Abbreviations are imperfectly handled (acceptable trade-off)

    Args:
        text: Input text string.

    Returns:
        List of sentence strings (non-empty).
    """
    # Split on sentence-ending punctuation followed by whitespace + capital
    pattern = r"(?<=[.!?۔؟])\s+(?=[A-Za-z\u0600-\u06FF\"\'\""])"
    sents = re.split(pattern, text)

    # Further split on paragraph boundaries
    result: List[str] = []
    for sent in sents:
        for sub in re.split(r"\n{2,}", sent):
            s = sub.strip()
            if s:
                result.append(s)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core Chunking Engine
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text:          str,
    max_words:     int = 400,
    overlap_words: int = 50,
) -> List[str]:
    """
    Split ``text`` into overlapping word-limited chunks.

    Algorithm:
      1. Split text into sentences.
      2. Pack sentences greedily until word budget is exceeded.
      3. When the budget is exceeded, save the current chunk.
      4. Start the next chunk with the last ``overlap_words`` words
         from the previous chunk (maintains context).
      5. Single sentences larger than ``max_words`` are hard-split by words.

    Args:
        text         : Full document text.
        max_words    : Maximum words per chunk (default 400 ≈ 520 tokens).
        overlap_words: Words of overlap between adjacent chunks (default 50).

    Returns:
        List of text chunk strings.
    """
    if not text.strip():
        return []

    sentences = split_into_sentences(text)
    chunks:    List[str] = []
    buffer:    List[str] = []
    buf_wc = 0

    for sent in sentences:
        sent_wc = len(sent.split())

        # Giant single sentence — hard split by words
        if sent_wc > max_words:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer, buf_wc = [], 0
            words = sent.split()
            step  = max_words - overlap_words
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i: i + max_words]))
            continue

        # Would adding this sentence overflow the buffer?
        if buf_wc + sent_wc > max_words and buffer:
            chunks.append(" ".join(buffer))
            # Build overlap from tail of previous buffer
            overlap_sents: List[str] = []
            overlap_wc = 0
            for s in reversed(buffer):
                wc = len(s.split())
                if overlap_wc + wc > overlap_words:
                    break
                overlap_sents.insert(0, s)
                overlap_wc += wc
            buffer, buf_wc = overlap_sents, overlap_wc

        buffer.append(sent)
        buf_wc += sent_wc

    if buffer:
        chunks.append(" ".join(buffer))

    return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Task-Specific Chunk Profiles
# ─────────────────────────────────────────────────────────────────────────────

def chunk_for_summarization(text: str) -> List[str]:
    """
    Larger chunks for BART/T5 (accepts up to ~1024 tokens).
    600 words × 1.3 ≈ 780 tokens — safe margin below BART's 1024 limit.
    """
    return chunk_text(text, max_words=600, overlap_words=60)


def chunk_for_qa(text: str) -> List[str]:
    """
    Smaller chunks for BERT QA (512-token limit).
    380 words + question overhead ≈ 494 tokens — safely within BERT.
    """
    return chunk_text(text, max_words=380, overlap_words=40)


def chunk_for_translation(text: str) -> List[str]:
    """
    Small chunks for MarianMT (512-token limit, sentence-level is best).
    80 words ≈ 104 tokens — keeps sentence structure intact.
    """
    return chunk_text(text, max_words=80, overlap_words=0)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Fast word-based token estimate: 1 word ≈ 1.3 BPE tokens."""
    return int(len(text.split()) * 1.3)


def chunk_summary(chunks: List[str]) -> dict:
    """
    Return diagnostic information about a list of chunks.

    Returns:
        dict with: count, min_words, max_words, avg_words, total_words.
    """
    if not chunks:
        return {"count": 0, "min_words": 0, "max_words": 0,
                "avg_words": 0, "total_words": 0}
    word_counts = [len(c.split()) for c in chunks]
    return {
        "count":       len(chunks),
        "min_words":   min(word_counts),
        "max_words":   max(word_counts),
        "avg_words":   round(sum(word_counts) / len(word_counts), 1),
        "total_words": sum(word_counts),
    }
