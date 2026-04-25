"""
translator.py
=============
Bidirectional English ↔ Urdu translation using Helsinki-NLP Opus-MT models.

Models
------
  EN → UR : Helsinki-NLP/opus-mt-en-ur
  UR → EN : Helsinki-NLP/opus-mt-ur-en

Both are MarianMT models — lightweight (~300 MB each), no API key needed,
fully offline after the first download.

Translation approach
--------------------
MarianMT models translate best at the sentence/paragraph level.
For long texts we:
  1. Split on paragraph and sentence boundaries (chunk_for_translation).
  2. Translate each chunk independently.
  3. Re-join preserving paragraph structure.

Language auto-detection
-----------------------
A heuristic based on Unicode character ranges:
  Arabic script (U+0600–U+06FF) → Urdu
  Latin script                   → English
  Mixed or unrecognisable        → 'mixed' / 'unknown'
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from modules.chunker import chunk_for_translation
from modules.utils   import get_cached_pipeline, safe_progress


# ─────────────────────────────────────────────────────────────────────────────
# Direction Registry
# ─────────────────────────────────────────────────────────────────────────────

TRANSLATION_DIRECTIONS: dict[str, dict] = {
    "English → Urdu": {
        "model":    "Helsinki-NLP/opus-mt-en-ur",
        "src_code": "en",
        "tgt_code": "ur",
    },
    "Urdu → English": {
        "model":    "Helsinki-NLP/opus-mt-ur-en",
        "src_code": "ur",
        "tgt_code": "en",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Single Chunk Translation
# ─────────────────────────────────────────────────────────────────────────────

def _translate_chunk(text: str, pipeline) -> str:
    """Translate a single chunk; returns the translated string."""
    result = pipeline(text, max_length=512)
    return result[0]["translation_text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Full Text Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_text(
    text:              str,
    direction:         str = "English → Urdu",
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Translate ``text`` in the given direction.

    Args:
        text              : Source text to translate.
        direction         : One of the keys in TRANSLATION_DIRECTIONS.
        progress_callback : Optional callable(current, total).

    Returns:
        dict:
          - 'translated_text' : Full translated string.
          - 'model'           : Model identifier used.
          - 'src_code'        : Source language code.
          - 'tgt_code'        : Target language code.
          - 'chunk_count'     : Number of chunks translated.
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    config = TRANSLATION_DIRECTIONS.get(direction)
    if not config:
        raise ValueError(
            f"Unknown direction '{direction}'. "
            f"Available: {list(TRANSLATION_DIRECTIONS.keys())}"
        )

    pipeline   = get_cached_pipeline("translation", config["model"])
    chunks     = chunk_for_translation(text)
    translated: List[str] = []

    for idx, chunk in enumerate(chunks):
        safe_progress(progress_callback, idx + 1, len(chunks))
        translated.append(_translate_chunk(chunk, pipeline))

    full_translation = "\n\n".join(translated)

    return {
        "translated_text": full_translation,
        "model":           config["model"],
        "src_code":        config["src_code"],
        "tgt_code":        config["tgt_code"],
        "chunk_count":     len(chunks),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Language Detection (Heuristic, No External Dependency)
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Heuristically detect whether text is primarily Urdu or English.

    Uses Unicode character range membership:
      Arabic/Urdu block : U+0600–U+06FF
      Latin block       : ASCII alphabetics

    Returns:
        'ur', 'en', 'mixed', or 'unknown'.
    """
    urdu_chars  = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    latin_chars = sum(1 for c in text if c.isalpha() and c.isascii())
    total       = urdu_chars + latin_chars

    if total == 0:
        return "unknown"

    ratio = urdu_chars / total
    if ratio > 0.60:
        return "ur"
    if ratio < 0.20:
        return "en"
    return "mixed"


def suggest_direction(text: str) -> str:
    """
    Suggest a translation direction string based on detected language.

    Returns:
        'English → Urdu' or 'Urdu → English' or '' (cannot determine).
    """
    lang = detect_language(text)
    if lang == "en":
        return "English → Urdu"
    if lang == "ur":
        return "Urdu → English"
    return ""
