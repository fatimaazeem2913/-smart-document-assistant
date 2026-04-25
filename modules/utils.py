"""
utils.py  —  Shared helpers: text cleaning, stats, model cache, labels.
"""
import re, unicodedata
from typing import Callable, Optional

_MODEL_CACHE: dict = {}

def get_cached_pipeline(task: str, model_name: str):
    key = f"{task}::{model_name}"
    if key not in _MODEL_CACHE:
        from transformers import pipeline
        _MODEL_CACHE[key] = pipeline(task, model=model_name, tokenizer=model_name)
    return _MODEL_CACHE[key]

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.rstrip() for l in text.split("\n")]
    collapsed, blank_run = [], 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)
    return "\n".join(collapsed).strip()

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def document_stats(text: str) -> dict:
    chars      = len(text)
    words      = len(text.split())
    sentences  = max(len(re.findall(r"[.!?]+", text)), 1)
    paragraphs = max(len([p for p in text.split("\n\n") if p.strip()]), 1)
    return {
        "characters": chars, "words": words,
        "sentences": sentences, "paragraphs": paragraphs,
        "estimated_tokens": int(words * 1.3),
    }

def confidence_label(score: float) -> tuple:
    if score >= 0.75: return "✅ High Confidence",   "#27ae60"
    if score >= 0.45: return "🟡 Medium Confidence", "#f39c12"
    if score >= 0.15: return "🟠 Low Confidence",    "#e67e22"
    return "❌ Very Low", "#e74c3c"

def ocr_quality_label(char_count: int) -> str:
    if char_count >= 500: return "✅ Good OCR result"
    if char_count >= 100: return "🟡 Partial — image may be low-resolution"
    return "❌ Very little text — check image quality / resolution"

def safe_progress(cb: Optional[Callable], current: int, total: int) -> None:
    if cb:
        try: cb(current, total)
        except Exception: pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS   = {".pdf"}
TEXT_EXTENSIONS  = {".txt", ".text", ".md"}

def detect_file_type(filename: str) -> str:
    from pathlib import Path
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTENSIONS:   return "pdf"
    if ext in IMAGE_EXTENSIONS: return "image"
    if ext in TEXT_EXTENSIONS:  return "text"
    return "unknown"
