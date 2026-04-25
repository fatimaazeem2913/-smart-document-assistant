"""
ocr_engine.py
=============
Optical Character Recognition (OCR) for image inputs.

Two backends are supported — choose based on your environment:

  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │ Backend         │ Notes                                                │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │ pytesseract     │ Wraps Google Tesseract. Fast, CPU-only, requires     │
  │ (default)       │ Tesseract binary installed on the system.            │
  │                 │ Great for clean, high-resolution printed text.       │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │ easyocr         │ Deep-learning OCR, no external binary needed.        │
  │                 │ Better on noisy/handwritten text; slower first run   │
  │                 │ (downloads model weights ~100 MB).                   │
  └─────────────────┴──────────────────────────────────────────────────────┘

Pre-processing pipeline (applied before OCR):
  1. Convert to RGB (handle RGBA / greyscale / palette images).
  2. Resize: upscale small images so Tesseract/EasyOCR work reliably.
  3. Greyscale + contrast enhancement (Pillow ImageEnhance).
  4. Adaptive thresholding via OpenCV (optional, improves accuracy ~10–20%).

Language support:
  - pytesseract : set TESSERACT_LANG (default 'eng+urd').
  - EasyOCR     : set EASYOCR_LANGS list (default ['en', 'ur']).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Union

from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants (can be overridden via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

# Tesseract language string  (e.g. 'eng', 'eng+urd', 'urd')
TESSERACT_LANG: str = os.getenv("TESSERACT_LANG", "eng+urd")

# EasyOCR language list  (e.g. ['en'], ['en', 'ur'])
EASYOCR_LANGS: list[str] = ["en", "ur"]

# Minimum dimension (px) — images smaller than this are upscaled before OCR
MIN_OCR_DIM: int = 1000

# Contrast enhancement factor (1.0 = no change, 2.0 = double contrast)
CONTRAST_FACTOR: float = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Image Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Enhance an image for better OCR accuracy.

    Steps:
      1. Convert to RGB.
      2. Upscale if the smaller dimension < MIN_OCR_DIM.
      3. Boost contrast.
      4. Convert to greyscale.

    Args:
        image: PIL Image object.

    Returns:
        Pre-processed PIL Image (greyscale).
    """
    from PIL import ImageEnhance

    # 1 — Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2 — Upscale small images
    w, h = image.size
    short_side = min(w, h)
    if short_side < MIN_OCR_DIM:
        scale = MIN_OCR_DIM / short_side
        image = image.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS,
        )

    # 3 — Boost contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(CONTRAST_FACTOR)

    # 4 — Greyscale
    image = image.convert("L")

    return image


def apply_adaptive_threshold(image: Image.Image) -> Image.Image:
    """
    Apply OpenCV adaptive thresholding to binarize the image.
    Returns the original if OpenCV is not installed.

    This step significantly improves OCR on documents with uneven lighting.
    """
    try:
        import cv2
        import numpy as np

        img_array = np.array(image)
        binarized = cv2.adaptiveThreshold(
            img_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,   # neighbourhood size
            C=10,           # constant subtracted from weighted mean
        )
        return Image.fromarray(binarized)
    except ImportError:
        # opencv-python is optional — fall back to greyscale only
        return image


def load_image(source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """
    Load an image from a file path, raw bytes, or an existing PIL Image.
    """
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, (str, Path)):
        return Image.open(source)
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source))
    # Streamlit UploadedFile has a .read() method
    if hasattr(source, "read"):
        return Image.open(io.BytesIO(source.read()))
    raise TypeError(f"Unsupported image source type: {type(source)}")


# ─────────────────────────────────────────────────────────────────────────────
# pytesseract Backend
# ─────────────────────────────────────────────────────────────────────────────

def ocr_with_tesseract(
    image: Image.Image,
    lang: str = TESSERACT_LANG,
    use_adaptive_threshold: bool = True,
) -> str:
    """
    Extract text from a PIL Image using pytesseract (Google Tesseract).

    Args:
        image                  : Pre-loaded PIL Image.
        lang                   : Tesseract language string (e.g. 'eng+urd').
        use_adaptive_threshold : Apply OpenCV binarization before OCR.

    Returns:
        Extracted text string.

    Raises:
        ImportError : If pytesseract is not installed.
        RuntimeError: If Tesseract binary is not found on PATH.
    """
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is not installed. Run: pip install pytesseract\n"
            "Also install the Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    # Pre-process
    img = preprocess_image(image)
    if use_adaptive_threshold:
        img = apply_adaptive_threshold(img)

    # Tesseract configuration:
    #   --oem 3  = LSTM + Legacy engine (best accuracy)
    #   --psm 3  = Fully automatic page segmentation (default)
    config = r"--oem 3 --psm 3"

    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR Backend
# ─────────────────────────────────────────────────────────────────────────────

# Module-level EasyOCR reader cache (loading is slow ~5 s)
_easyocr_reader = None


def _get_easyocr_reader(langs: list[str] = EASYOCR_LANGS):
    """Load and cache an EasyOCR Reader object."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "easyocr is not installed. Run: pip install easyocr"
            )
        # gpu=False ensures CPU-only inference
        _easyocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)
    return _easyocr_reader


def ocr_with_easyocr(
    image: Image.Image,
    langs: list[str] = EASYOCR_LANGS,
) -> str:
    """
    Extract text from a PIL Image using EasyOCR (deep-learning OCR).

    EasyOCR returns a list of (bounding_box, text, confidence) tuples.
    We sort detections top-to-bottom, left-to-right to reconstruct
    reading order, then join with spaces / newlines.

    Args:
        image : Pre-loaded PIL Image.
        langs : List of language codes (e.g. ['en', 'ur']).

    Returns:
        Extracted text string.
    """
    import numpy as np

    reader = _get_easyocr_reader(langs)

    # Convert PIL → numpy array (EasyOCR expects numpy or file path)
    img_preprocessed = preprocess_image(image)
    img_array = np.array(img_preprocessed)

    # detail=1 returns (bbox, text, confidence)
    results = reader.readtext(img_array, detail=1, paragraph=False)

    if not results:
        return ""

    # Sort by vertical position (top of bounding box), then horizontal
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    # Build text — insert newline when vertical gap between detections > 20 px
    lines, prev_y = [], None
    for (bbox, text, conf) in results:
        y_top = bbox[0][1]
        if prev_y is not None and (y_top - prev_y) > 20:
            lines.append("\n")
        lines.append(text)
        prev_y = y_top

    return " ".join(lines).replace(" \n ", "\n").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Unified OCR Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_image(
    source,
    backend: str = "tesseract",
    lang: str = TESSERACT_LANG,
) -> dict:
    """
    High-level OCR function.  Loads the image, preprocesses it, and
    runs the selected OCR backend.

    Args:
        source  : File path, bytes, UploadedFile, or PIL Image.
        backend : 'tesseract' (default) or 'easyocr'.
        lang    : Tesseract language string (ignored for EasyOCR).

    Returns:
        dict:
          - 'text'    : Extracted text string (cleaned).
          - 'backend' : Backend used.
          - 'chars'   : Character count of extracted text.
          - 'quality' : Human-readable quality label.
    """
    from modules.utils import clean_text, ocr_quality_label

    image = load_image(source)

    if backend == "easyocr":
        raw_text = ocr_with_easyocr(image)
    else:
        raw_text = ocr_with_tesseract(image, lang=lang)

    cleaned = clean_text(raw_text)
    return {
        "text":    cleaned,
        "backend": backend,
        "chars":   len(cleaned),
        "quality": ocr_quality_label(len(cleaned)),
    }
