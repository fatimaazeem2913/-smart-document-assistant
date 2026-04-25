"""
pdf_handler.py
==============
Smart PDF text extraction with automatic OCR fallback.

Strategy
--------
  1. Try native text extraction with PyMuPDF (fitz).
     — Fast, no quality loss, works for born-digital PDFs.

  2. Per-page text density check:
     — If a page returns fewer than MIN_CHARS_NATIVE characters,
       it's likely scanned / image-based.
     — Render that page to a high-resolution PIL Image and run OCR.

  3. Merge results from all pages into a single cleaned string.

This hybrid approach handles mixed PDFs (some pages native, some scanned)
without any manual configuration by the user.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Optional, Union

# Minimum chars from native extraction to trust it (below → use OCR)
MIN_CHARS_NATIVE: int = 50

# DPI used when rendering PDF pages to images for OCR
OCR_RENDER_DPI: int = 200


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading
# ─────────────────────────────────────────────────────────────────────────────

def _open_pdf(source):
    """
    Open a PDF from a file path, bytes, or Streamlit UploadedFile.

    Returns:
        fitz.Document object.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required. Install with: pip install PyMuPDF"
        )

    if isinstance(source, (str, Path)):
        return fitz.open(str(source))

    # File-like object (Streamlit UploadedFile)
    raw = source.read() if hasattr(source, "read") else bytes(source)
    return fitz.open(stream=raw, filetype="pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Per-Page Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_native(page) -> str:
    """Extract text from a single PyMuPDF page object."""
    return page.get_text("text").strip()


def _render_page_to_image(page, dpi: int = OCR_RENDER_DPI):
    """
    Render a PyMuPDF page to a high-resolution PIL Image for OCR.

    Args:
        page : fitz.Page object.
        dpi  : Resolution for rendering (higher = better OCR, slower).

    Returns:
        PIL.Image.Image
    """
    from PIL import Image

    # Create a transformation matrix scaled by DPI/72 (72 is PDF's native DPI)
    zoom   = dpi / 72
    matrix = __import__("fitz").Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    img_bytes = pixmap.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def _extract_page_with_ocr(page, backend: str, lang: str) -> str:
    """
    Render a PDF page to an image, then run OCR on it.

    Args:
        page    : fitz.Page.
        backend : 'tesseract' or 'easyocr'.
        lang    : Language string for Tesseract.

    Returns:
        OCR-extracted text string.
    """
    from modules.ocr_engine import extract_text_from_image

    pil_image = _render_page_to_image(page)
    result    = extract_text_from_image(pil_image, backend=backend, lang=lang)
    return result["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Full PDF Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(
    source,
    ocr_backend:       str = "tesseract",
    ocr_lang:          str = "eng+urd",
    force_ocr:         bool = False,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Extract text from every page of a PDF.

    For each page:
      - If native text is sufficient → use it (fast).
      - Otherwise → render to image and OCR it (slower but complete).

    Args:
        source            : File path, bytes, or UploadedFile.
        ocr_backend       : 'tesseract' or 'easyocr'.
        ocr_lang          : Tesseract language string.
        force_ocr         : If True, always OCR every page.
        progress_callback : Optional callable(current_page, total_pages).

    Returns:
        dict:
          - 'text'         : Full extracted text (all pages joined).
          - 'page_count'   : Number of pages in the PDF.
          - 'ocr_pages'    : Number of pages that needed OCR.
          - 'native_pages' : Number of pages extracted natively.
          - 'page_texts'   : List of per-page text strings.
    """
    from modules.utils import clean_text, safe_progress

    doc       = _open_pdf(source)
    total     = len(doc)
    page_texts: list[str] = []
    ocr_count = 0

    for page_num in range(total):
        safe_progress(progress_callback, page_num + 1, total)
        page = doc[page_num]

        native_text = _extract_page_native(page)

        if force_ocr or len(native_text) < MIN_CHARS_NATIVE:
            # Scanned / image page — use OCR
            try:
                text = _extract_page_with_ocr(page, backend=ocr_backend, lang=ocr_lang)
                ocr_count += 1
            except Exception as exc:
                text = f"[OCR failed on page {page_num + 1}: {exc}]"
        else:
            text = native_text

        page_texts.append(clean_text(text))

    doc.close()

    full_text = "\n\n".join(
        f"--- Page {i + 1} ---\n{t}"
        for i, t in enumerate(page_texts)
        if t.strip()
    )

    return {
        "text":         full_text,
        "page_count":   total,
        "ocr_pages":    ocr_count,
        "native_pages": total - ocr_count,
        "page_texts":   page_texts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plain Text File Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_txt(source) -> str:
    """
    Read a plain text file from a path or file-like object.

    Tries UTF-8, then latin-1, then cp1252 to handle common encodings.
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    raw = source.read() if hasattr(source, "read") else bytes(source)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, AttributeError):
            continue
    raise ValueError("Could not decode text file with any common encoding.")


# ─────────────────────────────────────────────────────────────────────────────
# Unified Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def extract_document(
    source,
    file_name:         str,
    ocr_backend:       str = "tesseract",
    ocr_lang:          str = "eng+urd",
    force_ocr:         bool = False,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Top-level extraction function.  Detects file type and routes accordingly.

    Supports: .pdf, .txt, .md, .jpg, .jpeg, .png, .bmp, .tiff, .webp

    Returns:
        dict with at minimum:
          - 'text'      : Extracted text string.
          - 'file_type' : Detected file type ('pdf', 'image', 'text').
          - 'source'    : Original file_name.
        Plus any extra keys from the specific extractor.
    """
    from modules.utils import detect_file_type, clean_text
    from modules.ocr_engine import extract_text_from_image

    file_type = detect_file_type(file_name)

    if file_type == "pdf":
        result = extract_text_from_pdf(
            source,
            ocr_backend=ocr_backend,
            ocr_lang=ocr_lang,
            force_ocr=force_ocr,
            progress_callback=progress_callback,
        )
        result["file_type"] = "pdf"
        result["source"]    = file_name
        return result

    elif file_type == "image":
        result = extract_text_from_image(source, backend=ocr_backend, lang=ocr_lang)
        result["file_type"]   = "image"
        result["source"]      = file_name
        result["page_count"]  = 1
        result["ocr_pages"]   = 1
        result["native_pages"] = 0
        return result

    elif file_type == "text":
        raw_text = extract_text_from_txt(source)
        text     = clean_text(raw_text)
        return {
            "text":          text,
            "file_type":     "text",
            "source":        file_name,
            "page_count":    1,
            "ocr_pages":     0,
            "native_pages":  1,
        }

    else:
        raise ValueError(
            f"Unsupported file type for '{file_name}'. "
            "Please upload a PDF, image (JPG/PNG/BMP/TIFF), or TXT file."
        )
