"""
tests/test_cases.py  —  Smart Document Assistant with OCR
=========================================================
Run:   pytest tests/test_cases.py -v

All tests are offline (no model downloads, no internet).
OCR tests skip gracefully when binaries are absent.
"""

import sys, os, io
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Shared sample data ────────────────────────────────────────────────────────

ENGLISH_TEXT = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines. "
    "It contrasts with natural intelligence displayed by animals and humans. "
    "AI research focuses on building rational agents that perceive their "
    "environment and take actions maximizing their chance of achieving goals. "
    "Applications include speech recognition, NLP, computer vision."
)

URDU_TEXT = (
    "مصنوعی ذہانت وہ ذہانت ہے جو مشینوں میں پائی جاتی ہے۔ "
    "یہ انسانی اور حیوانی فطری ذہانت کے برعکس ہے۔"
)

LONG_TEXT = (ENGLISH_TEXT + "\n\n") * 25


# ═══════════════════════════════════════════════════════════════
# 1. utils.py
# ═══════════════════════════════════════════════════════════════

class TestUtils:
    def test_clean_removes_nulls(self):
        from modules.utils import clean_text
        assert "\x00" not in clean_text("Hello\x00World")

    def test_clean_collapses_blank_lines(self):
        from modules.utils import clean_text
        assert "\n\n\n" not in clean_text("A\n\n\n\n\nB")

    def test_clean_empty(self):
        from modules.utils import clean_text
        assert clean_text("") == ""

    def test_normalize_whitespace(self):
        from modules.utils import normalize_whitespace
        assert normalize_whitespace("  a   b\t\nc  ") == "a b c"

    def test_doc_stats_fields(self):
        from modules.utils import document_stats
        s = document_stats(ENGLISH_TEXT)
        for key in ("characters", "words", "sentences", "paragraphs"):
            assert key in s
        assert s["words"] > 0

    def test_doc_stats_empty(self):
        from modules.utils import document_stats
        assert document_stats("")["words"] == 0

    def test_confidence_labels(self):
        from modules.utils import confidence_label
        assert "High"     in confidence_label(0.90)[0]
        assert "Medium"   in confidence_label(0.55)[0]
        assert "Low"      in confidence_label(0.25)[0]
        assert "Very Low" in confidence_label(0.05)[0]

    def test_ocr_quality_labels(self):
        from modules.utils import ocr_quality_label
        assert "Good"    in ocr_quality_label(1000)
        assert "Partial" in ocr_quality_label(200)
        assert ocr_quality_label(5)  # at least something returned

    def test_detect_file_types(self):
        from modules.utils import detect_file_type
        assert detect_file_type("x.pdf")  == "pdf"
        assert detect_file_type("x.png")  == "image"
        assert detect_file_type("x.jpg")  == "image"
        assert detect_file_type("x.txt")  == "text"
        assert detect_file_type("x.csv")  == "unknown"

    def test_safe_progress_none(self):
        from modules.utils import safe_progress
        safe_progress(None, 1, 5)  # must not raise

    def test_safe_progress_crashing_cb(self):
        from modules.utils import safe_progress
        safe_progress(lambda a, b: 1 / 0, 1, 5)  # must not propagate


# ═══════════════════════════════════════════════════════════════
# 2. chunker.py
# ═══════════════════════════════════════════════════════════════

class TestChunker:
    def test_split_sentences(self):
        from modules.chunker import split_into_sentences
        sents = split_into_sentences("Hello. How are you? Fine!")
        assert len(sents) >= 2

    def test_chunk_empty(self):
        from modules.chunker import chunk_text
        assert chunk_text("") == []

    def test_chunk_short_is_single(self):
        from modules.chunker import chunk_text
        chunks = chunk_text("One sentence. Another.", max_words=200)
        assert len(chunks) == 1

    def test_chunk_long_is_multiple(self):
        from modules.chunker import chunk_text
        chunks = chunk_text(LONG_TEXT, max_words=100, overlap_words=10)
        assert len(chunks) > 1

    def test_chunk_no_empties(self):
        from modules.chunker import chunk_text
        chunks = chunk_text(LONG_TEXT, max_words=100)
        assert all(c.strip() for c in chunks)

    def test_chunk_word_budget(self):
        from modules.chunker import chunk_text
        chunks = chunk_text(LONG_TEXT, max_words=120)
        for c in chunks:
            assert len(c.split()) <= 140  # small sentence-boundary slack

    def test_overlap_exists(self):
        from modules.chunker import chunk_text
        chunks = chunk_text(LONG_TEXT, max_words=80, overlap_words=20)
        if len(chunks) >= 2:
            tail = set(chunks[0].split()[-15:])
            head = set(chunks[1].split()[:30])
            assert len(tail & head) > 0

    def test_summarization_budget(self):
        from modules.chunker import chunk_for_summarization
        for c in chunk_for_summarization(LONG_TEXT):
            assert len(c.split()) <= 680

    def test_qa_budget(self):
        from modules.chunker import chunk_for_qa
        for c in chunk_for_qa(LONG_TEXT):
            assert len(c.split()) <= 420

    def test_translation_budget(self):
        from modules.chunker import chunk_for_translation
        for c in chunk_for_translation(LONG_TEXT):
            assert len(c.split()) <= 100

    def test_estimate_tokens(self):
        from modules.chunker import estimate_tokens
        assert estimate_tokens("hello world") > 0

    def test_chunk_summary_stats(self):
        from modules.chunker import chunk_text, chunk_summary
        chunks = chunk_text(LONG_TEXT, max_words=100)
        info = chunk_summary(chunks)
        assert info["count"] == len(chunks)
        assert info["max_words"] >= info["min_words"]

    def test_chunk_summary_empty(self):
        from modules.chunker import chunk_summary
        assert chunk_summary([])["count"] == 0


# ═══════════════════════════════════════════════════════════════
# 3. translator.py  (no model download)
# ═══════════════════════════════════════════════════════════════

class TestTranslator:
    def test_detect_english(self):
        from modules.translator import detect_language
        assert detect_language(ENGLISH_TEXT) == "en"

    def test_detect_urdu(self):
        from modules.translator import detect_language
        assert detect_language(URDU_TEXT) == "ur"

    def test_detect_digits_unknown(self):
        from modules.translator import detect_language
        assert detect_language("12345") in ("unknown", "en")

    def test_suggest_direction_en(self):
        from modules.translator import suggest_direction
        assert suggest_direction(ENGLISH_TEXT) == "English → Urdu"

    def test_suggest_direction_ur(self):
        from modules.translator import suggest_direction
        assert suggest_direction(URDU_TEXT) == "Urdu → English"

    def test_directions_keys_exist(self):
        from modules.translator import TRANSLATION_DIRECTIONS
        assert "English → Urdu" in TRANSLATION_DIRECTIONS
        assert "Urdu → English" in TRANSLATION_DIRECTIONS

    def test_bad_direction_raises(self):
        from modules.translator import translate_text
        with pytest.raises(ValueError, match="Unknown direction"):
            translate_text("hello", direction="French → German")

    def test_empty_text_raises(self):
        from modules.translator import translate_text
        with pytest.raises(ValueError):
            translate_text("   ")


# ═══════════════════════════════════════════════════════════════
# 4. qa_engine.py  (no model download)
# ═══════════════════════════════════════════════════════════════

class TestQAEngine:
    def test_models_dict_populated(self):
        from modules.qa_engine import QA_MODELS
        assert len(QA_MODELS) >= 2
        for v in QA_MODELS.values():
            assert isinstance(v, str)

    def test_empty_question_raises(self):
        from modules.qa_engine import answer_question
        with pytest.raises(ValueError, match="empty"):
            answer_question("", "Some context.")

    def test_empty_document_raises(self):
        from modules.qa_engine import answer_question
        with pytest.raises(ValueError, match="empty"):
            answer_question("What is AI?", "")


# ═══════════════════════════════════════════════════════════════
# 5. summarizer.py  (no model download)
# ═══════════════════════════════════════════════════════════════

class TestSummarizer:
    def test_models_dict_populated(self):
        from modules.summarizer import SUMMARIZATION_MODELS
        assert len(SUMMARIZATION_MODELS) >= 2

    def test_t5_prefix(self):
        from modules.summarizer import _T5_PREFIX
        assert "summarize" in _T5_PREFIX

    def test_empty_text_yields_no_chunks(self):
        from modules.chunker import chunk_for_summarization
        assert chunk_for_summarization("") == []


# ═══════════════════════════════════════════════════════════════
# 6. ocr_engine.py  (image preprocessing — no binary needed)
# ═══════════════════════════════════════════════════════════════

class TestOCREngine:
    def _blank_image(self, w=200, h=100):
        from PIL import Image
        return Image.new("RGB", (w, h), "white")

    def test_preprocess_to_greyscale(self):
        from modules.ocr_engine import preprocess_image
        result = preprocess_image(self._blank_image())
        assert result.mode == "L"

    def test_preprocess_upscales_tiny(self):
        from modules.ocr_engine import preprocess_image, MIN_OCR_DIM
        result = preprocess_image(self._blank_image(50, 30))
        assert min(result.size) >= MIN_OCR_DIM

    def test_load_from_pil(self):
        from modules.ocr_engine import load_image
        from PIL import Image
        img = Image.new("RGB", (100, 100))
        assert isinstance(load_image(img), Image.Image)

    def test_load_from_bytes(self):
        from modules.ocr_engine import load_image
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="PNG")
        assert load_image(buf.getvalue()) is not None

    def test_load_from_path(self, tmp_path):
        from modules.ocr_engine import load_image
        from PIL import Image
        p = tmp_path / "img.png"
        Image.new("RGB", (100, 100)).save(p)
        assert load_image(str(p)) is not None

    @pytest.mark.skipif(
        not __import__("shutil").which("tesseract"),
        reason="Tesseract not installed"
    )
    def test_tesseract_smoke(self):
        from modules.ocr_engine import ocr_with_tesseract
        from PIL import Image, ImageDraw
        img  = Image.new("RGB", (300, 80), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Test OCR", fill="black")
        result = ocr_with_tesseract(img, lang="eng")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════
# 7. pdf_handler.py  (text routing — no PDF binary needed)
# ═══════════════════════════════════════════════════════════════

class TestPDFHandler:
    def test_txt_from_path(self, tmp_path):
        from modules.pdf_handler import extract_text_from_txt
        p = tmp_path / "test.txt"
        p.write_text("Hello from txt.", encoding="utf-8")
        assert "Hello" in extract_text_from_txt(str(p))

    def test_txt_from_bytes_io(self):
        from modules.pdf_handler import extract_text_from_txt
        content = "UTF-8 content: café"
        buf = io.BytesIO(content.encode("utf-8"))
        assert "café" in extract_text_from_txt(buf)

    def test_unsupported_ext_raises(self):
        from modules.pdf_handler import extract_document
        with pytest.raises(ValueError, match="Unsupported"):
            extract_document(io.BytesIO(b"x"), "doc.xlsx")

    def test_text_routing(self, tmp_path):
        from modules.pdf_handler import extract_document
        p = tmp_path / "doc.txt"
        p.write_text("AI is amazing.", encoding="utf-8")
        result = extract_document(str(p), "doc.txt")
        assert result["file_type"]    == "text"
        assert "AI"                   in result["text"]
        assert result["ocr_pages"]    == 0
        assert result["native_pages"] == 1


# ═══════════════════════════════════════════════════════════════
# 8. Integration
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    def test_qa_chunks_within_bert_budget(self):
        from modules.chunker import chunk_for_qa
        for i, c in enumerate(chunk_for_qa(LONG_TEXT)):
            assert len(c.split()) <= 420, f"Chunk {i} exceeds BERT budget"

    def test_translation_chunks_fit_marian(self):
        from modules.chunker import chunk_for_translation
        for c in chunk_for_translation(LONG_TEXT):
            assert len(c.split()) <= 100

    def test_clean_then_chunk(self):
        from modules.utils   import clean_text
        from modules.chunker import chunk_for_qa
        dirty   = "Hello\x00World.\x01 This is NLP.\r\nAnother sentence."
        cleaned = clean_text(dirty)
        chunks  = chunk_for_qa(cleaned)
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_doc_stats_after_extraction(self, tmp_path):
        from modules.pdf_handler import extract_document
        from modules.utils       import document_stats
        p = tmp_path / "sample.txt"
        p.write_text(ENGLISH_TEXT, encoding="utf-8")
        result = extract_document(str(p), "sample.txt")
        stats  = document_stats(result["text"])
        assert stats["words"] > 0
        assert stats["characters"] > stats["words"]
