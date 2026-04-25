# 🧠 Smart Document Assistant — OCR + NLP

A production-grade NLP pipeline that reads **native PDFs, scanned documents,
and photos**, then summarizes, answers questions, and translates the content.

---

## ✨ Feature Matrix

| Feature | Technology | Notes |
|---------|-----------|-------|
| PDF text extraction | PyMuPDF (fitz) | Preserves page structure |
| OCR — scanned PDFs + images | pytesseract / EasyOCR | Auto-fallback per page |
| Abstractive summarization | BART-large / DistilBART / T5 | Chunk-and-merge for long docs |
| Extractive QA | RoBERTa / BERT SQuAD2 | Best answer from all chunks |
| EN ↔ UR translation | Helsinki-NLP Opus-MT | Sentence-level chunking |
| Web UI | Streamlit | Dark theme, progress bars |

---

## 🗂️ Project Structure

```
smart_doc_ocr/
│
├── app.py                    ← Streamlit UI (5 tabs, dark theme)
├── requirements.txt          ← All Python + system dependencies
├── README.md                 ← This file
│
├── modules/
│   ├── __init__.py           ← Package marker
│   ├── utils.py              ← Shared: clean, stats, model cache, labels
│   ├── ocr_engine.py         ← Tesseract + EasyOCR backends + preprocessing
│   ├── pdf_handler.py        ← PDF/image/txt routing with per-page OCR fallback
│   ├── chunker.py            ← Sentence-aware overlapping text splitter
│   ├── summarizer.py         ← BART / T5 abstractive summarization
│   ├── qa_engine.py          ← BERT / RoBERTa extractive QA
│   └── translator.py         ← Helsinki Opus-MT EN ↔ UR translation
│
└── tests/
    └── test_cases.py         ← pytest suite (offline, no GPU, 55+ tests)
```

---

## ⚙️ Step-by-Step Setup Guide

### Step 1 — Python Environment

```bash
# Requires Python 3.9+
python --version

python -m venv venv
source venv/bin/activate        # Linux / macOS
# Windows: venv\Scripts\activate
```

### Step 2 — Install Python Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **First run:** HuggingFace auto-downloads model weights to
> `~/.cache/huggingface/` — one-time download of 300 MB – 1.6 GB per model.

### Step 3 — Install Tesseract Binary

Tesseract is required for the **pytesseract** OCR backend.
(EasyOCR needs no system binary — use it as fallback if Tesseract is unavailable.)

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-urd
# Verify:
tesseract --version
```

**macOS (Homebrew)**
```bash
brew install tesseract
brew install tesseract-lang   # includes Urdu
```

**Windows**
1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. During install → check **Additional script data → Arabic** (covers Urdu)
3. Add `C:\Program Files\Tesseract-OCR` to your system PATH
4. Verify: `tesseract --version` in a new terminal

### Step 4 — Run the Application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Step 5 — Run Tests

```bash
pytest tests/test_cases.py -v
```

All 55+ tests pass offline with no GPU and no model downloads.

---

## 🔁 Architecture: Full Pipeline

```
Upload (PDF / Image / TXT)
         │
         ▼
  ┌─────────────────┐
  │  pdf_handler.py │  — detects file type, routes to extractor
  └────────┬────────┘
           │
     ┌─────▼───────────────────────────────────────┐
     │  Per-page extraction strategy:              │
     │  • Native text (≥50 chars)  → use directly  │
     │  • Scanned / image page     → render + OCR  │
     └─────┬────────────────────────────┬──────────┘
           │  native text              │  rendered image
           │                    ┌──────▼──────────────┐
           │                    │   ocr_engine.py      │
           │                    │  ┌─────────────────┐ │
           │                    │  │  preprocess:    │ │
           │                    │  │  upscale →      │ │
           │                    │  │  contrast →     │ │
           │                    │  │  greyscale →    │ │
           │                    │  │  threshold      │ │
           │                    │  └────────┬────────┘ │
           │                    │  tesseract│easyocr   │
           │                    └──────┬────────────────┘
           │                           │
           └───────────────────────────┘
                        │ clean text (utils.clean_text)
                        ▼
               ┌────────────────┐
               │  chunker.py    │  sentence-aware overlapping windows
               └───┬────────┬───┘
                   │        │
       ┌───────────▼┐  ┌────▼──────────┐  ┌────────────────┐
       │summarizer  │  │  qa_engine    │  │  translator    │
       │BART / T5   │  │  RoBERTa/BERT │  │  Helsinki MT   │
       │chunk→merge │  │  best score   │  │  EN ↔ UR       │
       └────────────┘  └───────────────┘  └────────────────┘
```

---

## 🧪 Sample Inputs & Expected Outputs

### 📝 Summarization

**Input** (500-word AI article excerpt):
```
Artificial intelligence (AI) is intelligence demonstrated by machines.
AI research has been defined as the field of study of intelligent agents.
Applications include speech recognition, recommendation systems,
self-driving cars, computer vision, and language translation...
```

**Output (BART-large-CNN):**
```
Artificial intelligence refers to machine-demonstrated intelligence that
mimics human cognitive functions. AI research aims to build rational agents
that perceive their environment and act to maximize their objectives.
Key applications span speech recognition, autonomous vehicles, and NLP.
```
*Chunks processed: 1 · Model: bart-large-cnn · Time: ~4 s (CPU)*

---

### ❓ Question Answering

**Document:** Climate change research paper (8 pages, ~4,200 words)

| Question | Answer | Confidence |
|----------|--------|-----------|
| What is the primary cause of climate change? | The burning of fossil fuels | ✅ 91% |
| When was the Paris Agreement signed? | 2015 | ✅ 87% |
| What temperature rise is considered dangerous? | 1.5°C above pre-industrial levels | 🟡 64% |
| Who funded this research? | *(not in document)* | ❌ 4% |

---

### 🌐 Translation

**English → Urdu:**
```
Input:  Artificial intelligence is transforming the world.
Output: مصنوعی ذہانت دنیا کو تبدیل کر رہی ہے۔
```

**Urdu → English:**
```
Input:  یہ ایک کمپیوٹر سائنس کا مضمون ہے۔
Output: This is a computer science subject.
```

---

### 🔍 OCR Extraction

| Input | Backend | Characters | Quality |
|-------|---------|-----------|---------|
| Clear printed page (300 DPI) | tesseract | 1,840 | ✅ Good |
| Handwritten notes | easyocr | 620 | 🟡 Partial |
| Low-resolution photo (72 DPI) | tesseract | 95 | ❌ Low |
| Scanned Urdu newspaper | tesseract (eng+urd) | 1,120 | ✅ Good |

**Pre-processing pipeline applied before OCR:**
1. Convert to RGB
2. Upscale if shortest dimension < 1000 px
3. Boost contrast × 1.5
4. Convert to greyscale
5. OpenCV adaptive threshold (Gaussian, block=31)

---

## 🤖 Models Reference

### Summarization
| Display Name | Model ID | Size | Speed |
|-------------|---------|------|-------|
| BART-Large CNN | `facebook/bart-large-cnn` | 1.6 GB | Slow |
| DistilBART | `sshleifer/distilbart-cnn-12-6` | 1.2 GB | Medium |
| T5-Base | `t5-base` | 900 MB | Medium |
| T5-Small | `t5-small` | 240 MB | Fast |

### Question Answering
| Display Name | Model ID | Size | F1 (SQuAD2) |
|-------------|---------|------|------------|
| RoBERTa-Base | `deepset/roberta-base-squad2` | 500 MB | 83.0 |
| BERT-Base | `deepset/bert-base-cased-squad2` | 430 MB | 75.1 |
| MiniLM | `deepset/minilm-uncased-squad2` | 120 MB | 76.1 |

### Translation
| Direction | Model ID | Size |
|-----------|---------|------|
| EN → UR | `Helsinki-NLP/opus-mt-en-ur` | ~300 MB |
| UR → EN | `Helsinki-NLP/opus-mt-ur-en` | ~300 MB |

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|---------|
| `TesseractNotFoundError` | Install Tesseract binary; add to PATH |
| `ModuleNotFoundError: fitz` | `pip install PyMuPDF` |
| `ModuleNotFoundError: easyocr` | `pip install easyocr` |
| `ModuleNotFoundError: sentencepiece` | `pip install sentencepiece` |
| Slow first run | One-time model download; subsequent runs use cache |
| PDF shows no text | Tick **Force OCR on every page** in the Extract tab |
| Poor Urdu OCR | Ensure `tesseract-ocr-urd` language pack is installed |
| Out of memory | Use T5-Small or MiniLM models; they require < 1 GB RAM |
| Garbled translation | Input may be mixed-language; split and translate separately |

---

## 📄 License

MIT — free for academic and commercial use.
