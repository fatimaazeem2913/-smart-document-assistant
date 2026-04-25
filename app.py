"""
app.py  —  Smart Document Assistant with OCR + NLP
===================================================
Run:   streamlit run app.py
"""

# ── Standard library ─────────────────────────────────────────────────────────
import io
from pathlib import Path

# ── Streamlit (must be imported before any st.* call) ────────────────────────
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Smart Document Assistant",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — dark editorial theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
  --bg-dark:    #0d0f14;
  --bg-card:    #161921;
  --bg-input:   #1e2230;
  --border:     #2c3145;
  --accent:     #4f8ef7;
  --accent2:    #34d399;
  --accent3:    #f59e0b;
  --text-main:  #e8eaf0;
  --text-muted: #8891a8;
  --danger:     #f87171;
  --success:    #34d399;
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--text-main);
}
.main { background: var(--bg-dark) !important; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Headings ── */
h1, h2, h3 {
  font-family: 'DM Serif Display', serif;
  color: var(--text-main);
  letter-spacing: -0.02em;
}
h1 { font-size: 2.1rem; line-height: 1.2; }
h2 { font-size: 1.45rem; }
h3 { font-size: 1.1rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--bg-card);
  border-radius: 10px;
  padding: 6px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 8px !important;
  color: var(--text-muted) !important;
  font-weight: 500;
  padding: 6px 18px;
  transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: white !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.92rem;
  padding: 0.55rem 1.4rem;
  transition: all 0.18s ease;
  width: 100%;
}
.stButton > button:hover {
  background: #3a7ce8;
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(79,142,247,0.35);
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-main) !important;
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div, .stSlider {
  color: var(--text-main) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--bg-card);
  border: 2px dashed var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
  transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 18px;
}
[data-testid="metric-container"] * { color: var(--text-main) !important; }

/* ── Expanders ── */
details { background: var(--bg-card) !important; border-radius: 8px; border: 1px solid var(--border) !important; }
summary  { color: var(--text-main) !important; }

/* ── Custom content cards ── */
.result-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  margin: 0.6rem 0;
  line-height: 1.7;
}
.result-card.green  { border-left: 4px solid var(--success); }
.result-card.amber  { border-left: 4px solid var(--accent3); }
.result-card.blue   { border-left: 4px solid var(--accent); }
.result-card.purple { border-left: 4px solid #a78bfa; }

.tag {
  display: inline-block;
  background: var(--bg-input);
  border: 1px solid var(--border);
  color: var(--text-muted);
  border-radius: 6px;
  padding: 2px 10px;
  font-size: 0.78rem;
  margin: 2px;
}

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent) !important; }

/* ── Info / Warning / Success banners ── */
.stAlert { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "doc_text":      "",
    "doc_stats":     {},
    "file_name":     "",
    "file_type":     "",
    "ocr_pages":     0,
    "native_pages":  0,
    "page_count":    0,
    "summary_result": None,
    "qa_result":      None,
    "trans_result":   None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
      <div style='font-size:2.8rem;'>🧠</div>
      <div style='font-family:DM Serif Display,serif; font-size:1.2rem; margin-top:4px;'>
        Smart Document<br>Assistant
      </div>
      <div style='color:#8891a8; font-size:0.78rem; margin-top:4px;'>
        OCR · NLP · Translation
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if st.session_state["file_name"]:
        st.markdown("### 📄 Loaded Document")
        st.markdown(
            f"<div class='result-card blue'>"
            f"<b>{st.session_state['file_name']}</b><br>"
            f"<span style='color:#8891a8; font-size:0.82rem;'>"
            f"{st.session_state['file_type'].upper()} file</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        s = st.session_state["doc_stats"]
        if s:
            c1, c2 = st.columns(2)
            c1.metric("Words",  f"{s.get('words', 0):,}")
            c2.metric("Chars",  f"{s.get('characters', 0):,}")
            c1.metric("Paras",  str(s.get('paragraphs', 0)))
            c2.metric("Tokens~", f"{s.get('estimated_tokens', 0):,}")
        if st.session_state.get("page_count", 0) > 0:
            pc = st.session_state["page_count"]
            oc = st.session_state["ocr_pages"]
            st.caption(
                f"📑 {pc} page(s) · "
                f"🔍 {oc} OCR'd · "
                f"📝 {pc-oc} native"
            )
    else:
        st.info("No document loaded.\nUse the **Extract** tab to begin.")

    st.divider()
    st.markdown("""
    <div style='color:#8891a8; font-size:0.8rem; line-height:1.8;'>
    <b>Models Available</b><br>
    📝 BART-large-CNN<br>
    📝 DistilBART / T5<br>
    ❓ RoBERTa-SQuAD2<br>
    🌐 Helsinki Opus-MT
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_extract, tab_summary, tab_qa, tab_trans, tab_about = st.tabs([
    "🔍 Extract",
    "📝 Summarize",
    "❓ Q&A",
    "🌐 Translate",
    "📖 About",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXTRACT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_extract:
    st.header("🔍 Document Text Extraction")
    st.caption("Upload a PDF, scanned image, or plain text file to begin.")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop your file here",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp", "txt"],
        help="PDF (native + OCR fallback) · Image (full OCR) · TXT",
    )

    if uploaded:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### ⚙️ Extraction Settings")

            from modules.utils import detect_file_type
            ftype = detect_file_type(uploaded.name)

            # OCR settings — only relevant for PDF / image
            if ftype in ("pdf", "image"):
                ocr_backend = st.radio(
                    "OCR Engine",
                    ["tesseract", "easyocr"],
                    index=0,
                    horizontal=True,
                    help=(
                        "**Tesseract** — fast, requires Tesseract binary installed.\n\n"
                        "**EasyOCR** — no binary needed, better on noisy images."
                    ),
                )
                if ocr_backend == "tesseract":
                    ocr_lang = st.selectbox(
                        "Tesseract Language",
                        ["eng+urd", "eng", "urd"],
                        help="Use 'eng+urd' for documents containing both languages.",
                    )
                else:
                    ocr_lang = "eng+urd"

                if ftype == "pdf":
                    force_ocr = st.checkbox(
                        "Force OCR on every page",
                        value=False,
                        help="By default, native text is used where available.",
                    )
                else:
                    force_ocr = True
            else:
                ocr_backend, ocr_lang, force_ocr = "tesseract", "eng+urd", False

            # Image preview
            if ftype == "image":
                st.markdown("#### 🖼️ Image Preview")
                from PIL import Image
                img = Image.open(uploaded)
                st.image(img, use_container_width=True)
                uploaded.seek(0)  # reset for OCR

            if st.button("🚀 Extract Text", type="primary"):
                prog = st.progress(0, text="Initializing…")

                def _progress(cur, tot):
                    pct  = int(cur / max(tot, 1) * 100)
                    prog.progress(pct / 100, text=f"Processing page {cur}/{tot}…")

                with st.spinner("Extracting text…"):
                    try:
                        from modules.pdf_handler import extract_document
                        from modules.utils      import document_stats

                        uploaded.seek(0)
                        result = extract_document(
                            uploaded,
                            file_name         = uploaded.name,
                            ocr_backend       = ocr_backend,
                            ocr_lang          = ocr_lang,
                            force_ocr         = force_ocr,
                            progress_callback = _progress,
                        )
                        text  = result["text"]
                        stats = document_stats(text)

                        # Persist in session state
                        st.session_state["doc_text"]     = text
                        st.session_state["doc_stats"]    = stats
                        st.session_state["file_name"]    = uploaded.name
                        st.session_state["file_type"]    = result["file_type"]
                        st.session_state["page_count"]   = result.get("page_count", 1)
                        st.session_state["ocr_pages"]    = result.get("ocr_pages", 0)
                        st.session_state["native_pages"] = result.get("native_pages", 1)
                        prog.progress(1.0, text="Done!")
                        st.success("✅ Text extracted successfully!")
                    except Exception as exc:
                        prog.empty()
                        st.error(f"Extraction failed: {exc}")

        with col_right:
            st.markdown("### 📋 Extracted Text")
            if st.session_state["doc_text"]:
                preview_text = st.session_state["doc_text"][:3000]
                suffix = "\n\n…*(truncated for preview)*" if len(st.session_state["doc_text"]) > 3000 else ""
                st.text_area("", value=preview_text + suffix, height=420, disabled=True)

                st.download_button(
                    "⬇️  Download Extracted Text",
                    data      = st.session_state["doc_text"],
                    file_name = Path(st.session_state["file_name"]).stem + "_extracted.txt",
                    mime      = "text/plain",
                )
            else:
                st.markdown(
                    "<div class='result-card' style='text-align:center; "
                    "color:#8891a8; padding:3rem;'>Upload a file and click "
                    "<b>Extract Text</b> to see results here.</div>",
                    unsafe_allow_html=True,
                )

    # ── Manual input fallback ─────────────────────────────────────────────────
    st.divider()
    st.subheader("✏️ Or Paste Text Directly")
    manual = st.text_area("Paste your text:", height=150,
                           placeholder="Paste any English or Urdu text…")
    if st.button("✅ Use This Text"):
        if manual.strip():
            from modules.utils import document_stats
            st.session_state["doc_text"]  = manual
            st.session_state["doc_stats"] = document_stats(manual)
            st.session_state["file_name"] = "manual_input.txt"
            st.session_state["file_type"] = "text"
            st.session_state["page_count"]   = 1
            st.session_state["ocr_pages"]    = 0
            st.session_state["native_pages"] = 1
            st.success("✅ Text loaded!")
        else:
            st.warning("Please paste some text first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARIZE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.header("📝 Document Summarization")

    if not st.session_state["doc_text"]:
        st.warning("⚠️ Load a document in the **Extract** tab first.")
    else:
        from modules.summarizer import SUMMARIZATION_MODELS

        col_cfg, col_out = st.columns([1, 2])

        with col_cfg:
            st.subheader("⚙️ Model Settings")
            model_key = st.selectbox("Model", list(SUMMARIZATION_MODELS.keys()))
            max_len   = st.slider("Max output tokens",  50, 350, 150, 10)
            min_len   = st.slider("Min output tokens",  10, 100,  40,  5)
            show_chunks = st.checkbox("Show per-chunk summaries")

            if st.button("🚀 Summarize", type="primary"):
                prog = st.progress(0, text="Loading model…")

                def _sum_prog(cur, tot):
                    prog.progress(cur / max(tot, 1),
                                  text=f"Summarizing chunk {cur}/{tot}…")

                with st.spinner("Generating summary…"):
                    try:
                        from modules.summarizer import summarize_document
                        res = summarize_document(
                            st.session_state["doc_text"],
                            model_key         = model_key,
                            max_length        = max_len,
                            min_length        = min_len,
                            progress_callback = _sum_prog,
                        )
                        st.session_state["summary_result"] = res
                        prog.progress(1.0, text="Done!")
                    except Exception as e:
                        prog.empty()
                        st.error(f"Summarization error: {e}")

        with col_out:
            st.subheader("📋 Summary")
            if st.session_state["summary_result"]:
                r = st.session_state["summary_result"]
                st.markdown(
                    f"<div class='result-card green'>{r['final_summary']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span class='tag'>Model: {r['model'].split('/')[-1]}</span>"
                    f"<span class='tag'>{r['chunk_count']} chunk(s)</span>"
                    f"<span class='tag'>{'2-pass ✓' if r['two_pass'] else '1-pass'}</span>",
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇️  Download Summary",
                    data=r["final_summary"],
                    file_name="summary.txt", mime="text/plain",
                )
                if show_chunks and len(r["chunk_summaries"]) > 1:
                    st.subheader("🧩 Per-Chunk Summaries")
                    for i, cs in enumerate(r["chunk_summaries"], 1):
                        with st.expander(f"Chunk {i} of {len(r['chunk_summaries'])}"):
                            st.write(cs)
            else:
                st.markdown(
                    "<div class='result-card' style='color:#8891a8; text-align:center;"
                    " padding:2.5rem;'>Configure settings and click <b>Summarize</b>.</div>",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUESTION ANSWERING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.header("❓ Question Answering")

    if not st.session_state["doc_text"]:
        st.warning("⚠️ Load a document in the **Extract** tab first.")
    else:
        from modules.qa_engine import QA_MODELS

        col_q, col_a = st.columns([1, 1])

        with col_q:
            st.subheader("💬 Ask a Question")
            qa_model   = st.selectbox("QA Model", list(QA_MODELS.keys()))
            question   = st.text_area(
                "Your question:",
                height=130,
                placeholder="e.g. What is the main topic of this document?",
            )
            show_cands = st.checkbox("Show top-3 candidate answers", value=False)

            if st.button("🔍 Find Answer", type="primary"):
                if not question.strip():
                    st.error("Please enter a question.")
                else:
                    prog = st.progress(0, text="Searching…")

                    def _qa_prog(cur, tot):
                        prog.progress(cur / max(tot, 1),
                                      text=f"Scanning chunk {cur}/{tot}…")

                    with st.spinner("Running QA model…"):
                        try:
                            from modules.qa_engine import answer_question
                            qa_res = answer_question(
                                question,
                                st.session_state["doc_text"],
                                model_key         = qa_model,
                                progress_callback = _qa_prog,
                            )
                            st.session_state["qa_result"] = qa_res
                            prog.progress(1.0, text="Done!")
                        except Exception as e:
                            prog.empty()
                            st.error(f"QA error: {e}")

        with col_a:
            st.subheader("💡 Answer")
            if st.session_state["qa_result"]:
                qa = st.session_state["qa_result"]
                st.markdown(
                    f"<div class='result-card amber'>"
                    f"<div style='font-size:1.1rem; font-weight:600; margin-bottom:8px;'>"
                    f"{qa['answer']}</div>"
                    f"<span class='tag'>{qa['confidence_label']}</span>"
                    f"<span class='tag'>Score: {qa['score']:.1%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                with st.expander("📖 Source passage"):
                    st.markdown(
                        f"<div style='color:#8891a8; font-size:0.9rem; "
                        f"line-height:1.7;'>{qa['context_snippet']}…</div>",
                        unsafe_allow_html=True,
                    )

                if show_cands and len(qa.get("all_candidates", [])) > 1:
                    st.subheader("🔬 Top Candidates")
                    for i, c in enumerate(qa["all_candidates"][:3], 1):
                        st.markdown(
                            f"<div class='result-card' style='margin-bottom:6px;'>"
                            f"<b>#{i}</b>&nbsp; {c['answer']}&nbsp;"
                            f"<span class='tag'>{c['score']:.1%}</span></div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    "<div class='result-card' style='color:#8891a8; text-align:center;"
                    " padding:2.5rem;'>Enter a question and click <b>Find Answer</b>.</div>",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRANSLATE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trans:
    st.header("🌐 English ↔ Urdu Translation")

    from modules.translator import TRANSLATION_DIRECTIONS, suggest_direction

    col_src, col_tgt = st.columns(2)

    with col_src:
        st.subheader("📥 Source")

        # Auto-detect direction if doc is loaded
        default_dir = "English → Urdu"
        if st.session_state["doc_text"]:
            suggested = suggest_direction(st.session_state["doc_text"][:500])
            if suggested:
                default_dir = suggested

        direction = st.radio(
            "Translation direction",
            list(TRANSLATION_DIRECTIONS.keys()),
            index=list(TRANSLATION_DIRECTIONS.keys()).index(default_dir),
            horizontal=True,
        )

        use_doc = st.checkbox(
            "Use loaded document text",
            disabled=not bool(st.session_state["doc_text"]),
        )

        if use_doc and st.session_state["doc_text"]:
            src_text = st.text_area(
                "Text (from document — editable):",
                value  = st.session_state["doc_text"][:2000],
                height = 320,
            )
        else:
            src_text = st.text_area(
                "Enter text to translate:",
                height=320,
                placeholder="Type or paste English or Urdu text here…",
            )

        if st.button("🌐 Translate", type="primary"):
            if not src_text.strip():
                st.error("Please enter text to translate.")
            else:
                prog = st.progress(0, text="Loading translation model…")

                def _trans_prog(cur, tot):
                    prog.progress(cur / max(tot, 1),
                                  text=f"Translating chunk {cur}/{tot}…")

                with st.spinner("Translating…"):
                    try:
                        from modules.translator import translate_text
                        res = translate_text(
                            src_text,
                            direction         = direction,
                            progress_callback = _trans_prog,
                        )
                        st.session_state["trans_result"] = res
                        prog.progress(1.0, text="Done!")
                    except Exception as e:
                        prog.empty()
                        st.error(f"Translation error: {e}")

    with col_tgt:
        st.subheader("📤 Translation")
        if st.session_state["trans_result"]:
            tr = st.session_state["trans_result"]
            st.markdown(
                f"<div class='result-card purple' style='font-family:\"Noto Nastaliq Urdu\","
                f"\"Segoe UI\",sans-serif; font-size:1.05rem; line-height:2;'>"
                f"{tr['translated_text']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='tag'>Model: {tr['model'].split('/')[-1]}</span>"
                f"<span class='tag'>{tr['src_code'].upper()} → {tr['tgt_code'].upper()}</span>"
                f"<span class='tag'>{tr['chunk_count']} chunk(s)</span>",
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️  Download Translation",
                data      = tr["translated_text"],
                file_name = "translation.txt",
                mime      = "text/plain",
            )
        else:
            st.markdown(
                "<div class='result-card' style='color:#8891a8; text-align:center;"
                " padding:2.5rem;'>Enter text and click <b>Translate</b>.</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.header("📖 About Smart Document Assistant")

    st.markdown("""
<div class="result-card blue">
<h3 style="margin-top:0;">What is this?</h3>
A complete NLP pipeline that processes <b>native PDFs, scanned documents, and images</b>
using OCR — then applies state-of-the-art language models for summarization,
question answering, and English↔Urdu translation.
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🗂️ Project Structure
```
smart_doc_ocr/
│
├── app.py                  ← Streamlit UI
├── requirements.txt
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── utils.py            ← Shared helpers + model cache
│   ├── ocr_engine.py       ← Tesseract / EasyOCR backends
│   ├── pdf_handler.py      ← PDF + image extraction router
│   ├── chunker.py          ← Overlapping text splitter
│   ├── summarizer.py       ← BART / T5 summarization
│   ├── qa_engine.py        ← BERT / RoBERTa QA
│   └── translator.py       ← Helsinki Opus-MT translation
│
└── tests/
    └── test_cases.py       ← pytest unit tests
```
""")

    with c2:
        st.markdown("""
### 🤖 Models
| Task | Model | Size |
|------|-------|------|
| Summarization | `facebook/bart-large-cnn` | 1.6 GB |
| Summarization | `sshleifer/distilbart-cnn-12-6` | 1.2 GB |
| Summarization | `t5-small / t5-base` | 240 / 900 MB |
| QA | `deepset/roberta-base-squad2` | 500 MB |
| QA | `deepset/bert-base-cased-squad2` | 430 MB |
| Translation | `Helsinki-NLP/opus-mt-en-ur` | 300 MB |
| Translation | `Helsinki-NLP/opus-mt-ur-en` | 300 MB |

### ⚡ Quick Start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
""")

    st.markdown("""
### 🧠 Architecture Overview
```
Upload ──► pdf_handler ──► chunker ──► summarizer  →  Summary
                │                 └──► qa_engine    →  Answer
                └── ocr_engine                      
                     (tesseract│                    
                      easyocr)  └──── translator    →  Translation
```
""")
