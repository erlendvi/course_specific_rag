import hashlib
import re
from pathlib import Path
from typing import Dict, List

import streamlit as st

from config import OPENAI_API_KEY, client
from ocr_store import index_pdf_ocr_text
from text_store import (
    count_chunks_by_source,
    delete_chunks_by_source,
    index_pdf_text,
    retrieve_relevant_chunk_records,
)

APP_TITLE = "Course-Specific RAG Lab"
UPLOAD_DIR = Path("data/uploads")
PDF_DIR = Path("data/pdfs")


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return normalized or "document"


def compute_doc_id(path: Path) -> str:
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{slugify(path.stem)}-{digest}"


def list_local_pdfs() -> List[Path]:
    candidates = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    uploads = sorted(UPLOAD_DIR.glob("*.pdf")) if UPLOAD_DIR.exists() else []
    seen = set()
    ordered_paths = []
    for path in candidates + uploads:
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        ordered_paths.append(path)
    return ordered_paths


def save_uploaded_pdf(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def build_context(records: List[Dict]) -> str:
    context_parts = []
    for record in records:
        metadata = record["metadata"]
        header = (
            f"Source type: {metadata.get('type', 'unknown')}, "
            f"page: {metadata.get('page', 'n/a')}"
        )
        context_parts.append(header + "\n" + record["document"])
    return "\n\n---\n\n".join(context_parts)


def answer_question(question: str, records: List[Dict]) -> str:
    prompt = f"""You are helping evaluate a course-specific RAG pipeline.
Use only the retrieved context below.
If the answer is incomplete, say what is uncertain.
Answer in English.

QUESTION:
{question}

RETRIEVED CONTEXT:
{build_context(records)}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def render_retrieved_chunks(records: List[Dict]):
    st.subheader("Retrieved Chunks")
    for index, record in enumerate(records, start=1):
        metadata = record["metadata"]
        label = (
            f"{index}. {metadata.get('type', 'unknown')} "
            f"· page {metadata.get('page', 'n/a')}"
        )
        with st.expander(label):
            st.caption(f"Chunk index: {metadata.get('chunk_index', 'n/a')}")
            st.write(record["document"])


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(239, 196, 123, 0.18), transparent 28rem),
                linear-gradient(180deg, #f7f1e6 0%, #f4eee2 100%);
            color: #1e1d1a;
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            background: rgba(255, 252, 245, 0.8);
            border: 1px solid rgba(76, 60, 33, 0.12);
            box-shadow: 0 16px 32px rgba(76, 60, 33, 0.08);
            margin-bottom: 1.2rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.55rem 0 0 0;
            font-size: 1rem;
            color: #5b5347;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="wide",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Course-Specific RAG Lab</h1>
            <p>Test PDF extraction, OCR indexing, retrieval, and question answering on your own lecture slides.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY mangler i `.env`.")
        st.stop()

    with st.sidebar:
        st.header("Document")
        source_mode = st.radio(
            "Choose input",
            options=["Use local PDF", "Upload PDF"],
            index=0,
        )

        selected_path = None

        if source_mode == "Use local PDF":
            local_pdfs = list_local_pdfs()
            if not local_pdfs:
                st.warning("Ingen lokale PDF-er funnet i `data/pdfs/` eller `data/uploads/`.")
            else:
                selected_path = Path(
                    st.selectbox(
                        "Available PDFs",
                        options=[str(path) for path in local_pdfs],
                    )
                )
        else:
            uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
            if uploaded_file is not None:
                selected_path = save_uploaded_pdf(uploaded_file)
                st.success(f"Saved to `{selected_path}`")

        replace_existing = st.checkbox("Re-index document if it already exists", value=True)
        retrieve_k = st.slider("Retrieved chunks", min_value=3, max_value=10, value=6)

    if selected_path is None:
        st.info("Choose or upload a PDF to begin.")
        st.stop()

    doc_id = compute_doc_id(selected_path)
    existing_count = count_chunks_by_source(doc_id)

    info_col, action_col = st.columns([1.6, 1.0], gap="large")

    with info_col:
        st.subheader("Selected Document")
        st.write(f"`{selected_path}`")
        st.caption(f"Document ID: `{doc_id}`")
        st.caption(f"Indexed chunks currently stored: `{existing_count}`")

    with action_col:
        st.subheader("Index")
        if st.button("Index Text + OCR", use_container_width=True):
            st.session_state["active_doc_id"] = doc_id
            st.session_state["active_pdf_path"] = str(selected_path)
            if existing_count and not replace_existing:
                st.info("Reusing the existing index for this document.")
            else:
                with st.spinner("Extracting text, running OCR, and writing embeddings..."):
                    if replace_existing:
                        delete_chunks_by_source(doc_id)

                    text_chunks = index_pdf_text(str(selected_path), doc_id)
                    try:
                        ocr_chunks = index_pdf_ocr_text(str(selected_path), doc_id)
                        ocr_warning = None
                    except RuntimeError as exc:
                        ocr_chunks = 0
                        ocr_warning = str(exc)

                st.success(
                    f"Indexed `{text_chunks}` PDF-text chunks and `{ocr_chunks}` OCR chunks."
                )
                if ocr_warning:
                    st.warning(ocr_warning)

    question = st.text_area(
        "Ask a question about the selected PDF",
        value="What helper algorithms are described here, especially the local and large neighborhood methods?",
        height=120,
    )

    if st.button("Ask Question", type="primary", use_container_width=True):
        active_doc_id = st.session_state.get("active_doc_id", doc_id)
        if count_chunks_by_source(active_doc_id) == 0:
            st.warning("Index the document first.")
            st.stop()

        with st.spinner("Retrieving context and generating answer..."):
            records = retrieve_relevant_chunk_records(
                question,
                k=retrieve_k,
                source=active_doc_id,
            )
            if not records:
                st.warning("No chunks were retrieved for this document.")
                st.stop()
            answer = answer_question(question, records)

        st.subheader("Answer")
        st.write(answer)
        render_retrieved_chunks(records)


if __name__ == "__main__":
    main()
