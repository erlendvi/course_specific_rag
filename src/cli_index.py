# src/cli_index.py
from pathlib import Path
from text_store import index_pdf_text
from clip_store import index_pdf_figures
from ocr_store import index_pdf_ocr_text

def index_all_pdfs(pdf_dir: str = "data/pdfs/TDT4215"):
    pdf_dir = Path(pdf_dir)
    for pdf_path in pdf_dir.glob("*.pdf"):
        doc_id = pdf_path.stem
        print(f"Indexing text for {pdf_path}...")
        index_pdf_text(str(pdf_path), doc_id)
        print(f"Indexing figures for {pdf_path}...")
        index_pdf_figures(str(pdf_path), doc_id)
        print(f"Indexing OCR text for {pdf_path} when pages look image-based...")
        try:
            chunk_count = index_pdf_ocr_text(str(pdf_path), doc_id)
            print(f"Indexed {chunk_count} OCR chunk(s) for {pdf_path}.")
        except RuntimeError as exc:
            print(f"Skipping OCR for {pdf_path}: {exc}")

if __name__ == "__main__":
    index_all_pdfs()
