# src/cli_index.py
from pathlib import Path
from text_store import index_pdf_text
from clip_store import index_pdf_figures

def index_all_pdfs(pdf_dir: str = "data/pdfs/TDT4215"):
    pdf_dir = Path(pdf_dir)
    for pdf_path in pdf_dir.glob("*.pdf"):
        doc_id = pdf_path.stem
        print(f"Indexing text for {pdf_path}...")
        index_pdf_text(str(pdf_path), doc_id)
        print(f"Indexing figures for {pdf_path}...")
        index_pdf_figures(str(pdf_path), doc_id)

if __name__ == "__main__":
    index_all_pdfs()
