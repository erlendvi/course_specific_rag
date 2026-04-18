# src/pdf_utils.py
from pypdf import PdfReader
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io

# --- Tekst fra PDF ---

def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text() or ""
        text += t + "\n"
    return text

def chunk_text(text: str, max_chars=2000, overlap=200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# --- Bilder (figurer) fra PDF ---

def extract_images_from_pdf(pdf_path: str, output_dir: str = "data/images"):
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    images_info = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            filename = f"{pdf_path.stem}_page{page_index+1}_img{img_index+1}.{image_ext}"
            filepath = out_dir / filename

            image.save(filepath)

            images_info.append({
                "pdf": pdf_path.name,
                "page": page_index + 1,
                "image_index": img_index + 1,
                "path": str(filepath),
            })

    return images_info