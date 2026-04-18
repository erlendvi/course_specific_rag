import os
from pathlib import Path
from typing import Any, Dict, List

from pdf_utils import extract_pdf_text_by_page, render_pdf_pages_to_images, chunk_text

MIN_EXISTING_TEXT_CHARS = 40
MIN_OCR_CONFIDENCE = 0.5

_ocr_engine = None


def _ensure_local_paddle_cache():
    cache_dir = Path(".paddlex_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    return cache_dir


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    _ensure_local_paddle_cache()

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install PaddlePaddle first, then install "
            "`paddleocr` to enable OCR indexing."
        ) from exc

    _ocr_engine = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    return _ocr_engine


def _extract_lines_from_prediction(prediction: Any) -> List[Dict[str, Any]]:
    result_json = getattr(prediction, "json", None)
    if not isinstance(result_json, dict):
        return []

    res = result_json.get("res", {})
    texts = res.get("rec_texts", []) or []
    scores = res.get("rec_scores", []) or []
    boxes = res.get("rec_boxes", []) or []

    lines = []
    for index, text in enumerate(texts):
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            continue

        score = float(scores[index]) if index < len(scores) else None
        box = boxes[index] if index < len(boxes) else None

        lines.append({
            "text": cleaned_text,
            "score": score,
            "box": box,
        })

    return lines


def ocr_image(image_path: str, min_confidence: float = MIN_OCR_CONFIDENCE) -> List[Dict[str, Any]]:
    ocr_engine = _get_ocr_engine()
    predictions = ocr_engine.predict(image_path)
    if not predictions:
        return []

    lines = _extract_lines_from_prediction(predictions[0])
    return [
        line for line in lines
        if line["score"] is None or line["score"] >= min_confidence
    ]


def index_pdf_ocr_text(
    pdf_path: str,
    doc_id_prefix: str,
    min_existing_text_chars: int = MIN_EXISTING_TEXT_CHARS,
):
    from text_store import add_text_chunks

    page_texts = extract_pdf_text_by_page(pdf_path)
    candidate_pages = {
        page_number
        for page_number, text in enumerate(page_texts, start=1)
        if len(text) < min_existing_text_chars
    }

    if not candidate_pages:
        return 0

    rendered_pages = render_pdf_pages_to_images(pdf_path, page_numbers=candidate_pages)

    indexed_chunks = []
    indexed_ids = []
    indexed_metadatas = []

    for page_info in rendered_pages:
        ocr_lines = ocr_image(page_info["path"])
        if not ocr_lines:
            continue

        page_text = "\n".join(line["text"] for line in ocr_lines)
        chunks = chunk_text(page_text, max_chars=1500, overlap=150)

        for chunk_index, chunk in enumerate(chunks):
            indexed_chunks.append(chunk)
            indexed_ids.append(
                f"{doc_id_prefix}_ocr_p{page_info['page']}_chunk_{chunk_index}"
            )
            indexed_metadatas.append({
                "source": doc_id_prefix,
                "chunk_index": chunk_index,
                "page": page_info["page"],
                "path": page_info["path"],
                "type": "ocr_text",
            })

    add_text_chunks(indexed_chunks, indexed_ids, indexed_metadatas)
    return len(indexed_chunks)
