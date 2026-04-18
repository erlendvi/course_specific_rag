# src/clip_store.py
from typing import List, Dict, Any

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

from config import chroma_client, FIGURE_COLLECTION_NAME
from pdf_utils import extract_images_from_pdf

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_collection = chroma_client.get_or_create_collection(FIGURE_COLLECTION_NAME)

def embed_image_clip(image: Image.Image) -> List[float]:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].cpu().tolist()

def embed_text_clip(text: str) -> List[float]:
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].cpu().tolist()

def index_pdf_figures(pdf_path: str, doc_id_prefix: str):
    infos = extract_images_from_pdf(pdf_path)
    ids, embeddings, docs, metas = [], [], [], []

    for info in infos:
        img = Image.open(info["path"]).convert("RGB")
        vec = embed_image_clip(img)

        fig_id = f"{doc_id_prefix}_p{info['page']}_img{info['image_index']}"

        ids.append(fig_id)
        embeddings.append(vec)
        docs.append(f"Figure from {info['pdf']} page {info['page']} (image {info['image_index']})")
        metas.append({
            "source": doc_id_prefix,
            "pdf": info["pdf"],
            "page": info["page"],
            "image_index": info["image_index"],
            "path": info["path"],
            "type": "figure",
        })

    if ids:
        clip_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metas,
        )

def retrieve_relevant_figures(query: str, k: int = 3) -> Dict[str, Any]:
    q_vec = embed_text_clip(query)
    results = clip_collection.query(
        query_embeddings=[q_vec],
        n_results=k,
    )
    return {
        "ids": results["ids"][0],
        "metadatas": results["metadatas"][0],
        "documents": results["documents"][0],
    }
