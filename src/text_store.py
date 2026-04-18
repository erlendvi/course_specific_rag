# src/text_store.py
from typing import List
from config import client, chroma_client, TEXT_COLLECTION_NAME
from pdf_utils import extract_pdf_text, chunk_text

text_collection = chroma_client.get_or_create_collection(TEXT_COLLECTION_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in response.data]

def index_pdf_text(path: str, doc_id_prefix: str):
    text = extract_pdf_text(path)
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)
    ids = [f"{doc_id_prefix}_chunk_{i}" for i in range(len(chunks))]

    text_collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": doc_id_prefix, "chunk_index": i} for i in range(len(chunks))]
    )

def retrieve_relevant_chunks(query: str, k: int = 5) -> List[str]:
    query_embedding = embed_texts([query])[0]
    results = text_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    return results["documents"][0]  # liste av tekstbiter
