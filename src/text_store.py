# src/text_store.py
from typing import Any, Dict, List, Optional
from config import client, chroma_client, TEXT_COLLECTION_NAME
from pdf_utils import extract_pdf_text, chunk_text

text_collection = chroma_client.get_or_create_collection(TEXT_COLLECTION_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in response.data]

def add_text_chunks(chunks: List[str], ids: List[str], metadatas: List[Dict[str, Any]]):
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    text_collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def count_chunks_by_source(source: str) -> int:
    existing = text_collection.get(where={"source": source})
    return len(existing["ids"])


def delete_chunks_by_source(source: str) -> int:
    deleted_count = count_chunks_by_source(source)
    if deleted_count:
        text_collection.delete(where={"source": source})
    return deleted_count

def index_pdf_text(path: str, doc_id_prefix: str):
    text = extract_pdf_text(path)
    chunks = chunk_text(text)

    ids = [f"{doc_id_prefix}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": doc_id_prefix, "chunk_index": i, "type": "pdf_text"}
        for i in range(len(chunks))
    ]

    return add_text_chunks(
        chunks=chunks,
        ids=ids,
        metadatas=metadatas,
    )


def retrieve_relevant_chunk_records(
    query: str,
    k: int = 5,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query_embedding = embed_texts([query])[0]
    query_args: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": k,
    }
    if source is not None:
        query_args["where"] = {"source": source}

    results = text_collection.query(
        **query_args,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    records = []
    for index, (doc_id, document, metadata) in enumerate(zip(ids, documents, metadatas)):
        records.append({
            "id": doc_id,
            "document": document,
            "metadata": metadata,
            "distance": distances[index] if index < len(distances) else None,
        })
    return records


def retrieve_relevant_chunks(query: str, k: int = 5) -> List[str]:
    records = retrieve_relevant_chunk_records(query, k=k)
    return [record["document"] for record in records]
