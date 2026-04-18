# src/rag.py
from typing import List
from config import client
from text_store import retrieve_relevant_chunks
from clip_store import retrieve_relevant_figures

def answer_with_rag(question: str) -> str:
    # 1. Tekstkontekst
    text_chunks: List[str] = retrieve_relevant_chunks(question, k=5)
    text_context = "\n\n---\n\n".join(text_chunks)

    # 2. Figur-kontekst
    figures = retrieve_relevant_figures(question, k=3)
    figure_lines = []
    for meta, doc in zip(figures["metadatas"], figures["documents"]):
        figure_lines.append(
            f"- Possible relevant figure from {meta['pdf']} page {meta['page']} (image {meta['image_index']}): {doc}"
        )
    figure_context = "\n".join(figure_lines) if figure_lines else "No relevant figures found."

    prompt = f"""
You are a helpful assistant. Use both the text context and any relevant figures.

TEXT CONTEXT:
{text_context}

FIGURE CONTEXT:
{figure_context}

QUESTION:
{question}

If figures seem relevant, mention them (e.g. "In one of the figures...").
Answer in English.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
