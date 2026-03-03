from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_chroma import Chroma

from config import get_settings
from embeddings import create_embedding_function


@dataclass
class RetrievalResult:
    context_text: str
    sources: list[str]
    scores: list[float]


def retrieve_context(query_text: str) -> RetrievalResult:
    settings = get_settings()
    if not os.path.exists(settings.chroma_path):
        raise FileNotFoundError(
            f"Vector database not found at '{settings.chroma_path}'. Run create_database.py first."
        )

    db = Chroma(
        persist_directory=settings.chroma_path,
        embedding_function=create_embedding_function(),
    )
    raw_results = db.similarity_search_with_relevance_scores(
        query_text,
        k=settings.retrieval_top_k,
    )

    filtered_results = [
        (doc, score)
        for doc, score in raw_results
        if score is None or score >= settings.retrieval_score_threshold
    ]

    context_documents = filtered_results
    if not context_documents and raw_results:
        context_documents = raw_results[:1]

    context_text = "\n\n---\n\n".join(doc.page_content for doc, _score in context_documents)
    sources = _dedupe_sources(
        [doc.metadata.get("source") for doc, _score in context_documents if doc.metadata.get("source")]
    )
    scores = [float(score) for _doc, score in context_documents if score is not None]

    return RetrievalResult(context_text=context_text, sources=sources, scores=scores)


def _dedupe_sources(sources: list[str]) -> list[str]:
    return list(dict.fromkeys(sources))
