"""Cross-Encoder Reranker integration using BAAI/bge-reranker-v2-m3."""
from __future__ import annotations

import logging
from typing import Sequence
from langchain_core.documents import Document
from app.core.config import settings

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODELS = {}


def get_cross_encoder(model_name: str | None = None):
    """Lazy-load and cache the CrossEncoder instance."""
    resolved_name = model_name or settings.reranker_model
    if resolved_name not in _CROSS_ENCODER_MODELS:
        from sentence_transformers import CrossEncoder
        logger.info("Initializing CrossEncoder: %s", resolved_name)
        _CROSS_ENCODER_MODELS[resolved_name] = CrossEncoder(resolved_name)
    return _CROSS_ENCODER_MODELS[resolved_name]


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_n: int = 5,
    model_name: str | None = None,
) -> list[Document]:
    """
    Reranks candidate documents with cross-encoder and returns top_n documents.
    """
    if not documents:
        return []
    if len(documents) <= 1:
        return list(documents)[:top_n]

    model = get_cross_encoder(model_name)
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)

    ranked: list[Document] = []
    for doc, score in zip(documents, scores):
        doc_copy = Document(
            page_content=doc.page_content,
            metadata=dict(doc.metadata),
        )
        doc_copy.metadata["rerank_score"] = float(score)
        ranked.append(doc_copy)

    ranked.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return ranked[:top_n]
