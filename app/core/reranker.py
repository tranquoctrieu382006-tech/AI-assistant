"""Cross-Encoder Reranker integration using BAAI/bge-reranker-v2-m3."""
from __future__ import annotations

import logging
from typing import Sequence
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_classic.retrievers import ContextualCompressionRetriever
from pydantic import Field

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL = None


def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Lazy-load and cache the CrossEncoder instance."""
    global _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_MODEL is None:
        from sentence_transformers import CrossEncoder
        logger.info("Initializing CrossEncoder: %s", model_name)
        _CROSS_ENCODER_MODEL = CrossEncoder(model_name)
    return _CROSS_ENCODER_MODEL


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_n: int = 5,
    model_name: str = "BAAI/bge-reranker-v2-m3",
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


class CrossEncoderCompressor(BaseDocumentCompressor):
    """Document compressor adapter for LangChain ContextualCompressionRetriever."""
    top_n: int = Field(default=5)
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3")

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        return rerank_documents(query, documents, top_n=self.top_n, model_name=self.model_name)


def get_reranked_retriever(
    base_retriever,
    top_n: int = 5,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> ContextualCompressionRetriever:
    """Builds a ContextualCompressionRetriever wrapping any base retriever."""
    compressor = CrossEncoderCompressor(top_n=top_n, model_name=model_name)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
