"""Small PGVector store backed by PostgreSQL and Ollama bge-m3 embeddings."""
from __future__ import annotations

from functools import lru_cache

import psycopg
from pgvector import Vector
from langchain_ollama import OllamaEmbeddings
from pgvector.psycopg import register_vector

from app.core.config import settings


_POOL = None


class VectorStoreNotReady(RuntimeError):
    """Raised when pgvector or the ingestion table has not been initialized."""


def _configure_connection(conn) -> None:
    # `register_vector` queries PostgreSQL's vector type. It must be skipped
    # until ingestion has installed the extension in this database.
    try:
        has_vector_type = conn.execute("SELECT to_regtype('vector')").fetchone()[0]
        if has_vector_type:
            register_vector(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _pool():
    global _POOL
    if not settings.database_url.strip():
        raise RuntimeError("DATABASE_URL is not configured; set it in the local .env file")
    if _POOL is None:
        from psycopg_pool import ConnectionPool
        _POOL = ConnectionPool(settings.database_url, min_size=1, max_size=8,
                               kwargs={"autocommit": False}, configure=_configure_connection,
                               open=True)
    return _POOL


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def _connect():
    return _pool().connection()


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)


def ensure_vector_schema() -> None:
    # DDL is performed during ingestion, not for every question.
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("""CREATE TABLE IF NOT EXISTS rag_chunks (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER,
            embedding VECTOR(1024) NOT NULL,
            UNIQUE (source, chunk_index)
        )""")
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx ON rag_chunks USING hnsw (embedding vector_cosine_ops)")


def replace_corpus_chunks(chunks: list[dict], sources: list[str]) -> int:
    """Atomically replace the indexed corpus, including removed/renamed sources."""
    ensure_vector_schema()
    batch_size = settings.rag_embedding_batch_size
    embeddings = []
    for start in range(0, len(chunks), batch_size):
        embeddings.extend(get_embeddings().embed_documents(
            [item["content"] for item in chunks[start:start + batch_size]]))
    with _connect() as conn:
        with conn.cursor() as cursor:
            if sources:
                # This table represents the single corpus configured by RAG_DOCUMENT_DIR.
                cursor.execute("DELETE FROM rag_chunks WHERE source <> ALL(%s)", (sources,))
                cursor.execute("DELETE FROM rag_chunks WHERE source = ANY(%s)", (sources,))
            else:
                cursor.execute("DELETE FROM rag_chunks")
            if chunks:
                cursor.executemany(
                    """INSERT INTO rag_chunks(content, source, chunk_index, page_number, embedding)
                       VALUES (%s, %s, %s, %s, %s)""",
                    [(item["content"], item["source"], item["chunk_index"], item.get("page_number"), Vector(vector))
                     for item, vector in zip(chunks, embeddings)],
                )
    return len(chunks)


def similarity_search(query: str, k: int | None = None) -> list[dict]:
    vector = get_embeddings().embed_query(query)
    with _connect() as conn:
        ready = conn.execute(
            "SELECT to_regtype('vector'), to_regclass('public.rag_chunks')"
        ).fetchone()
        if not ready[0] or not ready[1]:
            raise VectorStoreNotReady(
                "RAG vector store is not initialized; install pgvector and run `python -m scripts.ingest_documents`."
            )
        # The pool may have opened this connection before another process
        # installed pgvector. Register its adapter after confirming readiness.
        register_vector(conn)
        rows = conn.execute(
            """SELECT content, source, chunk_index, page_number, 1 - (embedding <=> %s) AS score
               FROM rag_chunks ORDER BY embedding <=> %s LIMIT %s""",
            (Vector(vector), Vector(vector), k or settings.rag_top_k),
        ).fetchall()
    return [{"content": r[0], "source": r[1], "chunk_index": r[2],
             "page_number": r[3], "score": float(r[4])} for r in rows]
