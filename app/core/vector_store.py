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


def _register_vector_if_needed(conn) -> None:
    if getattr(conn, "_vector_adapter_registered", False) is not True:
        register_vector(conn)
        conn._vector_adapter_registered = True


def _configure_connection(conn) -> None:
    # `register_vector` queries PostgreSQL's vector type. It must be skipped
    # until ingestion has installed the extension in this database.
    try:
        has_vector_type = conn.execute("SELECT to_regtype('vector')").fetchone()[0]
        if has_vector_type:
            _register_vector_if_needed(conn)
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
        _register_vector_if_needed(conn)
        conn.execute("""CREATE TABLE IF NOT EXISTS rag_chunks (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER,
            section TEXT,
            embedding VECTOR(1024) NOT NULL,
            UNIQUE (source, chunk_index)
        )""")
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS section TEXT")
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
        if chunks:
            _register_vector_if_needed(conn)
        with conn.cursor() as cursor:
            if sources:
                # This table represents the single corpus configured by RAG_DOCUMENT_DIR.
                cursor.execute("DELETE FROM rag_chunks WHERE source <> ALL(%s)", (sources,))
                cursor.execute("DELETE FROM rag_chunks WHERE source = ANY(%s)", (sources,))
            else:
                cursor.execute("DELETE FROM rag_chunks")
            if chunks:
                cursor.executemany(
                    """INSERT INTO rag_chunks(content, source, chunk_index, page_number, section, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    [(item["content"], item["source"], item["chunk_index"], item.get("page_number"),
                      item.get("section"), Vector(vector))
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
        # installed pgvector. Register it once after confirming readiness.
        _register_vector_if_needed(conn)
        rows = conn.execute(
            """SELECT content, source, chunk_index, page_number, section,
                      1 - (embedding <=> %s) AS score
               FROM rag_chunks ORDER BY embedding <=> %s LIMIT %s""",
            (Vector(vector), Vector(vector), k or settings.rag_top_k),
        ).fetchall()
    return [{"content": r[0], "source": r[1], "chunk_index": r[2],
             "page_number": r[3], "section": r[4], "score": float(r[5])} for r in rows]


def score_chunks(query: str, chunk_keys: list[tuple[str, int]]) -> dict[tuple[str, int], float]:
    """Return cosine similarities for specific chunks, including non-vector top-k hits."""
    if not chunk_keys:
        return {}

    vector = Vector(get_embeddings().embed_query(query))
    values = ", ".join(["(%s, %s)"] * len(chunk_keys))
    params = [vector]
    for source, chunk_index in chunk_keys:
        params.extend((source, chunk_index))

    with _connect() as conn:
        ready = conn.execute(
            "SELECT to_regtype('vector'), to_regclass('public.rag_chunks')"
        ).fetchone()
        if not ready[0] or not ready[1]:
            raise VectorStoreNotReady(
                "RAG vector store is not initialized; install pgvector and run `python -m scripts.ingest_documents`."
            )
        _register_vector_if_needed(conn)
        rows = conn.execute(
            f"""SELECT source, chunk_index, 1 - (embedding <=> %s) AS score
                FROM rag_chunks
                WHERE (source, chunk_index) IN (VALUES {values})""",
            params,
        ).fetchall()
    return {(source, chunk_index): float(score) for source, chunk_index, score in rows}


def get_corpus_revision() -> tuple[int, int]:
    """Return a cheap revision token that changes after replace_corpus_chunks()."""
    with _connect() as conn:
        ready = conn.execute(
            "SELECT to_regtype('vector'), to_regclass('public.rag_chunks')"
        ).fetchone()
        if not ready[0] or not ready[1]:
            raise VectorStoreNotReady(
                "RAG vector store is not initialized; install pgvector and run `python -m scripts.ingest_documents`."
            )
        _register_vector_if_needed(conn)
        count, max_id = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM rag_chunks"
        ).fetchone()
    return int(count), int(max_id)


def get_all_chunks() -> list[dict]:
    with _connect() as conn:
        ready = conn.execute(
            "SELECT to_regtype('vector'), to_regclass('public.rag_chunks')"
        ).fetchone()
        if not ready[0] or not ready[1]:
            raise VectorStoreNotReady(
                "RAG vector store is not initialized; install pgvector and run `python -m scripts.ingest_documents`."
            )
        _register_vector_if_needed(conn)
        rows = conn.execute(
            """SELECT content, source, chunk_index, page_number, section
               FROM rag_chunks ORDER BY source, chunk_index"""
        ).fetchall()
    return [{"content": r[0], "source": r[1], "chunk_index": r[2],
             "page_number": r[3], "section": r[4]} for r in rows]
