"""Small PGVector store backed by PostgreSQL and Ollama bge-m3 embeddings."""
from __future__ import annotations

from functools import lru_cache
from math import ceil

import psycopg
from pgvector import Vector
from langchain_ollama import OllamaEmbeddings
from pgvector.psycopg import register_vector

from app.core.config import settings


_POOL = None
_STORE_READY: bool | None = None


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
        timeout = settings.database_connection_timeout_seconds
        _POOL = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=8,
            timeout=timeout,
            kwargs={"autocommit": False, "connect_timeout": ceil(timeout)},
            configure=_configure_connection,
            open=True,
        )
    return _POOL


def close_pool() -> None:
    global _POOL, _STORE_READY
    if _POOL is not None:
        _POOL.close()
        _POOL = None
    _STORE_READY = None


def _connect():
    return _pool().connection()


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.embedding_timeout_seconds},
    )


def embed_query(query: str) -> list[float]:
    vector = get_embeddings().embed_query(query)
    if len(vector) != settings.embedding_dimensions:
        raise ValueError(
            f"Embedding model returned {len(vector)} dimensions; EMBEDDING_DIMENSIONS is "
            f"{settings.embedding_dimensions}. Update the setting and rebuild the vector table."
        )
    return vector


def ensure_vector_schema() -> None:
    global _STORE_READY
    # DDL is performed during ingestion, not for every question.
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        _register_vector_if_needed(conn)
        conn.execute(f"""CREATE TABLE IF NOT EXISTS rag_chunks (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER,
            section TEXT,
            corpus TEXT NOT NULL DEFAULT 'demo',
            display_name TEXT,
            embedding VECTOR({settings.embedding_dimensions}) NOT NULL
        )""")
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS section TEXT")
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS corpus TEXT NOT NULL DEFAULT 'demo'")
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS display_name TEXT")
        conn.execute("ALTER TABLE rag_chunks DROP CONSTRAINT IF EXISTS rag_chunks_source_chunk_index_key")
        conn.execute("""DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'rag_chunks_corpus_source_chunk_index_key'
            ) THEN
                ALTER TABLE rag_chunks ADD CONSTRAINT rag_chunks_corpus_source_chunk_index_key UNIQUE (corpus, source, chunk_index);
            END IF;
        END $$;""")
        conn.execute("CREATE INDEX IF NOT EXISTS rag_chunks_corpus_idx ON rag_chunks(corpus)")
        column_type = conn.execute(
            """SELECT format_type(a.atttypid, a.atttypmod)
               FROM pg_attribute a
               WHERE a.attrelid = 'public.rag_chunks'::regclass
                 AND a.attname = 'embedding' AND NOT a.attisdropped"""
        ).fetchone()[0]
        expected_type = f"vector({settings.embedding_dimensions})"
        if column_type != expected_type:
            raise VectorStoreNotReady(
                f"Stored embedding column is {column_type}, but EMBEDDING_DIMENSIONS is "
                f"{settings.embedding_dimensions}; recreate the rag_chunks table and re-ingest."
            )
        conn.execute("CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx ON rag_chunks USING hnsw (embedding vector_cosine_ops)")
    _STORE_READY = True


def _require_store_ready(conn) -> None:
    """Cache schema readiness once per process and register pgvector per connection."""
    global _STORE_READY
    if _STORE_READY is not True:
        ready = conn.execute(
            "SELECT to_regtype('vector'), to_regclass('public.rag_chunks')"
        ).fetchone()
        _STORE_READY = bool(ready[0] and ready[1])
        if not _STORE_READY:
            raise VectorStoreNotReady(
                "RAG vector store is not initialized; install pgvector and run `python -m scripts.ingest_documents`."
            )
    _register_vector_if_needed(conn)


def replace_corpus_chunks(chunks: list[dict], *, corpus: str = "demo") -> int:
    """Atomically replace the indexed corpus, scoped to `corpus`."""
    if not chunks:
        raise ValueError("Refusing to replace the indexed corpus with an empty chunk list.")
    ensure_vector_schema()
    batch_size = settings.rag_embedding_batch_size
    embeddings = []
    for start in range(0, len(chunks), batch_size):
        embeddings.extend(get_embeddings().embed_documents(
            [item["content"] for item in chunks[start:start + batch_size]]))
    invalid_dimensions = next(
        (len(vector) for vector in embeddings if len(vector) != settings.embedding_dimensions),
        None,
    )
    if invalid_dimensions is not None:
        raise ValueError(
            f"Embedding model returned {invalid_dimensions} dimensions; EMBEDDING_DIMENSIONS is "
            f"{settings.embedding_dimensions}. Update the setting and rebuild the vector table."
        )
    with _connect() as conn:
        _register_vector_if_needed(conn)
        with conn.transaction():
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM rag_chunks WHERE corpus = %s", (corpus,))
                cursor.executemany(
                    """INSERT INTO rag_chunks(content, source, chunk_index, page_number, section, corpus, display_name, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [(item["content"], item["source"], item["chunk_index"], item.get("page_number"),
                      item.get("section"), corpus, item.get("display_name"), Vector(vector))
                     for item, vector in zip(chunks, embeddings)],
                )
    return len(chunks)


def upsert_document_chunks(
    source: str,
    chunks: list[dict],
    embeddings: list[list[float]] | None = None,
    *,
    corpus: str = "user",
    display_name: str | None = None,
) -> int:
    """Atomically upsert chunks for a specific document (source) within a given corpus."""
    ensure_vector_schema()
    if embeddings is None and chunks:
        batch_size = settings.rag_embedding_batch_size
        embeddings = []
        for start in range(0, len(chunks), batch_size):
            embeddings.extend(get_embeddings().embed_documents(
                [item["content"] for item in chunks[start:start + batch_size]]))
    if embeddings is not None and len(chunks) != len(embeddings):
        raise ValueError("Length of chunks and embeddings must match")

    with _connect() as conn:
        _register_vector_if_needed(conn)
        with conn.transaction():
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM rag_chunks WHERE corpus = %s AND source = %s",
                    (corpus, source),
                )
                if chunks and embeddings:
                    cursor.executemany(
                        """INSERT INTO rag_chunks(content, source, chunk_index, page_number, section, corpus, display_name, embedding)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        [(item["content"], source, item["chunk_index"], item.get("page_number"),
                          item.get("section"), corpus, display_name or item.get("display_name"), Vector(vec))
                         for item, vec in zip(chunks, embeddings)],
                    )
    return len(chunks)


def similarity_search(
    query: str,
    k: int | None = None,
    *,
    embedding: list[float] | None = None,
    corpus: str | list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    vector = embedding if embedding is not None else embed_query(query)
    with _connect() as conn:
        _require_store_ready(conn)
        if isinstance(corpus, (list, tuple)):
            rows = conn.execute(
                """SELECT content, source, chunk_index, page_number, section,
                          1 - (embedding <=> %s) AS score, corpus, display_name
                   FROM rag_chunks WHERE corpus = ANY(%s) ORDER BY embedding <=> %s LIMIT %s""",
                (Vector(vector), list(corpus), Vector(vector), k or settings.rag_top_k),
            ).fetchall()
        elif corpus is not None:
            rows = conn.execute(
                """SELECT content, source, chunk_index, page_number, section,
                          1 - (embedding <=> %s) AS score, corpus, display_name
                   FROM rag_chunks WHERE corpus = %s ORDER BY embedding <=> %s LIMIT %s""",
                (Vector(vector), corpus, Vector(vector), k or settings.rag_top_k),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT content, source, chunk_index, page_number, section,
                          1 - (embedding <=> %s) AS score, corpus, display_name
                   FROM rag_chunks ORDER BY embedding <=> %s LIMIT %s""",
                (Vector(vector), Vector(vector), k or settings.rag_top_k),
            ).fetchall()

    results = []
    for r in rows:
        item = {
            "content": r[0],
            "source": r[1],
            "chunk_index": r[2],
            "page_number": r[3],
            "section": r[4],
            "score": float(r[5]),
        }
        if len(r) > 6:
            item["corpus"] = r[6]
        if len(r) > 7:
            item["display_name"] = r[7]
        results.append(item)
    return results


def score_chunks(
    query: str,
    chunk_keys: list[tuple[str, int]],
    *,
    embedding: list[float] | None = None,
) -> dict[tuple[str, int], float]:
    """Return cosine similarities for specific chunks, including non-vector top-k hits."""
    if not chunk_keys:
        return {}

    vector = Vector(embedding if embedding is not None else embed_query(query))
    values = ", ".join(["(%s, %s)"] * len(chunk_keys))
    params = [vector]
    for source, chunk_index in chunk_keys:
        params.extend((source, chunk_index))

    with _connect() as conn:
        _require_store_ready(conn)
        rows = conn.execute(
            f"""SELECT source, chunk_index, 1 - (embedding <=> %s) AS score
                FROM rag_chunks
                WHERE (source, chunk_index) IN (VALUES {values})""",
            params,
        ).fetchall()
    return {(source, chunk_index): float(score) for source, chunk_index, score in rows}


def get_corpus_revision(*, corpus: str | list[str] | tuple[str, ...] | None = None) -> tuple[int, int]:
    """Return a cheap revision token that changes after replace_corpus_chunks()."""
    with _connect() as conn:
        _require_store_ready(conn)
        if isinstance(corpus, (list, tuple)):
            count, max_id = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM rag_chunks WHERE corpus = ANY(%s)",
                (list(corpus),),
            ).fetchone()
        elif corpus is not None:
            count, max_id = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM rag_chunks WHERE corpus = %s",
                (corpus,),
            ).fetchone()
        else:
            count, max_id = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM rag_chunks"
            ).fetchone()
    return int(count), int(max_id)


def get_all_chunks(*, corpus: str | list[str] | tuple[str, ...] | None = None) -> list[dict]:
    with _connect() as conn:
        _require_store_ready(conn)
        if isinstance(corpus, (list, tuple)):
            rows = conn.execute(
                """SELECT content, source, chunk_index, page_number, section, corpus, display_name
                   FROM rag_chunks WHERE corpus = ANY(%s) ORDER BY source, chunk_index""",
                (list(corpus),),
            ).fetchall()
        elif corpus is not None:
            rows = conn.execute(
                """SELECT content, source, chunk_index, page_number, section, corpus, display_name
                   FROM rag_chunks WHERE corpus = %s ORDER BY source, chunk_index""",
                (corpus,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT content, source, chunk_index, page_number, section, corpus, display_name
                   FROM rag_chunks ORDER BY source, chunk_index"""
            ).fetchall()
    results = []
    for r in rows:
        item = {
            "content": r[0],
            "source": r[1],
            "chunk_index": r[2],
            "page_number": r[3],
            "section": r[4],
        }
        if len(r) > 5:
            item["corpus"] = r[5]
        if len(r) > 6:
            item["display_name"] = r[6]
        results.append(item)
    return results
