"""Small PGVector store backed by PostgreSQL and Ollama bge-m3 embeddings."""
from __future__ import annotations

from functools import lru_cache

import psycopg
from langchain_ollama import OllamaEmbeddings
from pgvector.psycopg import register_vector

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)


def ensure_vector_schema() -> None:
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("""CREATE TABLE IF NOT EXISTS rag_chunks (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            embedding VECTOR(1024) NOT NULL,
            UNIQUE (source, chunk_index)
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx ON rag_chunks USING hnsw (embedding vector_cosine_ops)")


def upsert_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    embeddings = get_embeddings().embed_documents([item["content"] for item in chunks])
    ensure_vector_schema()
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        conn.executemany(
            """INSERT INTO rag_chunks(content, source, chunk_index, embedding)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (source, chunk_index) DO UPDATE SET
               content=EXCLUDED.content, embedding=EXCLUDED.embedding""",
            [(item["content"], item["source"], item["chunk_index"], vector)
             for item, vector in zip(chunks, embeddings)],
        )
    return len(chunks)


def similarity_search(query: str, k: int | None = None) -> list[dict]:
    vector = get_embeddings().embed_query(query)
    ensure_vector_schema()
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        rows = conn.execute(
            """SELECT content, source, chunk_index, 1 - (embedding <=> %s) AS score
               FROM rag_chunks ORDER BY embedding <=> %s LIMIT %s""",
            (vector, vector, k or settings.rag_top_k),
        ).fetchall()
    return [{"content": r[0], "source": r[1], "chunk_index": r[2], "score": float(r[3])} for r in rows]

