-- Kích hoạt extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Khởi tạo bảng rag_chunks cho vector store
CREATE TABLE IF NOT EXISTS rag_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    section TEXT,
    corpus TEXT NOT NULL DEFAULT 'demo',
    display_name TEXT,
    embedding vector(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT rag_chunks_corpus_source_chunk_index_key UNIQUE (corpus, source, chunk_index)
);
CREATE INDEX IF NOT EXISTS rag_chunks_corpus_idx ON rag_chunks(corpus);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx ON rag_chunks USING hnsw (embedding vector_cosine_ops);
