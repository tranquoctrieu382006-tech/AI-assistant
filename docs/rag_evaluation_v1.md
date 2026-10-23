# RAG evaluation v1

## Golden sets

- RAG: `data/rag_golden_set.json` contains five questions with expected source or expected abstention.
- NLQ: `data/nlq_golden_set.json` contains four questions, reference read-only SQL, and expected aggregate values from the demo seed.

## Retrieval metrics

Run `python -m scripts.evaluate_rag_retrieval` after PostgreSQL/pgvector has been seeded with documents and Ollama is serving the configured `bge-m3` embedding model. The script reports Hit@k (expected source appears in top-k; abstention cases count as hits only when retrieval is below the API threshold) and MRR.

## Initial run

Not run on 2026-10-23: PostgreSQL (5432) and Ollama (11434) were not reachable, so there is no defensible retrieval score yet. Run the ingest command first, then this evaluation; do not treat placeholder values as scores.
