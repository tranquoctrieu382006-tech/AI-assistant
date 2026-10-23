# RAG evaluation v1

## Golden sets

- RAG: `data/rag_golden_set.json` contains five questions with expected source or expected abstention.
- NLQ: `data/nlq_golden_set.json` contains four questions, reference read-only SQL, and expected aggregate values from the demo seed.

## Retrieval metrics

Run `python -m scripts.evaluate_rag_retrieval` after PostgreSQL/pgvector has been seeded with documents and Ollama is serving the configured `bge-m3` embedding model. The script reports Hit@k (expected source appears in top-k; abstention cases count as hits only when retrieval is below the API threshold) and MRR.

## Initial run (2026-10-23)

- PostgreSQL 18.6 and pgvector 0.8.7 were reachable. Ingestion stored 21 chunks from 15 files, with 1024-dimensional embeddings.
- Before tuning abstention, retrieval Hit@5 and MRR were both 0.8. Four answerable cases ranked the expected source first. The negative case had a top cosine similarity of 0.438 against an unrelated shipping policy.
- `RAG_MIN_RELEVANCE_SCORE=0.5` is the initial v1 cutoff: it sits above that negative score and below the weakest positive score (0.623). This is calibrated on only five cases and should be retuned as the set grows.
- Run `python -m scripts.evaluate_nlq_golden` to execute the checked-in reference SELECTs in a read-only transaction and compare results.

## After abstention calibration

- At the 0.5 cutoff, all five RAG cases pass: Hit@5 = 1.0 and MRR = 1.0. The negative case is counted as a hit when the API abstains below the configured cutoff.
- All four NLQ reference queries match their expected values (accuracy = 1.0).
- A local `POST /api/rag/ask` smoke test returned an answer and sources for an in-scope question. A separate local API request returned the expected no-answer response with no sources for the out-of-scope case.
- On 2026-10-23, after explicit user authorization, one live RAG smoke trace was flushed to Langfuse Cloud successfully. This confirms one delivery path, not sustained delivery reliability or a privacy-safe configuration; traces may include the question, retrieved context, and generated answer.
