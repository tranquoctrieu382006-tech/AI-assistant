# RAG evaluation v1

## What these checks measure

- `data/rag_golden_set.json` has four answerable retrieval cases and one expected abstention. Retrieval Hit@k and MRR now use only answerable cases. Abstention is reported separately as `abstention_accuracy` using the configured relevance cutoff.
- The cutoff `RAG_MIN_RELEVANCE_SCORE=0.5` was selected against this same five-case development set (21 chunks in the initial run). Results are calibration observations, not independent evidence of quality. Use a held-out set before making a quality claim.
- `python -m scripts.evaluate_rag_answers` sends the golden questions to the running API and checks `answer_contains`. This is a simple substring smoke check, not semantic evaluation. The expected text is deliberately specific; for example, the exchange window is `7 ngày lịch`.
- `python -m scripts.evaluate_nlq_golden` executes checked-in reference SQL in a read-only transaction. Its rate is **seed consistency**, not NLQ model accuracy: the script does not generate SQL or call an LLM.

## Initial development-set run (2026-10-23)

The original run used 21 chunks from 15 files. Four answerable cases ranked their expected source first; the negative case had an unrelated top similarity of 0.438. The 0.5 cutoff separated those five examples. Earlier figures of Hit@5/MRR 1.0 included the abstention case and overstated retrieval performance; do not use them as a quality claim.

Rerun retrieval after ingestion with `python -m scripts.evaluate_rag_retrieval`. It requires PostgreSQL/pgvector and the configured `bge-m3` embedding service. Run the answer smoke check only when the API is available; this invokes the configured RAG model and may send data to a hosted provider if `RAG_MODEL` selects one. Langfuse tracing may also export question, context, and answer.
