# AI Assistant evaluation v1 — 2026-10-23

## Environment and corpus

- PostgreSQL 18.6 at the configured local database and pgvector 0.8.7 were verified using read-only checks.
- Ollama exposed `bge-m3:latest` and `gemma3:12b` locally. The default corpus is the 12 business documents in `data/demo_sales_warehouse`; repo/data-description Markdown is excluded. `data/documents/` is a separate optional corpus.
- Ingestion completed with **12 files and 27 chunks**. A retrieval smoke query for the return window ranked `04_chinh_sach_doi_tra.docx` first (cosine score 0.743); PDF source page metadata was returned for PDF matches.

## Retrieval and abstention

| Measure | Development-set result | Scope |
|---|---:|---|
| Answerable cases | 33 | `data/rag_golden_set.json` |
| Hit@5 | 1.000 | Expected source appears in top five |
| MRR | 0.939 | Four expected sources were at rank 2 |
| Abstention accuracy | 1.000 (2/2) | At `RAG_MIN_RELEVANCE_SCORE=0.5` |

These are development-set results; the threshold was calibrated on the same corpus. They are not a held-out quality estimate. Earlier five-case Hit@5/MRR values that counted the abstention case are superseded.

## NLQ

- **30/30 reference SQL cases** matched the fixed seed in a read-only transaction. The set now includes joins, groupings, date filters and aggregates. This is seed consistency, not model accuracy.
- A single local `gemma3:12b` smoke evaluation generated a count query whose result matched the golden output. This is only an evaluator smoke test, not a meaningful model score.
- Use `python -m scripts.evaluate_nlq_model --model-role local` to run the model-generated SQL evaluation over the full golden set. SQL is restricted to a single SELECT/WITH statement, read-only transaction and 10-second timeout.

## Answer quality and tracing

- One local RAG smoke answer correctly stated the seven-calendar-day return window and returned sources.
- Faithfulness and answer relevancy have **not** been scored. The 35-case `answer_contains` check is a substring smoke metric, not a semantic judge.
- No two-model RAG comparison or Langfuse Dataset/Experiment was run. No Langfuse trace was sent in this verification pass. The previously authorized one-trace smoke test only demonstrates an earlier delivery path.
- Retrieval has not yet been recorded as its own Langfuse observation; the current callback traces LLM execution.

## Tests and limitations

- Full pytest suite: **53 passed** at the time of the full run.
- Python compile check passed. Live ingestion and retrieval were run locally; full 35-question answer evaluation was not run.
- The benchmark script now warm-ups each model and reports median latency over three measured samples. Cost remains comparable only for the two Groq models with configured rates; other models report no cost estimate.
- API authentication and rate limiting are not implemented. The demo binds to loopback by default and must not be exposed externally without those controls.

## Next work

1. Add faithfulness and answer-relevancy judge scores, with judge/model and prompt recorded.
2. Run a same-question, same-corpus comparison for two approved models; review cloud cost and data-sharing before selecting a hosted model.
3. Create Langfuse Dataset/Experiment only after data-sharing scope and Cloud write access are confirmed.
4. Re-evaluate on a held-out question set before claiming quality improvements.
