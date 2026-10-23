# AI Assistant evaluation v1 — 2026-10-23

## Environment and corpus

- PostgreSQL 18.6 at the configured local database and pgvector 0.8.7 were verified using read-only checks.
- Ollama exposed `bge-m3:latest` and `gemma3:12b` locally. The default corpus is the 12 business documents in `data/demo_sales_warehouse`; repo/data-description Markdown is excluded. `data/documents/` is a separate optional corpus.
- Ingestion completed with **12 files and 27 chunks**. A retrieval smoke query for the return window ranked `04_chinh_sach_doi_tra.docx` first (cosine score 0.743); PDF source page metadata was returned for PDF matches.

## Retrieval and abstention (previous 35-case set)

| Measure | Development-set result | Scope |
|---|---:|---|
| Answerable cases | 33 | `data/rag_golden_set.json` |
| Hit@5 | 1.000 | Expected source appears in top five |
| MRR | 0.939 | Four expected sources were at rank 2 |
| Abstention accuracy | 1.000 (2/2) | At `RAG_MIN_RELEVANCE_SCORE=0.5` |

These are development-set results; the threshold was calibrated on the same corpus. They are not a held-out quality estimate. Earlier five-case Hit@5/MRR values that counted the abstention case are superseded.

The golden set has since been expanded to 47 explicitly categorized cases (33 simple, four paraphrases, two multi-document, eight out-of-scope). The live retrieval run below applies to this expanded set. The small 27-chunk corpus means Hit@k remains a development diagnostic, not a robust generalization estimate.

## Retrieval and abstention (expanded 47-case set)

The run used local `bge-m3:latest`, PostgreSQL 18.6 and pgvector 0.8.7. The answerable set contains 39 cases; eight are expected abstentions.

| Measure | Result | Count / scope |
|---|---:|---|
| Hit@1 | 0.744 | 29/39 answerable cases |
| Hit@3 | 0.846 | 33/39 answerable cases |
| Hit@5 | 0.872 | 34/39 answerable cases |
| MRR | 0.808 | Answerable cases |
| Expected-source coverage@5 | 0.872 | Required sources across answerable cases |
| Abstention accuracy | 1.000 | 8/8 out-of-scope cases at cutoff 0.5 |

Failure analysis: all four paraphrase questions missed their expected source in the top five. Of the two multi-document cases, one retrieved both expected sources at ranks 2 and 4; the other retrieved neither in the top five. The 33 simple cases had Hit@3 and Hit@5 of 1.0. These are development-set results, and the abstention sample remains small.

## NLQ

- **30/30 reference SQL cases** matched the fixed seed in a read-only transaction. The set now includes joins, groupings, date filters and aggregates. This is seed consistency, not model accuracy.
- A single local `gemma3:12b` smoke evaluation generated a count query whose result matched the golden output. This is only an evaluator smoke test, not a meaningful model score.
- Use `python -m scripts.evaluate_nlq_model --model-role local` to run the model-generated SQL evaluation over the full golden set. SQL is restricted to a single SELECT/WITH statement, read-only transaction and 10-second timeout.

## Answer quality and tracing

- One local RAG smoke answer correctly stated the seven-calendar-day return window and returned sources.
- The full 47-case AI08 comparison is recorded below. `answer_contains` remains a strict substring check, not a semantic correctness metric.
- RAG requests now have a request span with a child retrieval span; retrieved context, sources and scores are included when Langfuse is configured.

## Tests and limitations

- Current suite: **72 passed in 4.67s**, captured with `.venv\Scripts\python.exe -m pytest -q`. `pytest -q` is the reproducible command.
- Python compile check passed. Live ingestion, retrieval, and the full 47-item model comparison were run.
- The benchmark script now warm-ups each model and reports median latency over three measured samples. Cost remains comparable only for the two Groq models with configured rates; other models report no cost estimate.
- API key authentication and a 60-requests/minute per-IP in-process limit are now implemented. Multi-worker deployments still need a shared limiter or proxy-level enforcement.

## AI08 follow-up

The Langfuse Dataset `ai-assistant-rag-golden-v1` and both experiments contain 47 items. Both candidates used the same retrieved contexts and RAG prompt; the `fast` role (`groq/openai/gpt-oss-20b`) judged faithfulness and answer relevancy. The candidate roles were `local` (`ollama/gemma3:12b`) and `strong` (`groq/openai/gpt-oss-120b`).

| Category | Local answer_contains | Strong answer_contains | Local abstention correct | Strong abstention correct | Local faithfulness / relevancy | Strong faithfulness / relevancy |
|---|---:|---:|---:|---:|---:|---:|
| Simple | 0.909 | 0.364 | 0.970 | 0.970 | 1.000 / 1.000 | 1.000 / 1.000 |
| Paraphrase | 0.000 | 0.000 | 0.000 | 0.000 | Not scored | Not scored |
| Multi-document | 0.000 | 0.000 | 0.500 | 1.000 | 1.000 / 0.800 | 0.975 / 0.950 |
| Out of scope | 0.250 | 0.250 | 1.000 | 1.000 | Not scored | Not scored |

Mean generation latency was 8.06s local / 0.97s strong for simple cases and 9.83s / 1.55s for multi-document cases. Substring scores can undercount correct paraphrased answers; the zero paraphrase and multi-document scores also align with retrieval misses described above. Abstention and exact-answer checks are separate signals.

**Rate-limit caveat:** Groq returned HTTP 429 during two judge evaluations. The experiment completed all 47 candidate items for each model, but that run's aggregate output does not expose the number of successful judge evaluations per category. Treat the reported judge means as provisional/possibly partial, not full-denominator quality estimates. The evaluator now configures up to five provider retries and emits evaluated-case counts alongside means; this change is covered by tests but has not yet been verified with another full Cloud run. The same judge was used for both candidates, but it was unblinded and has not been validated against human ratings.

- Local experiment: https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=054d3c7108e3b020
- Strong experiment: https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=3689dd9781ee1a9e

These results use the synthetic development set and do not establish held-out quality or justify a production-model choice. Follow-up: rerun after rate limits reset or are raised, confirm judge counts cover all eligible cases, and validate a held-out sample with human review.
