# AI Assistant evaluation — 2026-10-23

## Environment and corpus

- PostgreSQL 18.6 at the configured local database and pgvector 0.8.7 were verified using read-only checks.
- Ollama exposed `bge-m3:latest` and `gemma3:12b` locally. The default corpus is the 12 business documents in `data/demo_sales_warehouse`; repo/data-description Markdown is excluded. `data/documents/` is a separate optional corpus.
- Ingestion completed with **12 files and 30 chunks** after retaining section boundaries. A retrieval smoke query for the return window ranked `04_chinh_sach_doi_tra.docx` first; page metadata is available for PDF matches.

## Retrieval and abstention (previous 35-case set)

| Measure | Development-set result | Scope |
|---|---:|---|
| Answerable cases | 33 | `data/rag_golden_set.json` |
| Hit@5 | 1.000 | Expected source appears in top five |
| MRR | 0.939 | Four expected sources were at rank 2 |
| Abstention accuracy | 1.000 (2/2) | At `RAG_MIN_RELEVANCE_SCORE=0.5` |

These are development-set results; the threshold was calibrated on the same corpus. They are not a held-out quality estimate. Earlier five-case Hit@5/MRR values that counted the abstention case are superseded.

The golden set has since been expanded to 47 explicitly categorized cases (33 simple, four paraphrases, two multi-document, eight out-of-scope). The small 30-chunk corpus means Hit@k remains a development diagnostic, not a robust generalization estimate.

## Retrieval and abstention (corrected 47-case set)

The run used local `bge-m3:latest`, PostgreSQL 18.6 and pgvector 0.8.7. The answerable set contains 39 cases; eight are expected abstentions. During review, 12 recently added golden questions were found to contain literal `?` characters replacing Vietnamese diacritics; those questions and expected fragments have been corrected. The results below supersede the earlier expanded-set retrieval run.

| Measure | Result | Count / scope |
|---|---:|---|
| Hit@1 | 0.897 | 35/39 answerable cases |
| Hit@3 | 1.000 | 39/39 answerable cases |
| Hit@5 | 1.000 | 39/39 answerable cases |
| MRR | 0.962 | Answerable cases |
| Expected-source coverage@5 | 1.000 | All required sources across answerable cases |
| Abstention accuracy | 0.625 | 5/8 out-of-scope cases at cutoff 0.5 |

Failure analysis: all four corrected paraphrases and both multi-document questions retrieved every expected source by rank 3. Three out-of-scope questions were false positives at the current 0.5 cutoff: `rag-042` (maternity leave) retrieved `01_quy_che_lam_viec.md` at 0.584; `rag-043` (customer-care phone) retrieved `10_huong_dan_don_hang_khieu_nai.docx` at 0.556; and `rag-044` (base salary) retrieved `03_quy_che_luong_cong_tac_phi.md` at 0.549. These development results must not be used to tune a production threshold without held-out data. Complete per-question details are in `eval/rag_retrieval_2026-10-23.json`.

## NLQ

- **30/30 reference SQL cases** matched the fixed seed in a read-only transaction. The set now includes joins, groupings, date filters and aggregates. This is seed consistency, not model accuracy.
- Full local `gemma3:12b` evaluation completed on all 30 questions: **16 passed, 14 failed, 0 rejected, 0 execution errors (16/30, 0.533)**. All 30 generated statements executed read-only; SQL execution accuracy is not semantic answer quality beyond comparison with the fixed expected rows.
- Example failures: `nlq-017` asks for monthly counts but the model returned one annual count; `nlq-023` asks for one total but returned 300 per-order rows; `nlq-022` asks for average order value but queried average shipping fee.
- Full generated SQL and expected/actual rows are stored in `eval/nlq_model_local_2026-10-23.json`. With the user's approval, Dataset `ai-assistant-nlq-golden-v1` and a 30-item replay experiment were uploaded to Langfuse. The experiment replays these local results; it does not call the model again. It reports execution accuracy 16/30, with 16 pass, 14 fail, 0 rejected, 0 execution errors.
- NLQ Langfuse experiment: https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=d4ef3f9b8e58150c

## Answer quality and tracing

- One local RAG smoke answer correctly stated the seven-calendar-day return window and returned sources.
- `answer_contains` remains a strict substring check, not a semantic correctness metric. The corrected v3 model comparison is complete; judge coverage is uneven because Groq returned HTTP 429, as detailed below.
- `/api/rag/ask` now uses an LCEL chain (`ChatPromptTemplate | ChatModel | StrOutputParser`); the API answer remains a string and `sources` are still assembled from retrieval.
- A direct HTTP request to `/api/rag/ask` returned 200 using only synthetic question/context and a mocked model/retriever. The Langfuse Cloud API then returned six observations for the trace: request, retrieval, LCEL sequence, prompt, synthetic model and output parser. [View the synthetic trace](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/traces/470eb0d013621fc6b3d6112a83b0cdd4). This confirms Cloud delivery and request/callback wiring, not live retrieval or a real provider-model trace.
- Compared section-packed chunking with paragraph-block-preserving chunking over the same 47-case set and local bge-m3 embeddings. Results: packed paragraph — 30 chunks, Hit@1 0.897, Hit@3 1.000, Hit@5 1.000, MRR 0.962, abstention 5/8; preserved blocks — 33 chunks, Hit@1 0.897, Hit@3 0.974, Hit@5 1.000, MRR 0.962, abstention 5/8. The block variant changed expected-source ranks for one case and did not improve retrieval, so the existing default remains. This is a development-set result, not held-out evidence; the active vector store was not modified. Full output: `eval/chunking_comparison_2026-10-23.json`.

## Tests and limitations

- Current suite: **84 passed**, captured with `.venv\Scripts\python.exe -m pytest -q`. `pytest -q` is the reproducible command.
- Python compile check passed. Live ingestion, retrieval, and the full 30-case NLQ evaluation were run. The earlier RAG model comparison used malformed new golden questions; its category scores are superseded pending the v3 rerun.
- The benchmark script now warm-ups each model and reports median latency over three measured samples. Cost remains comparable only for the two Groq models with configured rates; other models report no cost estimate.
- API key authentication and a 60-requests/minute per-IP in-process limit are now implemented. Multi-worker deployments still need a shared limiter or proxy-level enforcement.

## AI08 follow-up

The previous Langfuse v1 run used malformed additions; v2 was stopped with partially uploaded items and is superseded. The corrected v3 Dataset contains all 47 cases. Both candidates ran all 47 items with the same retrieved context/prompt: local Gemma and Groq GPT-OSS-120B. Judge was Groq GPT-OSS-20B.

| Category | Candidate | Answer substring | Correct expected behavior | Faithfulness | Answer relevancy |
|---|---|---:|---:|---:|---:|
| Simple | Local | 0.909 (30/33) | 0.970 (32/33) | 1.000 (32/32 scored) | 1.000 (32/32 scored) |
| Simple | Strong | 0.333 (11/33) | 0.970 (32/33) | 1.000 (1/1 scored) | 1.000 (1/1 scored) |
| Paraphrase | Local | 1.000 (4/4) | 1.000 (4/4) | 1.000 (4/4 scored) | 0.975 (4/4 scored) |
| Paraphrase | Strong | 0.250 (1/4) | 1.000 (4/4) | Not scored (0/4) | Not scored (0/4) |
| Multi-document | Local | 1.000 (2/2) | 1.000 (2/2) | 1.000 (2/2 scored) | 1.000 (2/2 scored) |
| Multi-document | Strong | 0.000 (0/2) | 1.000 (2/2) | Not scored (0/2) | Not scored (0/2) |
| Out of scope | Local | 1.000 (8/8) | 1.000 (8/8) | Not applicable | Not applicable |
| Out of scope | Strong | 1.000 (8/8) | 1.000 (8/8) | Not applicable | Not applicable |

The Groq judge hit its daily token limit (HTTP 429) during the run. Counts show the judge denominator explicitly: it scored 32/33 local simple cases, all local paraphrase and multi-document cases, but only 1/33 strong simple cases and none of the strong paraphrase/multi-document cases. Do not compare the two candidates' judge means as if they had equal coverage. The runner was subsequently changed to stop calling a judge after its first 429; this circuit breaker was not active during this v3 run.

The exact-answer metric is substring matching and is not semantic correctness; out-of-scope substring matches reflect the fixed refusal text. The single unblinded LLM judge is not human-validated. Results are development-set signals, not held-out evidence or a production model recommendation.

- Local experiment: https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=37eb4cf19217b692
- Strong experiment: https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=cb9132e5ab224119
