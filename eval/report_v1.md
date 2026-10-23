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
- `answer_contains` remains a strict substring check, not a semantic correctness metric. The v3 run is historical; v4 below reruns both candidates with the current Pydantic prompt. Judge coverage remains uneven because Groq returned HTTP 429.
- `/api/rag/ask` now uses an LCEL chain (`ChatPromptTemplate | ChatModel | PydanticOutputParser`) to validate generated `{answer, abstain}`. The public API remains `answer: str` and `sources` assembled from retrieval; malformed model output fails as HTTP 502.
- A direct HTTP request to `/api/rag/ask` returned 200 using only synthetic question/context and a mocked model/retriever. The Langfuse Cloud API confirmed six observations: request, retrieval, LCEL sequence, prompt, synthetic model and `PydanticOutputParser`. [View the structured-output synthetic trace](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/traces/32281c435ef6271194aab67b1b5e910e). This confirms Cloud delivery and request/callback wiring, not live retrieval or a real provider-model trace.
- Validated the current structured prompt against 30 frozen v3 contexts with the actual local `ollama/gemma3:12b`: 28 contexts invoked the model and **0/28 parse failures** (0 generation failures); two items had no frozen context, including one answerable retrieval miss. The end-to-end normalized answer-fragment result was **26/30**; it is a lexical benchmark, not a semantic judge. Per-case statuses/answers remain local in `eval/rag_structured_local_2026-10-23.json`; this run did not upload generated outputs or traces.
- Separately, a real authenticated `/api/rag/ask` request used live pgvector retrieval and local Gemma, returned HTTP 200 with an answer and two sources, and was verified in Langfuse Cloud. [View the live RAG trace](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/traces/71ec01dcb45e7f54a4d191e1137732c6); it includes request, retrieval, `ChatOllama`, LCEL and Pydantic parser observations. All corpus content is synthetic demo data.
- Completed a new 47-item Langfuse v4 experiment using the current structured-output prompt. Both models made 41 calls and had **0 parse failures**; six cases had no frozen retrieval context. Normalized substring matches: local **42/47** (simple 28/33, paraphrase 4/4, multi-document 2/2, out-of-scope 8/8); strong **29/47** (18/33, 2/4, 1/2, 8/8). This is lexical scoring, not semantic answer accuracy. The Groq judge hit HTTP 429: faithfulness/relevancy coverage was local 2/33 simple, 4/4 paraphrase, 2/2 multi-document, and strong only 1/2 multi-document. AI08 judge evidence remains incomplete. [Local v4 experiment](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=4f7bde5098a1be3e) · [Strong v4 experiment](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=365be450c96433de). The saved summary is `eval/rag_experiment_v4_2026-10-23.json`.
- Compared the two AI06-requested strategies over the same 47 cases with local bge-m3. Fixed-size+overlap produced 13 chunks: Hit@1/3/5 0.846/1.000/1.000, MRR 0.936, abstention 5/8. Heading-bounded chunking produced 30 chunks: Hit@1/3/5 0.897/1.000/1.000, MRR 0.962, abstention 5/8. Heading-bounded was better on Hit@1/MRR and remains the default. The comparison is development evidence, not held-out quality; it does not modify the active vector store. Full output: `eval/chunking_fixed_size_vs_heading_2026-10-23.json`. The earlier paragraph-block comparison remains in `eval/chunking_comparison_2026-10-23.json` as historical analysis.

## Tests and limitations

- Current suite: **91 passed** with `.venv\Scripts\python.exe -m pytest -q` (one upstream `langchain-community` sunset warning). `pytest -q` is the reproducible command.
- Python compile check passed at the previous checkpoint. Live ingestion, retrieval, and the full 30-case NLQ evaluation were run. The earlier RAG model comparison used malformed new golden questions; those scores were superseded by the completed corrected v3 comparison.
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

### Manual review of strong-model answer misses

The stored strong-candidate v3 outputs were reviewed against their dataset expectations; no model was rerun and no new data was uploaded. Strict substring scoring is sensitive to formatting and expected-fragment wording:

| Category | Strict raw substring | After deterministic format normalization |
|---|---:|---:|
| Simple | 11/33 | 18/33 |
| Paraphrase | 1/4 | 2/4 |
| Multi-document | 0/2 | 0/2 |
| Out of scope | 8/8 | 8/8 |
| Total | 20/47 | 28/47 |

The normalizer handles Unicode compatibility spaces/dashes and Markdown emphasis only; it deliberately does not infer synonyms. For example, `rag-001` answers the expected `08:30–17:30` as `**08:30 – 17:30**`, so typography alone caused the raw miss. Other raw misses use different but equivalent wording: `rag-003` says “đơn mua và phiếu điều chuyển” where the expected fragment joins terms with `/`; `rag-006` says “12 ngày mỗi năm” instead of “12 ngày/năm”; `rag-007` says “12:00 đến 13:00” instead of a dash-separated range. This shows the strict score understates some correct responses, but does not explain every miss.

Some misses are end-to-end retrieval failures rather than answer formatting: `rag-029` is a supported inventory question, but its stored golden context is empty after the relevance cutoff, so the model abstains. Both multi-document exact-fragment checks still fail after normalization. The judge scored too few strong answers to conclude whether the remaining misses are factual errors or valid paraphrases. Future experiments emit both `answer_contains` (unchanged for continuity) and `answer_contains_normalized`; the normalized historical counts above were recalculated from existing Cloud outputs without rerunning the model.

- Local experiment: https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=37eb4cf19217b692
- Strong experiment: https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=cb9132e5ab224119
