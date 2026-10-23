# RAG evaluation v1

## What these checks measure

- `data/rag_golden_set.json` currently has 47 classified cases: 33 simple, four paraphrases, two multi-document questions and eight expected abstentions. Hit@1/3/5 and MRR include answerable questions; a multi-document hit requires every expected source. Source coverage is reported separately so partial multi-source retrieval remains visible.
- The eight out-of-scope questions are still a small sample; abstention rates are descriptive only, not a stable quality estimate.
- `data/nlq_golden_set.json` has 30 cases covering joins, groupings, date ranges and aggregates, with read-only reference SQL and expected results.
- The cutoff `RAG_MIN_RELEVANCE_SCORE=0.5` was selected on the development corpus. These results are calibration observations, not held-out quality evidence.
- `python -m scripts.evaluate_rag_answers` sends the golden questions to the running API and checks `answer_contains`. This is a simple substring smoke check, not semantic evaluation. The expected text is deliberately specific; for example, the exchange window is `7 ngày lịch`.
- `python -m scripts.evaluate_rag_experiment --roles local strong --judge-role fast` creates/updates a versioned Langfuse Dataset and runs comparable experiments with answer substring, abstention, faithfulness, answer relevancy, and latency scores.
- `python -m scripts.evaluate_nlq_golden` executes checked-in reference SQL read-only. Its rate is **seed consistency**, not model accuracy. `python -m scripts.evaluate_nlq_model --model-role local` separately evaluates generated SQL against those expected results.
- The model evaluator supplies the actual `CREATE TABLE` statements from `schema_seed.sql`, compares rows in order when generated SQL has a top-level `ORDER BY`, and allows semicolons inside quoted strings/comments while rejecting multiple statements.

## Previous development-set run (2026-10-23; original 35-case set)

The refreshed corpus contains 12 business documents. On the original 35-case set, Hit@5 was 1.0 and MRR was 0.9394; both out-of-scope questions were rejected at the 0.5 cutoff (abstention accuracy 1.0). All 30 reference SQL cases matched the seed. This remains development-set evidence because the threshold was calibrated on this corpus.

One local RAG answer smoke test returned the correct exchange window with sources. One local NLQ model smoke test generated a count query with matching execution; one case is not a model-quality estimate.

## Expanded golden set and evaluator changes

The 47-case set adds paraphrases, two questions requiring evidence from two separate documents, and six additional out-of-scope questions. Every case has a category. The evaluator reports Hit@1/3/5, MRR, expected-source coverage, and per-category results. Answer smoke evaluation checks every required substring for multi-part answers.

Review found literal question marks replacing Vietnamese diacritics in the 12 recently added cases. Those prompts and answer fragments were corrected before rerunning. After reingestion with retained section metadata, the corpus contains 30 chunks. Corrected-set retrieval with local bge-m3 and PostgreSQL/pgvector: 39 answerable cases, Hit@1 **0.897**, Hit@3 **1.000**, Hit@5 **1.000**, MRR **0.962**, expected-source coverage@5 **1.000**; abstention accuracy **5/8** at cutoff 0.5. Every corrected paraphrase and both multi-document cases retrieved all expected sources by rank 3. Three out-of-scope prompts were false positives: `rag-042` returned `01_quy_che_lam_viec.md` (0.584), `rag-043` returned `10_huong_dan_don_hang_khieu_nai.docx` (0.556), and `rag-044` returned `03_quy_che_luong_cong_tac_phi.md` (0.549). These are development diagnostics on the same corpus, not held-out evidence; do not recalibrate the production threshold from these eight negative examples. Per-case details are stored in `eval/rag_retrieval_2026-10-23.json`.

## AI05–AI07 follow-up

- A full local `gemma3:12b` model-generated SQL evaluation ran all 30 NLQ cases read-only: 16 pass, 14 fail, no rejected query or execution error (0.533 exact-result accuracy). Example failures: `nlq-017` returned one annual order count instead of monthly rows; `nlq-023` returned per-order rows instead of one total; `nlq-022` averaged shipping fee instead of order value. Full SQL and actual/expected rows are in `eval/nlq_model_local_2026-10-23.json`. With user approval, Dataset `ai-assistant-nlq-golden-v1` and a 30-item Langfuse experiment replay were created; the replay reused saved outputs and did not call the model again. It records 16 pass, 14 fail, 0 rejected, 0 errors (0.533 accuracy).
- Ingestion stores a `section` for Markdown headings and DOCX Heading styles, plus PDF page numbers; section is returned with sources and placed into the RAG context. PDFs do not infer headings. No Docling dependency was added because no requirement has been established.
- Chunking comparison on the 47-case set: current section-packed paragraphs made 30 chunks and scored Hit@1/3/5 0.897/1.000/1.000, MRR 0.962 and abstention 5/8. The paragraph-block-preserving variant made 33 chunks and scored 0.897/0.974/1.000, MRR 0.962 and abstention 5/8. The existing default remains; this is development-set evidence only. Full metrics and failure examples are in `eval/chunking_comparison_2026-10-23.json`.
- `/api/rag/ask` now uses LCEL (`ChatPromptTemplate | ChatModel | StrOutputParser`); API `answer` remains a string and sources continue to come from retrieval. A direct local HTTP request with synthetic question/context and a mocked retriever/model produced a trace confirmed through the Langfuse Cloud observations API. The Cloud trace includes request, retrieval, LCEL sequence, prompt, synthetic model and parser observations: https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/traces/470eb0d013621fc6b3d6112a83b0cdd4. This confirms trace delivery but does not validate a real model-provider call or live retrieval.

The RAG model-comparison script stores the synthetic questions, retrieved context, expected answer fragments, and generated answers in Langfuse Cloud. The current demo documents and seed are synthetic. Do not point the experiment at real or confidential corpora without confirming authorization and retention requirements.

## Full AI08 comparison

The earlier v1 47-item comparison used 12 added Vietnamese golden questions and expected answer fragments containing literal question marks in place of diacritics. Category metrics for those cases are invalid and superseded. A v2 run was stopped with a partially populated dataset; it is also superseded. The corrected v3 Dataset and both experiments contain 47 items:

| Category | Candidate | Answer substring | Correct expected behavior | Faithfulness | Answer relevancy |
|---|---|---:|---:|---:|---:|
| Simple | Local Gemma | 0.909 (30/33) | 0.970 (32/33) | 1.000 (32/32) | 1.000 (32/32) |
| Simple | Groq strong | 0.333 (11/33) | 0.970 (32/33) | 1.000 (1/1) | 1.000 (1/1) |
| Paraphrase | Local Gemma | 1.000 (4/4) | 1.000 (4/4) | 1.000 (4/4) | 0.975 (4/4) |
| Paraphrase | Groq strong | 0.250 (1/4) | 1.000 (4/4) | Not scored (0/4) | Not scored (0/4) |
| Multi-document | Local Gemma | 1.000 (2/2) | 1.000 (2/2) | 1.000 (2/2) | 1.000 (2/2) |
| Multi-document | Groq strong | 0.000 (0/2) | 1.000 (2/2) | Not scored (0/2) | Not scored (0/2) |
| Out of scope | Both | 1.000 (8/8) | 1.000 (8/8) | Not applicable | Not applicable |

The Groq judge reached its daily token limit (HTTP 429). It scored 32/33 local simple cases, all local paraphrase/multi-document cases, but only 1/33 strong simple cases and none of the strong paraphrase/multi-document cases. Judge means are therefore not a balanced comparison. The current runner now stops further judge calls after a 429; the completed v3 run predates that circuit breaker.

`answer_contains` is substring matching, not semantic correctness; the out-of-scope substring result is the expected fixed refusal text. The single unblinded judge is not human-validated. These results are development diagnostics, not held-out evidence or a production recommendation.

- [Local Gemma experiment](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=37eb4cf19217b692)
- [Groq strong experiment](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=cb9132e5ab224119)

`pytest -q` is the reproducible test command. Latest full run: **84 passed**. Automated HTTP-boundary tests use mocked tracing/model components; the separate synthetic HTTP check above verifies actual Langfuse Cloud delivery.

Faithfulness and answer relevancy are LLM-as-a-judge metrics; they are comparative indicators, not human-validated truth. Use the same judge model and fixed corpus across candidate runs, review disagreements manually, and keep held-out questions separate from the development set.
