# RAG evaluation v1

## What these checks measure

- `data/rag_golden_set.json` currently has 47 classified cases: 33 simple, four paraphrases, two multi-document questions and eight expected abstentions. Hit@1/3/5 and MRR include answerable questions; a multi-document hit requires every expected source. Source coverage is reported separately so partial multi-source retrieval remains visible.
- The eight out-of-scope questions are still a small sample; abstention rates are descriptive only, not a stable quality estimate.
- `data/nlq_golden_set.json` has 30 cases covering joins, groupings, date ranges and aggregates, with read-only reference SQL and expected results.
- The cutoff `RAG_MIN_RELEVANCE_SCORE=0.5` was selected on the development corpus. These results are calibration observations, not held-out quality evidence.
- `python -m scripts.evaluate_rag_answers` sends the golden questions to the running API and checks `answer_contains`. This is a simple substring smoke check, not semantic evaluation. The expected text is deliberately specific; for example, the exchange window is `7 ngày lịch`.
- `python -m scripts.evaluate_rag_experiment --roles local strong --judge-role fast` creates/updates the Langfuse Dataset and runs comparable experiments with answer substring, abstention, faithfulness, answer relevancy, and latency scores.
- `python -m scripts.evaluate_nlq_golden` executes checked-in reference SQL read-only. Its rate is **seed consistency**, not model accuracy. `python -m scripts.evaluate_nlq_model --model-role local` separately evaluates generated SQL against those expected results.
- The model evaluator supplies the actual `CREATE TABLE` statements from `schema_seed.sql`, compares rows in order when generated SQL has a top-level `ORDER BY`, and allows semicolons inside quoted strings/comments while rejecting multiple statements.

## Previous development-set run (2026-10-23; original 35-case set)

The refreshed corpus contains 12 business documents and 27 chunks. On 33 answerable questions, Hit@5 was 1.0 and MRR was 0.9394; both out-of-scope questions were rejected at the 0.5 cutoff (abstention accuracy 1.0). Four answerable questions had the expected source at rank 2. All 30 reference SQL cases matched the seed. This remains development-set evidence because the threshold was calibrated on this corpus.

One local RAG answer smoke test returned the correct exchange window with sources. One local NLQ model smoke test generated a count query with matching execution; one case is not a model-quality estimate.

## Expanded golden set and evaluator changes

The 47-case set adds paraphrases, two questions requiring evidence from two separate documents, and six additional out-of-scope questions. Every case has a category. The evaluator reports Hit@1/3/5, MRR, expected-source coverage, and per-category results. Answer smoke evaluation checks every required substring for multi-part answers.

Live expanded-set retrieval run with local bge-m3 and PostgreSQL/pgvector: 39 answerable cases, Hit@1 **0.744**, Hit@3 **0.846**, Hit@5 **0.872**, MRR **0.808**, expected-source coverage@5 **0.872**; abstention accuracy **8/8**. Simple questions reached Hit@3/5 **1.0**. All four paraphrases missed their expected source in top five. One multi-document case retrieved both sources at ranks 2 and 4; the other missed both. These are development diagnostics on the same 27-chunk corpus, not held-out results. The eight negative cases are too few for a stable abstention estimate.

The RAG model-comparison script stores the synthetic questions, retrieved context, expected answer fragments, and generated answers in Langfuse Cloud. The current demo documents and seed are synthetic. Do not point the experiment at real or confidential corpora without confirming authorization and retention requirements.

## Full AI08 comparison

The 47-item Langfuse run compared `ollama/gemma3:12b` (`local`) and `groq/openai/gpt-oss-120b` (`strong`) on the same retrieved contexts, using `groq/openai/gpt-oss-20b` (`fast`) as the unblinded judge. The dataset and experiments were created with the user's approval to upload this synthetic evaluation data.

| Category | Local answer substring | Strong answer substring | Local abstention accuracy | Strong abstention accuracy | Local faithfulness / relevancy | Strong faithfulness / relevancy |
|---|---:|---:|---:|---:|---:|---:|
| Simple | 0.909 | 0.364 | 0.970 | 0.970 | 1.000 / 1.000 | 1.000 / 1.000 |
| Paraphrase | 0.000 | 0.000 | 0.000 | 0.000 | Not scored | Not scored |
| Multi-document | 0.000 | 0.000 | 0.500 | 1.000 | 1.000 / 0.800 | 0.975 / 0.950 |
| Out of scope | 0.250 | 0.250 | 1.000 | 1.000 | Not scored | Not scored |

Mean generation latency was 8.06s local / 0.97s strong for simple cases and 9.83s / 1.55s for multi-document cases. The strict substring metric is not semantic correctness and may penalize valid paraphrases.

Groq returned HTTP 429 during two judge evaluations. All 47 candidate items ran for each model, but that run's aggregate does not include successful judge-evaluation denominators, so faithfulness/relevancy means must be treated as provisional and potentially partial. The evaluator now configures up to five provider retries and emits metric counts alongside means; this change is tested but has not yet been validated with another full Cloud run. The judge is unblinded and has not been checked against human ratings. The runs are development-set evidence, not held-out quality estimates.

- [Local model experiment](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=054d3c7108e3b020)
- [Strong model experiment](https://cloud.langfuse.com/project/cmvj0s4r600axad0jz9ji8ki1/experiments/results?baseline=3689dd9781ee1a9e)

`pytest -q` is the reproducible test command. Latest full run: **72 passed in 4.67s**; use pytest's output as the source of truth.

Faithfulness and answer relevancy are LLM-as-a-judge metrics; they are comparative indicators, not human-validated truth. Use the same judge model and fixed corpus across candidate runs, review disagreements manually, and keep held-out questions separate from the development set.
