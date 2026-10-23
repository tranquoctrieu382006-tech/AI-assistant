# RAG evaluation v1

## What these checks measure

- `data/rag_golden_set.json` currently has 47 classified cases: 33 simple, four paraphrases, two multi-document questions and eight expected abstentions. Hit@1/3/5 and MRR include answerable questions; a multi-document hit requires every expected source. Source coverage is reported separately so partial multi-source retrieval remains visible.
- The eight out-of-scope questions are still a small sample; abstention rates are descriptive only, not a stable quality estimate.
- `data/nlq_golden_set.json` has 30 cases covering joins, groupings, date ranges and aggregates, with read-only reference SQL and expected results.
- The cutoff `RAG_MIN_RELEVANCE_SCORE=0.5` was selected on the development corpus. These results are calibration observations, not held-out quality evidence.
- `python -m scripts.evaluate_rag_answers` sends the golden questions to the running API and checks `answer_contains`. This is a simple substring smoke check, not semantic evaluation. The expected text is deliberately specific; for example, the exchange window is `7 ngày lịch`.
- `python -m scripts.evaluate_nlq_golden` executes checked-in reference SQL read-only. Its rate is **seed consistency**, not model accuracy. `python -m scripts.evaluate_nlq_model --model-role local` separately evaluates generated SQL against those expected results.
- The model evaluator supplies the actual `CREATE TABLE` statements from `schema_seed.sql`, compares rows in order when generated SQL has a top-level `ORDER BY`, and allows semicolons inside quoted strings/comments while rejecting multiple statements.

## Previous development-set run (2026-10-23; original 35-case set)

The refreshed corpus contains 12 business documents and 27 chunks. On 33 answerable questions, Hit@5 was 1.0 and MRR was 0.9394; both out-of-scope questions were rejected at the 0.5 cutoff (abstention accuracy 1.0). Four answerable questions had the expected source at rank 2. All 30 reference SQL cases matched the seed. This remains development-set evidence because the threshold was calibrated on this corpus.

One local RAG answer smoke test returned the correct exchange window with sources. One local NLQ model smoke test generated a count query with matching execution; one case is not a model-quality estimate.

## Expanded golden set and evaluator changes

The 47-case set adds paraphrases, two questions requiring evidence from two separate documents, and six additional out-of-scope questions. Every case has a category. The evaluator reports Hit@1/3/5, MRR, expected-source coverage, and per-category results. Answer smoke evaluation checks every required substring for multi-part answers.

Live expanded-set retrieval run with local bge-m3 and PostgreSQL/pgvector: 39 answerable cases, Hit@1 **0.744**, Hit@3 **0.846**, Hit@5 **0.872**, MRR **0.808**, expected-source coverage@5 **0.872**; abstention accuracy **8/8**. Simple questions reached Hit@3/5 **1.0**. All four paraphrases missed their expected source in top five. One multi-document case retrieved both sources at ranks 2 and 4; the other missed both. These are development diagnostics on the same 27-chunk corpus, not held-out results. The eight negative cases are too few for a stable abstention estimate.

`pytest -q` is the reproducible test command. The latest run after these evaluator changes reported **62 passed**: 57 test functions plus five parameterized instances. Use pytest's collection/run output as the source of truth.

Faithfulness, answer relevancy, a two-model RAG comparison and a Langfuse Dataset/Experiment have not been run. The local evaluator does not upload traces. Verify the project's data-sharing scope and explicitly configure upload before any Cloud experiment.
