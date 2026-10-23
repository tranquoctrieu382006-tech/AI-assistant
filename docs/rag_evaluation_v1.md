# RAG evaluation v1

## What these checks measure

- `data/rag_golden_set.json` has 35 cases (33 answerable and two expected abstentions), each with expected source and answer substring. Hit@k/MRR include only answerable cases; abstention is reported separately.
- `data/nlq_golden_set.json` has 30 cases covering joins, groupings, date ranges and aggregates, with read-only reference SQL and expected results.
- The cutoff `RAG_MIN_RELEVANCE_SCORE=0.5` was selected on the development corpus. These results are calibration observations, not held-out quality evidence.
- `python -m scripts.evaluate_rag_answers` sends the golden questions to the running API and checks `answer_contains`. This is a simple substring smoke check, not semantic evaluation. The expected text is deliberately specific; for example, the exchange window is `7 ngày lịch`.
- `python -m scripts.evaluate_nlq_golden` executes checked-in reference SQL read-only. Its rate is **seed consistency**, not model accuracy. `python -m scripts.evaluate_nlq_model --model-role local` separately evaluates generated SQL against those expected results.

## Development-set run (2026-10-23)

The refreshed corpus contains 12 business documents and 27 chunks. On 33 answerable questions, Hit@5 was 1.0 and MRR was 0.9394; both out-of-scope questions were rejected at the 0.5 cutoff (abstention accuracy 1.0). Four answerable questions had the expected source at rank 2. All 30 reference SQL cases matched the seed. This remains development-set evidence because the threshold was calibrated on this corpus.

One local RAG answer smoke test returned the correct exchange window with sources. One local NLQ model smoke test generated a count query with matching execution; one case is not a model-quality estimate.

Faithfulness, answer relevancy, a two-model RAG comparison and a Langfuse Dataset/Experiment have not been run. The local evaluator does not upload traces. Verify the project's data-sharing scope and explicitly configure upload before any Cloud experiment.
