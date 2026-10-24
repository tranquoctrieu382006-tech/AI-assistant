# Production-path evaluation notes

The frozen M1 retrieval evaluator measures vector similarity only. The
aggregate M2 evaluator preserves that exact scorer for auditability. Current
API-path development measurements are available through
`scripts/evaluate_rag_production.py` or the `P-current` row from
`scripts/evaluate_rerank_configs.py`. They call `retrieve_and_gate` with the
active vector/hybrid mode, cosine gate, and production reranker candidate pool.
These fresh development results do not replace or amend M1/M2.

For multi-source questions, source Hit@k requires every expected source.
First-relevant-source MRR requires only the first expected source and can look
optimistic when some required sources are missing. Interpret it alongside
context recall and expected-source coverage.

The current 12-document, 30-chunk demo corpus is too small for top-k metrics to
support generalization claims. The historical hybrid comparison predates the
per-chunk cosine gate. Historical R10/HR10 used 10 candidates, unlike the API's
current reranker pool of 20 and top 5 output. Run the production-path evaluator
against a live vector store before making a new configuration decision.

The saved M2 manifest also does not validate against the current branch: its
recorded M1/M2 code hashes differ from the checked-out versions (including the
manifest writers and protocol/test files). The frozen evaluator correctly
stops on this mismatch. Do not rewrite those hashes to force a pass; treat a
new run as a fresh development milestone and create a newly reviewed held-out
set before making claims from a later configuration.
