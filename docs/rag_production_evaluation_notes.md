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

`RERANKER_POOL_SIZE` (default 20) controls the maximum number of candidate
pairs sent to the cross-encoder, including hybrid mode. Increasing it can
increase CPU latency; benchmark it on the deployment hardware before raising
the default.

## Runtime review follow-up

- PDF page overlap now keeps the whole next-page chunk when it fits the budget,
  avoiding a one-word remainder chunk.
- Structured answer parsing uses `OutputFixingParser` with parser feedback;
  its repair prompt is format-only and forbids adding or inferring facts. Since
  it does not receive the original context/question, this is not a factual
  verification step. Repair calls are tagged `rag-output-format-repair` so
  their frequency can be reviewed alongside a human-checked sample.
- Database pool checkout and PostgreSQL connection attempts use the configured
  `DATABASE_CONNECTION_TIMEOUT_SECONDS` (default 3 seconds).
- Long follow-ups containing Vietnamese pronouns such as “nó”, “họ”, “ấy”, or
  “kia” are sent through the question rewriter.
- Stream closure handles both task cancellation and generator closure. A turn
  is saved only if at least one non-empty token was generated; saved partial
  answers are explicitly marked incomplete in conversation history.
- BM25 rebuild and invalidation are serialized; warm-path revision checks and
  copies stay outside the lock using a local reference. Empty corpus replacement
  now raises instead of deleting the indexed table, and the unused `sources`
  parameter was removed.
- `X-Forwarded-For` still assumes the trusted proxy overwrites the header or
  appends the client address at the end; multi-proxy chains require a trusted
  proxy-chain policy before changing that interpretation.

The saved M2 manifest also does not validate against the current branch: its
recorded M1/M2 code hashes differ from the checked-out versions (including the
manifest writers and protocol/test files). The frozen evaluator correctly
stops on this mismatch. Do not rewrite those hashes to force a pass; treat a
new run as a fresh development milestone and create a newly reviewed held-out
set before making claims from a later configuration.
