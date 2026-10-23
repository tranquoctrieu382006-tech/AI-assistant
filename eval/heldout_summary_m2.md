# Held-out RAG M2 — exploratory result

## Frozen setup

- Corpus: 12 synthetic business documents, 30 indexed chunks.
- Retrieval: heading chunking, 900-character chunks, 150-character overlap,
  `bge-m3`, top-k 5, relevance cutoff 0.5.
- Held-out: 50 questions, 10 each for simple, paraphrase, multi-document,
  out-of-scope far, and in-scope missing.
- The held-out set was committed before its first evaluation. Its aggregate-only
  output is `eval/rag_heldout_m2_results.json`; the held-out manifest and hash
  are `eval/heldout_manifest_m2.json`.

## Results

| Measure | Result |
|---|---:|
| Context recall@5 | 0.917 (30 answerable cases) |
| Case-level bootstrap 95% interval | 0.850–0.983 |
| False positives, in-scope missing | 10/10 (1.000; Wilson 95% interval 0.722–1.000) |
| False positives, out-of-scope far | 0/10 (0.000; Wilson 95% interval 0.000–0.278) |
| False negatives, answerable cases | 0/30 |

Here, a false positive means retrieval returns a result at or above the cutoff
for a question expected to be refused. The 10/10 result is a strong warning
that the current cosine cutoff does not reject these same-domain missing-info
questions. The far-topic result is based on only 10 examples.

## Limits and next action

Codex authored this held-out set from the frozen corpus; an independent person
has not reviewed it. The corpus is synthetic, and the test measures retrieval
and its cutoff, not answer-generation quality. The bootstrap interval is
case-level; several questions share the same small document set, so dependence
may make it optimistic. Treat all metrics as exploratory.

Keep M2 sealed while investigating threshold or routing changes on development.
If the held-out findings influence a system change, M2 becomes development
evidence; create a fresh held-out set for the next independent evaluation.
