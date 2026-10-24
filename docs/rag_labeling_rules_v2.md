# RAG golden-set labeling rules (next benchmark revision)

These rules apply when creating the next benchmark revision. They do not
retroactively change the frozen M1 labels or its results. Any relabeling or
question rewrite must create a new version and manifest.

## 1. Decide from the frozen corpus only

Reviewers receive the exact corpus version and questions containing only `id`
and `question`. They must not see current labels, expected answers, evidence
annotations, retrieved chunks, scores, model answers, or prompt identities until
their independent review is saved. Record the corpus version/hash with the
review.

For each question, first split the request into atomic facts or actions the
answer must cover. Do not infer unstated facts from general knowledge. Direct
paraphrase and deterministic, elementary calculation from explicit evidence
are allowed; assumptions about policy, intent, or real-world practice are not.

## 2. Assign one primary label

### `ANSWERABLE`

Use when the frozen corpus directly supports every essential part of the
question. For a multi-part question, all parts must be supported. Record the
shortest sufficient verbatim evidence span for each part and its source.

### `IN_SCOPE_MISSING`

Use when the question is clear and at least one passage substantively covers
the same subject and functional domain, but one or more essential requested
details are absent or only partly supported. Record the closest passage(s),
their source(s), and the exact missing detail. A passage that only shares a
word, mentions the company generally, or concerns a merely adjacent process is
not enough to establish in-scope coverage.

Examples of the distinction: a shipping policy that gives standard delivery
times is in scope for a question about same-day delivery availability; a
general workplace-hours rule is not, by itself, coverage of a company's
regional office locations.

### `OUT_OF_SCOPE_FAR`

Use when no passage substantively covers the question's subject and functional
domain. Mere lexical overlap, a broad corporate connection, or an unrelated
nearby topic does not make a question in scope. Record the closest passage
considered, if any, and explain why it concerns a different subject or process.

### `AMBIGUOUS`

Use only when the question has two or more reasonable interpretations, a
material typo/unclear referent changes the answer, or applicable source
passages conflict or are themselves materially ambiguous. Explain the
alternative interpretations and what clarification would resolve them.

Missing evidence alone is not ambiguity: a clear question whose requested
detail is absent from an otherwise relevant topic is `IN_SCOPE_MISSING`.
Likewise, do not mark a question ambiguous merely because its answer is
unexpected or the reviewer is uncertain; state low confidence and request
adjudication instead.

## 3. Multi-part and mixed-scope questions

- Mark a question `ANSWERABLE` only if every essential sub-question is
  answerable.
- If any essential part is missing but the corpus substantively covers the
  same functional domain, use `IN_SCOPE_MISSING` and list the unsupported part.
- If the question combines distinct topics, record a label for each atomic
  part in the review notes. For a single primary label, use `IN_SCOPE_MISSING`
  when at least one part is in scope but unanswered; use `OUT_OF_SCOPE_FAR`
  only when none of the unanswered parts has substantive corpus coverage.
- For multi-document questions, cite evidence from every required source and
  identify which sub-question each source supports.

## 4. Evidence and review record

For every case record:

- `label` and `confidence` (`high`, `medium`, or `low`);
- atomic sub-questions, when the question has multiple parts;
- exact source path and verbatim quote for each supported part;
- for abstention labels, the closest passage considered and the precise
  missing detail or scope mismatch;
- a short rationale, without model or retrieval results.

Do not cite a chunk index as evidence identity; chunking can change. Evidence
must be tied to source text and the frozen corpus version.

## 5. Duplicates, near-duplicates, and dependent cases

- Exact duplicate questions receive a shared `duplicate_cluster` identifier.
  Keep both only when they intentionally test retrieval of different sources;
  otherwise remove one from the scored set.
- Near-duplicates and paraphrases may remain when they test meaningful wording
  variation, but must share a cluster for reporting. Do not count members of a
  cluster as independent evidence when summarizing uncertainty.
- Ambiguous wording or likely typos should be rewritten before the next
  benchmark is frozen. Keep the original separately for traceability.

## 6. Adjudication and freezing

Have two reviewers label independently and blindly when possible. Resolve
disagreements by returning to the cited source text and these rules, not by
looking at model performance. If reviewers cannot resolve a case, mark it
`AMBIGUOUS` or exclude it from primary metrics with the reason documented.

Freeze the corpus, questions, labels, evidence, rules, evaluator, and retrieval
configuration together before running a new held-out evaluation. Do not tune
against held-out case-level results.
