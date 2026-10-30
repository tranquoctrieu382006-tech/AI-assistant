# RAG evaluation protocol (development M1)

## Frozen development baseline

The current development baseline targets the 12 business documents in
`data/demo_sales_warehouse` (excluding the repository README, data dictionary,
dataset notes, and SQL seed). It uses heading-bounded chunking, 900-character
chunks, 150-character overlap, `bge-m3`, and top-k 5. This is a development
benchmark, not held-out evidence.

`eval/development_manifest_m1.json` records SHA-256 hashes for the corpus,
development sets, scorer, and data checks. Regenerate it after any intended
development change with:

```powershell
.venv\Scripts\python.exe -m scripts.write_development_manifest
```

Manifest hashes are canonical: text files are hashed with LF line endings and
binary files (`.pdf`, `.docx`, `.png`, `.zip`) verbatim, so the baseline does
not depend on the checkout's line endings. `.gitattributes` enforces LF for
text, and CI checks out on Linux. After cloning on Windows, or after changing
`.gitattributes`, normalise the index once:

```powershell
git add --renormalize .
```

Before regenerating a baseline, inspect drift without overwriting anything:

```powershell
.venv\Scripts\python.exe -m scripts.report_manifest_drift
```

That command prints, per recorded entry, whether the canonical hash still
matches and which line endings the working file uses. It exits non-zero when
anything drifted. Do not regenerate a baseline to make drift disappear: review
each drifted file, commit the intended change, then freeze a new baseline with a
new label (`--label`) and write the held-out manifest against it.

The M1 manifest and its matching Git commit must be committed before creating
or running a held-out set. A later held-out manifest must record its own hash
and the M1 commit/hash. A corpus or scorer change requires a new M1 baseline and
a newly reviewed held-out set.

## Labels and metrics

Answerable cases carry source-linked evidence text, independent of chunk IDs.
The deterministic validator verifies that each evidence passage exists in its
source. `context_recall@5` checks whether each evidence passage occurs in the
retrieved top five, joining consecutive chunks from the same source to handle
chunk boundaries. The retrieval evaluator also reports source-level Hit@k and
MRR for continuity.

Abstention labels distinguish `out_of_scope_far` from `in_scope_missing`.
False positive means retrieval returns a result above the configured cutoff
for a case that should be refused. The report emits counts for all groups and
suppresses rates for groups with fewer than 10 cases. This is especially
important for the five `out_of_scope_far` examples. The development set now has
19 `in_scope_missing` examples; held-out results must meet the same minimum
before reporting a rate.

The expanded development questions and evidence labels were authored during
implementation and have not had an independent reviewer. Treat this result as
exploratory until a separate writer/reviewer creates or reviews held-out items.

## Held-out execution rules

- Create held-out only after corpus, development set, scorer, chunking settings,
  and top-k are frozen and recorded in M1.
- Prefer an independent question writer/reviewer. If unavailable, document
  self-review as a limitation.
- Commit the held-out hash as a separate milestone before its first run.
- Unit-test and run the full scoring pipeline on development before held-out.
- Held-out output before final report contains aggregates only. Per-case output
  may be inspected after the report is finalized; using it to change the system
  converts that set to development data.
- Infrastructure retries/resume are allowed only without changing inputs,
  prompts, configuration, or scoring code. Record each attempt.
- The initial held-out target is 30–50 RAG cases and is exploratory. Report
  paired bootstrap intervals and limitations; do not interpret a small sample
  as proof of equivalence or a small quality difference.
