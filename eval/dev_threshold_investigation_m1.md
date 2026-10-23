# Development-only retrieval cutoff investigation

This analysis uses only the 65-case development set and the frozen M1 corpus.
It does not change the production cutoff and does not use held-out M2 results.
“Accepted” means the highest retrieved chunk score is at or above the candidate
cutoff. A false positive is an accepted `in_scope_missing` or
`out_of_scope_far` case; a false negative is a rejected answerable case.

## Score overlap

| Group | n | Min | P25 | Median | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Answerable | 41 | 0.479 | 0.632 | 0.659 | 0.691 | 0.785 |
| In-scope missing | 19 | 0.362 | 0.485 | 0.573 | 0.593 | 0.778 |
| Out-of-scope far | 5 | 0.261 | 0.393 | 0.434 | 0.448 | 0.499 |

The in-scope-missing and answerable score ranges overlap substantially. The
highest missing-info score (0.778) exceeds the answerable median (0.659), so a
single cosine cutoff cannot cleanly separate these classes on this corpus.

## Fixed threshold sweep

| Cutoff | Answerable rejected | In-scope missing false positives | Far-topic false positives |
|---:|---:|---:|---:|
| 0.45 | 0/41 | 14/19 | 1/5 |
| 0.50 (current) | 1/41 | 14/19 | 0/5 |
| 0.55 | 2/41 | 12/19 | 0/5 |
| 0.60 | 5/41 | 4/19 | 0/5 |
| 0.65 | 19/41 | 1/19 | 0/5 |
| 0.70 | 34/41 | 1/19 | 0/5 |
| 0.75 | 40/41 | 1/19 | 0/5 |

These are development counts, not validated rates. The far-topic group has only
five examples, so its entries are counts only.

At 0.60, the five answerable cases rejected are `rag-022`, `rag-025`, `rag-029`,
`rag-030`, and `rag-038`. The four in-scope-missing false positives left are
`rag-054` (same-day delivery), `rag-055` (return request channel), `rag-056`
(warranty after the stated period; score 0.778), and `rag-059` (who signs an
intake discrepancy record). These examples support the score-overlap finding:
the missing questions retrieve highly topical chunks, while some supported
questions score below 0.60.

## Interpretation

Raising the single cutoff from 0.50 to 0.60 would reduce development
false positives by 10 cases, but reject four additional supported questions.
At 0.65, compared with 0.50, it would remove 13 additional missing-info
false positives, while rejecting 19/41 supported cases in total (18 more
rejections than at 0.50). A global threshold increase is therefore not a good
fix by itself.

The next development experiment should test a second-stage evidence/support
check or reranker while keeping M2 unchanged. Any configuration change informed
by M2 needs a fresh held-out set for independent confirmation. All questions in
this development analysis were authored in-project and have not received an
independent review.
