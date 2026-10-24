# AI label-review reconciliation for the next benchmark revision

This note compares the attached AI review with the frozen M1 golden set using
`docs/rag_labeling_rules_v2.md`. It is an adjudication proposal, not a relabeling
of M1. The M1 golden file, its manifest, and held-out results remain unchanged.

## Version and format checks

- The repository's `data/rag_golden_set.json` parses successfully and contains
  65 cases: 35 `simple`, 4 `paraphrase`, 2 `multi_document`, 19
  `in_scope_missing`, and 5 `out_of_scope_far`.
- The pasted question list had an extra quote in `rag-064`; this was a copy
  error in the chat response. The repository JSON is valid.
- The review's 13 `IN_SCOPE_MISSING`, 4 `OUT_OF_SCOPE_FAR`, and 2 `AMBIGUOUS`
  among the original 19 missing-info candidates sum to 19. This is a
  reclassification, not evidence of a different golden-set version.
- `rag-033` and `rag-049` are exact duplicate questions. Their expected source
  documents differ, so they can remain as a source-attribution pair, but must
  share a duplicate cluster and must not count as independent question
  evidence.

## Proposed dispositions for differences

| ID | M1 label | AI review | Proposal for next revision | Reason |
|---|---|---|---|---|
| `rag-005` | `in_scope_missing` | `OUT_OF_SCOPE_FAR` | `OUT_OF_SCOPE_FAR` | The corpus has workplace rules but no substantive coverage of company office locations. |
| `rag-035` | `in_scope_missing` | `AMBIGUOUS` | Rewrite or exclude pending clarification | “Khách” could mean customers or be a typo for the company; the referent changes the scope. |
| `rag-045` | `in_scope_missing` | `OUT_OF_SCOPE_FAR` | `OUT_OF_SCOPE_FAR` | A generic mention of business results as a bonus factor is not substantive coverage of monthly revenue reporting. |
| `rag-046` | `in_scope_missing` | `OUT_OF_SCOPE_FAR` | `OUT_OF_SCOPE_FAR` | Employee policies do not cover branches, rosters, or headcount. |
| `rag-047` | `in_scope_missing` | `OUT_OF_SCOPE_FAR` | `OUT_OF_SCOPE_FAR` | A general data-protection rule is not an employee-record lookup source. |
| `rag-060` | `in_scope_missing` | `AMBIGUOUS` | Rewrite; likely `IN_SCOPE_MISSING` after clarification | The question asks when points are credited; the source phrase about points being converted at transaction time is unclear about earning versus redemption. |

These proposals follow the new scope rule and source inspection, but the AI
review was not fully blind for `rag-042`, `043`, `044`, `059`, or `060`. Do not
count its agreement/disagreement on those items as independent confirmation.

## Borderline cases to include in independent adjudication

- `rag-043`: keep as a provisional `IN_SCOPE_MISSING`. The order/complaint
  material covers customer support, but gives no hotline number. Its current
  label was already known to the reviewer, so obtain a fresh blind judgment.
- `rag-052`: monthly meal allowance versus per-day meals in travel expenses.
  The reviewer calls it missing but explicitly notes these are different
  benefits. A second reviewer should decide whether the shared compensation
  domain is substantive enough under the rule.
- `rag-053`: accepted e-wallets versus refunds to the original payment method.
  These are related payment topics but different operations. Adjudicate before
  keeping `IN_SCOPE_MISSING`; under the strict same-process reading, this may
  be `OUT_OF_SCOPE_FAR`.
- `rag-054`: provisional `IN_SCOPE_MISSING`; the shipping policy directly gives
  delivery timing but does not say whether same-day service exists. Its label
  also was known to the reviewer.
- `rag-059`: retain as `IN_SCOPE_MISSING` provisionally. Source documents say
  to create a discrepancy record but do not say who signs it. The reviewer had
  already seen this label, so this is not an independent confirmation.

## Duplicate and question-quality handling

- Keep `rag-033` and `rag-049` only if testing which of two documents supplies
  the answer is intentional. Assign a shared duplicate cluster; otherwise remove
  one from the answerability metric.
- Rewrite or exclude `rag-035` until its subject is clear. If its intended
  wording is the company-office question, it duplicates `rag-005` and should
  not count as an independent case.
- Rewrite `rag-060` to ask explicitly when points are recorded to the account
  after payment, then label against the frozen source text without looking at
  retrieval or model outcomes.

## Next step

Have an independent reviewer adjudicate the borderline cases while blind to
model outputs. Apply agreed changes only in a new golden-set revision, attach
duplicate-cluster metadata, regenerate its manifest, and rerun development
checks before creating a held-out set. Do not edit M1 in place.
