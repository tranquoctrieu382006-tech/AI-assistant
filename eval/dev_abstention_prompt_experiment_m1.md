# Development prompt experiment — M1

## Setup

- Development M1 only; no held-out M2 examples or results used to select the candidate.
- Fixed retrieval: current M1 corpus, `bge-m3`, top-k 5, cutoff 0.50. Each prompt
  variant received the same retrieved context for each question; retrieval was
  run once per question and its context reused for both variants.
- Local `gemma3:12b`, temperature 0. A deterministic stratified sample selected
  8 `in_scope_missing`, 8 `simple`, 4 `paraphrase`, 2 `multi_document`, and 2
  `out_of_scope_far` cases (24 total).
- Candidate is the existing prompt with one added rule: check every essential
  requested fact, and abstain if a requested value, time, entity, condition, or
  procedure is not explicitly supported. Production prompt/configuration was
  not changed.

## Results

| Measure | Baseline | Candidate |
|---|---:|---:|
| Correct abstention decision | 22/24 | 23/24 |
| False answers on `in_scope_missing` | 2/8 | 1/8 |
| False abstentions on answerable cases | 0/14 | 0/14 |
| Expected-fragment matches on answerable cases | 12/14 | 13/14 |
| Generation/parse errors | 0 | 0 |

The one abstention decision that changed was `rag-057`: baseline returned “Hoàn
tiền về phương thức thanh toán ban đầu” despite the question also asking which
bank and whether a fee applies; the candidate correctly abstained. `rag-031`
also changed from a short correct answer that failed the normalized
expected-fragment check to a fuller answer that matched it. No answerable case
was newly refused.

## Interpretation and limits

This small, deterministic development sample does not establish a quality
gain. Only 8 same-topic missing-info examples were included; expected-fragment
matching is lexical, not semantic; all documents are synthetic; and baseline
was run before candidate for each case. The far-topic cases had no retrieved
context, so they do not test the candidate prompt itself. Review the full
per-case output in `eval/dev_abstention_prompt_experiment_m1.json`.

Next: expand the development sample for `in_scope_missing` and manually review
all disagreement cases, then freeze exactly one candidate prompt and evaluator
before any new held-out run. If M2 findings are used to guide further changes,
M2 is no longer independent and a fresh held-out set is required.
