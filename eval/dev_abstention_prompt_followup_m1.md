# Development prompt experiment follow-up — M1

This follow-up evaluates the 11 `in_scope_missing` development questions that
were not included in the initial stratified prompt sample. It uses the same
frozen M1 corpus, retrieval settings, local model, baseline prompt, and single
candidate instruction. The initial sample remains recorded separately in
`eval/dev_abstention_prompt_experiment_m1.json`.

## Full in-scope-missing group

Combining the initial 8 cases with these 11 gives:

| Measure | Baseline | Candidate |
|---|---:|---:|
| False answers | 3/19 | 2/19 |
| Correct refusals | 16/19 | 17/19 |
| Generation/parse errors | 0 | 0 |

The candidate fixes `rag-057`; it does not change the two remaining false
answers, `rag-059` and `rag-060`. No candidate regressions occurred in the
19-case group. In the 14 answerable controls from the initial sample, neither
prompt falsely abstained; normalized expected-fragment matches were 12/14 for
the baseline and 13/14 for the candidate.

## Manual label audit of remaining errors

- `rag-059` asks who must sign an inventory discrepancy record. The receiving
  procedure says to create the record and that an employee enters the receipt
  while a manager approves it; it does not identify who signs the record. The
  `in_scope_missing` label is appropriate.
- `rag-060` asks when loyalty points are credited after payment. The loyalty
  document says points are converted according to the program at transaction
  time; it does not specify when they are credited. The `in_scope_missing`
  label is appropriate.

These are still small, synthetic development results and do not establish a
reliable improvement. Four of the 19 missing-info cases have no retrieved
context and are refused before prompt evaluation; the model-level refusal
challenge is therefore represented by at most 15 cases. The 14 answerable
controls and two far-topic cases were run only in the initial sample. The
answer-fragment metric is lexical, not semantic. Baseline was always run before
candidate for each question.

The existing development set already contains 19 `in_scope_missing` cases, so
the immediate gap is not raw count. Next, independently review the question
labels and add examples only for unrepresented failure patterns; then freeze
one candidate prompt and scoring code before preparing a new held-out set.
M2 was not used in this follow-up and was not modified.
