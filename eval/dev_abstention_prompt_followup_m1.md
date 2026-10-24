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

There was exactly one discordant pair: the candidate fixed `rag-057`, with zero
regressions. The exact two-sided paired sign/McNemar test is p=1.0, so this
single disagreement provides no evidence that the candidate is better than
chance. Wilson 95% intervals are 5.5–37.6% for baseline (3/19) and 2.9–31.4%
for candidate (2/19); they overlap substantially.

On the same 14 answerable controls from the initial sample, false abstentions
were 0/14 for both prompts (Wilson 95% interval 0–21.5% for each). This is a
limited control sample, not all 41 answerable development questions. Normalized
expected-fragment matches were 12/14 for baseline and 13/14 for candidate;
that lexical score is not a semantic accuracy measure.

## Manual label audit of remaining errors

- `rag-059` asks who must sign an inventory discrepancy record. The receiving
  procedure says to create the record and that an employee enters the receipt
  while a manager approves it; it does not identify who signs the record. The
  `in_scope_missing` label is appropriate.
- `rag-060` asks when loyalty points are credited after payment. The loyalty
  document says points are converted according to the program at transaction
  time; it does not specify when they are credited. The `in_scope_missing`
  label is appropriate.

These small synthetic development results do not establish a reliable
improvement; the appropriate conclusion is **no detected regression**, not
that the candidate is better. Four of the 19 missing-info cases have no
retrieved context and are refused before prompt evaluation; the model-level
refusal challenge is therefore represented by at most 15 cases. The 14
answerable controls and two far-topic cases were run only in the initial
sample. Baseline was always run before candidate for each question.

The existing development set already contains 19 `in_scope_missing` cases, so
the immediate gap is not raw count. Next, independently review the question
labels and add examples only for unrepresented failure patterns; then freeze
one candidate prompt and scoring code before preparing a new held-out set.
M2 was not used in this follow-up and was not modified.
