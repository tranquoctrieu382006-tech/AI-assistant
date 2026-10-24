# Development prompt experiment follow-up — M1

This follow-up first evaluates the 11 `in_scope_missing` questions omitted from
the initial sample, then the 27 answerable questions omitted from it. It uses
the same frozen M1 corpus, retrieval settings, local model, baseline prompt,
and single candidate instruction. Each question is retrieved once and the same
context is used for both prompt variants. Results are in
`eval/dev_abstention_prompt_missing_followup_m1.json` and
`eval/dev_abstention_prompt_answerable_followup_m1.json`.

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

Across all 41 answerable cases, end-to-end false abstentions were 1/41 for both
prompts (Wilson 95% interval 0.4–12.6%). This shared refusal was `rag-029`,
where retrieval returned no context at the configured cutoff; neither prompt
was called. On the other 40 answerable cases that reached the model, neither
prompt abstained (0/40; Wilson upper bound 8.8%). Thus no prompt-induced false
abstention was observed in this development run.

Normalized expected-fragment matches were 36/41 for baseline and 37/41 for
candidate. The one lexical-match difference is `rag-031`; this substring metric
is not a semantic accuracy measure and does not establish a quality gain.

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
refusal challenge is therefore represented by at most 15 cases. Only two of
the five far-topic cases were run, and both had no context, so they do not test
the candidate prompt. Baseline was always run before candidate for each
question.

The existing development set already contains 19 `in_scope_missing` cases, so
the immediate gap is not raw count. Next, independently review the question
labels and add examples only for unrepresented failure patterns; then freeze
one candidate prompt and scoring code before preparing a new held-out set.
M2 was not used in this follow-up and was not modified.
