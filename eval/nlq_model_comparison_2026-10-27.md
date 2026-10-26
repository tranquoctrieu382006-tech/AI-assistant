# NLQ model comparison — task date 2026-10-27 (benchmark run 2026-10-26)

This is a seed-DDL SQL-generation benchmark, not an evaluation of the production LangGraph/schema-linking API. The artifacts use the planned task date (2026-10-27); benchmark runs were performed on 2026-10-26.

Compared the 30 synthetic cases in `data/nlq_golden_set.json` against the sample PostgreSQL database using the `ai_nlq_reader` role. Each generated query ran in a read-only transaction with a 10-second statement timeout. Historical results below used that 10-second setting; production defaults to 5 seconds, so latency and completion rates are not directly comparable. Accuracy is exact result matching with row-order ignored unless SQL specifies `ORDER BY`.

| Model | Strict execution accuracy | Relaxed execution accuracy* | Mean SQL generation latency | Input / output tokens | Estimated paid-tier cost per 30 cases |
|---|---:|---:|---:|---:|---:|---:|
| `ollama/gemma3:12b` | 16/30 (53.3%) | 18/30 (60.0%) | 8.052 s | 16,661 / 1,440 | $0 API cost* |
| `google_genai/gemini-3.5-flash-lite` | 18/30 (60.0%) | 24/30 (80.0%) | 0.946 s | 16,391 / 1,341 | $0.00827 |
| `google_genai/gemini-3.1-flash-lite` | 20/30 (66.7%) | 24/30 (80.0%) | 4.361 s | 16,391 / 4,802 | $0.01130 |

All three complete runs had 30 generated and executed cases, with zero SQL rejections or provider errors. Gemini runs were paced at 4.2 seconds between requests to stay below the observed 15 requests/minute free-tier limit. Cost is a paid-tier token equivalent; actual Gemini Free Tier billing was $0 for these requests. The local estimate excludes electricity and hardware.

**Recommendation:** use `gemini-3.5-flash-lite` for the demo default based on latency and cost, not benchmark accuracy. The strict gap between Gemini versions disappears under this relaxed diagnostic (24/30 each), so these results do not establish an accuracy winner. Gemini 3.1 Flash-Lite returned a temporary 503 high-demand error during the final end-to-end run. Both models completed their saved 30-case benchmark runs. The 30-case set is a development benchmark only; neither score is production accuracy.

Google Free Tier limits still apply. The unpaced Gemini 3.5 run hit the 15 RPM limit; the paced rerun completed all cases. Gemini 3.6 Flash was excluded from the comparison because this project key reached its 20-requests/day quota before a full 30-case run. Provider quotas can change by model and project; see [Google rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

Paid-tier cost estimates use Google's standard text rates: Gemini 3.5 Flash-Lite $0.30/$2.50 and Gemini 3.1 Flash-Lite $0.25/$1.50 per million input/output tokens. *Local inference has no API charge. See [Google Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).

The SQL guard rejected DELETE, multiple statements, `pg_sleep`, and access to `rag_chunks`; it accepted a normal SELECT and enforced `LIMIT 200`. The read-only role also lacks SELECT privilege on customer `email` and `full_name`. A fake-model graph check confirmed one repair turn changes a blocked DELETE to a valid SELECT and returns the expected count. No write query was executed.

The final Gemini 3.5 Flash-Lite end-to-end run returned one result row and completed with no repair. Langfuse accepted and returned trace `ec5320e8e3640b1ac7531a157c1c0c5d` with 18 observations, including schema linking, SQL generation, validation, execution, and answer interpretation.

Detailed per-case outputs: `nlq_model_local_metrics_2026-10-27.json`, `nlq_model_gemini35lite_2026-10-27.json`, and `nlq_model_gemini31lite_2026-10-27.json`. The earlier Groq comparison is historical and superseded by this Gemini comparison.


## Evaluator review and follow-up

The previous comparator ignored aliases but compared each row by position and required identical row width. This incorrectly failed answers with harmless extra columns. Retrospective relaxed rescoring raises Gemini 3.5 from 18 to 24 and Gemini 3.1 from 20 to 24; local Gemma rises from 16 to 18. They do **not** vindicate every reported mismatch:

- Gemini 3.5 cases 010, 013, 015, 017, 019, and 022 become relaxed matches. Case 014 remains a real business-definition disagreement: generated SQL excludes canceled orders while the golden revenue includes them. Confirm the intended definition before changing either prompt or expected SQL.
- Cases 016, 028, and similar ID/name-only outputs still omit a requested measure. They remain failures; treating an identifier alone as the whole answer would overstate quality. Case 026 also remains unresolved because a warehouse ID cannot safely be equated to its displayed warehouse name without an explicit mapping.
- Cases 021 and 027 contain ties. Their saved question wording did not specify a tie-break, although the golden query did. The questions now explicitly request ascending city/supplier ID as the tie-break. The old runs cannot be considered tie-fair and require rerunning for comparable scores.

The relaxed diagnostic ignores aliases and extra projected columns, accepts numeric differences up to one cent (or 1e-9 relative), and normalizes month-only values such as `1` to `2026-01` by month number. It still requires every golden value to appear, so missing requested metrics remain failures. This is diagnostic, not a substitute for semantic review.

The production and benchmark SQL prompts now request only asked dimensions/metrics, two-decimal rounding for currency/average prices, and deterministic secondary ordering for top-N ties. These changes guide future generations; they do not change historical outputs. A complete three-model rerun was not possible because neither `GOOGLE_API_KEY` nor `NLQ_DATABASE_URL` is configured in this workspace. Configure them locally and rerun to obtain comparable results with revised questions and prompts.
