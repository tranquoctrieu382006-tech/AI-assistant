# NLQ model comparison — 2026-10-27

Compared the 30 synthetic cases in `data/nlq_golden_set.json` against the sample PostgreSQL database using the `ai_nlq_reader` role. Each generated query ran in a read-only transaction with a 10-second statement timeout. Accuracy is exact result matching with row-order ignored unless SQL specifies `ORDER BY`.

| Model | Exact execution accuracy | Mean SQL generation latency | Input / output tokens | Estimated paid-tier cost per 30 cases |
|---|---:|---:|---:|---:|
| `ollama/gemma3:12b` | 16/30 (53.3%) | 8.052 s | 16,661 / 1,440 | $0 API cost* |
| `google_genai/gemini-3.5-flash-lite` | 18/30 (60.0%) | 0.946 s | 16,391 / 1,341 | $0.00827 |
| `google_genai/gemini-3.1-flash-lite` | 20/30 (66.7%) | 4.361 s | 16,391 / 4,802 | $0.01130 |

All three complete runs had 30 generated and executed cases, with zero SQL rejections or provider errors. Gemini runs were paced at 4.2 seconds between requests to stay below the observed 15 requests/minute free-tier limit. Cost is a paid-tier token equivalent; actual Gemini Free Tier billing was $0 for these requests. The local estimate excludes electricity and hardware.

**Recommendation:** use `gemini-3.5-flash-lite` for the demo default. Gemini 3.1 Flash-Lite scored two more cases, but returned a temporary 503 high-demand error during the final end-to-end run. Gemini 3.5 Flash-Lite completed all 30 cases and the full NLQ graph/trace smoke run, with lower latency and cost. The 30-case set is a development benchmark only; 12 of 30 still failed exact-result comparison, so do not treat the score as production accuracy.

Google Free Tier limits still apply. The unpaced Gemini 3.5 run hit the 15 RPM limit; the paced rerun completed all cases. Gemini 3.6 Flash was excluded from the comparison because this project key reached its 20-requests/day quota before a full 30-case run. Provider quotas can change by model and project; see [Google rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

Paid-tier cost estimates use Google's standard text rates: Gemini 3.5 Flash-Lite $0.30/$2.50 and Gemini 3.1 Flash-Lite $0.25/$1.50 per million input/output tokens. *Local inference has no API charge. See [Google Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).

The SQL guard rejected DELETE, multiple statements, `pg_sleep`, and access to `rag_chunks`; it accepted a normal SELECT and enforced `LIMIT 200`. The read-only role also lacks SELECT privilege on customer `email` and `full_name`. A fake-model graph check confirmed one repair turn changes a blocked DELETE to a valid SELECT and returns the expected count. No write query was executed.

The final Gemini 3.5 Flash-Lite end-to-end run returned one result row and completed with no repair. Langfuse accepted and returned trace `ec5320e8e3640b1ac7531a157c1c0c5d` with 18 observations, including schema linking, SQL generation, validation, execution, and answer interpretation.

Detailed per-case outputs: `nlq_model_local_metrics_2026-10-27.json`, `nlq_model_gemini35lite_2026-10-27.json`, and `nlq_model_gemini31lite_2026-10-27.json`. The earlier Groq comparison is historical and superseded by this Gemini comparison.
