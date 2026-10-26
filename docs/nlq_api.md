# NLQ API

`POST /api/nlq/ask` runs the schema-linking, SQL generation, validation, read-only execution, bounded repair (up to two attempts), and Vietnamese interpretation graph. The request uses the existing API key protection. The response contains `sql`, `columns`, `rows`, `answer`, and `repaired`.

## Read-only database account

Set `NLQ_DATABASE_URL` to a dedicated PostgreSQL login. The application does not fall back to `DATABASE_URL`. For the sample sales database, create a role and grant only business data columns. Run as a database administrator and set a unique password outside source control:

```sql
CREATE ROLE ai_nlq_reader LOGIN PASSWORD 'replace-with-a-secret';
GRANT CONNECT ON DATABASE ai_assistant_demo TO ai_nlq_reader;
GRANT USAGE ON SCHEMA public TO ai_nlq_reader;
GRANT SELECT ON categories, suppliers, products, warehouses, orders,
  order_items, inventory, stock_movements TO ai_nlq_reader;
GRANT SELECT (customer_id, city, member_tier, joined_on)
  ON customers TO ai_nlq_reader;
```

This account has no write grants and cannot read customer names or email addresses. Do not grant it access to `rag_chunks`. The API independently starts a read-only transaction, enforces a statement timeout, caps rows at `NLQ_MAX_ROWS`, rejects multiple statements and non-SELECT AST nodes, restricts referenced tables to the selected schema, and blocks sensitive result columns. Keep the database grants narrow even with these application checks.

Schema linking samples up to six short values from text or enum columns whose names indicate categorical fields (such as status, type, tier, or category). These sample values are included in the Gemini schema prompt and Langfuse trace when tracing is configured. `user_id` from the request is also propagated as a Langfuse user attribute. Send only the identifier you intend to associate with that trace.

Configure `NLQ_MODEL` to `fast`, `strong`, `local`, or a model name already supported by the LLM registry. `NLQ_MAX_REPAIRS` is limited to 0–2. Each NLQ model call defaults to a 60-second timeout and zero retries (`NLQ_MODEL_TIMEOUT_SECONDS`, `NLQ_MODEL_MAX_RETRIES`) so serial calls cannot hang for minutes on provider retries. Application logs report each graph stage and its duration without logging user IDs, SQL, or prompt contents. Langfuse tracing is attached to the graph and its model calls when credentials are configured; request session IDs are propagated. Flush traces at application shutdown or after a bounded manual smoke run, not inside the API request.

Example:

```http
POST /api/nlq/ask
X-API-Key: <configured API key>
Content-Type: application/json

{"question":"Doanh thu theo tháng trong năm 2025 là bao nhiêu?","session_id":"demo-27-10","user_id":"demo-user-01"}
```

The endpoint is unavailable until the separate read-only database URL is configured. The `scripts/evaluate_nlq_model.py` evaluator remains the offline model comparison path; it can evaluate `local`, `fast`, and `strong` roles against the checked-in synthetic golden set.
