# NLQ API

`POST /api/nlq/ask` runs a local scope check, SQL generation, validation, read-only execution, bounded repair (up to two attempts), and Vietnamese interpretation graph. It supplies the full nine-table schema to SQL generation so scope checking does not add an LLM call. The request uses the existing API key protection. The response contains `sql`, `columns`, `rows`, `answer`, `repaired`, and `truncated`.

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

This account has no write grants and cannot read customer names or email addresses. Do not grant it access to `rag_chunks`. The API independently starts a read-only transaction, enforces a statement timeout, caps rows at `NLQ_MAX_ROWS`, rejects multiple statements and non-SELECT AST nodes, restricts referenced tables and columns to the business schema available to the reader, and blocks sensitive result columns. The scope check is lexical; all business tables are sent to SQL generation after an in-scope match. Keep the database grants narrow even with these application checks.

Schema metadata and up to six short categorical samples are cached for ten minutes; the NLQ reader uses a dedicated pool with at most eight connections. The local scope check uses Vietnamese-aware lexical matching. Each in-scope request calls the model for SQL and interpretation, with up to two repair calls. Interpretation receives only the first 25 result rows and is told when the query result was capped. Because this endpoint does not authenticate an end-user identity, client-supplied `user_id` is ignored and is not sent to Langfuse.

Configure `NLQ_MODEL` to `fast`, `strong`, `local`, or a model name already supported by the LLM registry. `NLQ_MAX_REPAIRS` is limited to 0–2. Each NLQ model call defaults to a 60-second timeout. One retry is allowed for HTTP 429 and 5xx provider errors; timeout errors are not retried (`NLQ_MODEL_TIMEOUT_SECONDS`, `NLQ_MODEL_MAX_RETRIES`). Application logs report each graph stage and its duration without logging user IDs, SQL, or prompt contents. Langfuse tracing is attached to the graph and its model calls when credentials are configured; request session IDs are propagated. Flush traces at application shutdown or after a bounded manual smoke run, not inside the API request.

Example:

```http
POST /api/nlq/ask
X-API-Key: <configured API key>
Content-Type: application/json

{"question":"Doanh thu theo tháng trong năm 2025 là bao nhiêu?","session_id":"demo-27-10"}
```

Response (`HTTP 200 OK`):
```http
HTTP/1.1 200 OK
Content-Type: application/json
X-Trace-Id: 2cf23a0491e0773c3458cc76c295e8d7

{
  "sql": "SELECT DATE_TRUNC('month', order_date) AS month, SUM(total_amount) AS revenue FROM orders WHERE EXTRACT(year FROM order_date) = 2025 GROUP BY 1 ORDER BY 1;",
  "columns": ["month", "revenue"],
  "rows": [
    {"month": "2025-01-01", "revenue": 125000000},
    {"month": "2025-02-01", "revenue": 98000000}
  ],
  "answer": "Tổng doanh thu năm 2025 theo từng tháng được liệt kê trong bảng kết quả.",
  "repaired": false,
  "truncated": false,
  "trace_id": "2cf23a0491e0773c3458cc76c295e8d7"
}
```

- `trace_id` *(string | null)*: Mã trace 32 ký tự hex trên Langfuse (hoặc `null` nếu hệ thống chưa bật cấu hình Langfuse). Header `X-Trace-Id` được trả kèm khi `trace_id` khả dụng.


The endpoint is unavailable until the separate read-only database URL is configured. The `scripts/evaluate_nlq_model.py` evaluator remains the offline model comparison path; it can evaluate `local`, `fast`, and `strong` roles against the checked-in synthetic golden set.
