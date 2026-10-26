"""Run one synthetic NLQ request and verify its observation in Langfuse."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from uuid import uuid4

from langfuse.langchain import CallbackHandler

from app.core.config import settings
from app.core.tracing import get_langfuse_client, init_tracing
import app.core.nlq_graph as nlq_graph


def main() -> int:
    init_tracing()
    settings.nlq_max_repairs = 1
    settings.nlq_model_timeout_seconds = 60
    handler = CallbackHandler()
    nlq_graph.get_langfuse_handler = lambda: handler
    session_id = f"nlq-connectivity-smoke-{uuid4().hex[:10]}"

    error = None
    try:
        result = nlq_graph.ask_nlq(
            "Có bao nhiêu đơn hàng?",
            session_id=session_id,
            user_id="synthetic-nlq-connectivity-check",
        )
        print(
            "NLQ E2E: OK "
            f"(rows={len(result.get('rows', []))}, repaired={result.get('repaired', False)})"
        )
    except Exception as exc:
        error = exc
        print(f"NLQ E2E: FAIL ({type(exc).__name__})")

    trace_id = handler.last_trace_id
    if not trace_id:
        print("LANGFUSE TRACE: callback returned no trace ID")
        return 2 if error is None else 1
    print(f"LANGFUSE TRACE ID: {trace_id}")

    client = get_langfuse_client()
    client.flush()
    print("LANGFUSE FLUSH: OK")

    start = datetime.now(timezone.utc) - timedelta(minutes=5)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    for attempt in range(7):
        response = client.api.observations.get_many(
            session_id=session_id,
            from_start_time=start,
            to_start_time=end,
            limit=100,
        )
        observations = response.data or []
        if observations:
            names = sorted({item.name for item in observations if item.name})
            print(f"LANGFUSE TRACE: VISIBLE ({len(observations)} observations)")
            print("LANGFUSE OBSERVATIONS: " + ", ".join(names))
            return 1 if error is not None else 0
        if attempt < 6:
            time.sleep(5)

    print("LANGFUSE TRACE: flushed but not yet readable after 30 seconds")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
