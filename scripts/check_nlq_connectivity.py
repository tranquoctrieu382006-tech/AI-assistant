"""Bounded connectivity checks for the NLQ database, Gemini, and Langfuse."""
from __future__ import annotations

import sys
import re

from app.core.config import settings


def main() -> int:
    failures = 0
    try:
        from app.core.nlq_schema import _connect

        with _connect() as conn:
            role, read_only = conn.execute(
                "SELECT current_user, current_setting('transaction_read_only')"
            ).fetchone()
        print(f"POSTGRES: OK (role={role}, read_only={read_only})")
    except Exception as exc:
        print(f"POSTGRES: FAIL ({type(exc).__name__})")
        failures += 1

    try:
        from app.core.llm import _resolve_model_name, get_llm

        configured_model = _resolve_model_name(settings.nlq_model)
        candidates = [configured_model]
        if configured_model != "google_genai/gemini-3.6-flash":
            candidates.append("google_genai/gemini-3.6-flash")
        for model in candidates:
            try:
                llm = get_llm(
                    model,
                    temperature=0,
                    timeout=45,
                    max_retries=0,
                    thinking_config={"thinking_level": "low"},
                )
                response = llm.invoke("Reply with exactly OK.")
                print(f"GEMINI: OK (model={model}, response_received={bool(response.content)})")
                break
            except Exception as exc:
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                message = str(getattr(exc, "message", "") or "").replace("\n", " ")[:240]
                message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", message)
                detail = f", code={code}" if code is not None else ""
                detail += f", detail={message}" if message else ""
                print(f"GEMINI: FAIL (model={model}, {type(exc).__name__}{detail})")
                if model == candidates[-1]:
                    failures += 1
    except Exception as exc:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        message = str(getattr(exc, "message", "") or "").replace("\n", " ")[:240]
        message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", message)
        detail = f", code={code}" if code is not None else ""
        detail += f", detail={message}" if message else ""
        print(f"GEMINI: FAIL ({type(exc).__name__}{detail})")
        failures += 1

    try:
        from app.core.tracing import (
            get_langfuse_client,
            init_tracing,
            is_langfuse_configured,
        )

        init_tracing()
        if not is_langfuse_configured():
            print("LANGFUSE: NOT CONFIGURED")
            failures += 1
        else:
            authenticated = get_langfuse_client().auth_check()
            print(f"LANGFUSE: {'OK' if authenticated else 'AUTH_FAILED'}")
            failures += not authenticated
    except Exception as exc:
        print(f"LANGFUSE: FAIL ({type(exc).__name__})")
        failures += 1

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
