"""Run opt-in live HTTP checks against PostgreSQL-backed RAG/NLQ and Gemini."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import httpx
from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import PROJECT_ROOT

NO_ANSWER = "Tài liệu hiện có không chứa đủ thông tin để trả lời câu hỏi này."


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_server(env_file: Path, port: int):
    values = {key: value for key, value in dotenv_values(env_file).items() if value is not None}
    child_env = os.environ.copy()
    child_env.update(values)
    if int(child_env.get("APP_WORKERS", "1")) != 1:
        raise RuntimeError("Live smoke requires APP_WORKERS=1 because sessions and rate limits are in memory.")
    command = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=child_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    ), child_env


def _trace_id(response: httpx.Response) -> str:
    return response.headers.get("x-langfuse-trace-id") or response.headers.get("x-trace-id") or "not exposed by API"


def _record(records, name, response, elapsed, assertion, detail=""):
    record = {
        "case": name,
        "status": response.status_code,
        "seconds": round(elapsed, 3),
        "passed": bool(assertion),
        "trace_id": _trace_id(response),
        "detail": detail[:180],
    }
    records.append(record)
    print(
        f"{name}: HTTP {response.status_code}, {record['seconds']:.2f}s, "
        f"{'PASS' if assertion else 'FAIL'} {detail[:100]}",
        flush=True,
    )


def _record_error(records, name, exc, elapsed):
    records.append({
        "case": name,
        "status": "TIMEOUT" if isinstance(exc, httpx.TimeoutException) else "TRANSPORT_ERROR",
        "seconds": round(elapsed, 3),
        "passed": False,
        "trace_id": "not exposed by API",
        "detail": type(exc).__name__,
    })
    print(f"{name}: {type(exc).__name__} after {elapsed:.2f}s", flush=True)


def run_smoke(
    base_url: str,
    api_key: str,
    interval: float = 4.2,
    quick: bool = False,
    only_case: str | None = None,
) -> list[dict]:
    if not api_key:
        raise RuntimeError("APP_API_KEY is missing from the selected environment file.")
    records: list[dict] = []
    headers = {"X-API-Key": api_key}
    session_id = f"e2e-{uuid.uuid4()}"
    with httpx.Client(timeout=httpx.Timeout(75, connect=5), follow_redirects=False) as client:
        started = time.perf_counter()
        health = client.get(f"{base_url}/api/health")
        health_detail = health.text
        healthy = health.status_code == 200
        if healthy:
            health_body = health.json()
            healthy = health_body.get("status") == "ok"
            health_detail = json.dumps({
                "status": health_body.get("status"),
                "database": health_body.get("database_status", {}).get("status"),
                "nlq_database": health_body.get("nlq_database_status", {}).get("status"),
                "ollama": health_body.get("ollama_status", {}).get("status"),
            })
        _record(records, "health_dependencies", health, time.perf_counter() - started, healthy, health_detail)

        unauth_started = time.perf_counter()
        unauth = client.post(f"{base_url}/api/nlq/ask", json={"question": "Có bao nhiêu sản phẩm?"})
        _record(records, "auth_missing_key", unauth, time.perf_counter() - unauth_started,
                unauth.status_code == 401, unauth.text)

        cases = [
            ("rag_answerable", "rag", {"question": "Khách được yêu cầu đổi hoặc trả trong thời hạn bao lâu kể từ khi giao thành công?"}),
            ("rag_followup_same_session", "rag", {"question": "Còn trường hợp hàng lỗi thì sao?", "session_id": session_id}),
            ("rag_refusal", "rag", {"question": "Thời tiết trên Sao Hỏa ngày mai thế nào?"}),
            ("nlq_basic", "nlq", {"question": "Có bao nhiêu sản phẩm?"}),
            ("nlq_out_of_scope", "nlq", {"question": "Thời tiết ngày mai ở Hà Nội thế nào?"}),
        ]
        for name, endpoint, payload in cases:
            setup_turn = only_case == "rag_followup_same_session" and name == "rag_answerable"
            if only_case and name != only_case and not setup_turn:
                continue
            if quick and name == "rag_followup_same_session":
                continue
            if name == "rag_answerable":
                payload["session_id"] = session_id
            started = time.perf_counter()
            try:
                response = client.post(f"{base_url}/api/{endpoint}/ask", json=payload, headers=headers)
            except httpx.HTTPError as exc:
                _record_error(records, name, exc, time.perf_counter() - started)
                time.sleep(interval)
                continue
            elapsed = time.perf_counter() - started
            assertion = response.status_code == 200
            detail = response.text
            if assertion and name == "rag_answerable":
                body = response.json()
                assertion = bool(body.get("sources")) and body.get("answer") != NO_ANSWER
                detail = f"sources={len(body.get('sources', []))}; answer={body.get('answer', '')}"
            elif assertion and name == "rag_refusal":
                body = response.json()
                assertion = body.get("answer") == NO_ANSWER and not body.get("sources")
                detail = body.get("answer", "")
            elif assertion:
                body = response.json()
                detail = body.get("answer", "")
            _record(records, "rag_followup_setup" if setup_turn else name, response, elapsed, assertion, detail)
            time.sleep(interval)

        injection_cases = json.loads((PROJECT_ROOT / "eval/nlq_prompt_injection_cases.json").read_text(encoding="utf-8"))
        selected_injections = {"inject-001", "inject-002"} if quick else {"inject-001", "inject-002", "inject-003"}
        if only_case:
            selected_injections &= {only_case}
        injection_cases = [case for case in injection_cases if case["id"] in selected_injections]
        for case in injection_cases:
            started = time.perf_counter()
            try:
                response = client.post(
                    f"{base_url}/api/nlq/ask",
                    json={"question": case["question"]},
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                _record_error(records, case["id"], exc, time.perf_counter() - started)
                time.sleep(interval)
                continue
            elapsed = time.perf_counter() - started
            safe = response.status_code in (200, 422)
            detail = response.text
            if response.status_code == 200:
                body = response.json()
                sql = body.get("sql", "").casefold()
                safe = not any(token in sql for token in ("delete ", "drop ", "update ", "insert ", "pg_sleep", "query_to_xml"))
                safe = safe and "email" not in sql and "full_name" not in sql
                detail = f"safe_sql={safe}; status=200"
            _record(records, case["id"], response, elapsed, safe, detail)
            time.sleep(interval)

        stream_started = time.perf_counter()
        if only_case and only_case != "rag_stream":
            return records
        try:
            with client.stream(
                "POST", f"{base_url}/api/rag/ask/stream",
                json={"question": "Khách được yêu cầu đổi hoặc trả trong thời hạn bao lâu kể từ khi giao thành công?"},
                headers=headers,
            ) as response:
                chunks = "".join(response.iter_text())
                complete = "data: [DONE]" in chunks and '"type": "token"' in chunks
                _record(records, "rag_stream", response, time.perf_counter() - stream_started,
                        response.status_code == 200 and complete, chunks[-180:])
        except httpx.HTTPError as exc:
            _record_error(records, "rag_stream", exc, time.perf_counter() - stream_started)
    return records


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Use an already-running API; default starts a single-worker local API.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--interval", type=float, default=4.2, help="Delay between Gemini calls to respect free-tier RPM.")
    parser.add_argument("--quick", action="store_true", help="Run the core smoke set; omit follow-up and pg_sleep injection case.")
    parser.add_argument("--only-case", help="Run one case name (still checks dependency health and API-key rejection).")
    args = parser.parse_args()

    process = None
    env = os.environ.copy()
    if args.env_file.is_file():
        env.update({key: value for key, value in dotenv_values(args.env_file).items() if value is not None})
    if not env.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is required to confirm the real Gemini path.")

    def model_name(role_name: str, default: str) -> str:
        role = env.get(role_name, default)
        if role in {"fast", "strong", "local", "default"}:
            return env.get({"fast": "FAST_MODEL", "strong": "STRONG_MODEL", "local": "LOCAL_MODEL", "default": "DEFAULT_MODEL"}[role], default)
        return role

    rag_model = model_name("RAG_MODEL", "fast")
    nlq_model = model_name("NLQ_MODEL", "fast")
    if not rag_model.startswith("google_genai/") or not nlq_model.startswith("google_genai/"):
        raise SystemExit(f"Smoke requires Gemini for both paths; configured RAG={rag_model}, NLQ={nlq_model}.")
    base_url = args.base_url
    if not base_url:
        if not args.env_file.is_file():
            raise SystemExit(f"No environment file found: {args.env_file}")
        port = _free_port()
        process, env = _start_server(args.env_file, port)
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise SystemExit("API exited before becoming reachable; inspect its normal startup logs and configuration.")
            try:
                httpx.get(f"{base_url}/", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            process.terminate()
            raise SystemExit("API did not become reachable within 45 seconds.")

    try:
        api_key = env.get("APP_API_KEY", "")
        records = run_smoke(base_url.rstrip("/"), api_key, args.interval, args.quick, args.only_case)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    elapsed = [record["seconds"] for record in records]
    sorted_elapsed = sorted(elapsed)
    p95 = sorted_elapsed[min(len(sorted_elapsed) - 1, int(0.95 * len(sorted_elapsed)))] if sorted_elapsed else 0
    report_path = args.report or PROJECT_ROOT / "eval" / f"e2e_smoke_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Live RAG/NLQ HTTP smoke test",
        "",
        f"- Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- API: `{base_url}`",
        f"- RAG model: `{rag_model}`; NLQ model: `{nlq_model}`",
        f"- Cases: {len(records)}; passed: {sum(r['passed'] for r in records)}; failed: {sum(not r['passed'] for r in records)}",
        f"- Request latency p50/p95: {sorted_elapsed[len(sorted_elapsed)//2]:.3f}s / {p95:.3f}s" if sorted_elapsed else "- No requests completed.",
        "- Trace ID is recorded only if the API returns an `X-Langfuse-Trace-Id` or `X-Trace-Id` header.",
        "",
        "| Case | HTTP | Seconds | Result | Trace ID | Detail |",
        "|---|---:|---:|---|---|---|",
    ]
    for record in records:
        detail = record["detail"].replace("|", "\\|").replace("\n", " ")
        report.append(f"| {record['case']} | {record['status']} | {record['seconds']:.3f} | {'PASS' if record['passed'] else 'FAIL'} | {record['trace_id']} | {detail} |")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)
    print(f"live HTTP smoke: {sum(r['passed'] for r in records)}/{len(records)} passed")
    return 0 if all(record["passed"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
