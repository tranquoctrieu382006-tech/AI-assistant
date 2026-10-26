"""API key authentication and per-client request limiting."""
from collections import OrderedDict
import ipaddress
from secrets import compare_digest
from threading import Lock
from time import monotonic
from typing import Any, Awaitable, Callable

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.responses import JSONResponse

from app.core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    configured_key = settings.app_api_key
    if len(configured_key) < 32:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured; set APP_API_KEY to a random value of at least 32 characters.",
        )
    if not api_key or not compare_digest(
        api_key.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


class RateLimitMiddleware:
    """Pure ASGI process-local sliding-window limit, safe for streaming responses."""

    _MAX_TRACKED_CLIENTS = 10_000

    def __init__(self, app):
        self.app = app
        self._requests: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")
        if not path.startswith("/api/") or path == "/api/health/live" or method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        request_headers = {key.lower(): value for key, value in scope.get("headers", [])}
        if client_ip in settings.trusted_proxy_ips:
            forwarded = request_headers.get(b"x-forwarded-for", b"").decode("latin-1")
            # Only honor the proxy header when the immediate peer is trusted.
            # Trusted proxies must overwrite this header or append the actual
            # client address as its final value.
            candidate = forwarded.rsplit(",", 1)[-1].strip()
            try:
                client_ip = str(ipaddress.ip_address(candidate))
            except ValueError:
                pass
        now = monotonic()
        window_start = now - 60
        limit = settings.api_rate_limit_per_minute
        limited_response = None
        with self._lock:
            recent = [
                timestamp
                for timestamp in self._requests.pop(client_ip, [])
                if timestamp > window_start
            ]
            if len(recent) >= limit:
                self._requests[client_ip] = recent
                retry_after = max(1, int(60 - (now - recent[0])))
                limited_response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded; try again later."},
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            else:
                recent.append(now)
                self._requests[client_ip] = recent
                if len(self._requests) > self._MAX_TRACKED_CLIENTS:
                    self._requests.popitem(last=False)
                remaining = limit - len(recent)

        if limited_response is not None:
            await limited_response(scope, receive, send)
            return

        async def send_with_rate_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() not in {b"x-ratelimit-limit", b"x-ratelimit-remaining"}
                ]
                headers.extend([
                    (b"x-ratelimit-limit", str(limit).encode("ascii")),
                    (b"x-ratelimit-remaining", str(remaining).encode("ascii")),
                ])
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_rate_headers)
