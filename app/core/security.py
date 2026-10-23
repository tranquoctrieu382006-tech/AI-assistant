"""API key authentication and per-client request limiting."""
from collections import OrderedDict
from secrets import compare_digest
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
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
    if not api_key or not compare_digest(api_key, configured_key):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply a process-local sliding-window limit to API requests by client IP."""

    _MAX_TRACKED_CLIENTS = 10_000

    def __init__(self, app):
        super().__init__(app)
        self._requests: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/") or path == "/api/health/live" or request.method == "OPTIONS":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = monotonic()
        window_start = now - 60
        limit = settings.api_rate_limit_per_minute
        with self._lock:
            recent = [
                timestamp
                for timestamp in self._requests.pop(client_ip, [])
                if timestamp > window_start
            ]
            if len(recent) >= limit:
                self._requests[client_ip] = recent
                retry_after = max(1, int(60 - (now - recent[0])))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded; try again later."},
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            recent.append(now)
            self._requests[client_ip] = recent
            if len(self._requests) > self._MAX_TRACKED_CLIENTS:
                self._requests.popitem(last=False)
            remaining = limit - len(recent)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
