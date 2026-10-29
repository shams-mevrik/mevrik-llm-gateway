"""
Async HTTP client pool for proxying requests to vLLM backends.

Concurrency model (Python equivalent of Go goroutines + buffered channels):
- One shared httpx.AsyncClient with a large connection pool.
- Per-backend asyncio.Semaphore acts like a Go buffered channel —
  caps in-flight requests so one slow backend can't starve others.
- All I/O is non-blocking via asyncio + httpx (like goroutines on netpoll).
- uvloop (if installed) replaces the default event loop for ~2x throughput.

Backend addresses never appear in client-facing errors — they are logged
only. Callers get the model id and what went wrong.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Union

import httpx

from app.config import settings
from app.registry import ModelEntry

logger = logging.getLogger("llm-gateway")

_client: httpx.AsyncClient | None = None
_semaphores: dict[str, asyncio.Semaphore] = {}

# Per-backend concurrency cap (like a Go buffered channel size)
_MAX_INFLIGHT_PER_BACKEND = 200


# ---------------------------------------------------------------------------
# Structured proxy result (avoids raw httpx.Response leaking into routers)
# ---------------------------------------------------------------------------
@dataclass
class ProxyResult:
    """Encapsulates a proxied response with pre-parsed data."""

    status_code: int
    body: Optional[Dict[str, Any]]  # parsed JSON, or None if not JSON
    raw_text: str  # raw response text for debugging
    error: Optional[str] = None  # set if the proxy call itself failed

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300

    @property
    def is_backend_error(self) -> bool:
        """True when the backend responded but with a non-2xx status."""
        return self.error is None and self.status_code >= 400


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------
async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(settings.request_timeout),
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=settings.max_connections,
                max_keepalive_connections=200,
                keepalive_expiry=60,
            ),
            follow_redirects=True,
            http2=True,
            trust_env=False,
        )
    return _client


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


def _sem(backend_url: str) -> asyncio.Semaphore:
    """Get or create a per-backend semaphore for backpressure."""
    if backend_url not in _semaphores:
        _semaphores[backend_url] = asyncio.Semaphore(_MAX_INFLIGHT_PER_BACKEND)
    return _semaphores[backend_url]


def _url(model: ModelEntry, path: str, root: bool = False) -> str:
    """Backend URL for ``path`` — under backend_url, or the server root."""
    base = model.backend_root if root else model.backend_url.rstrip("/")
    return f"{base}{path}"


def _headers(model: ModelEntry) -> Optional[dict]:
    """Auth header for backends started with vLLM --api-key."""
    if model.backend_api_key:
        return {"Authorization": f"Bearer {model.backend_api_key}"}
    return None


def _network_error(model: ModelEntry, url: str, exc: BaseException) -> ProxyResult:
    """Map a transport failure to a ProxyResult. The URL is logged only."""
    if isinstance(exc, httpx.ConnectError):
        logger.error("Connection refused [%s] %s: %s", model.id, url, exc)
        return ProxyResult(
            status_code=502,
            body=None,
            raw_text="",
            error=f"Cannot connect to the backend for model '{model.id}' — is vLLM running?",
        )
    if isinstance(exc, httpx.TimeoutException):
        logger.error("Timeout [%s] %s: %s", model.id, url, exc)
        return ProxyResult(
            status_code=504,
            body=None,
            raw_text="",
            error=f"Backend timed out ({settings.request_timeout}s) for model '{model.id}'",
        )
    if isinstance(exc, httpx.HTTPError):
        logger.error("HTTP error [%s] %s: %s", model.id, url, exc)
        return ProxyResult(
            status_code=502,
            body=None,
            raw_text="",
            error=f"Backend HTTP error for model '{model.id}': {type(exc).__name__}",
        )
    logger.exception("Unexpected proxy error [%s] %s", model.id, url, exc_info=exc)
    return ProxyResult(
        status_code=500,
        body=None,
        raw_text="",
        error=f"Unexpected proxy error for model '{model.id}': {type(exc).__name__}",
    )


def _build_result(model: ModelEntry, url: str, resp: httpx.Response) -> ProxyResult:
    """Parse a completed backend response into a ProxyResult."""
    raw_text = resp.text
    parsed: Optional[Dict[str, Any]] = None

    try:
        parsed = resp.json()
    except Exception:
        logger.warning(
            "Non-JSON response from [%s] %s (status %d): %.200s",
            model.id,
            url,
            resp.status_code,
            raw_text,
        )

    if resp.status_code >= 400:
        # Extract vLLM error message if available
        detail = ""
        if parsed and isinstance(parsed, dict):
            detail = (
                parsed.get("message")
                or parsed.get("detail")
                or (
                    parsed.get("error", {}).get("message")
                    if isinstance(parsed.get("error"), dict)
                    else ""
                )
                or raw_text[:500]
            )
        else:
            detail = raw_text[:500]

        logger.warning(
            "Backend error [%s] %s → %d: %s",
            model.id,
            url,
            resp.status_code,
            detail,
        )

    return ProxyResult(
        status_code=resp.status_code,
        body=parsed,
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# Core proxy functions
# ---------------------------------------------------------------------------
async def forward(
    model: ModelEntry, path: str, body: dict, root: bool = False
) -> httpx.Response:
    """
    Proxy a JSON request to the upstream vLLM backend.
    Returns the raw httpx.Response.  Callers that need safer handling
    should prefer `forward_safe()` instead.
    """
    client = await get_client()
    url = _url(model, path, root)
    async with _sem(model.backend_url):
        return await client.post(url, json=body, headers=_headers(model))


async def forward_safe(
    model: ModelEntry, path: str, body: dict, root: bool = False
) -> ProxyResult:
    """
    Proxy a JSON request and return a structured ProxyResult.

    - ``root=True`` sends to the backend server root (``/classify``,
      ``/pooling``) instead of under backend_url's ``/v1``.
    - Catches network/timeout errors and wraps them.
    - Safely parses JSON (returns raw text if parsing fails).
    - Always returns a ProxyResult — never raises.
    """
    url = _url(model, path, root)
    logger.debug("→ %s  %s  model=%s", path, url, body.get("model", "?"))

    try:
        client = await get_client()
        async with _sem(model.backend_url):
            resp = await client.post(url, json=body, headers=_headers(model))
    except Exception as exc:
        return _network_error(model, url, exc)

    return _build_result(model, url, resp)


class StreamHandle:
    """An open, successful (2xx) backend stream.

    Holds the backend semaphore slot until closed. ``aclose()`` is
    idempotent; ``iter_bytes()`` closes on completion, error or client
    disconnect, and routers also register ``aclose`` as a background task
    so the slot is freed even if the body is never iterated.
    """

    def __init__(self, resp: httpx.Response, sem: asyncio.Semaphore):
        self.status_code = resp.status_code
        self._resp = resp
        self._sem = sem
        self._closed = False

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._resp.aiter_bytes():
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._resp.aclose()
        finally:
            self._sem.release()


async def open_stream(
    model: ModelEntry, path: str, body: dict
) -> Union[StreamHandle, ProxyResult]:
    """Start a streaming request and look at the status before streaming.

    Returns a StreamHandle for a 2xx response, otherwise a ProxyResult with
    the backend's error body — so the client gets the real status code and
    vLLM's message instead of a 200 stream carrying a generic failure.
    """
    url = _url(model, path)
    sem = _sem(model.backend_url)
    await sem.acquire()
    try:
        client = await get_client()
        request = client.build_request("POST", url, json=body, headers=_headers(model))
        resp = await client.send(request, stream=True)
    except Exception as exc:
        sem.release()
        return _network_error(model, url, exc)
    except BaseException:  # cancelled while connecting
        sem.release()
        raise

    if resp.status_code >= 400:
        try:
            await resp.aread()
            return _build_result(model, url, resp)
        finally:
            await resp.aclose()
            sem.release()

    return StreamHandle(resp, sem)


async def forward_stream(
    model: ModelEntry, path: str, body: dict
) -> AsyncIterator[bytes]:
    """Proxy a streaming request, yielding raw SSE chunks.

    Raises RuntimeError (without the backend address) on a failed start.
    """
    handle = await open_stream(model, path, body)
    if isinstance(handle, ProxyResult):
        raise RuntimeError(
            handle.error
            or f"Backend returned {handle.status_code}: {handle.raw_text[:300]}"
        )
    async for chunk in handle.iter_bytes():
        yield chunk


async def probe(model: ModelEntry) -> dict:
    """GET the backend's /health (server root). Never raises."""
    url = _url(model, "/health", root=True)
    t0 = time.perf_counter()
    try:
        client = await get_client()
        resp = await client.get(
            url, headers=_headers(model), timeout=settings.health_timeout
        )
        ok = resp.status_code == 200
        result = {
            "status": "ok" if ok else "unhealthy",
            "http_status": resp.status_code,
        }
    except httpx.TimeoutException:
        result = {"status": "timeout"}
    except Exception as exc:
        logger.warning("Health probe failed [%s] %s: %s", model.id, url, exc)
        result = {"status": "unreachable"}
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
