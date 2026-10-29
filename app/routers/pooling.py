"""Classification & pooling endpoints — vLLM ``--runner pooling`` models.

vLLM mounts ``/classify`` and ``/pooling`` at the server root, not under
the OpenAI ``/v1`` prefix. The gateway serves both spellings:

    POST /classify      POST /v1/classify   → backend {root}/classify
    POST /pooling       POST /v1/pooling    → backend {root}/pooling

``{root}`` is the model's backend_url without its trailing ``/v1``, so a
classifier is registered exactly like every other model. Clients that strip
``/v1`` from their base URL (e.g. mevrik-ai's classifier client posts to
``{base}/classify``) and clients that keep it both work.

Bodies are forwarded as sent; the model's default_params fill fields the
client left out (e.g. ``truncation_side: "right"`` for a router model whose
input must only ever lose its oldest turns).
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.deps import resolve_model, verify_api_key
from app.proxy import forward_safe
from app.registry import ModelEntry
from app.routers.openai import (
    _error_response,
    _extract_usage,
    _record_stat,
    _user_key,
    raw_body,
)
from app.schemas import ClassifyRequest, PoolingRequest

logger = logging.getLogger("llm-gateway")

router = APIRouter(tags=["pooling"], dependencies=[Depends(verify_api_key)])

_CLASSIFY_TYPES = {"classify"}
# /pooling returns the raw pooler output and works for any pooling model
_POOLING_TYPES = {"pooling", "classify", "embedding"}


async def _proxy_root(
    request: Request, model: ModelEntry, path: str, payload: dict
) -> JSONResponse:
    """Forward to the backend server root and relay the response as-is."""
    user = _user_key(request)
    t0 = time.perf_counter()

    result = await forward_safe(model, path, payload, root=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not result.ok:
        logger.warning(
            "Proxy error [%s] %s → %d (%s) %.0fms",
            model.id,
            path,
            result.status_code,
            result.error or "backend error",
            elapsed_ms,
        )
        return _error_response(result)

    if result.body is None:
        return _error_response(result)

    tokens_in, _ = _extract_usage(result.body)
    await _record_stat(user, model.id, tokens_in, 0, elapsed_ms)
    return JSONResponse(content=result.body, status_code=result.status_code)


@router.post("/classify")
@router.post("/v1/classify")
async def classify(body: ClassifyRequest, request: Request):
    """Sequence classification (intent routers, guards, sentiment, ...).

    Returns vLLM's response unchanged:
    ``{"data": [{"index": 0, "label": "...", "probs": [...], "num_classes": N}],
    "usage": {...}}``
    """
    model, payload = resolve_model(
        await raw_body(request), allowed_types=_CLASSIFY_TYPES
    )
    return await _proxy_root(request, model, "/classify", payload)


@router.post("/pooling")
@router.post("/v1/pooling")
async def pooling(body: PoolingRequest, request: Request):
    """Raw pooler output of a pooling model."""
    model, payload = resolve_model(
        await raw_body(request), allowed_types=_POOLING_TYPES
    )
    return await _proxy_root(request, model, "/pooling", payload)
