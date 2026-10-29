"""Rerank & Score endpoints — dedicated router with batching + normalization.

Improvements over the basic proxy:
1. Model type validation — only rerank-type models accepted.
2. Automatic document batching — splits large doc sets to avoid vLLM's
   max_num_batched_tokens limit.
3. Response normalization — consistent output format regardless of
   backend quirks (Cohere/Jina/vLLM).
4. return_documents support — optionally includes document text in results.
5. Actionable error messages for common rerank failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.deps import apply_default_params, resolve_model, verify_api_key
from app.proxy import ProxyResult, forward_safe
from app.registry import ModelEntry
from app.routers.openai import raw_body
from app.schemas import RerankRequest, ScoreRequest
from app.stats import RequestStat, tracker

logger = logging.getLogger("llm-gateway")

router = APIRouter(
    prefix="/v1", tags=["rerank"], dependencies=[Depends(verify_api_key)]
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default batch size for document splitting.
# vLLM typically has a max_num_batched_tokens limit (commonly 32768).
# Sending too many documents at once causes 400 errors.
# This can be overridden per-request via X-Rerank-Batch-Size header.
_DEFAULT_BATCH_SIZE = 64

# Maximum number of documents we accept in a single request.
_MAX_DOCUMENTS = 10000

# Allowed model types for rerank endpoints
_RERANK_TYPES = {"rerank"}
_SCORE_TYPES = {"rerank", "score"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_key(request: Request) -> str:
    """Extract a user identifier from the Authorization header for stats."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return f"key_...{token[-8:]}" if len(token) > 8 else token
    return "__anonymous__"


def _get_batch_size(request: Request) -> int:
    """Read optional X-Rerank-Batch-Size header, fallback to default."""
    raw = request.headers.get("x-rerank-batch-size", "")
    if raw.isdigit():
        val = int(raw)
        if 1 <= val <= 1000:
            return val
    return _DEFAULT_BATCH_SIZE


def _normalize_results(
    raw_results: List[Dict[str, Any]],
    documents: list,
    return_documents: bool,
    top_n: Optional[int],
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Normalize backend results into a consistent format.

    Handles:
    - vLLM format: {"results": [{"index": i, "relevance_score": s}]}
    - Cohere format: {"results": [{"index": i, "relevance_score": s, "document": {...}}]}
    - Missing fields gracefully.

    Args:
        raw_results: Backend response results array.
        documents: Original documents list for return_documents.
        return_documents: Whether to include document text in output.
        top_n: If set, only return top_n results.
        offset: Index offset for batched results.
    """
    normalized = []
    for item in raw_results:
        idx = item.get("index", 0)
        score = item.get("relevance_score", item.get("score", 0.0))
        global_idx = idx + offset

        entry: Dict[str, Any] = {
            "index": global_idx,
            "relevance_score": float(score),
        }

        if return_documents:
            # Try to include the document text
            if "document" in item:
                entry["document"] = item["document"]
            elif global_idx < len(documents):
                doc = documents[global_idx]
                if isinstance(doc, str):
                    entry["document"] = {"text": doc}
                elif isinstance(doc, dict):
                    entry["document"] = doc
                else:
                    entry["document"] = {"text": str(doc)}

        normalized.append(entry)

    return normalized


def _merge_batched_results(
    all_results: List[Dict[str, Any]],
    top_n: Optional[int],
) -> List[Dict[str, Any]]:
    """Merge results from multiple batches and re-sort by score descending."""
    # Sort by relevance_score descending
    all_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    if top_n is not None:
        all_results = all_results[:top_n]

    return all_results


def _build_rerank_error(result: ProxyResult, model_id: str) -> JSONResponse:
    """Build a rich error response for rerank failures."""
    raw_msg = ""
    if result.body and isinstance(result.body, dict):
        err_obj = result.body.get("error", result.body)
        if isinstance(err_obj, dict):
            raw_msg = err_obj.get("message", "")
        elif isinstance(err_obj, str):
            raw_msg = err_obj
        if not raw_msg:
            raw_msg = result.body.get("message", "") or result.body.get("detail", "")
    if not raw_msg:
        raw_msg = result.error or result.raw_text[:500]

    msg_lower = raw_msg.lower()

    # Detect batch/token size errors
    batch_hints = (
        "too large",
        "batch size",
        "max_num_batched_tokens",
        "too long",
        "exceed",
    )
    if any(h in msg_lower for h in batch_hints):
        return JSONResponse(
            status_code=result.status_code,
            content={
                "error": {
                    "message": raw_msg,
                    "type": "input_too_large",
                    "code": result.status_code,
                    "hint": (
                        "The document batch exceeds the backend's limit. "
                        "The gateway will attempt automatic batching, but if individual "
                        "documents are too long, try truncating them before sending. "
                        "You can also send X-Rerank-Batch-Size header to control batch size."
                    ),
                }
            },
        )

    # Detect model not loaded errors
    if "not found" in msg_lower or "not loaded" in msg_lower:
        return JSONResponse(
            status_code=result.status_code,
            content={
                "error": {
                    "message": raw_msg,
                    "type": "model_not_found",
                    "code": result.status_code,
                    "hint": (
                        f"The backend model for '{model_id}' may not be loaded. "
                        "Check that vLLM has the rerank model running."
                    ),
                }
            },
        )

    # Generic error pass-through
    if result.body is not None:
        return JSONResponse(content=result.body, status_code=result.status_code)

    return JSONResponse(
        status_code=result.status_code,
        content={
            "error": {
                "message": result.error or f"Backend returned {result.status_code}",
                "type": "proxy_error",
                "code": result.status_code,
            }
        },
    )


async def _record_stat(
    user_key: str,
    model_id: str,
    latency_ms: float,
    doc_count: int = 0,
):
    """Record rerank stats."""
    try:
        stat = RequestStat(
            model_id=model_id,
            tokens_in=doc_count,  # Use tokens_in to track document count
            tokens_out=0,
            latency_ms=round(latency_ms, 2),
            ttfb_ms=round(latency_ms, 2),
        )
        await tracker.record(user_key, stat)
    except Exception as exc:
        logger.debug("Stats record failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Rerank endpoint
# ---------------------------------------------------------------------------


@router.post("/rerank")
async def rerank(body: RerankRequest, request: Request):
    """Rerank documents against a query.

    Supports:
    - vLLM, Cohere, and Jina rerank API formats
    - Automatic document batching for large sets
    - return_documents flag to include document text in results
    - X-Rerank-Batch-Size header to control batch size

    The response format is Cohere-compatible:
    ```json
    {
      "id": "rerank-...",
      "results": [
        {"index": 0, "relevance_score": 0.95, "document": {"text": "..."}},
        ...
      ],
      "meta": {"api_version": {"version": "1"}, "billed_units": {"search_units": 1}},
      "usage": {"prompt_tokens": ..., "total_tokens": ...}
    }
    ```
    """
    # Validate model type
    model, payload = resolve_model(
        body.model_dump(exclude_none=True),
        allowed_types=_RERANK_TYPES,
    )

    doc_count = len(body.documents)
    if doc_count == 0:
        return JSONResponse(
            content={
                "id": f"rerank-{uuid.uuid4().hex[:12]}",
                "results": [],
                "meta": {"api_version": {"version": "1"}},
            }
        )

    if doc_count > _MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Too many documents ({doc_count}). Maximum is {_MAX_DOCUMENTS}.",
                    "type": "invalid_request_error",
                }
            },
        )

    user = _user_key(request)
    batch_size = _get_batch_size(request)
    return_documents = body.return_documents or False
    t0 = time.perf_counter()

    # --- Single batch (most common case) ---
    if doc_count <= batch_size:
        vllm_payload = body.to_vllm_payload()
        # Rewrite model to backend name
        vllm_payload["model"] = model.backend_model
        vllm_payload = apply_default_params(model, vllm_payload)

        result = await forward_safe(model, "/rerank", vllm_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not result.ok:
            return _build_rerank_error(result, model.id)

        raw_results = (result.body or {}).get("results", [])
        normalized = _normalize_results(
            raw_results,
            body.documents,
            return_documents,
            body.top_n,
        )
        if body.top_n is not None:
            normalized.sort(key=lambda x: x["relevance_score"], reverse=True)
            normalized = normalized[: body.top_n]

        await _record_stat(user, model.id, elapsed_ms, doc_count)

        response_body = {
            "id": f"rerank-{uuid.uuid4().hex[:12]}",
            "results": normalized,
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"search_units": 1},
            },
            "usage": (result.body or {}).get("usage"),
        }
        return JSONResponse(content=response_body)

    # --- Multi-batch: split documents and merge results ---
    logger.info(
        "Rerank batching: %d docs into %d-doc batches for model %s",
        doc_count,
        batch_size,
        model.id,
    )

    all_results: List[Dict[str, Any]] = []
    errors: List[str] = []

    # Create batches
    batches = []
    for i in range(0, doc_count, batch_size):
        batch_docs = body.documents[i : i + batch_size]
        batch_payload = {
            "model": model.backend_model,
            "query": body.query,
            "documents": batch_docs,
        }
        # For batches, don't apply top_n per-batch — we merge and apply globally
        batch_payload = apply_default_params(model, batch_payload)
        batch_payload.pop("top_n", None)
        batches.append((i, batch_payload))

    # Execute batches concurrently (with limited parallelism)
    sem = asyncio.Semaphore(4)  # max 4 concurrent batch requests

    async def _run_batch(
        offset: int, payload: dict
    ) -> tuple[int, Optional[ProxyResult]]:
        async with sem:
            return offset, await forward_safe(model, "/rerank", payload)

    tasks = [_run_batch(offset, payload) for offset, payload in batches]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in batch_results:
        if isinstance(item, Exception):
            errors.append(str(item))
            continue

        offset, result = item
        if result is None:
            errors.append(f"Batch at offset {offset}: no response")
            continue

        if not result.ok:
            errors.append(
                f"Batch at offset {offset}: {result.error or result.status_code}"
            )
            continue

        raw = (result.body or {}).get("results", [])
        normalized = _normalize_results(
            raw,
            body.documents,
            return_documents,
            top_n=None,
            offset=offset,
        )
        all_results.extend(normalized)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not all_results and errors:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"All rerank batches failed: {'; '.join(errors[:3])}",
                    "type": "proxy_error",
                }
            },
        )

    # Merge, sort, apply top_n
    merged = _merge_batched_results(all_results, body.top_n)

    await _record_stat(user, model.id, elapsed_ms, doc_count)

    response_body: Dict[str, Any] = {
        "id": f"rerank-{uuid.uuid4().hex[:12]}",
        "results": merged,
        "meta": {
            "api_version": {"version": "1"},
            "billed_units": {"search_units": len(batches)},
            "batching": {
                "total_documents": doc_count,
                "batch_size": batch_size,
                "batches": len(batches),
                "succeeded": len(batches) - len(errors),
                "failed": len(errors),
            },
        },
    }

    if errors:
        response_body["meta"]["warnings"] = errors[:5]
        logger.warning(
            "Rerank partial failure: %d/%d batches failed for %s",
            len(errors),
            len(batches),
            model.id,
        )

    return JSONResponse(content=response_body)


# ---------------------------------------------------------------------------
# Score endpoint (cross-encoder)
# ---------------------------------------------------------------------------


@router.post("/score")
async def score(body: ScoreRequest, request: Request):
    """Cross-encoder scoring endpoint.

    Accepts vLLM's /score format with text_1 / text_2 pairs.
    """
    model, payload = resolve_model(
        await raw_body(request),
        allowed_types=_SCORE_TYPES,
    )

    user = _user_key(request)
    t0 = time.perf_counter()

    result = await forward_safe(model, "/score", payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not result.ok:
        return _build_rerank_error(result, model.id)

    # Count pairs for stats
    pair_count = 0
    if body.text_1 is not None:
        pair_count = len(body.text_1) if isinstance(body.text_1, list) else 1

    await _record_stat(user, model.id, elapsed_ms, pair_count)

    return JSONResponse(content=result.body, status_code=result.status_code)
