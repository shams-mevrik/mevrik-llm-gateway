"""
Benchmark & stats endpoints.

Load tests run as background tasks — the endpoint returns a job_id
immediately, and you poll GET /v1/jobs/{job_id} for results.
This avoids nginx/proxy gateway timeouts on long-running tests.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.deps import verify_admin_key, verify_api_key
from app.proxy import forward, forward_stream
from app.registry import registry
from app.stats import RequestStat, tracker

logger = logging.getLogger("llm-gateway")

router = APIRouter(
    prefix="/v1", tags=["benchmark"], dependencies=[Depends(verify_api_key)]
)

_TEST_PROMPT = "Say 'hello' in one sentence."

# Tests send /chat/completions — other model types cannot be benchmarked here
_BENCH_TYPES = {"chat"}

# Strong references so running load tests are not garbage-collected
_background_tasks: set[asyncio.Task] = set()

# ---------------------------------------------------------------------------
# Job store (in-memory, capped)
# ---------------------------------------------------------------------------
_MAX_JOBS = 200

_jobs: dict[str, dict] = {}


def _benchmark_entry(model_id: str):
    """Look up a model and check it can be benchmarked."""
    entry = registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    if entry.model_type not in _BENCH_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_id}' is type '{entry.model_type}'; "
                f"benchmarks support {sorted(_BENCH_TYPES)} models only"
            ),
        )
    return entry


def _store_job(job_id: str, data: dict):
    _jobs[job_id] = data
    # Evict oldest if over cap
    if len(_jobs) > _MAX_JOBS:
        oldest = next(iter(_jobs))
        _jobs.pop(oldest, None)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TestRequest(BaseModel):
    prompt: str = Field(_TEST_PROMPT, description="Test prompt to send")
    max_tokens: int = Field(50, ge=1, le=4096)


class LoadTestRequest(BaseModel):
    prompt: str = Field(_TEST_PROMPT, description="Prompt for each request")
    max_tokens: int = Field(50, ge=1, le=4096)
    concurrency: int = Field(5, ge=1, le=50, description="Parallel requests")
    total_requests: int = Field(10, ge=1, le=200, description="Total requests to send")


# ---------------------------------------------------------------------------
# 1. Test single model (quick, inline response)
# ---------------------------------------------------------------------------
@router.post("/models/{model_id}/test")
async def test_model(model_id: str, body: TestRequest):
    """Single request — measure latency, TTFB, tokens/sec."""
    entry = _benchmark_entry(model_id)

    payload = {
        "model": entry.backend_model,
        "messages": [{"role": "user", "content": body.prompt}],
        "max_tokens": body.max_tokens,
        "stream": True,
    }

    ttfb_ms = 0.0
    chunks: list[bytes] = []
    t_start = time.perf_counter()
    first_chunk = True

    try:
        async for chunk in forward_stream(entry, "/chat/completions", payload):
            if first_chunk:
                ttfb_ms = (time.perf_counter() - t_start) * 1000
                first_chunk = False
            chunks.append(chunk)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backend error: {exc}")

    total_ms = (time.perf_counter() - t_start) * 1000
    raw = b"".join(chunks).decode(errors="replace")
    tokens_out = _count_tokens_from_sse(raw)
    tps = (tokens_out / total_ms) * 1000 if total_ms > 0 and tokens_out > 0 else 0

    stat = RequestStat(
        model_id=model_id,
        tokens_in=len(body.prompt.split()),
        tokens_out=tokens_out,
        latency_ms=round(total_ms, 2),
        ttfb_ms=round(ttfb_ms, 2),
    )
    await tracker.record("__benchmark__", stat)

    return {
        "model": model_id,
        "status": "ok",
        "latency_ms": round(total_ms, 2),
        "ttfb_ms": round(ttfb_ms, 2),
        "tokens_generated": tokens_out,
        "tokens_per_sec": round(tps, 2),
    }


# ---------------------------------------------------------------------------
# 2. Load test (background job — returns immediately)
# ---------------------------------------------------------------------------
@router.post(
    "/models/{model_id}/loadtest",
    status_code=202,
    dependencies=[Depends(verify_admin_key)],
)
async def load_test_model(model_id: str, body: LoadTestRequest):
    """Start a load test in the background. Returns a job_id to poll."""
    entry = _benchmark_entry(model_id)

    job_id = str(uuid.uuid4())[:8]
    _store_job(
        job_id,
        {
            "job_id": job_id,
            "model": model_id,
            "status": "running",
            "config": body.model_dump(),
            "started_at": time.time(),
        },
    )

    # Fire and forget — runs in the event loop background
    task = asyncio.create_task(_run_loadtest(job_id, entry, body))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "job_id": job_id,
        "status": "running",
        "poll_url": f"/v1/jobs/{job_id}",
    }


async def _run_loadtest(job_id: str, entry, body: LoadTestRequest):
    """Execute load test concurrently, store results in job store."""
    sem = asyncio.Semaphore(body.concurrency)

    async def _single(idx: int) -> dict:
        async with sem:
            payload = {
                "model": entry.backend_model,
                "messages": [{"role": "user", "content": body.prompt}],
                "max_tokens": body.max_tokens,
            }
            t0 = time.perf_counter()
            try:
                resp = await forward(entry, "/chat/completions", payload)
                elapsed = (time.perf_counter() - t0) * 1000
                data = resp.json()
                usage = data.get("usage", {})
                tokens_out = usage.get("completion_tokens", 0)

                stat = RequestStat(
                    model_id=entry.id,
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=tokens_out,
                    latency_ms=round(elapsed, 2),
                    ttfb_ms=round(elapsed, 2),
                )
                await tracker.record("__loadtest__", stat)

                return {
                    "index": idx,
                    "status": resp.status_code,
                    "latency_ms": round(elapsed, 2),
                    "tokens_out": tokens_out,
                }
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                return {
                    "index": idx,
                    "status": "error",
                    "latency_ms": round(elapsed, 2),
                    "error": str(exc),
                }

    try:
        tasks = [_single(i) for i in range(body.total_requests)]
        t_wall = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_wall_ms = (time.perf_counter() - t_wall) * 1000

        successes = [r for r in results if r.get("status") == 200]
        failures = [r for r in results if r.get("status") != 200]
        latencies = sorted(r["latency_ms"] for r in successes) if successes else [0]
        n = len(successes)

        def _pct(arr: list, p: float) -> float:
            return round(arr[min(int(len(arr) * p), len(arr) - 1)], 2) if arr else 0

        _store_job(
            job_id,
            {
                "job_id": job_id,
                "model": entry.id,
                "status": "completed",
                "config": body.model_dump(),
                "wall_time_ms": round(total_wall_ms, 2),
                "succeeded": n,
                "failed": len(failures),
                "avg_latency_ms": round(sum(latencies) / n, 2) if n else 0,
                "min_latency_ms": round(latencies[0], 2) if n else 0,
                "max_latency_ms": round(latencies[-1], 2) if n else 0,
                "p50_latency_ms": _pct(latencies, 0.50),
                "p95_latency_ms": _pct(latencies, 0.95),
                "p99_latency_ms": _pct(latencies, 0.99),
                "requests_per_sec": round((n / total_wall_ms) * 1000, 2)
                if total_wall_ms > 0
                else 0,
                "results": list(results),
                "completed_at": time.time(),
            },
        )
        logger.info(
            "Load test %s completed: %d/%d succeeded", job_id, n, body.total_requests
        )

    except Exception as exc:
        _store_job(
            job_id,
            {
                "job_id": job_id,
                "model": entry.id,
                "status": "failed",
                "error": str(exc),
                "completed_at": time.time(),
            },
        )
        logger.error("Load test %s failed: %s", job_id, exc)


# ---------------------------------------------------------------------------
# 3. Poll job results
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll load test job status and results."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.get("/jobs")
async def list_jobs():
    """List all jobs (most recent first)."""
    items = list(_jobs.values())
    items.reverse()
    return {"jobs": items}


# ---------------------------------------------------------------------------
# 4. Per-user token speed stats
# ---------------------------------------------------------------------------
@router.get("/stats")
async def get_stats(
    user: str = Query("__benchmark__", description="User key to query stats for"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
):
    """Get average token speed and latency stats for a user."""
    return {
        "user": user,
        "model_filter": model_id,
        **tracker.summary(user, model_id),
    }


@router.get("/stats/users")
async def list_stat_users():
    """List all user keys that have recorded stats."""
    return {"users": tracker.all_users()}


@router.delete("/stats", dependencies=[Depends(verify_admin_key)])
async def clear_stats(
    user: Optional[str] = Query(
        None, description="Clear stats for specific user, or all if omitted"
    ),
):
    """Clear recorded stats."""
    await tracker.clear(user)
    return {"message": f"Stats cleared for {'all users' if not user else user}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_tokens_from_sse(raw: str) -> int:
    """Best-effort token count from SSE stream chunks."""
    count = 0
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            obj = _json.loads(data_str)
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens"):
                return usage["completion_tokens"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                # reasoning models stream thinking separately from content
                if (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                ):
                    count += 1
        except (_json.JSONDecodeError, TypeError):
            continue
    return count
