"""Health check router."""

import asyncio

from fastapi import APIRouter, Depends

from app.deps import verify_admin_key
from app.proxy import probe
from app.registry import registry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": len(registry.list_all()),
    }


@router.get("/health/backends", dependencies=[Depends(verify_admin_key)])
async def health_backends():
    """Probe every model's backend (vLLM GET /health), concurrently.

    One line per model id — e.g. confirm a newly started classifier
    container is up before pointing an application at it.
    """
    models = registry.list_all()
    results = await asyncio.gather(*(probe(m) for m in models))
    data = {m.id: {"model_type": m.model_type, **r} for m, r in zip(models, results)}
    healthy = sum(1 for r in results if r["status"] == "ok")
    return {
        "status": "ok" if healthy == len(models) else "degraded",
        "healthy": healthy,
        "total": len(models),
        "models": data,
    }
