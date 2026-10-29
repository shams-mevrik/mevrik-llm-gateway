"""Model management — CRUD endpoints that persist to models.json.

Listing is open to every API key; the backend location (URL, backend model
name, default_params) is shown only to admin callers. Add / update / delete
need the admin key (ADMIN_API_KEY, or API_KEY when no admin key is set).
backend_api_key is write-only — it is never returned.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.deps import caller_is_admin, verify_admin_key, verify_api_key
from app.registry import ModelCreate, ModelUpdate, registry

router = APIRouter(
    prefix="/v1/models", tags=["models"], dependencies=[Depends(verify_api_key)]
)


@router.get("")
async def list_models(admin: bool = Depends(caller_is_admin)):
    """OpenAI-compatible GET /v1/models."""
    data = [m.public_view(include_backend=admin) for m in registry.list_all()]
    return {"object": "list", "data": data}


@router.get("/{model_id}")
async def get_model(model_id: str, admin: bool = Depends(caller_is_admin)):
    entry = registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return entry.public_view(include_backend=admin)


@router.post("", status_code=201, dependencies=[Depends(verify_admin_key)])
async def add_model(body: ModelCreate):
    """Add a new model at runtime — saved to models.json immediately."""
    try:
        entry = await registry.add(body)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "message": f"Model '{entry.id}' added",
        "model": entry.public_view(include_backend=True),
    }


@router.patch("/{model_id}", dependencies=[Depends(verify_admin_key)])
async def update_model(model_id: str, body: ModelUpdate):
    """Update an existing model's fields."""
    try:
        entry = await registry.update(model_id, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:  # merged entry failed validation
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "message": f"Model '{entry.id}' updated",
        "model": entry.public_view(include_backend=True),
    }


@router.delete("/{model_id}", dependencies=[Depends(verify_admin_key)])
async def delete_model(model_id: str):
    """Remove a model — takes effect immediately."""
    try:
        entry = await registry.delete(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"message": f"Model '{entry.id}' deleted"}
