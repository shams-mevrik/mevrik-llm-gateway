"""Shared FastAPI dependencies."""

from __future__ import annotations

import copy
import logging
import secrets
from typing import Optional, Set

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.registry import ModelEntry, registry

logger = logging.getLogger("llm-gateway")

# auto_error=False → missing header returns None instead of 403
_bearer = HTTPBearer(auto_error=False)


def _key_matches(given: str, expected: str) -> bool:
    """Constant-time comparison — no timing side channel on the key."""
    return secrets.compare_digest(given.encode(), expected.encode())


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
):
    """Validate Bearer token against API_KEY (or ADMIN_API_KEY) from .env."""
    if not settings.api_key and not settings.admin_api_key:
        return  # no key configured = open access
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = credentials.credentials
    if settings.api_key and _key_matches(token, settings.api_key):
        return
    if settings.admin_api_key and _key_matches(token, settings.admin_api_key):
        return
    raise HTTPException(status_code=401, detail="Invalid API key")


def is_admin(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """True when the caller may manage models and see backend details.

    With ADMIN_API_KEY set, only that key is admin. Without it, any caller
    that passes verify_api_key is admin (the original single-key behaviour).
    """
    if not settings.admin_api_key:
        return True
    return bool(credentials) and _key_matches(
        credentials.credentials, settings.admin_api_key
    )


def verify_admin_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
):
    """Require the admin key for management endpoints."""
    verify_api_key(credentials)
    if not is_admin(credentials):
        raise HTTPException(status_code=403, detail="Admin API key required")


def caller_is_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> bool:
    """Dependency form of is_admin, for endpoints that only change output."""
    return is_admin(credentials)


def apply_default_params(entry: ModelEntry, body: dict) -> dict:
    """Fill the model's default_params into a request body.

    Client values always win. Dict-valued defaults (e.g.
    chat_template_kwargs) are merged one level deep, so a client sending
    {"chat_template_kwargs": {"foo": 1}} still gets the operator's
    {"enable_thinking": false} unless it sets that key itself.
    """
    if not entry.default_params:
        return body
    out = dict(body)
    for key, default in entry.default_params.items():
        if key not in out or out[key] is None:
            out[key] = copy.deepcopy(default)
        elif isinstance(default, dict) and isinstance(out[key], dict):
            out[key] = {**copy.deepcopy(default), **out[key]}
    return out


def resolve_model(
    body: dict, allowed_types: Optional[Set[str]] = None
) -> tuple[ModelEntry, dict]:
    """
    Look up model by id, rewrite model field to backend name.

    Args:
        body: Request body dict containing "model" field.
        allowed_types: Optional set of allowed model_type values.
            If provided, the resolved model's type must be in this set.

    Returns:
        (ModelEntry, rewritten_body) where body["model"] is replaced
        with the actual model name the vLLM backend expects and the
        model's default_params are filled in.
    """
    model_id = body.get("model", "")
    if not model_id or not isinstance(model_id, str):
        raise HTTPException(
            status_code=400,
            detail="Request body must include a 'model' field",
        )

    entry = registry.get(model_id)
    if entry is None:
        available = [m.id for m in registry.list_all()]
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_id}' not found",
                    "type": "invalid_request_error",
                    "available_models": available,
                }
            },
        )

    # Validate model type if constraints are provided
    if allowed_types and entry.model_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"Model '{model_id}' is type '{entry.model_type}', "
                        f"but this endpoint requires one of: {sorted(allowed_types)}"
                    ),
                    "type": "invalid_request_error",
                    "model_type": entry.model_type,
                    "allowed_types": sorted(allowed_types),
                }
            },
        )

    # Rewrite model to the backend's expected name
    rewritten = apply_default_params(entry, {**body, "model": entry.backend_model})
    return entry, rewritten
