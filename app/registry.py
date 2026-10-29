"""Model registry for managing LLM configurations.

Thread-safe via asyncio.Lock inside a worker and an exclusive file lock
across Uvicorn workers. All mutations persist to disk immediately using
atomic file replacements.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import settings

try:  # POSIX only — the gateway runs on Linux; Windows dev falls back to no-op
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

logger = logging.getLogger("llm-gateway.registry")

# Every model type the gateway can route. Pooling-runner models (vLLM
# ``--runner pooling``) are embedding / rerank / score / classify / pooling.
MODEL_TYPES = (
    "chat",
    "completion",
    "embedding",
    "rerank",
    "score",
    "classify",
    "pooling",
)

# Request fields a model's default_params may never set.
_RESERVED_PARAMS = {"model", "stream"}


# ---------------------------------------------------------------------------
# Validation helpers (shared by entry / create / update)
# ---------------------------------------------------------------------------
def _check_model_type(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    v = v.strip().lower()
    if v not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {list(MODEL_TYPES)}, got {v!r}")
    return v


def _check_backend_url(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    v = v.strip()
    parsed = urlparse(v)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("backend_url must be an http(s) URL, e.g. http://host:8000/v1")
    return v.rstrip("/")


def _check_default_params(v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if v is None:
        return v
    reserved = _RESERVED_PARAMS.intersection(v)
    if reserved:
        raise ValueError(f"default_params cannot set {sorted(reserved)}")
    return v


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class _ValidatedFields(BaseModel):
    """Validators shared by ModelEntry / ModelCreate / ModelUpdate."""

    @field_validator("model_type", check_fields=False)
    @classmethod
    def validate_model_type(cls, v):
        return _check_model_type(v)

    @field_validator("backend_url", check_fields=False)
    @classmethod
    def validate_backend_url(cls, v):
        return _check_backend_url(v)

    @field_validator("default_params", check_fields=False)
    @classmethod
    def validate_default_params(cls, v):
        return _check_default_params(v)


class ModelEntry(_ValidatedFields):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique model identifier clients will use")
    name: str = Field("", description="Human-readable display name")
    backend_url: str = Field(..., description="vLLM base URL, e.g. http://host:8010/v1")
    backend_model: str = Field(
        ..., description="Actual model name on vLLM (--served-model-name)"
    )
    model_type: str = Field("chat", description=" | ".join(MODEL_TYPES))
    owned_by: str = Field("organization")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    default_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Request fields added when the client does not send them, e.g. "
            '{"truncation_side": "right"} or {"chat_template_kwargs": {"enable_thinking": false}}'
        ),
    )
    backend_api_key: Optional[str] = Field(
        None,
        description="Bearer key for a backend started with vLLM --api-key (never returned)",
    )

    @property
    def backend_root(self) -> str:
        """Server root of the backend (backend_url without a trailing /v1).

        vLLM mounts /classify, /pooling, /health, /tokenize at the root,
        not under the OpenAI /v1 prefix.
        """
        url = self.backend_url.rstrip("/")
        return url[:-3] if url.endswith("/v1") else url

    def public_view(self, include_backend: bool = False) -> dict:
        """Client-facing description. Backend location only for admins."""
        out = {
            "id": self.id,
            "object": "model",
            "created": self.created_at,
            "owned_by": self.owned_by,
            "name": self.name,
            "model_type": self.model_type,
        }
        if include_backend:
            out["backend_url"] = self.backend_url
            out["backend_model"] = self.backend_model
            out["default_params"] = self.default_params
            out["backend_auth"] = bool(self.backend_api_key)
        return out


class ModelCreate(_ValidatedFields):
    id: str
    name: str = ""
    backend_url: str
    backend_model: str
    model_type: str = "chat"
    owned_by: str = "organization"
    default_params: Dict[str, Any] = Field(default_factory=dict)
    backend_api_key: Optional[str] = None

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("id must not be empty")
        return v


class ModelUpdate(_ValidatedFields):
    name: Optional[str] = None
    backend_url: Optional[str] = None
    backend_model: Optional[str] = None
    model_type: Optional[str] = None
    owned_by: Optional[str] = None
    default_params: Optional[Dict[str, Any]] = None
    backend_api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, path: str = "data/models.json"):
        self._filepath = Path(path)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._lockpath = self._filepath.with_name(self._filepath.name + ".lock")

        # Async lock for task safety within the current worker
        self._lock = asyncio.Lock()

        # In-memory cache
        self._models: Dict[str, ModelEntry] = {}
        self._signature: Optional[tuple] = None

        # Initial boot load
        self._load()

    @contextmanager
    def _file_lock(self):
        """Exclusive lock shared by every worker process (POSIX flock)."""
        if fcntl is None:
            yield
            return
        with open(self._lockpath, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _load(self, force: bool = False) -> None:
        """
        Loads models from disk.
        Re-reads only when the file changed (mtime, size or inode), unless forced.
        One invalid entry is skipped and logged; it never hides the others.
        """
        if not self._filepath.exists():
            return

        try:
            st = self._filepath.stat()
            signature = (st.st_mtime_ns, st.st_size, st.st_ino)
            if not force and signature == self._signature:
                return
            with open(self._filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("models file must contain a JSON list")

            models: Dict[str, ModelEntry] = {}
            for pos, item in enumerate(data):
                try:
                    # Stable "created" for hand-written entries without one
                    entry = ModelEntry(**{"created_at": int(st.st_mtime), **item})
                except (ValidationError, TypeError) as e:
                    logger.error(
                        "Skipping invalid model entry #%d in %s: %s",
                        pos,
                        self._filepath,
                        e,
                    )
                    continue
                if entry.id in models:
                    logger.error(
                        "Duplicate model id '%s' in %s — keeping the first",
                        entry.id,
                        self._filepath,
                    )
                    continue
                models[entry.id] = entry

            self._models = models
            self._signature = signature
            logger.debug("Loaded %d models from disk.", len(self._models))
        except Exception as e:
            logger.error("Failed to load models from disk: %s", e)

    def _save(self) -> None:
        """
        Saves models to disk using an atomic write to prevent file corruption.
        """
        temp_filepath = self._filepath.with_suffix(".tmp")
        try:
            # Dump to a temporary file first
            data = [
                model.model_dump(exclude_none=True) for model in self._models.values()
            ]
            with open(temp_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic swap (replaces the old file with the new temp file instantly)
            os.replace(temp_filepath, self._filepath)

            st = self._filepath.stat()
            self._signature = (st.st_mtime_ns, st.st_size, st.st_ino)

        except Exception as e:
            logger.error("Failed to save models to disk: %s", e)
            if temp_filepath.exists():
                os.remove(temp_filepath)
            raise

    # --- Read Operations ---

    def get(self, model_id: str) -> Optional[ModelEntry]:
        """Retrieve a specific model by ID."""
        self._load()  # Sync cache if file was modified by another worker
        return self._models.get(model_id)

    def list_all(self) -> List[ModelEntry]:
        """List all available models."""
        self._load()  # Sync cache if file was modified by another worker
        return list(self._models.values())

    # --- Write Operations ---

    async def add(self, data: ModelCreate) -> ModelEntry:
        """Add a new model to the registry."""
        async with self._lock:
            with self._file_lock():
                # Strictly sync to prevent overwriting other worker's changes
                self._load(force=True)

                if data.id in self._models:
                    raise ValueError(f"Model '{data.id}' already exists")

                entry = ModelEntry(**data.model_dump())
                self._models[entry.id] = entry
                self._save()

        logger.info("Added model: %s [%s]", entry.id, entry.model_type)
        return entry

    async def update(self, model_id: str, data: ModelUpdate) -> ModelEntry:
        """Update an existing model's properties."""
        async with self._lock:
            with self._file_lock():
                self._load(force=True)

                if model_id not in self._models:
                    raise KeyError(f"Model '{model_id}' not found")

                current = self._models[model_id]

                # Only fields the caller sent; re-validate the merged entry
                updates = data.model_dump(exclude_unset=True)
                updated = ModelEntry(**{**current.model_dump(), **updates})

                self._models[model_id] = updated
                self._save()

        logger.info("Updated model: %s", model_id)
        return updated

    async def delete(self, model_id: str) -> ModelEntry:
        """Remove a model from the registry."""
        async with self._lock:
            with self._file_lock():
                self._load(force=True)

                if model_id not in self._models:
                    raise KeyError(f"Model '{model_id}' not found")

                entry = self._models.pop(model_id)
                self._save()

        logger.info("Deleted model: %s", model_id)
        return entry


# Global singleton instance
registry = ModelRegistry(settings.models_path)
