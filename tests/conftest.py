"""Test setup: isolated data dir + a fake vLLM backend behind httpx.MockTransport.

pip install -r requirements-dev.txt && pytest
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Must run before `app` is imported: settings and the registry read these.
_DATA = Path(tempfile.mkdtemp(prefix="gateway-test-"))
os.environ["DATA_DIR"] = str(_DATA)
os.environ["MODELS_PATH"] = str(_DATA / "models.json")
os.environ["STATS_PATH"] = str(_DATA / "stats.json")
os.environ["API_KEY"] = "user-key"
os.environ["ADMIN_API_KEY"] = "admin-key"

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import proxy  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.registry import registry  # noqa: E402

BACKEND = "http://backend.invalid:8000"

MODELS = [
    {
        "id": "spark-chat",
        "backend_url": f"{BACKEND}/v1",
        "backend_model": "generation",
        "model_type": "chat",
        "default_params": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    {
        "id": "spark-decision",
        "backend_url": f"{BACKEND}/v1",
        "backend_model": "decision",
        "model_type": "classify",
        "default_params": {"truncation_side": "right"},
        "backend_api_key": "backend-secret",
    },
    {
        "id": "spark-embed",
        "backend_url": f"{BACKEND}/v1",
        "backend_model": "embedding",
        "model_type": "embedding",
    },
    {
        "id": "spark-rerank",
        "backend_url": f"{BACKEND}/v1",
        "backend_model": "rerank",
        "model_type": "rerank",
    },
]

USER = {"Authorization": "Bearer user-key"}
ADMIN = {"Authorization": "Bearer admin-key"}


class FakeBackend:
    """Records requests; answers like vLLM. Override `responses` per test."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.responses: dict[str, httpx.Response] = {}
        self.fail_with: Exception | None = None

    def last_json(self) -> dict:
        return json.loads(self.requests[-1].content)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        path = request.url.path
        if path in self.responses:
            return self.responses[path]
        body = json.loads(request.content) if request.content else {}
        if path == "/classify":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "index": 0,
                            "label": "kb",
                            "probs": [0.9, 0.1],
                            "num_classes": 2,
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "total_tokens": 7},
                },
            )
        if path == "/v1/chat/completions" and body.get("stream"):
            chunks = [
                b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"".join(chunks),
            )
        if path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            )
        if path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        if path == "/v1/rerank":
            docs = body.get("documents", [])
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": i, "relevance_score": 1.0 / (i + 1)}
                        for i in range(len(docs))
                    ]
                },
            )
        if path == "/health":
            return httpx.Response(200)
        return httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    monkeypatch.setattr(proxy, "_client", client)
    monkeypatch.setattr(proxy, "_semaphores", {})
    return fake


@pytest.fixture(autouse=True)
def models_file(monkeypatch):
    """Fresh models.json and keys for every test."""
    Path(os.environ["MODELS_PATH"]).write_text(json.dumps(MODELS), encoding="utf-8")
    registry._load(force=True)
    monkeypatch.setattr(settings, "api_key", "user-key")
    monkeypatch.setattr(settings, "admin_api_key", "admin-key")
    yield


@pytest.fixture
def client(backend):
    with TestClient(app) as c:
        yield c
