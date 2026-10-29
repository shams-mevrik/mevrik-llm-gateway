"""End-to-end tests through the FastAPI app against a fake vLLM backend."""

from __future__ import annotations

import json

import httpx

from app import proxy
from app.registry import ModelEntry, registry
from tests.conftest import ADMIN, BACKEND, USER


# ---------------------------------------------------------------------------
# /classify
# ---------------------------------------------------------------------------
def test_classify_root_path_goes_to_backend_root(client, backend):
    r = client.post(
        "/classify",
        headers=USER,
        json={"model": "spark-decision", "input": ["[app|customer|auth]\n>> balance?"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["probs"] == [0.9, 0.1]

    sent = backend.requests[-1]
    assert str(sent.url) == f"{BACKEND}/classify"  # /v1 stripped
    body = json.loads(sent.content)
    assert body["model"] == "decision"  # rewritten to backend name
    assert body["truncation_side"] == "right"  # default_params applied
    assert sent.headers["authorization"] == "Bearer backend-secret"


def test_classify_v1_alias_and_client_params_win(client, backend):
    r = client.post(
        "/v1/classify",
        headers=USER,
        json={
            "model": "spark-decision",
            "input": "hello",
            "truncation_side": "left",
            "truncate_prompt_tokens": 2048,
        },
    )
    assert r.status_code == 200, r.text
    body = backend.last_json()
    assert body["truncation_side"] == "left"
    assert body["truncate_prompt_tokens"] == 2048
    assert str(backend.requests[-1].url) == f"{BACKEND}/classify"


def test_classify_rejects_wrong_type_and_unknown_model(client, backend):
    r = client.post(
        "/classify", headers=USER, json={"model": "spark-chat", "input": "x"}
    )
    assert r.status_code == 400
    r = client.post("/classify", headers=USER, json={"model": "nope", "input": "x"})
    assert r.status_code == 404
    r = client.post("/classify", headers=USER, json={"model": "spark-decision"})
    assert r.status_code == 422  # neither input nor messages
    assert backend.requests == []


def test_classify_requires_api_key(client, backend):
    r = client.post("/classify", json={"model": "spark-decision", "input": "x"})
    assert r.status_code == 401
    r = client.post(
        "/classify",
        headers={"Authorization": "Bearer wrong"},
        json={"model": "spark-decision", "input": "x"},
    )
    assert r.status_code == 401


def test_classify_backend_error_passes_through(client, backend):
    backend.responses["/classify"] = httpx.Response(
        400,
        json={
            "error": {"message": "This model's maximum context length is 8192 tokens"}
        },
    )
    r = client.post(
        "/classify", headers=USER, json={"model": "spark-decision", "input": "x"}
    )
    assert r.status_code == 400
    assert "maximum context length" in r.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Chat — exact pass-through, default_params, streaming
# ---------------------------------------------------------------------------
def test_chat_forwards_body_exactly_with_defaults_merged(client, backend):
    payload = {
        "model": "spark-chat",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": ["string", "null"], "default": None}
                        },
                    },
                },
            }
        ],
        "chat_template_kwargs": {"custom": 1},
        "temperature": 0,
    }
    r = client.post("/v1/chat/completions", headers=USER, json=payload)
    assert r.status_code == 200, r.text

    sent = backend.last_json()
    assert sent["model"] == "generation"
    assert sent["messages"] == payload["messages"]  # content: null kept
    assert sent["tools"] == payload["tools"]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False, "custom": 1}
    assert str(backend.requests[-1].url) == f"{BACKEND}/v1/chat/completions"
    assert "authorization" not in backend.requests[-1].headers


def test_chat_stream_relays_sse(client, backend):
    r = client.post(
        "/v1/chat/completions",
        headers=USER,
        json={
            "model": "spark-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert '"content":"Hi"' in r.text and "[DONE]" in r.text
    assert proxy._semaphores[f"{BACKEND}/v1"]._value == proxy._MAX_INFLIGHT_PER_BACKEND


def test_chat_stream_backend_error_returns_real_status(client, backend):
    backend.responses["/v1/chat/completions"] = httpx.Response(
        400, json={"error": {"message": "tool_choice not supported"}}
    )
    r = client.post(
        "/v1/chat/completions",
        headers=USER,
        json={
            "model": "spark-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "tool_choice not supported"
    assert proxy._semaphores[f"{BACKEND}/v1"]._value == proxy._MAX_INFLIGHT_PER_BACKEND


def test_chat_on_classifier_rejected(client, backend):
    r = client.post(
        "/v1/chat/completions",
        headers=USER,
        json={
            "model": "spark-decision",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 400
    assert backend.requests == []


def test_network_error_does_not_leak_backend_address(client, backend):
    backend.fail_with = httpx.ConnectError("connection refused to backend.invalid")
    for stream in (False, True):
        r = client.post(
            "/v1/chat/completions",
            headers=USER,
            json={
                "model": "spark-chat",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": stream,
            },
        )
        assert r.status_code == 502
        assert "backend.invalid" not in r.text
        assert "spark-chat" in r.text
    r = client.post(
        "/classify", headers=USER, json={"model": "spark-decision", "input": "x"}
    )
    assert r.status_code == 502 and "backend.invalid" not in r.text
    assert proxy._semaphores[f"{BACKEND}/v1"]._value == proxy._MAX_INFLIGHT_PER_BACKEND


# ---------------------------------------------------------------------------
# Embeddings / rerank keep working
# ---------------------------------------------------------------------------
def test_embeddings_and_rerank(client, backend):
    r = client.post(
        "/v1/embeddings", headers=USER, json={"model": "spark-embed", "input": "x"}
    )
    assert r.status_code == 200 and r.json()["data"][0]["embedding"] == [0.1, 0.2]

    r = client.post(
        "/v1/rerank",
        headers=USER,
        json={
            "model": "spark-rerank",
            "query": "q",
            "documents": ["a", "b", "c"],
            "top_n": 2,
        },
    )
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert [x["index"] for x in results] == [0, 1]
    assert str(backend.requests[-1].url) == f"{BACKEND}/v1/rerank"


# ---------------------------------------------------------------------------
# Model management + backend privacy
# ---------------------------------------------------------------------------
def test_listing_hides_backend_from_users(client, backend):
    r = client.get("/v1/models", headers=USER)
    assert r.status_code == 200
    assert "backend.invalid" not in r.text and "backend-secret" not in r.text
    types = {m["id"]: m["model_type"] for m in r.json()["data"]}
    assert types["spark-decision"] == "classify"

    r = client.get("/v1/models/spark-decision", headers=ADMIN)
    assert r.json()["backend_url"] == f"{BACKEND}/v1"
    assert r.json()["backend_auth"] is True
    assert "backend-secret" not in r.text  # write-only


def test_crud_requires_admin(client, backend):
    new = {
        "id": "spark-city-decision",
        "backend_url": f"{BACKEND}/v1/",
        "backend_model": "decision",
        "model_type": "classify",
    }
    assert client.post("/v1/models", headers=USER, json=new).status_code == 403
    r = client.post("/v1/models", headers=ADMIN, json=new)
    assert r.status_code == 201, r.text
    assert r.json()["model"]["backend_url"] == f"{BACKEND}/v1"  # trailing / trimmed

    assert (
        client.patch(
            "/v1/models/spark-city-decision", headers=USER, json={"name": "x"}
        ).status_code
        == 403
    )
    r = client.patch(
        "/v1/models/spark-city-decision", headers=ADMIN, json={"model_type": "bogus"}
    )
    assert r.status_code == 422
    r = client.patch(
        "/v1/models/spark-city-decision",
        headers=ADMIN,
        json={"default_params": {"truncation_side": "right"}},
    )
    assert r.status_code == 200
    assert registry.get("spark-city-decision").default_params == {
        "truncation_side": "right"
    }

    assert (
        client.delete("/v1/models/spark-city-decision", headers=USER).status_code == 403
    )
    assert (
        client.delete("/v1/models/spark-city-decision", headers=ADMIN).status_code
        == 200
    )
    assert registry.get("spark-city-decision") is None


def test_create_validation(client, backend):
    bad = [
        {"id": "a", "backend_url": "ftp://x", "backend_model": "m"},
        {
            "id": "a",
            "backend_url": f"{BACKEND}/v1",
            "backend_model": "m",
            "model_type": "llm",
        },
        {
            "id": "a",
            "backend_url": f"{BACKEND}/v1",
            "backend_model": "m",
            "default_params": {"model": "other"},
        },
        {"id": "  ", "backend_url": f"{BACKEND}/v1", "backend_model": "m"},
    ]
    for body in bad:
        assert client.post("/v1/models", headers=ADMIN, json=body).status_code == 422, (
            body
        )


def test_single_key_mode_keeps_old_behaviour(client, backend, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "admin_api_key", "")
    r = client.get("/v1/models/spark-chat", headers=USER)
    assert r.json()["backend_url"] == f"{BACKEND}/v1"
    new = {"id": "m2", "backend_url": f"{BACKEND}/v1", "backend_model": "m"}
    assert client.post("/v1/models", headers=USER, json=new).status_code == 201


def test_backend_health(client, backend):
    assert client.get("/health/backends", headers=USER).status_code == 403
    r = client.get("/health/backends", headers=ADMIN)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok" and data["healthy"] == 4
    assert "backend.invalid" not in r.text
    paths = {req.url.path for req in backend.requests}
    assert paths == {"/health"}


def test_benchmark_rejects_non_chat_models(client, backend):
    r = client.post("/v1/models/spark-decision/test", headers=USER, json={})
    assert r.status_code == 400
    r = client.post("/v1/models/spark-chat/loadtest", headers=USER, json={})
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_backend_root():
    def root(url):
        return ModelEntry(id="x", backend_url=url, backend_model="m").backend_root

    assert root("http://h:8011/v1") == "http://h:8011"
    assert root("http://h:8011/v1/") == "http://h:8011"
    assert root("http://h:8011") == "http://h:8011"


def test_invalid_entry_is_skipped_not_fatal(tmp_path):
    from app.registry import ModelRegistry

    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            [
                {"id": "ok", "backend_url": "http://h/v1", "backend_model": "m"},
                {
                    "id": "bad",
                    "backend_url": "http://h/v1",
                    "backend_model": "m",
                    "model_type": "x",
                },
                {"id": "missing-url", "backend_model": "m"},
            ]
        ),
        encoding="utf-8",
    )
    reg = ModelRegistry(str(path))
    assert [m.id for m in reg.list_all()] == ["ok"]
