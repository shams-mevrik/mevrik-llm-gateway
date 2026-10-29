# LLM Gateway

OpenAI-compatible reverse proxy for multiple vLLM backends. Models are stored in `data/models.json` and can be
added/removed at runtime via API.

Supported model types (`model_type`), one per vLLM serving mode:

| model_type   | vLLM mode                                          | Gateway endpoints                         |
|--------------|----------------------------------------------------|-------------------------------------------|
| `chat`       | generate (chat template)                           | `/v1/chat/completions`, `/v1/completions` |
| `completion` | generate                                           | `/v1/completions`                         |
| `embedding`  | `--runner pooling`                                 | `/v1/embeddings`, `/pooling`              |
| `rerank`     | `--runner pooling`                                 | `/v1/rerank`, `/v1/score`                 |
| `score`      | `--runner pooling`                                 | `/v1/score`                               |
| `classify`   | `--runner pooling` (`*ForSequenceClassification`)  | `/classify`, `/v1/classify`, `/pooling`   |
| `pooling`    | `--runner pooling`                                 | `/pooling`                                |

## Quick Start

### Docker

```bash
docker build -t llm-gateway .
docker run -d \
  --name llm-gateway \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  --restart unless-stopped \
  llm-gateway
```

> Mount the `data/` directory (the gateway reads `/app/data/models.json`) so runtime model changes and stats persist
> across restarts. Start from the example: `mkdir -p data && cp models.example.json data/models.json`, then set the
> backend URLs. `data/` is git-ignored, so backend addresses stay out of the repository.

### Without Docker

```bash
pip install -r requirements.txt
python run.py
```

## Managing Models

### Add a model at runtime

```bash
curl -X POST http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-your-secret-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "llama3-70b",
    "name": "Llama 3 70B",
    "backend_url": "http://<gpu-host>:8020/v1",
    "backend_model": "meta-llama/Llama-3-70B",
    "model_type": "chat",
    "owned_by": "meta"
  }'
```

### Model fields

| Field              | Required | Description                                                                             |
|--------------------|----------|-----------------------------------------------------------------------------------------|
| `id`               | yes      | Name clients send as `model`                                                            |
| `backend_url`      | yes      | vLLM base URL including `/v1`, e.g. `http://<gpu-host>:8011/v1`                         |
| `backend_model`    | yes      | vLLM `--served-model-name`                                                              |
| `model_type`       | no       | One of the types above (default `chat`)                                                 |
| `default_params`   | no       | Request fields added when the client leaves them out; client values always win          |
| `backend_api_key`  | no       | Sent as `Authorization: Bearer` to a backend started with vLLM `--api-key`; write-only  |
| `name`, `owned_by` | no       | Display fields                                                                          |

`default_params` examples:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
{"truncate_prompt_tokens": 2048, "truncation_side": "right"}
```

Dict values are merged one level deep, so a client's own `chat_template_kwargs` keys are kept alongside the defaults.

### Add a classifier (e.g. an intent router)

```bash
curl -X POST http://localhost:8000/v1/models \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "my-router",
    "backend_url": "http://<gpu-host>:8011/v1",
    "backend_model": "decision",
    "model_type": "classify",
    "default_params": {"truncation_side": "right"}
  }'

curl -X POST http://localhost:8000/classify \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-router", "input": ["first text", "second text"]}'
```

vLLM serves `/classify` at its server root, so the gateway strips `/v1` from `backend_url` for this route. It answers
on both `/classify` and `/v1/classify`. The response is vLLM's, unchanged (`data[].probs` in label-id order).

### List models

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-your-secret-key-change-me"
```

### Update a model

```bash
curl -X PATCH http://localhost:8000/v1/models/llama3-70b \
  -H "Authorization: Bearer sk-your-secret-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"backend_url": "http://<gpu-host>:8025/v1"}'
```

### Delete a model

```bash
curl -X DELETE http://localhost:8000/v1/models/llama3-70b \
  -H "Authorization: Bearer sk-your-secret-key-change-me"
```

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-server:8000/v1",
    api_key="sk-your-secret-key-change-me",
)

# Chat
resp = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Streaming
for chunk in client.chat.completions.create(
        model="qwen3.5-9b",
        messages=[{"role": "user", "content": "Write a poem"}],
        stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")

# Embeddings
emb = client.embeddings.create(model="qwen3-embedding", input="Hello world")

# List models
for m in client.models.list().data:
    print(m.id)
```

## Access control

| Key             | Grants                                                                                         |
|-----------------|------------------------------------------------------------------------------------------------|
| `API_KEY`       | Inference endpoints, model listing, single-model test, stats read                              |
| `ADMIN_API_KEY` | All of the above, plus add/update/delete models, load tests, stats reset, `/health/backends`, and backend details in model listings |

Without `ADMIN_API_KEY`, `API_KEY` has full access (single-key mode, the original behaviour). Backend URLs are never
included in client-facing error messages, and `backend_api_key` is never returned.

## API Endpoints

| Method | Endpoint                     | Description                                   |
|--------|------------------------------|-----------------------------------------------|
| GET    | `/health`                    | Health check                                  |
| GET    | `/health/backends`           | Probe every backend's `/health` (admin)       |
| GET    | `/v1/models`                 | List all models                               |
| GET    | `/v1/models/{id}`            | Get model info                                |
| POST   | `/v1/models`                 | Add model (persists to models.json, admin)    |
| PATCH  | `/v1/models/{id}`            | Update model fields (admin)                   |
| DELETE | `/v1/models/{id}`            | Remove model (admin)                          |
| POST   | `/v1/chat/completions`       | Chat (streaming supported)                    |
| POST   | `/v1/completions`            | Text completions (streaming supported)        |
| POST   | `/v1/embeddings`             | Embeddings                                    |
| POST   | `/v1/rerank`                 | Rerank documents                              |
| POST   | `/v1/score`                  | Cross-encoder scoring                         |
| POST   | `/classify`, `/v1/classify`  | Sequence classification                       |
| POST   | `/pooling`, `/v1/pooling`    | Raw pooler output                             |
| POST   | `/v1/models/{id}/test`       | Test chat model latency, TTFB, tokens/sec     |
| POST   | `/v1/models/{id}/loadtest`   | Concurrent stress test (admin)                |
| GET    | `/v1/stats`                  | Per-user token speed & latency stats          |
| GET    | `/v1/stats/users`            | List all tracked user keys                    |
| DELETE | `/v1/stats`                  | Clear stats (admin)                           |

Interactive docs available at `/docs`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```