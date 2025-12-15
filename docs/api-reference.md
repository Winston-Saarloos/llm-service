# API Reference

## Base URL

The service runs on port `6002` by default:

- Local: `http://localhost:6002`
- Docker: `http://localhost:6002` (mapped from container)

## Authentication

All endpoints except `/health` and `/healthz` require authentication via the `X-Server-Auth-Secret` header:

```
X-Server-Auth-Secret: your-secret-key-here
```

If the header is missing or incorrect, the service returns `401 Unauthorized`.

## Endpoints

### Health Check

#### `GET /health` or `GET /healthz`

Check if the service is running and healthy. Both endpoints are supported for compatibility with different monitoring systems.

**Authentication:** Not required

**Response:**

```json
{
  "status": "ok"
}
```

**Examples:**

```bash
# Using /health
curl http://localhost:6002/health

# Using /healthz (Kubernetes convention)
curl http://localhost:6002/healthz
```

---

### List Models

#### `GET /v1/models`

Retrieve a list of all models available in the connected Ollama instance.

**Authentication:** Required

**Response:**

```json
{
  "data": [{ "id": "llama3" }, { "id": "llama3.1:latest" }, { "id": "mistral" }]
}
```

**Example:**

```bash
curl -H "X-Server-Auth-Secret: your-secret-key-here" \
  http://localhost:6002/v1/models
```

**Error Responses:**

- `500 Internal Server Error`: Failed to connect to Ollama

---

### Chat Completion (Non-Streaming)

#### `POST /v1/chat/completions`

Generate a chat completion using the specified model. Returns a complete response.

**Authentication:** Required

**Request Body:**

```json
{
  "model": "llama3",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" }
  ],
  "temperature": 0.7,
  "max_tokens": 256
}
```

**Parameters:**

- `model` (string, required): Name of the model to use. Must match a model available in Ollama.
- `messages` (array, required): Array of message objects with `role` and `content`.
  - `role`: One of `"system"`, `"user"`, or `"assistant"`
  - `content`: The message text
- `temperature` (float, optional): Sampling temperature (0.0 to 2.0). Default: `0.7`
- `max_tokens` (integer, optional): Maximum tokens to generate. Default: `256`
- `stream` (boolean, optional): Ignored for this endpoint (always non-streaming). Default: `false`

**Response:**

```json
{
  "id": "chatcmpl-1",
  "object": "chat.completion",
  "model": "llama3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 8,
    "total_tokens": 23
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:6002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Server-Auth-Secret: your-secret-key-here" \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Error Responses:**

- `400 Bad Request`: Model not available in Ollama
- `401 Unauthorized`: Missing or invalid authentication header
- `500 Internal Server Error`: Ollama connection or processing error

---

### Chat Completion (Streaming)

#### `POST /v1/chat/completions/stream`

Generate a chat completion with streaming responses. Returns NDJSON (newline-delimited JSON) chunks.

**Authentication:** Required

**Request Body:**

```json
{
  "model": "llama3",
  "messages": [{ "role": "user", "content": "Tell me a story" }],
  "temperature": 0.7
}
```

**Parameters:**

- Same as non-streaming endpoint (see above)
- `stream` parameter is ignored (always streams)

**Response Format:**

NDJSON stream where each line is a JSON object:

```json
{"model":"llama3","created_at":"2024-01-01T00:00:00.000Z","response":"Once","done":false}
{"model":"llama3","created_at":"2024-01-01T00:00:00.100Z","response":" upon","done":false}
{"model":"llama3","created_at":"2024-01-01T00:00:00.200Z","response":" a","done":false}
{"model":"llama3","created_at":"2024-01-01T00:00:00.300Z","response":" time","done":false}
{"model":"llama3","created_at":"2024-01-01T00:00:00.400Z","response":"...","done":true,"total_duration":5000000000,"load_duration":1000000000,"prompt_eval_count":10,"prompt_eval_duration":200000000,"eval_count":50,"eval_duration":4000000000}
```

**Example:**

```bash
curl -X POST http://localhost:6002/v1/chat/completions/stream \
  -H "Content-Type: application/json" \
  -H "X-Server-Auth-Secret: your-secret-key-here" \
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Processing the Stream:**

In Python:

```python
import httpx
import json

response = httpx.post(
    "http://localhost:6002/v1/chat/completions/stream",
    headers={
        "Content-Type": "application/json",
        "X-Server-Auth-Secret": "your-secret-key-here"
    },
    json={
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hello!"}]
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        if not chunk.get("done"):
            print(chunk.get("response", ""), end="", flush=True)
```

**Error Responses:**

- `400 Bad Request`: Model not available in Ollama
- `401 Unauthorized`: Missing or invalid authentication header
- `500 Internal Server Error`: Ollama connection or processing error

---

## Response Format

### Chat Completion Response

The service returns responses in OpenAI-compatible format:

- `id`: Unique identifier for the completion
- `object`: Always `"chat.completion"`
- `model`: The model used for the completion
- `choices`: Array with a single choice object containing:
  - `index`: Always `0`
  - `message`: The assistant's message with `role` and `content`
  - `finish_reason`: Always `"stop"` for completed responses
- `usage`: Token usage statistics (if available from Ollama)

### Error Response

All errors return a JSON object with a `detail` field:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Validation errors may include additional fields:

```json
{
  "detail": "Request validation failed",
  "errors": [
    {
      "loc": ["body", "model"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "body": "{\"messages\": [...]}"
}
```
