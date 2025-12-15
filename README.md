# LLM Service

A generic FastAPI microservice that acts as a proxy in front of a local Ollama instance.  
It exposes OpenAI-compatible endpoints so any client can call a local LLM over HTTP.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completion endpoints
- **Docker-Ready**: Containerized service that runs out of the box
- **Model Validation**: Automatically validates that requested models exist in Ollama
- **Streaming Support**: Both streaming and non-streaming chat completion endpoints
- **Authentication**: Configurable server-side authentication via header
- **Health Checks**: Built-in health check endpoint for monitoring

---

## Endpoints

- `GET /health` or `GET /healthz`  
  Health check for the service. Both endpoints are supported for compatibility.

- `GET /v1/models`  
  Lists models available in Ollama.

- `POST /v1/chat/completions`  
  Non-streaming chat completion. Requires `{ model: string, messages: [...] }`.  
  The `model` field is required and must match a model available in Ollama.  
  Returns a 400 error if the model is not available.

- `POST /v1/chat/completions/stream`  
  Streaming version (NDJSON) that relays Ollama's streaming output.  
  Requires `{ model: string, messages: [...] }`.  
  The `model` field is required and must match a model available in Ollama.  
  Returns a 400 error if the model is not available.

---

## Prerequisites

- Docker (Desktop on Windows with WSL2 integration, or Docker on Linux)
- An Ollama container or service running and reachable at `http://ollama:11434` (Docker) or `http://localhost:11434` (local dev)
- At least one model pulled in Ollama, e.g.:
  ```bash
  docker exec -it ollama ollama pull llama3
  ```

## Environment Variables

Environment variables are managed using a `.env` file that is **not tracked in git**.

1. **Create the `.env` file:**

   ```bash
   cp .env.example .env
   ```

2. **Configure environment variables in `.env`:**

   ```bash
   # Required: Ollama host URL
   OLLAMA_HOST=http://host.docker.internal:11434

   # Required: Server authentication secret (must match client API)
   SERVER_AUTH_SECRET=your-secret-key-here
   ```

3. **The `.env` file is automatically loaded by docker-compose** (both local and prod configs)

**Important:**

- The `.env` file is not tracked in git (already in `.gitignore`)
- The `.env.example` file is tracked and serves as a template
- If `SERVER_AUTH_SECRET` is not set, the service will block all requests

## Deployment

### Local Development

For local testing with Docker:

```bash
docker-compose -f docker-compose.local.yml up -d
```

To stop:

```bash
docker-compose -f docker-compose.local.yml down
```

### Production Deployment

For production deployment, use `docker-compose.prod.yml`:

```bash
# Build and tag the image first
docker build -t your-registry/llm-service:main .

# Push to your registry (if needed)
docker push your-registry/llm-service:main

# Deploy using production compose file
docker-compose -f docker-compose.prod.yml up -d
```

**Note:**

- Update the image name in `docker-compose.prod.yml` to match the container registry
- The service runs on port `6002` by default
- Ollama must be accessible at the configured `OLLAMA_HOST` URL

## Documentation

For detailed usage instructions, API reference, and examples, see the [docs](./docs/) folder:

- [Getting Started](./docs/getting-started.md)
- [API Reference](./docs/api-reference.md)
- [Configuration](./docs/configuration.md)
- [Examples](./docs/examples.md)
