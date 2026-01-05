"""Embedding worker for Redis stream jobs using Ollama (GPU host)."""
import json
import os
import socket
import time
from urllib import error, parse, request

import redis


def env(name: str, default: str | None = None) -> str:
    """Read environment variable with optional default."""
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def http_json(method: str, url: str, payload: dict | None, headers: dict) -> dict:
    """Send HTTP request with JSON body and return JSON response."""
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def chunk_text(text: str, chunk_tokens: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    tokens = text.split()
    if len(tokens) <= chunk_tokens:
        return [(0, len(tokens), text.strip())]

    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= len(tokens):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def main() -> int:
    redis_url = env("REDIS_URL")
    memory_service_url = env("MEMORY_SERVICE_URL", "http://localhost:8000")
    memory_service_secret = env("MEMORY_SERVICE_SECRET")
    api_prefix = env("MEMORY_SERVICE_API_PREFIX", "/v1")
    ollama_url = env("OLLAMA_HOST", "http://localhost:11434")
    stream_name = env("EMBEDDING_JOBS_STREAM_NAME", "embedding_jobs")
    default_model = env("EMBEDDING_MODEL", env("EMBEDDING_MODEL_DEFAULT", "embeddinggemma:latest"))
    chunk_tokens = int(env("EMBEDDING_CHUNK_TOKENS", "350"))
    chunk_overlap = int(env("EMBEDDING_CHUNK_OVERLAP_TOKENS", "60"))
    group_name = env("EMBEDDING_JOBS_GROUP", "embedding_workers")
    consumer_name = env(
        "EMBEDDING_JOBS_CONSUMER",
        f"{socket.gethostname()}-{os.getpid()}",
    )

    if default_model.endswith(":latest"):
        raise RuntimeError("Embedding model must be pinned. Do not use :latest.")
    if chunk_overlap >= chunk_tokens:
        raise RuntimeError("EMBEDDING_CHUNK_OVERLAP_TOKENS must be less than EMBEDDING_CHUNK_TOKENS.")

    try:
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        try:
            redis_client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
    except Exception as exc:
        log(f"Redis unavailable, exiting: {exc}")
        return 1

    headers = {"Content-Type": "application/json", "X-Service-Secret": memory_service_secret}
    log(
        "Embedding worker online. "
        f"stream={stream_name} model={default_model} chunk_tokens={chunk_tokens} overlap={chunk_overlap}"
    )
    log(
        "Endpoints: "
        f"redis={redis_url} memory_service={memory_service_url} ollama={ollama_url}"
    )

    while True:
        try:
            response = redis_client.xreadgroup(
                group_name,
                consumer_name,
                {stream_name: ">"},
                count=1,
                block=5000,
            )
        except Exception as exc:
            log(f"Redis read failed, retrying: {exc}")
            time.sleep(5)
            continue
        if not response:
            continue

        for _, messages in response:
            for message_id, fields in messages:
                item_id = fields.get("item_id")
                tenant_id = fields.get("tenant_id")
                app_id = fields.get("app_id")
                user_id = fields.get("user_id")
                embedding_model = fields.get("embedding_model") or default_model
                if embedding_model.endswith(":latest"):
                    embedding_model = default_model
                last_url = ""

                if not all([item_id, tenant_id, app_id, user_id]):
                    redis_client.xack(stream_name, group_name, message_id)
                    continue

                try:
                    log(f"Embedding job received item_id={item_id} model={embedding_model}")
                    item_params = parse.urlencode(
                        {
                            "tenant_id": tenant_id,
                            "app_id": app_id,
                            "user_id": user_id,
                        }
                    )
                    item_url = f"{memory_service_url}{api_prefix}/items/{item_id}?{item_params}"
                    last_url = item_url
                    item_resp = http_json("GET", item_url, None, headers)
                    item_data = item_resp.get("data") or {}
                    content = item_data.get("content", "")
                    title = item_data.get("title")
                    metadata = item_data.get("metadata") or item_data.get("meta_data") or {}

                    if metadata.get("chunked") is True:
                        log(f"Item already chunked, skipping item_id={item_id}")
                        redis_client.xack(stream_name, group_name, message_id)
                        continue

                    if title:
                        content = f"{title}\n\n{content}"
                    if not content:
                        log(f"Empty content, skipping item_id={item_id}")
                        redis_client.xack(stream_name, group_name, message_id)
                        continue

                    should_chunk = metadata.get("is_chunk") is not True
                    chunks = (
                        chunk_text(content, chunk_tokens, chunk_overlap)
                        if should_chunk
                        else [(0, 0, content)]
                    )

                    if should_chunk and len(chunks) > 1:
                        log(f"Chunking item_id={item_id} chunks={len(chunks)}")
                        for idx, (start_idx, end_idx, chunk_text_value) in enumerate(chunks):
                            log(
                                f"Embedding chunk {idx + 1}/{len(chunks)} "
                                f"item_id={item_id} tokens={end_idx - start_idx}"
                            )
                            chunk_title = f"{title} (chunk {idx + 1})" if title else None
                            chunk_metadata = {
                                **metadata,
                                "is_chunk": True,
                                "chunk_index": idx,
                                "chunk_count": len(chunks),
                                "chunk_start_token": start_idx,
                                "chunk_end_token": end_idx,
                                "source_item_id": item_id,
                                "source_type": item_data.get("type"),
                            }
                            create_payload = {
                                "tenant_id": tenant_id,
                                "app_id": app_id,
                                "user_id": user_id,
                                "session_id": item_data.get("session_id"),
                                "type": f"{item_data.get('type', 'document')}_chunk",
                                "title": chunk_title,
                                "content": chunk_text_value,
                                "tags": item_data.get("tags") or [],
                                "source": item_data.get("source"),
                                "metadata": chunk_metadata,
                                "importance": item_data.get("importance", 0),
                                "confidence": item_data.get("confidence", 0.0),
                            }
                            created = http_json(
                                "POST",
                                f"{memory_service_url}{api_prefix}/items",
                                create_payload,
                                headers,
                            )
                            created_item = (created or {}).get("data") or {}
                            chunk_item_id = created_item.get("id")
                            if not chunk_item_id:
                                continue

                            embedding_resp = http_json(
                                "POST",
                                f"{ollama_url}/api/embeddings",
                                {"model": embedding_model, "prompt": chunk_text_value},
                                {"Content-Type": "application/json"},
                            )
                            embedding = embedding_resp.get("embedding")
                            if not embedding:
                                raise RuntimeError("No embedding returned from Ollama")

                            log(
                                f"Upserting embedding for chunk item_id={chunk_item_id} dims={len(embedding)}"
                            )
                            upsert_payload = {
                                "tenant_id": tenant_id,
                                "app_id": app_id,
                                "user_id": user_id,
                                "embedding": embedding,
                                "embedding_model": embedding_model,
                                "dims": len(embedding),
                            }
                            upsert_url = (
                                f"{memory_service_url}{api_prefix}/embeddings/{chunk_item_id}/embedding"
                            )
                            last_url = upsert_url
                            http_json("PUT", upsert_url, upsert_payload, headers)

                        patch_payload = {
                            "metadata": {
                                **metadata,
                                "chunked": True,
                                "chunk_count": len(chunks),
                                "embedding_model": embedding_model,
                            }
                        }
                        http_json(
                            "PATCH",
                            f"{memory_service_url}{api_prefix}/items/{item_id}"
                            f"?tenant_id={tenant_id}&app_id={app_id}&user_id={user_id}",
                            patch_payload,
                            headers,
                        )
                        last_url = f"{memory_service_url}{api_prefix}/items/{item_id}"
                        log(f"Chunked embeddings complete item_id={item_id}")
                    else:
                        log(f"Embedding single item_id={item_id}")
                        embedding_url = f"{ollama_url}/api/embeddings"
                        last_url = embedding_url
                        embedding_resp = http_json(
                            "POST",
                            embedding_url,
                            {"model": embedding_model, "prompt": content},
                            {"Content-Type": "application/json"},
                        )
                        embedding = embedding_resp.get("embedding")
                        if not embedding:
                            raise RuntimeError("No embedding returned from Ollama")

                        log(f"Upserting embedding for item_id={item_id} dims={len(embedding)}")
                        upsert_payload = {
                            "tenant_id": tenant_id,
                            "app_id": app_id,
                            "user_id": user_id,
                            "embedding": embedding,
                            "embedding_model": embedding_model,
                            "dims": len(embedding),
                        }
                        upsert_url = f"{memory_service_url}{api_prefix}/embeddings/{item_id}/embedding"
                        last_url = upsert_url
                        http_json("PUT", upsert_url, upsert_payload, headers)

                    redis_client.xack(stream_name, group_name, message_id)
                    log(f"Embedding job complete item_id={item_id}")
                except error.HTTPError as exc:
                    if exc.code in {401, 403, 404}:
                        redis_client.xack(stream_name, group_name, message_id)
                        continue
                    log(
                        f"HTTP error {exc.code} for item_id={item_id} url={last_url}, retrying"
                    )
                    time.sleep(2)
                except Exception as exc:
                    log(f"Unexpected error for item_id={item_id} url={last_url}: {exc}")
                    time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
