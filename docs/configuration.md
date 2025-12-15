# Configuration

## Environment Variables

The service is configured using environment variables, which can be set via:

- `.env` for local
- Docker Compose `environment` section
- Container environment variables
- System environment variables

### Required Variables

#### `OLLAMA_HOST`

The URL of the Ollama instance to connect to.

**Default:** `http://host.docker.internal:11434`

**Examples:**

- Docker: `http://host.docker.internal:11434` (when Ollama runs on the host)
- Docker network: `http://ollama:11434` (when Ollama is in the same Docker network)
- Local: `http://localhost:11434` (when running locally)

**Note:** The service will fail to connect if Ollama is not accessible at this URL.

#### `SERVER_AUTH_SECRET`

A secret key used to authenticate requests to the service. This must be provided in the `X-Server-Auth-Secret` header for all requests (except `/healthz`).

**Default:** None (service will block all requests if not set)

- If not set, the service will return `500 Internal Server Error` for all requests

### Optional Variables

#### `PORT`

The port the service listens on.

**Default:** `6002`

**Note:** This is set in the Dockerfile. To change it, modify the Dockerfile or override the CMD in the Docker Compose file.

## Configuration Examples

### Local Development

Create a `.env` file:

```env
OLLAMA_HOST=http://localhost:11434
SERVER_AUTH_SECRET=dev-secret-key-change-in-production
```

### Docker with Host Ollama

```env
OLLAMA_HOST=http://host.docker.internal:11434
SERVER_AUTH_SECRET=your-production-secret-here
```

### Docker Compose with Ollama Service

If Ollama is running as a Docker service in the same network:

```yaml
services:
  llm-service:
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - SERVER_AUTH_SECRET=${SERVER_AUTH_SECRET}
```

### Production Deployment

For production, use environment variables or secrets management:

```bash
export OLLAMA_HOST=http://ollama-service:11434
export SERVER_AUTH_SECRET=$(cat /run/secrets/llm_auth_secret)
```

## Docker Compose Configuration

### Local Development (`docker-compose.local.yml`)

```yaml
services:
  llm-service:
    build: .
    ports:
      - "6002:6002"
    env_file:
      - .env
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### Production (`docker-compose.prod.yml`)

```yaml
services:
  llm-service:
    image: llm-service:main
    container_name: llm-service
    restart: unless-stopped
    ports:
      - "6002:6002"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
      - SERVER_AUTH_SECRET=${SERVER_AUTH_SECRET}
```

## Network Configuration

### Connecting to Ollama

The service needs network access to Ollama. Common scenarios:

1. **Ollama on Host Machine**

   - Use `http://host.docker.internal:11434`
   - Requires `extra_hosts` in Docker Compose (already configured)

2. **Ollama in Same Docker Network**

   - Use `http://ollama:11434` (or the Ollama service name)
   - Ensure both services are in the same network

3. **Ollama on Remote Server**
   - Use the full URL: `http://remote-server:11434`
   - Ensure firewall rules allow access

### Port Configuration

The service listens on port `6002` by default. To change:

1. Update the `PORT` environment variable in the Dockerfile
2. Update the `CMD` in the Dockerfile to use the new port
3. Update port mappings in Docker Compose files

## Troubleshooting Configuration

### Service Can't Connect to Ollama

1. Verify Ollama is running:

   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Check the `OLLAMA_HOST` value matches the deployment setup

3. For Docker, verify `host.docker.internal` is accessible:
   ```bash
   docker exec llm-service ping -c 1 host.docker.internal
   ```

### Authentication Not Working

1. Verify `SERVER_AUTH_SECRET` is set:

   ```bash
   docker exec llm-service env | grep SERVER_AUTH_SECRET
   ```

2. Check that the header value matches exactly (case-sensitive)

3. Review service logs for authentication errors:
   ```bash
   docker logs llm-service
   ```

### Port Already in Use

If port 6002 is already in use:

1. Find what's using it:

   ```bash
   # Linux/Mac
   lsof -i :6002
   # Windows
   netstat -ano | findstr :6002
   ```

2. Change the port in Docker Compose:
   ```yaml
   ports:
     - "6003:6002" # Map host port 6003 to container port 6002
   ```
