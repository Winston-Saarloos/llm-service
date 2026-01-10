from typing import List, Optional, AsyncIterator
import logging

import os
import httpx
import jwt
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
SERVER_AUTH_SECRET = os.getenv("SERVER_AUTH_SECRET")
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY")

# -----------------------------------------------------------------------------
# Authentication Middleware
# -----------------------------------------------------------------------------

class ServerAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to verify server authentication via JWT token or server auth header."""
    
    def _validate_jwt_token(self, token: str) -> bool:
        """
        Validate JWT token and check for 'llm' scope.
        Returns True if token is valid and has 'llm' scope, False otherwise.
        """
        if not JWT_PUBLIC_KEY:
            logger.debug("JWT_PUBLIC_KEY not configured, skipping JWT validation")
            return False
        
        try:
            decoded_token = jwt.decode(
                token,
                JWT_PUBLIC_KEY,
                algorithms=["RS256", "ES256", "HS256"],
                options={"verify_signature": True}
            )
            
            # Check for 'llm' scope in the token
            scope = decoded_token.get("scope", "")
            if isinstance(scope, str):
                scopes = scope.split()
            elif isinstance(scope, list):
                scopes = scope
            else:
                scopes = []
            
            if "llm" in scopes:
                logger.info(f"JWT token validated successfully with 'llm' scope")
                return True
            else:
                logger.warning(f"JWT token validated but missing 'llm' scope. Found scopes: {scopes}")
                return False
                
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return False
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating JWT token: {e}")
            return False
    
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Incoming {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
        
        if request.url.path in ["/health", "/healthz"]:
            return await call_next(request)
        
        auth_header = request.headers.get("Authorization", "")
        jwt_valid = False
        
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            jwt_valid = self._validate_jwt_token(token)
        
        server_auth_header = request.headers.get("X-Server-Auth-Secret")
        server_auth_valid = False
        
        if SERVER_AUTH_SECRET and server_auth_header == SERVER_AUTH_SECRET:
            server_auth_valid = True
        
        if jwt_valid or server_auth_valid:
            auth_method = "JWT token" if jwt_valid else "X-Server-Auth-Secret header"
            logger.info(f"Request authenticated via {auth_method}")
            response = await call_next(request)
            logger.info(f"Response {response.status_code} for {request.method} {request.url.path}")
            return response
        
        if not JWT_PUBLIC_KEY and not SERVER_AUTH_SECRET:
            logger.error("Neither JWT_PUBLIC_KEY nor SERVER_AUTH_SECRET is configured. All requests will be blocked.")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Server authentication not configured"}
            )
        
        logger.error(f"Unauthorized request: Missing or invalid authentication from {request.client.host if request.client else 'unknown'}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Unauthorized: Missing or invalid authentication. Provide either a valid JWT token with 'llm' scope in Authorization header or a valid X-Server-Auth-Secret header"}
        )


# -----------------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------------

application = FastAPI(
    title="LLM Service",
    version="1.0.0",
    description="A generic LLM proxy service that wraps a local Ollama instance with OpenAI-compatible endpoints.",
)

# Add authentication middleware
application.add_middleware(ServerAuthMiddleware)


# -----------------------------------------------------------------------------
# Exception Handlers
# -----------------------------------------------------------------------------

@application.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with detailed logging."""
    logger.error(f"Validation error on {request.method} {request.url.path}")
    try:
        body = await request.body()
        body_str = body.decode('utf-8') if body else 'empty'
        logger.error(f"Request body: {body_str}")
    except Exception as e:
        logger.error(f"Could not read request body: {e}")
        body_str = None
    logger.error(f"Validation errors: {exc.errors()}")
    logger.error(f"Request headers: {dict(request.headers)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors(),
            "body": body_str
        }
    )


@application.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed logging."""
    logger.error(f"HTTP {exc.status_code} error on {request.method} {request.url.path}")
    logger.error(f"Error detail: {exc.detail}")
    logger.error(f"Request headers: {dict(request.headers)}")
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            logger.error(f"Request body: {body.decode('utf-8') if body else 'empty'}")
        except Exception as e:
            logger.error(f"Could not read request body: {e}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str  # Required: model name must be provided
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

async def validate_model_exists(model_name: str) -> bool:
    """
    Check if a model exists in the Ollama instance.
    Returns True if the model exists, False otherwise.
    """
    logger.debug(f"Checking if model '{model_name}' exists in Ollama at {OLLAMA_HOST_URL}")
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            ollama_response = await http_client.get(f"{OLLAMA_HOST_URL}/api/tags")
        
        if ollama_response.status_code != 200:
            logger.error(f"Failed to reach Ollama: status={ollama_response.status_code}, response={ollama_response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reach Ollama to validate model: {ollama_response.status_code}",
            )
        
        ollama_response_json = ollama_response.json()
        logger.debug(f"Raw Ollama /api/tags response: {ollama_response_json}")
        
        available_models = [
            model["name"] for model in ollama_response_json.get("models", [])
        ]
        
        logger.debug(f"Available models in Ollama: {available_models}")
        
        # Check exact match first
        model_exists = model_name in available_models
        
        # If not found, check if the base model name (without tag) exists
        # Ollama might accept "llama3.1:latest" even if tags list shows "llama3.1"
        if not model_exists and ":" in model_name:
            base_model = model_name.split(":")[0]
            model_exists = any(
                model.startswith(base_model) or model == base_model 
                for model in available_models
            )
            if model_exists:
                logger.info(f"Model '{model_name}' matched by base name '{base_model}'")
        
        if not model_exists:
            logger.warning(f"Model '{model_name}' not found. Available models: {available_models}")
        
        return model_exists
    except httpx.RequestError as e:
        logger.error(f"Network error connecting to Ollama at {OLLAMA_HOST_URL}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Ollama: {str(e)}",
        )


def build_prompt_from_chat_messages(chat_messages: List[ChatMessage]) -> str:
    """
    Convert a list of chat messages into a single text prompt for Ollama.
    This is a simple format that can be improved later to match specific
    chat templates.
    """
    prompt_segments: List[str] = []
    for chat_message in chat_messages:
        if chat_message.role == "system":
            prompt_segments.append(f"System: {chat_message.content}")
        elif chat_message.role == "user":
            prompt_segments.append(f"User: {chat_message.content}")
        elif chat_message.role == "assistant":
            prompt_segments.append(f"Assistant: {chat_message.content}")
    prompt_text: str = "\n".join(prompt_segments)
    return prompt_text


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@application.get("/health")
@application.get("/healthz")
async def health_check() -> dict:
    return {"status": "ok"}


@application.get("/v1/models")
async def list_available_models() -> dict:
    """
    Retrieve a list of all models known to the Ollama instance
    """
    async with httpx.AsyncClient(timeout=10.0) as http_client:
        ollama_response = await http_client.get(f"{OLLAMA_HOST_URL}/api/tags")

    if ollama_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to reach Ollama")

    ollama_response_json = ollama_response.json()
    model_descriptions = [
        {"id": model["name"]} for model in ollama_response_json.get("models", [])
    ]
    return {"data": model_descriptions}


@application.post("/v1/chat/completions")
async def create_chat_completion(chat_completion_request: ChatCompletionRequest) -> dict:
    """
    Non-streaming chat completion endpoint.
    Builds a prompt from the provided chat messages and forwards it to the
    Ollama instance. Returns a single response in an OpenAI-like shape.
    
    Requires a model name to be provided. Throws an error if the model
    is not available in the Ollama instance.
    """
    logger.info(f"Received chat completion request: model={chat_completion_request.model}, messages_count={len(chat_completion_request.messages)}")
    model_name: str = chat_completion_request.model
    
    # Validate that the model exists
    if not await validate_model_exists(model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not available. Use GET /v1/models to see available models.",
        )
    
    prompt_text: str = build_prompt_from_chat_messages(chat_completion_request.messages)

    ollama_request_payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": chat_completion_request.temperature,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        ollama_response = await http_client.post(
            f"{OLLAMA_HOST_URL}/api/generate",
            json=ollama_request_payload,
        )

    if ollama_response.status_code != 200:
        error_text = ollama_response.text
        logger.error(f"Ollama API error: status={ollama_response.status_code}, response={error_text}")
        logger.error(f"Request payload sent to Ollama: {ollama_request_payload}")
        raise HTTPException(
            status_code=500,
            detail=f"Ollama error: {error_text}",
        )

    ollama_response_json = ollama_response.json()
    generated_content: str = ollama_response_json.get("response", "")
    
    # Extract token usage from Ollama response
    prompt_eval_count: Optional[int] = ollama_response_json.get("prompt_eval_count")
    eval_count: Optional[int] = ollama_response_json.get("eval_count")
    total_tokens: Optional[int] = None
    if prompt_eval_count is not None and eval_count is not None:
        total_tokens = prompt_eval_count + eval_count

    # Return in OpenAI-like format so the orchestrator can swap backends later.
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": total_tokens,
        },
    }


@application.post("/v1/chat/completions/stream")
async def create_chat_completion_stream(
    chat_completion_request: ChatCompletionRequest,
):
    """
    Streaming chat completion endpoint.
    Forwards the request to Ollama with streaming enabled, and relays the
    line-delimited JSON chunks back to the client.
    
    Requires a model name to be provided. Throws an error if the model
    is not available in the Ollama instance.
    """
    logger.info(f"Received streaming chat completion request: model={chat_completion_request.model}, messages_count={len(chat_completion_request.messages)}")
    model_name: str = chat_completion_request.model
    
    # Validate that the model exists
    logger.info(f"Validating model existence: {model_name}")
    if not await validate_model_exists(model_name):
        logger.error(f"Model validation failed: '{model_name}' is not available")
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not available. Use GET /v1/models to see available models.",
        )
    logger.info(f"Model validation passed: {model_name}")
    
    prompt_text: str = build_prompt_from_chat_messages(chat_completion_request.messages)

    ollama_request_payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": True,
        "options": {
            "temperature": chat_completion_request.temperature,
        },
    }

    async def stream_from_ollama() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=None) as http_client:
            async with http_client.stream(
                "POST",
                f"{OLLAMA_HOST_URL}/api/generate",
                json=ollama_request_payload,
            ) as streaming_response:

                if streaming_response.status_code != 200:
                    error_text = await streaming_response.aread()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Ollama error: {error_text.decode()}",
                    )

                async for response_line in streaming_response.aiter_lines():
                    if not response_line:
                        continue
                    # Forward each line as NDJSON
                    # The last chunk with done=true will contain token usage info
                    yield (response_line + "\n").encode("utf-8")

    return StreamingResponse(stream_from_ollama(), media_type="application/x-ndjson")


class EmbeddingRequest(BaseModel):
    model: str
    prompt: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]


@application.post("/v1/embeddings")
async def create_embedding(embedding_request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embeddings for the provided text using the specified model.
    
    This endpoint wraps Ollama's /api/embeddings endpoint and provides
    a consistent API for embedding generation.
    """
    logger.info(f"Received embedding request: model={embedding_request.model}")
    
    ollama_request_payload = {
        "model": embedding_request.model,
        "prompt": embedding_request.prompt,
    }
    
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        ollama_response = await http_client.post(
            f"{OLLAMA_HOST_URL}/api/embeddings",
            json=ollama_request_payload,
        )
    
    if ollama_response.status_code != 200:
        error_text = ollama_response.text
        logger.error(f"Ollama API error: status={ollama_response.status_code}, response={error_text}")
        raise HTTPException(
            status_code=500,
            detail=f"Ollama error: {error_text}",
        )
    
    ollama_response_json = ollama_response.json()
    embedding: list[float] = ollama_response_json.get("embedding", [])
    
    if not embedding:
        logger.error("No embedding returned from Ollama")
        raise HTTPException(
            status_code=500,
            detail="No embedding returned from Ollama",
        )
    
    logger.info(f"Generated embedding with {len(embedding)} dimensions")
    return EmbeddingResponse(embedding=embedding)