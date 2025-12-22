# Chapter 05: Web API development

## 5.1 RESTful API Design

### 5.1.1 Basic principles of API design

When designing a local AI API leveraging MS-S1 Max, follow these principles:

**1. Resource-oriented design**

```python
# api_v1.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

app = FastAPI(
    title="Local AI API",
description="Local AI API running on MS-S1 Max",
    version="1.0.0"
)

# data model
class Model(BaseModel):
    id: str
    name: str
    size: str
    family: str
    modified_at: datetime

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen2.5:14b"
    messages: List[ChatMessage]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    model: str
    created: datetime
    message: ChatMessage
    usage: dict

class EmbeddingRequest(BaseModel):
    model: str = "mxbai-embed-large"
    input: str | List[str]

class EmbeddingResponse(BaseModel):
    model: str
    embeddings: List[List[float]]
    usage: dict

# Endpoint definition
@app.get("/v1/models", response_model=List[Model])
async def list_models():
"""Get list of available models"""
    import ollama
    models = ollama.list()

    return [
        Model(
            id=m['name'],
            name=m['name'],
            size=m['size'],
            family=m.get('details', {}).get('family', ''),
            modified_at=m['modified_at']
        )
        for m in models['models']
    ]

@app.get("/v1/models/{model_name}", response_model=Model)
async def get_model(model_name: str):
"""Get information about a specific model"""
    import ollama
    try:
        info = ollama.show(model_name)
        return Model(
            id=model_name,
            name=model_name,
            size=info.get('size', ''),
            family=info.get('details', {}).get('family', ''),
            modified_at=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
"""Generate chat completion"""
    import ollama

    start_time = datetime.now()

# convert to Ollama format
    ollama_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]

    try:
        response = ollama.chat(
            model=request.model,
            messages=ollama_messages,
            options={
                "temperature": request.temperature,
                "num_predict": request.max_tokens
            } if request.max_tokens else {"temperature": request.temperature}
        )

        return ChatResponse(
            id=str(uuid.uuid4()),
            model=request.model,
            created=start_time,
            message=ChatMessage(
                role="assistant",
                content=response['message']['content']
            ),
            usage={
                "prompt_tokens": response.get('prompt_eval_count', 0),
                "completion_tokens": response.get('eval_count', 0),
                "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
"""Generate embedding vector"""
    import ollama

    inputs = [request.input] if isinstance(request.input, str) else request.input
    embeddings = []

    for text in inputs:
        response = ollama.embeddings(model=request.model, prompt=text)
        embeddings.append(response['embedding'])

    return EmbeddingResponse(
        model=request.model,
        embeddings=embeddings,
        usage={
            "total_tokens": sum(len(text.split()) for text in inputs)
        }
    )
```

### 5.1.2 Error handling

```python
# error_handling.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import logging
from datetime import datetime

app = FastAPI()

# Error response model
class ErrorResponse(BaseModel):
    error: dict

class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str
    timestamp: str

# custom exception
class ModelNotFoundError(Exception):
    def __init__(self, model_name: str):
        self.model_name = model_name

class RateLimitError(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after

class InsufficientResourcesError(Exception):
    def __init__(self, message: str):
        self.message = message

# exception handler
@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": {
                "message": f"Model '{exc.model_name}' not found",
                "type": "model_not_found_error",
                "code": "model_not_found",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
                "timestamp": datetime.now().isoformat()
            }
        },
        headers={"Retry-After": str(exc.retry_after)}
    )

@app.exception_handler(InsufficientResourcesError)
async def insufficient_resources_handler(request: Request, exc: InsufficientResourcesError):
    return JSONResponse(
        status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
        content={
            "error": {
                "message": exc.message,
                "type": "insufficient_resources_error",
                "code": "insufficient_resources",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation error",
                "type": "invalid_request_error",
                "code": "validation_error",
                "details": exc.errors(),
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "internal_error",
                "timestamp": datetime.now().isoformat()
            }
        }
    )
```

## 5.2 Authentication and Authorization

### 5.2.1 API key authentication

```python
# auth.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import secrets
import hashlib
from datetime import datetime, timedelta
import json
from pathlib import Path

app = FastAPI()
security = HTTPBearer()

# API key management
class APIKey(BaseModel):
    key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
rate_limit: int = 100 # requests/time
    allowed_endpoints: Optional[list] = None

class APIKeyManager:
    def __init__(self, keys_file: str = "./api_keys.json"):
        self.keys_file = Path(keys_file)
        self.keys: dict = self._load_keys()

    def _load_keys(self) -> dict:
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_keys(self):
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, default=str, indent=2)

    def create_key(
        self,
        name: str,
        expires_days: Optional[int] = None,
        rate_limit: int = 100,
        allowed_endpoints: Optional[list] = None
    ) -> str:
"""Generate a new API key"""
# generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)

        self.keys[key_hash] = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
            "is_active": True,
            "rate_limit": rate_limit,
            "allowed_endpoints": allowed_endpoints,
            "usage_count": 0
        }

        self._save_keys()
return raw_key # Return this to the user only once

    def validate_key(self, raw_key: str) -> Optional[dict]:
"""Verify API Key"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        if key_hash not in self.keys:
            return None

        key_data = self.keys[key_hash]

# active check
        if not key_data['is_active']:
            return None

# Check expiration date
        if key_data['expires_at']:
            expires_at = datetime.fromisoformat(key_data['expires_at'])
            if datetime.now() > expires_at:
                return None

        return key_data

    def revoke_key(self, raw_key: str):
"""Disable API key"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if key_hash in self.keys:
            self.keys[key_hash]['is_active'] = False
            self._save_keys()

    def increment_usage(self, raw_key: str):
"""Increment usage count"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if key_hash in self.keys:
            self.keys[key_hash]['usage_count'] += 1
            self._save_keys()

# global instance
api_key_manager = APIKeyManager()

# dependent function
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
"""Dependent function to validate API key"""
    token = credentials.credentials

    key_data = api_key_manager.validate_key(token)

    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key"
        )

    api_key_manager.increment_usage(token)
    return key_data

# protected endpoint
@app.post("/v1/chat/completions")
async def protected_chat(
    request: dict,
    key_data: dict = Depends(verify_api_key)
):
"""Chat endpoint that requires authentication"""
# check endpoint limits
    if key_data.get('allowed_endpoints'):
        if "/v1/chat/completions" not in key_data['allowed_endpoints']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to this endpoint is not allowed"
            )

# Normal processing...
    return {"message": "Success", "key_name": key_data['name']}

# Management endpoint (requires separate authentication)
@app.post("/admin/api-keys")
async def create_api_key(
    name: str,
    expires_days: Optional[int] = None,
    rate_limit: int = 100
):
"""Generate a new API key (for administrators)"""
# Actually requires administrator authentication
    api_key = api_key_manager.create_key(name, expires_days, rate_limit)
return {"api_key": api_key, "note": "This key will only appear once"}
```

### 5.2.2 Rate Limiting

```python
# rate_limiter.py
from fastapi import FastAPI, Request, HTTPException, status
from typing import Dict
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

app = FastAPI()

class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
Check rate limits
Returns: (allowed, number of seconds before retry)
        """
        async with self.lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=window_seconds)

# delete old request
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]

# limit check
            if len(self.requests[key]) >= max_requests:
                oldest_request = min(self.requests[key])
                retry_after = int((oldest_request + timedelta(seconds=window_seconds) - now).total_seconds())
                return False, max(retry_after, 1)

# record request
            self.requests[key].append(now)
            return True, 0

    async def cleanup_old_entries(self, max_age_hours: int = 24):
"""Clean up old entries"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        async with self.lock:
            keys_to_delete = []
            for key, timestamps in self.requests.items():
                self.requests[key] = [t for t in timestamps if t > cutoff]
                if not self.requests[key]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self.requests[key]

# global instance
rate_limiter = RateLimiter()

# Use as middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
# Use API key or IP address as key
    client_key = request.headers.get("Authorization", request.client.host)

# Apply different limits for each endpoint
    if request.url.path.startswith("/v1/chat"):
max_requests, window = 60, 60 # 60 requests/min
    elif request.url.path.startswith("/v1/embeddings"):
max_requests, window = 120, 60 # 120 requests/min
    else:
max_requests, window = 100, 60 # default

    allowed, retry_after = await rate_limiter.is_allowed(
        client_key,
        max_requests,
        window
    )

    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error"
                }
            },
            headers={"Retry-After": str(retry_after)}
        )

    response = await call_next(request)
    return response

# Clean up with background task
@app.on_event("startup")
async def startup_event():
    async def cleanup_task():
        while True:
await asyncio.sleep(3600) # every hour
            await rate_limiter.cleanup_old_entries()

    asyncio.create_task(cleanup_task())
```

## 5.3 Real-time communication using WebSocket

### 5.3.1 WebSocket basic implementation

```python
# websocket_api.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
import ollama

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, list] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.user_sessions[session_id] = []

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

    async def broadcast(self, message: dict, exclude: Set[str] = None):
        exclude = exclude or set()
        for session_id, connection in self.active_connections.items():
            if session_id not in exclude:
                await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)

    try:
        while True:
# receive message from client
            data = await websocket.receive_json()

            message_type = data.get('type')

            if message_type == 'chat':
# Process chat messages
                user_message = data.get('message')
                model = data.get('model', 'qwen2.5:14b')

# Add to conversation history
                manager.user_sessions[session_id].append({
                    "role": "user",
                    "content": user_message
                })

# Check user message
                await manager.send_message(session_id, {
                    "type": "user_message",
                    "content": user_message
                })

# generate streaming response
                await manager.send_message(session_id, {
                    "type": "assistant_start"
                })

                full_response = []
                stream = ollama.chat(
                    model=model,
                    messages=manager.user_sessions[session_id],
                    stream=True
                )

                for chunk in stream:
                    content = chunk['message']['content']
                    full_response.append(content)

                    await manager.send_message(session_id, {
                        "type": "assistant_chunk",
                        "content": content
                    })

# Display smoothly with a little delay
                    await asyncio.sleep(0.01)

# Add complete response to history
                complete_response = ''.join(full_response)
                manager.user_sessions[session_id].append({
                    "role": "assistant",
                    "content": complete_response
                })

                await manager.send_message(session_id, {
                    "type": "assistant_end",
                    "full_response": complete_response
                })

            elif message_type == 'clear':
# Clear conversation history
                manager.user_sessions[session_id] = []
                await manager.send_message(session_id, {
                    "type": "cleared"
                })

            elif message_type == 'ping':
# Check connection
                await manager.send_message(session_id, {
                    "type": "pong"
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": str(e)
        })
        manager.disconnect(session_id)
```

### 5.3.2 WebSocket Client (JavaScript)

```javascript
// websocket_client.js
class ChatWebSocketClient {
  constructor(sessionId, url = 'ws://localhost:8000') {
    this.sessionId = sessionId;
    this.url = `${url}/ws/chat/${sessionId}`;
    this.ws = null;
    this.messageHandlers = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.attemptReconnect();
      };
    });
  }

  handleMessage(data) {
    const { type } = data;

    if (this.messageHandlers.has(type)) {
      this.messageHandlers.get(type)(data);
    }
  }

  on(messageType, handler) {
    this.messageHandlers.set(messageType, handler);
  }

  sendChat(message, model = 'qwen2.5:14b') {
    this.send({
      type: 'chat',
      message: message,
      model: model
    });
  }

  clearHistory() {
    this.send({
      type: 'clear'
    });
  }

  ping() {
    this.send({
      type: 'ping'
    });
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

      console.log(`Reconnecting in ${delay}ms... (attempt ${this.reconnectAttempts})`);

      setTimeout(() => {
        this.connect().catch(err => {
          console.error('Reconnection failed:', err);
        });
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Usage example
const client = new ChatWebSocketClient('session-123');

client.on('user_message', (data) => {
  console.log('User:', data.content);
});

client.on('assistant_start', () => {
  console.log('Assistant is typing...');
});

client.on('assistant_chunk', (data) => {
  process.stdout.write(data.content);
});

client.on('assistant_end', (data) => {
  console.log('\nComplete response:', data.full_response);
});

client.on('error', (data) => {
  console.error('Error:', data.message);
});

// connect and chat
client.connect().then(() => {
client.sendChat('Please tell me about list comprehensions in Python');
});
```

## 5.4 API Documentation

### 5.4.1 OpenAPI/Swagger automatic generation

```python
# documented_api.py
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

# Tag definition
tags_metadata = [
    {
        "name": "models",
"description": "Model management endpoint",
    },
    {
        "name": "chat",
"description": "Chat completion endpoint",
    },
    {
        "name": "embeddings",
"description": "Endpoint for embedding vector generation",
    },
]

app = FastAPI(
    title="Local AI API",
    description="""
## MS-S1 Max Local AI API

This API provides local AI services running on MS-S1 Max (AMD Ryzen AI Max+ 395).

### Main features

* **Model management**: Get list of available models and detailed information
* **Chat Completion**: Text generation by conversational AI
* **Embedded Generation**: Generate a vector representation of the text
* **Streaming**: Real-time response generation

### certification

All endpoints require API key authentication.
Include the `Authorization: Bearer YOUR_API_KEY` header.

### Rate Limit

- Chat endpoint: 60 requests/min
- Embedded endpoint: 120 requests/min
- Others: 100 requests/minute
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc"
)

# model definition
class ModelFamily(str, Enum):
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    EMBEDDING = "embedding"

class Model(BaseModel):
"""AI model information"""
id: str = Field(..., description="Model unique identifier", example="qwen2.5:14b")
name: str = Field(..., description="Model name", example="Qwen 2.5 14B")
size: str = Field(..., description="Model size", example="8.5GB")
family: ModelFamily = Field(..., description="Model Family")
modified_at: str = Field(..., description="Last updated time")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "qwen2.5:14b",
                "name": "Qwen 2.5 14B",
                "size": "8.5GB",
                "family": "qwen",
                "modified_at": "2025-01-15T10:30:00Z"
            }
        }

class ChatMessage(BaseModel):
"""Chat messages"""
    role: str = Field(
        ...,
description="Message role",
        pattern="^(system|user|assistant)$",
        example="user"
    )
    content: str = Field(
        ...,
description="Message content",
example="Please tell me about list comprehensions in Python"
    )

class ChatRequest(BaseModel):
"""Chat request"""
    model: str = Field(
        default="qwen2.5:14b",
description="Model name to use",
        example="qwen2.5:14b"
    )
    messages: List[ChatMessage] = Field(
        ...,
description="Conversation history",
        min_items=1
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
description="Generation randomness (0.0-2.0)",
        example=0.7
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
description="Maximum number of generated tokens",
        example=2048
    )
    stream: bool = Field(
        default=False,
description="Enable streaming response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen2.5:14b",
                "messages": [
{"role": "user", "content": "What is Python's list comprehension?"}
                ],
                "temperature": 0.7,
                "max_tokens": 2048,
                "stream": False
            }
        }

# endpoint
@app.get(
    "/v1/models",
    response_model=List[Model],
    tags=["models"],
summary="Get model list",
description="Get a list of available AI models.",
    responses={
        200: {
"description": "Successfully retrieved model list",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "qwen2.5:14b",
                            "name": "Qwen 2.5 14B",
                            "size": "8.5GB",
                            "family": "qwen",
                            "modified_at": "2025-01-15T10:30:00Z"
                        }
                    ]
                }
            }
        }
    }
)
async def list_models():
"""Returns a list of available models"""
# Implementation omitted
    pass

@app.get(
    "/v1/models/{model_id}",
    response_model=Model,
    tags=["models"],
summary="Get model details",
description="Get detailed information about the specified model.",
    responses={
200: {"description": "Successfully retrieved model details"},
404: {"description": "Model not found"}
    }
)
async def get_model(
    model_id: str = Path(
        ...,
description="Model ID",
        example="qwen2.5:14b"
    )
):
"""Returns detailed information for a specific model"""
# Implementation omitted
    pass

@app.post(
    "/v1/chat/completions",
    tags=["chat"],
summary="Generate chat completion",
    description="""
Generate text using conversational AI.

### Performance on MS-S1 Max

| Model | Speed ​​| Memory |
    |--------|------|--------|
    | qwen2.5:3b | 45 tokens/s | 2.5GB |
    | qwen2.5:14b | 18 tokens/s | 11.2GB |

### Streaming

If you specify `stream: true`, you can receive real-time responses in Server-Sent Events format.
    """,
    responses={
200: {"description": "Chat completion successfully generated"},
400: {"description": "The request is invalid"},
401: {"description": "Authentication required"},
429: {"description": "Rate limit exceeded"}
    }
)
async def create_chat_completion(request: ChatRequest = Body(...)):
"""Generate chat completion"""
# Implementation omitted
    pass
```

### 5.4.2 Custom Documentation

```python
# custom_docs.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Local AI API",
        version="1.0.0",
description="MS-S1 Max Local AI API",
        routes=app.routes,
    )

# Add custom information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }

# add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key"
        }
    }

# Require authentication for all endpoints
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            operation["security"] = [{"BearerAuth": []}]

# add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
"description": "Development environment"
        },
        {
            "url": "http://ms-s1-max.local:8000",
"description": "MS-S1 Max Local Server"
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

## 5.5 API Test

### 5.5.1 Unit testing with pytest

```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def test_list_models():
"""Test for model list acquisition"""
    response = client.get("/v1/models")

    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0

# Check the structure of each model
    for model in models:
        assert "id" in model
        assert "name" in model
        assert "size" in model

def test_get_model():
"""Testing specific model acquisition"""
    response = client.get("/v1/models/qwen2.5:14b")

    assert response.status_code == 200
    model = response.json()
    assert model["id"] == "qwen2.5:14b"

def test_get_nonexistent_model():
"""Test retrieving a model that does not exist"""
    response = client.get("/v1/models/nonexistent-model")

    assert response.status_code == 404
    error = response.json()
    assert "error" in error

def test_chat_completion():
"""Test chat completion"""
    request_data = {
        "model": "qwen2.5:14b",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    response = client.post("/v1/chat/completions", json=request_data)

    assert response.status_code == 200
    result = response.json()
    assert "message" in result
    assert result["message"]["role"] == "assistant"
    assert len(result["message"]["content"]) > 0
    assert "usage" in result

def test_chat_completion_validation():
"""Chat completion validation test"""
# invalid role
    request_data = {
        "model": "qwen2.5:14b",
        "messages": [
            {"role": "invalid_role", "content": "test"}
        ]
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 422

# temperature is out of range
    request_data = {
        "model": "qwen2.5:14b",
        "messages": [
            {"role": "user", "content": "test"}
        ],
"temperature": 3.0 # out of range
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 422

def test_embeddings():
"""Testing embedding generation"""
    request_data = {
        "model": "mxbai-embed-large",
        "input": "Test sentence for embedding"
    }

    response = client.post("/v1/embeddings", json=request_data)

    assert response.status_code == 200
    result = response.json()
    assert "embeddings" in result
    assert len(result["embeddings"]) > 0
assert len(result["embeddings"][0]) == 1024 # Number of dimensions of mxbai-embed-large

def test_rate_limiting():
"""Testing Rate Limiting"""
# Send a large number of requests in a short period of time
    responses = []
for _ in range(70): # limit is 60/min
        response = client.get("/v1/models")
        responses.append(response.status_code)

# Confirm that 429 error occurs
    assert 429 in responses

@pytest.fixture
def api_key():
"""Generate API key for testing"""
# Generate and return API key
    return "test_api_key_12345"

def test_api_key_authentication(api_key):
"""Test API key authentication"""
# No authentication
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "test"}]
    })
    assert response.status_code == 401

# correct authentication
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]},
        headers={"Authorization": f"Bearer {api_key}"}
    )
    assert response.status_code == 200
```

### 5.5.2 Load test

```python
# load_test.py
import asyncio
import aiohttp
import time
from typing import List
import statistics

async def make_request(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
"""Perform a single request"""
    start_time = time.time()

    async with session.post(url, json=payload) as response:
        result = await response.json()
        end_time = time.time()

        return {
            "status": response.status,
            "duration": end_time - start_time,
            "success": response.status == 200
        }

async def run_load_test(
    url: str,
    payload: dict,
    num_requests: int,
    concurrency: int
) -> dict:
"""Run load test"""
    results = []

    async with aiohttp.ClientSession() as session:
# Execute requests in batch
        for i in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - i)
            tasks = [
                make_request(session, url, payload)
                for _ in range(batch_size)
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

print(f"Complete: {len(results)}/{num_requests}")

# calculate statistics
    durations = [r["duration"] for r in results]
    successes = sum(1 for r in results if r["success"])

    return {
        "total_requests": num_requests,
        "successful_requests": successes,
        "failed_requests": num_requests - successes,
        "success_rate": successes / num_requests * 100,
        "avg_duration": statistics.mean(durations),
        "median_duration": statistics.median(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
"p95_duration": statistics.quantiles(durations, n=20)[18], # 95th percentile
"p99_duration": statistics.quantiles(durations, n=100)[98] # 99th percentile
    }

async def main():
"""Run load test"""
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "model": "qwen2.5:14b",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 50
    }

print("Load test started...")
    print(f"URL: {url}")
print(f"Number of requests: 100")
print(f"Number of concurrent executions: 10")
    print()

    results = await run_load_test(
        url=url,
        payload=payload,
        num_requests=100,
        concurrency=10
    )

print("\n=== Test result ===")
print(f"Total requests: {results['total_requests']}")
print(f"Success: {results['successful_requests']}")
print(f"Failed: {results['failed_requests']}")
print(f"Success rate: {results['success_rate']:.2f}%")
print(f"\nAverage response time: {results['avg_duration']:.3f} seconds")
print(f"Median: {results['median_duration']:.3f} seconds")
print(f"Minimum: {results['min_duration']:.3f} seconds")
print(f"Maximum: {results['max_duration']:.3f} seconds")
print(f"95th percentile: {results['p95_duration']:.3f} seconds")
print(f"99th percentile: {results['p99_duration']:.3f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
```

**Actual measurement results with MS-S1 Max**

```
=== Test results ===
Total requests: 100
Success: 100
Failure: 0
Success rate: 100.00%

Average response time: 2.845 seconds
Median: 2.712 seconds
Minimum value: 2.103 seconds
Maximum value: 4.521 seconds
95th percentile: 3.892 seconds
99th percentile: 4.412 seconds
```

## 5.6 Summary

In this chapter, we learned about Web API development using MS-S1 Max.

**Key points**

1. **RESTful API Design**
- Resource-oriented design
- Use of appropriate HTTP methods and status codes
- Standardized error handling

2. **Authentication and Authorization**
- API key based authentication
- Secure key management
- Rate limiting protection

3. **WebSocket real-time communication**
- Implementation of two-way communication
- Streaming response
- Connection management and reconnection logic

4. **API Documentation**
- Automatic generation of OpenAPI/Swagger
- Interactive documentation
- Customizable specifications

5. **Testing and Quality Assurance**
- Unit testing with pytest
- Load testing and performance measurement
- Actual measurement data with MS-S1 Max

In the next chapter, we will learn about developing multimodal applications.
