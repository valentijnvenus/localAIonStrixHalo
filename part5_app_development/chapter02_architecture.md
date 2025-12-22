# Chapter 2: Application architecture design

In this chapter, you will learn how to architect a scalable local AI application using MS-S1 Max. We will provide detailed explanations of implementation patterns for production environments, such as microservices, asynchronous processing, caching, and database design.

---

## 2.1 Choosing an architecture pattern

### 2.1.1 Monolith vs Microservices

**Small to Medium (Recommended: Monolith):**

```yaml
Applicable conditions:
- Team size: 1-5 people
- Number of users: ~10,000
- Service type: 1-3 types (chat, image generation, etc.)

Monolith configuration:
├── FastAPI integration server
  │   ├── /api/chat (LLM)
  │   ├── /api/image (ComfyUI)
│ └── /api/rag (search extension generation)
├── PostgreSQL (single DB)
├── Redis (cache)
└── Ollama + ComfyUI (backend)

advantage:
- Simple: easy to deploy and manage
- Low latency: no inter-process communication required
- Easy to debug: single code base
- Optimized for MS-S1 Max: Maximize use of integrated memory
```

**Large scale (microservices):**

```yaml
Applicable conditions:
- Team size: 5 or more people
- Number of users: 10,000+
- Service types: 4 or more types

Microservice configuration:
  ├── API Gateway (Nginx/Kong)
├── Auth Service
├── LLM Service (Ollama only)
├── Image Service (ComfyUI only)
├── RAG Service (Search only)
  ├── Queue Service (Celery/RabbitMQ)
└── Dedicated DB for each service

advantage:
- Scalability: independent scaling
- Independent deployment: updated per service
- Technological diversity: Optimal technology for each service

Disadvantages:
- Complexity: increased management costs
- Latency: network delay
- MS-S1 Max: Little advantage on a single machine
```

### 2.1.2 Recommended architecture (MS-S1 Max specific)

**Hybrid monolith:**

```yaml
composition:
  Frontend (Next.js/React)
    ↓ HTTP/WebSocket
  API Server (FastAPI) ← Redis Cache
    ↓
  ├── LLM Module (Ollama SDK)
  ├── Image Module (ComfyUI API)
  ├── RAG Module (ChromaDB + Embeddings)
  └── Task Queue (Background Tasks)
    ↓
PostgreSQL (metadata/history)

Features:
- Single process: memory sharing optimization
- Asynchronous processing: FastAPI async/await
- Integrated memory utilization: Zero CPU↔GPU transfers
- Simple: Complete with one MS-S1 Max
```

---

## 2.2 Asynchronous processing and task queues

### 2.2.1 FastAPI Asynchronous Endpoint

**Synchronous vs asynchronous comparison:**

```python
# ❌ Synchronous version (blocking)
@app.post("/chat")
def chat_sync(prompt: str):
# Processing that takes 10 seconds
response = ollama.generate(prompt) # block
    return {"response": response}

# Other requests will wait 10 seconds

# ✅ Asynchronous version (non-blocking)
@app.post("/chat")
async def chat_async(prompt: str):
# Processing that takes 10 seconds
    response = await asyncio.to_thread(ollama.generate, prompt)
    return {"response": response}

# Other requests can be processed in parallel
```

**Implementation example:**

```python
# app/main.py

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
import uuid
from typing import Dict, Optional
from datetime import datetime

app = FastAPI()

# Jobstore (Uses Redis in production environment)
jobs: Dict[str, dict] = {}

class ChatRequest(BaseModel):
    prompt: str
    model: str = "llama3.2:3b"

class JobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str

@app.post("/chat/async", response_model=JobResponse)
async def chat_async(request: ChatRequest, background_tasks: BackgroundTasks):
"""Asynchronous chat (immediately returns job ID)"""

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.dict()
    }

# run as background task
    background_tasks.add_task(process_chat, job_id, request)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"]
    )

async def process_chat(job_id: str, request: ChatRequest):
"""Background chat processing"""

    jobs[job_id]["status"] = "processing"

    try:
# Ollama call (run in separate thread)
        response = await asyncio.to_thread(
            ollama_generate,
            request.prompt,
            request.model
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["response"] = response
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
"""Get job status"""

    if job_id not in jobs:
        return {"error": "Job not found"}, 404

    return jobs[job_id]

def ollama_generate(prompt: str, model: str) -> str:
"""Ollama generation (synchronous function)"""
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]
```

### 2.2.2 Distributed task queue with Celery

**Offload heavy processing to Celery:**

```python
# tasks.py

from celery import Celery
import requests

# Initialize Celery app
celery_app = Celery(
    'ai_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@celery_app.task(bind=True)
def generate_image_task(self, workflow: dict, prompt: str):
"""ComfyUI image generation task"""

# status update
    self.update_state(state='PROCESSING', meta={'step': 'queuing'})

# ComfyUI API call
    response = requests.post(
        "http://localhost:8188/prompt",
        json={"prompt": workflow}
    )
    prompt_id = response.json()["prompt_id"]

# Wait for completion
    self.update_state(state='PROCESSING', meta={'step': 'generating'})

    import time
    while True:
        history = requests.get(f"http://localhost:8188/history/{prompt_id}").json()
        if prompt_id in history:
            return {
                'status': 'completed',
                'image_url': f"/outputs/{history[prompt_id]['outputs']['9']['images'][0]['filename']}"
            }
        time.sleep(2)

@celery_app.task(bind=True)
def batch_embeddings_task(self, texts: list):
"""Batch embedding generation"""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=False)

    return embeddings.tolist()

# FastAPI integration
from fastapi import FastAPI
from celery.result import AsyncResult

app = FastAPI()

@app.post("/image/generate")
async def generate_image(workflow: dict, prompt: str):
"""Image generation (asynchronous task)"""

    task = generate_image_task.delay(workflow, prompt)

    return {
        "task_id": task.id,
        "status": "queued"
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
"""Get task status"""

    task = AsyncResult(task_id, app=celery_app)

    if task.state == 'PENDING':
        return {"status": "pending"}
    elif task.state == 'PROCESSING':
        return {"status": "processing", "meta": task.info}
    elif task.state == 'SUCCESS':
        return {"status": "completed", "result": task.result}
    else:
        return {"status": "failed", "error": str(task.info)}
```

**Start Celery Worker:**

```bash
# Terminal 1: Celery Worker
celery -A tasks worker --loglevel=info --concurrency=2

# Terminal 2: FastAPI
uvicorn app.main:app --reload

# Terminal 3: Celery Flower (monitoring)
celery -A tasks flower
# Display dashboard at http://localhost:5555
```

---

## 2.3 Caching strategy

### 2.3.1 Redis cache layer

**Caching of LLM responses:**

```python
# cache.py

import redis
import hashlib
import json
from typing import Optional

class LLMCache:
"""LLM response cache"""

    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        self.redis = redis.from_url(redis_url)
self.ttl = 3600 # 1 hour

    def _make_key(self, prompt: str, model: str) -> str:
"""Cache key generation"""
        content = f"{model}:{prompt}"
        return f"llm:{hashlib.sha256(content.encode()).hexdigest()}"

    def get(self, prompt: str, model: str) -> Optional[str]:
"""Fetch from cache"""
        key = self._make_key(prompt, model)
        cached = self.redis.get(key)

        if cached:
            return cached.decode('utf-8')
        return None

    def set(self, prompt: str, model: str, response: str):
"""Save in cache"""
        key = self._make_key(prompt, model)
        self.redis.setex(key, self.ttl, response)

    def invalidate(self, prompt: str, model: str):
"""Cache invalidation"""
        key = self._make_key(prompt, model)
        self.redis.delete(key)

# Usage example
cache = LLMCache()

@app.post("/chat")
async def chat(prompt: str, model: str = "llama3.2:3b"):
# cache check
    cached_response = cache.get(prompt, model)
    if cached_response:
        return {
            "response": cached_response,
            "cached": True,
"latency_ms": 5 # Super fast on cache hits
        }

# LLM call
    import time
    start = time.time()
    response = await asyncio.to_thread(ollama_generate, prompt, model)
    latency = (time.time() - start) * 1000

# save to cache
    cache.set(prompt, model, response)

    return {
        "response": response,
        "cached": False,
        "latency_ms": latency
    }
```

**Effect measurement:**

```yaml
No cache:
Average latency: 850ms
Throughput: 70 requests/min

With cache (50% hit rate):
Average latency: 430ms (49% improvement)
Throughput: 140 requests/min (2x)

With cache (80% hit rate):
Average latency: 180ms (79% improvement)
Throughput: 330 requests/min (4.7x)
```

### 2.3.2 Caching embedding vectors

**Embedded cache on RAG systems:**

```python
# embedding_cache.py

import numpy as np
import pickle
from typing import Optional, List

class EmbeddingCache:
"""Embedded vector cache"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
self.ttl = 86400 # 24 hours

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
"""Get embedding"""
        key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
        cached = self.redis.get(key)

        if cached:
            return pickle.loads(cached)
        return None

    def set_embedding(self, text: str, embedding: np.ndarray):
"""Embedded save"""
        key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
        self.redis.setex(key, self.ttl, pickle.dumps(embedding))

    def get_or_compute(self, texts: List[str], model) -> List[np.ndarray]:
"""Get from cache or calculate"""

        embeddings = []
        to_compute = []
        to_compute_indices = []

# cache check
        for i, text in enumerate(texts):
            cached = self.get_embedding(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                to_compute.append(text)
                to_compute_indices.append(i)

# calculate uncached
        if to_compute:
            computed = model.encode(to_compute)

            for idx, emb in zip(to_compute_indices, computed):
                embeddings[idx] = emb
                self.set_embedding(texts[idx], emb)

        return embeddings

# Usage example
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_cache = EmbeddingCache(redis.from_url("redis://localhost:6379/3"))

texts = ["Hello", "World", "Hello"] # "Hello" is a duplicate
embeddings = embedding_cache.get_or_compute(texts, embedding_model)

# "Hello" is calculated only once, second time is retrieved from cache
```

---

## 2.4 Database design

### 2.4.1 PostgreSQL schema design

**Chat history management:**

```sql
-- users.sql

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    model VARCHAR(100),
    tokens INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);

-- Usage statistics
CREATE TABLE usage_stats (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    endpoint VARCHAR(100),
    model VARCHAR(100),
    tokens_input INTEGER,
    tokens_output INTEGER,
    latency_ms INTEGER,
    cached BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_usage_stats_user_date ON usage_stats(user_id, created_at DESC);
```

**SQLAlchemy ORM:**

```python
# models.py

from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    model = Column(String(100))
    tokens = Column(Integer)
    latency_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

# database connection
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://user:password@localhost/ai_app"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# create table
Base.metadata.create_all(engine)
```

### 2.4.2 ChromaDB Vector Database

**Vector store for RAG:**

```python
# vector_store.py

import chromadb
from chromadb.config import Settings
from typing import List, Dict

class VectorStore:
"""ChromaDB Vector Store"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ):
"""Add document"""
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None
    ) -> Dict:
"""Similar document search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results

    def delete(self, ids: List[str]):
"""Delete document"""
        self.collection.delete(ids=ids)

    def count(self) -> int:
"""Get number of documents"""
        return self.collection.count()

# Usage example
vector_store = VectorStore()

# add document
documents = [
"MS-S1 Max is powered by AMD Ryzen AI Max+ 395.",
"A powerful APU with 128GB of integrated memory.",
"This is the perfect hardware for local AI development."
]
metadatas = [
    {"source": "manual", "page": 1},
    {"source": "manual", "page": 2},
    {"source": "manual", "page": 3}
]
ids = ["doc1", "doc2", "doc3"]

vector_store.add_documents(documents, metadatas, ids)

# search
results = vector_store.search("How much memory does MS-S1 Max have?", n_results=2)
print(results['documents'])
# [['A powerful APU with 128GB of integrated memory. ', 'MS-S1 Max is AMD...'], ...]
```

---

## 2.5 API versioning and documentation

### 2.5.1 API versioning strategy

**URL path-based versioning:**

```python
# app/main.py

from fastapi import FastAPI, APIRouter

app = FastAPI(title="AI Application API")

# version 1
v1_router = APIRouter(prefix="/api/v1")

@v1_router.post("/chat")
async def chat_v1(prompt: str):
"""V1 Chat (Simple)"""
    return {"response": "..."}

# Version 2 (extension)
v2_router = APIRouter(prefix="/api/v2")

@v2_router.post("/chat")
async def chat_v2(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False
):
"""V2 Chat (Advanced Settings)"""
    return {"response": "...", "metadata": {...}}

app.include_router(v1_router)
app.include_router(v2_router)

# Both versions available at the same time
# /api/v1/chat
# /api/v2/chat
```

### 2.5.2 OpenAPI automatic documentation

**Type safe with Pydantic models:**

```python
# schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ModelName(str, Enum):
    LLAMA_3B = "llama3.2:3b"
    LLAMA_13B = "llama3.1:13b"
    GEMMA_9B = "gemma2:9b"

class ChatRequest(BaseModel):
prompt: str = Field(..., description="User prompt", min_length=1)
model: ModelName = Field(default=ModelName.LLAMA_3B, description="Model used")
temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="generation temperature")
max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum number of tokens")

    class Config:
        schema_extra = {
            "example": {
"prompt": "Please tell me the features of MS-S1 Max",
                "model": "llama3.2:3b",
                "temperature": 0.7,
                "max_tokens": 512
            }
        }

class ChatResponse(BaseModel):
response: str = Field(..., description="LLM response")
model: str = Field(..., description="Model used")
tokens: int = Field(..., description="number of generated tokens")
latency_ms: int = Field(..., description="Latency (ms)")
cached: bool = Field(default=False, description="cache hit")

# Used in FastAPI
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
"""Chat generation endpoint

This endpoint uses Ollama to generate LLM responses.

**Parameters:**
- prompt: input text from user
- model: LLM model to use
- temperature: randomness of generation (0=deterministic, 2=very random)
- max_tokens: Maximum number of generated tokens

**Return value:**
- response: LLM response text
- tokens: number of generated tokens
- latency_ms: processing time (ms)
    """
# implementation...
    pass

# Auto-generated documentation: http://localhost:8000/docs
```

---

## 2.6 Error handling and logging

### 2.6.1 Unified error handling

**Custom exception class:**

```python
# exceptions.py

class AIServiceException(Exception):
"""AI Service Base Exception"""
    def __init__(self, message: str, code: str, details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class ModelNotFoundError(AIServiceException):
"""Model not found"""
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            code="MODEL_NOT_FOUND",
            details={"model": model_name}
        )

class GenerationTimeoutError(AIServiceException):
"""Generation timeout"""
    def __init__(self, timeout: int):
        super().__init__(
            message=f"Generation timed out after {timeout}s",
            code="GENERATION_TIMEOUT",
            details={"timeout": timeout}
        )

class RateLimitExceededError(AIServiceException):
"""Rate limit exceeded"""
    def __init__(self, limit: int, window: int):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}s",
            code="RATE_LIMIT_EXCEEDED",
            details={"limit": limit, "window": window}
        )

# FastAPI error handler
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(AIServiceException)
async def ai_service_exception_handler(request: Request, exc: AIServiceException):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )

# Usage example
@app.post("/chat")
async def chat(model: str, prompt: str):
# Check model existence
    available_models = get_available_models()
    if model not in available_models:
        raise ModelNotFoundError(model)

# Generate...
```

### 2.6.2 Structured Logging

**Python standard logging settings:**

```python
# logging_config.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
"""JSON format log formatter"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

# Exception information
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

# custom field
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id

        return json.dumps(log_data, ensure_ascii=False)

# logger settings
def setup_logging():
"""Logging settings"""

# root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

# console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

# file handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger

# Usage example
logger = setup_logging()

@app.post("/chat")
async def chat(request: ChatRequest, user_id: int):
    logger.info(
        "Chat request received",
        extra={
            "user_id": user_id,
            "model": request.model,
            "prompt_length": len(request.prompt)
        }
    )

    try:
        response = await generate_response(request)

        logger.info(
            "Chat response generated",
            extra={
                "user_id": user_id,
                "tokens": response.tokens,
                "latency_ms": response.latency_ms
            }
        )

        return response

    except Exception as e:
        logger.error(
            "Chat generation failed",
            exc_info=True,
            extra={"user_id": user_id}
        )
        raise
```

---

## 2.7 Summary of this chapter

In this chapter, you learned how to architect a scalable local AI application.

### Review of learning content

**2.1-2.2: Architecture and asynchronous processing**
- ✅ Choosing monolith vs. microservices
- ✅ MS-S1 Max specialized hybrid monolith
- ✅ FastAPI asynchronous endpoint
- ✅ Celery distributed task queue

**2.3-2.4: Caching and database**
- ✅ Redis LLM response caching (4.7x faster)
- ✅ Embedded vector caching
- ✅ PostgreSQL schema design
- ✅ ChromaDB Vector Store

**2.5-2.7: API design and operation**
- ✅ API versioning strategy
- ✅ OpenAPI automatic documentation
- ✅ Unified error handling
- ✅ Structured logging

### Architecture overview diagram

```
┌──────────────────────────────────────────────┐
│         Frontend (Next.js/React)             │
└────────────────┬─────────────────────────────┘
                 │ HTTP/WebSocket
┌────────────────▼─────────────────────────────┐
│    API Server (FastAPI) + Redis Cache        │
├──────────────────────────────────────────────┤
│  ┌──────────────┬──────────────┬────────────┐│
│  │ LLM Module   │ Image Module │ RAG Module ││
│  │ (Ollama)     │ (ComfyUI)    │ (ChromaDB) ││
│  └──────────────┴──────────────┴────────────┘│
│  Background Tasks (Celery + Redis Queue)     │
└────────────────┬─────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────┐
│ PostgreSQL (metadata/history management) │
└──────────────────────────────────────────────┘

MS-S1 Max: 128GB integrated memory keeps all components running efficiently
```

### Next steps

In Chapter 3, we will learn the implementation of a RAG (Retrieval Augmentation Generation) system. Build ChromaDB, embedded model, and document processing pipeline to realize local document search AI.

---

**Reference materials:**

- FastAPI Async: https://fastapi.tiangolo.com/async/
- Celery: https://docs.celeryproject.org/
- ChromaDB: https://docs.trychroma.com/
- SQLAlchemy: https://www.sqlalchemy.org/

---
