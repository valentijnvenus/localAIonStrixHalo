# Chapter 09: Production operations and best practices

## 9.1 Security Best Practices

### 9.1.1 Strengthening authentication/authorization

```python
# advanced_auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import secrets

# Security settings
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    scopes: List[str] = []

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

def verify_password(plain_password: str, hashed_password: str) -> bool:
"""Verify password"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
"""Hash password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
"""Generate access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
"""Get current user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

# get user from database
    user = get_user_from_db(username)

    if user is None:
        raise credentials_exception

    return user

def check_permissions(required_scopes: List[str]):
"""Permission check decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: User = Depends(get_current_user), **kwargs):
            for scope in required_scopes:
                if scope not in current_user.scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission denied. Required scope: {scope}"
                    )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

# Usage example
@app.post("/admin/models")
@check_permissions(["admin:models"])
async def manage_models(current_user: User = Depends(get_current_user)):
"""Model management (administrators only)"""
    return {"message": "Model management", "user": current_user.username}
```

### 9.1.2 Input validation and sanitization

```python
# input_validation.py
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional
import re
import bleach

class SafeChatRequest(BaseModel):
"""Secure Chat Request"""

    model: str = Field(
        ...,
regex=r'^[a-zA-Z0-9\-\.:]+$', # Only allowed characters
        max_length=50
    )

    messages: List[dict] = Field(
        ...,
        min_items=1,
max_items=100 # limit maximum number of messages
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0
    )

    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768
    )

    @validator('messages')
    def validate_messages(cls, messages):
"""Verify message"""
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Message must have 'role' and 'content'")

            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role: {msg['role']}")

# Content length limit
            if len(msg['content']) > 10000:
                raise ValueError("Message content too long (max 10000 characters)")

# remove HTML tags
            msg['content'] = bleach.clean(msg['content'])

        return messages

    @root_validator
    def validate_total_tokens(cls, values):
"""Estimate and verify the total number of tokens"""
        messages = values.get('messages', [])
        estimated_tokens = sum(len(msg['content'].split()) * 1.3 for msg in messages)

        if estimated_tokens > 50000:
            raise ValueError("Total estimated tokens exceed limit")

        return values

class SafeFileUpload(BaseModel):
"""Secure File Upload"""

    filename: str = Field(..., max_length=255)
    content_type: str

    @validator('filename')
    def validate_filename(cls, filename):
"""Verify file name"""
# Prevent path traversal attacks
        if '..' in filename or '/' in filename or '\\' in filename:
            raise ValueError("Invalid filename")

# Allowed extensions
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf', '.txt']
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"File type not allowed. Allowed: {allowed_extensions}")

        return filename

    @validator('content_type')
    def validate_content_type(cls, content_type):
"""Validate content type"""
        allowed_types = [
            'image/jpeg',
            'image/png',
            'application/pdf',
            'text/plain'
        ]

        if content_type not in allowed_types:
            raise ValueError(f"Content type not allowed: {content_type}")

        return content_type
```

### 9.1.3 Security Header

```python
# security_middleware.py
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
"""Middleware that adds security headers"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

# add security header
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'

# Hide API version information
        response.headers.pop('Server', None)

        return response

app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)
```

## 9.2 Performance optimization

### 9.2.1 Caching strategy

```python
# advanced_caching.py
from functools import wraps
import hashlib
import json
import redis
from typing import Callable, Optional
import pickle

class CacheManager:
"""Advanced Cache Manager"""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)

    def cache_key(self, prefix: str, *args, **kwargs) -> str:
"""Generate cache key"""
# serialize arguments
        key_data = json.dumps({
            'args': args,
            'kwargs': sorted(kwargs.items())
        }, sort_keys=True)

# hashing
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()

        return f"{prefix}:{key_hash}"

    def cached(
        self,
        prefix: str,
        ttl: int = 3600,
        serialize: str = 'json'
    ):
"""Cache Decorator"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
# generate cache key
                cache_key = self.cache_key(prefix, *args, **kwargs)

# check cache
                cached_value = self.redis_client.get(cache_key)

                if cached_value is not None:
# cache hit
                    if serialize == 'json':
                        return json.loads(cached_value)
                    elif serialize == 'pickle':
                        return pickle.loads(cached_value)

# execute the function
                result = func(*args, **kwargs)

# save to cache
                if serialize == 'json':
                    self.redis_client.setex(
                        cache_key,
                        ttl,
                        json.dumps(result)
                    )
                elif serialize == 'pickle':
                    self.redis_client.setex(
                        cache_key,
                        ttl,
                        pickle.dumps(result)
                    )

                return result

            return wrapper
        return decorator

    def invalidate(self, prefix: str):
"""Invalidate cache matching prefix"""
        pattern = f"{prefix}:*"
        keys = self.redis_client.keys(pattern)

        if keys:
            self.redis_client.delete(*keys)

# Usage example
cache_manager = CacheManager(redis_url=settings.redis_url)

@cache_manager.cached(prefix="embeddings", ttl=86400) # 24 hours
def get_embedding(text: str, model: str) -> List[float]:
"""Get embedding vector (cached)"""
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']

@cache_manager.cached(prefix="model_list", ttl=300) # 5 minutes
def get_models() -> List[dict]:
"""Get model list (with cache)"""
    return ollama.list()['models']
```

### 9.2.2 Connection pooling

```python
# connection_pooling.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import redis
from typing import Generator

# database connection pool
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
pool_size=20, # Number of constant connections
max_overflow=40, # Number of additional connections that can be created
pool_timeout=30, # Connection wait timeout
pool_recycle=3600, # Connection reuse time
pool_pre_ping=True, # Ping and check before connecting
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db() -> Generator[Session, None, None]:
"""Get database session"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Redis connection pool
redis_pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=0,
    max_connections=50,
    decode_responses=True
)

def get_redis() -> redis.Redis:
"""Get Redis Connection"""
    return redis.Redis(connection_pool=redis_pool)
```

### 9.2.3 Asynchronous processing and batching

```python
# async_processing.py
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import ollama

class AsyncBatchProcessor:
"""Asynchronous Batch Processor"""

    def __init__(self, max_workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: int = 10
    ) -> List[Any]:
"""Batch processing"""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]

# Process batches in parallel
            batch_results = await asyncio.gather(*[
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    processor,
                    item
                )
                for item in batch
            ])

            results.extend(batch_results)

        return results

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: str = "mxbai-embed-large"
    ) -> List[List[float]]:
"""Batch generation of embedding vectors"""

        def get_embedding(text: str):
            return ollama.embeddings(model=model, prompt=text)['embedding']

        return await self.process_batch(texts, get_embedding, batch_size=32)

# Usage example
processor = AsyncBatchProcessor(max_workers=8)

@app.post("/batch/embeddings")
async def batch_embeddings(texts: List[str]):
"""Batch embed generation endpoint"""
    embeddings = await processor.generate_embeddings_batch(texts)
    return {"embeddings": embeddings}
```

## 9.3 Scalability

### 9.3.1 Horizontal Scaling

```yaml
# kubernetes/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: local-ai-api
  labels:
    app: local-ai-api
spec:
replicas: 3 # 3 replicas
  selector:
    matchLabels:
      app: local-ai-api
  template:
    metadata:
      labels:
        app: local-ai-api
    spec:
      containers:
      - name: api
        image: yourusername/local-ai-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
          limits:
            memory: "32Gi"
            cpu: "8"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: local-ai-api-service
spec:
  selector:
    app: local-ai-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: local-ai-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: local-ai-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 9.3.2 Load Balancing

```nginx
# nginx-lb.conf
upstream api_servers {
least_conn; # Route with least number of connections

    server api-1.local:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api-2.local:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api-3.local:8000 weight=1 max_fails=3 fail_timeout=30s;

keepalive 32; # keepalive connection
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://api_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

# timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;

# Buffering settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;

# error handling
        proxy_next_upstream error timeout http_500 http_502 http_503;
    }

# Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

## 9.4 Cost optimization

### 9.4.1 Optimizing resource usage

```python
# resource_optimizer.py
from typing import Dict, List
import psutil
import time

class ResourceOptimizer:
"""Resource optimization"""

    @staticmethod
    def optimize_model_selection(
        task_complexity: str,
        response_time_requirement: float
    ) -> str:
"""Choose the best model for the task"""

# Select a model based on task complexity and response time requirements
        model_specs = {
            "qwen2.5:3b": {
                "speed": 45,  # tokens/s
                "memory": 2.5,  # GB
                "quality": "good"
            },
            "qwen2.5:7b": {
                "speed": 32,
                "memory": 5.8,
                "quality": "very_good"
            },
            "qwen2.5:14b": {
                "speed": 18,
                "memory": 11.2,
                "quality": "excellent"
            }
        }

        if task_complexity == "simple" and response_time_requirement < 2:
            return "qwen2.5:3b"
        elif task_complexity == "moderate":
            return "qwen2.5:7b"
        else:
            return "qwen2.5:14b"

    @staticmethod
    def should_use_cache(query: str, cache_hit_rate: float) -> bool:
"""Determine whether to use cache"""
# Use cache if similar queries have a high cache hit rate
        return cache_hit_rate > 0.3

    @staticmethod
    def optimize_batch_size(
        available_memory: float,
        item_size: float
    ) -> int:
"""Calculate the optimal batch size"""
# use 80% of available memory
        usable_memory = available_memory * 0.8

# Calculate optimal batch size from item size
        batch_size = int(usable_memory / item_size)

# min 1, max 64
        return max(1, min(batch_size, 64))

# Cost tracking
class CostTracker:
"""Cost Tracking"""

    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_inference_time": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def track_request(
        self,
        tokens: int,
        inference_time: float,
        cache_hit: bool
    ):
"""Track your request"""
        self.metrics["total_requests"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_inference_time"] += inference_time

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_cost_report(self) -> Dict:
"""Generate cost report"""
        cache_hit_rate = (
            self.metrics["cache_hits"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )

        avg_tokens_per_request = (
            self.metrics["total_tokens"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )

        avg_inference_time = (
            self.metrics["total_inference_time"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )

# Cost considering MS-S1 Max power consumption (approximate)
# Power consumption: approximately 120W, electricity bill: 0.03 USD/kWh
        power_consumption_kwh = (self.metrics["total_inference_time"] / 3600) * 0.12
        electricity_cost = power_consumption_kwh * 0.03

        return {
            "total_requests": self.metrics["total_requests"],
            "total_tokens": self.metrics["total_tokens"],
            "cache_hit_rate": cache_hit_rate,
            "avg_tokens_per_request": avg_tokens_per_request,
            "avg_inference_time": avg_inference_time,
            "estimated_electricity_cost_usd": electricity_cost,
            "cost_savings_from_cache": self.metrics["cache_hits"] * avg_inference_time * 0.12 / 3600 * 0.03
        }
```

## 9.5 Documentation

### 9.5.1 API documentation automatic generation

```python
# api_documentation.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="Local AI API",
    description="""
# Local AI API

Local AI API running on MS-S1 Max (AMD Ryzen AI Max+ 395)

## function

* Chat completion
* Embedding vector generation
* Image analysis
* Multimodal RAG

## How to use

1. Get API key
2. Include `Authorization: Bearer YOUR_API_KEY` in the request header
3. Send request to endpoint

## Rate limiting

* Chat endpoint: 60 requests/min
* Embedded endpoint: 120 requests/min

## support

If you encounter any issues, please create a GitHub issue.
    """,
    version="1.0.0",
    terms_of_service="https://example.com/terms/",
    contact={
        "name": "API Support",
        "url": "https://example.com/support/",
        "email": "[email protected]"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Schema with example requests
class ChatRequestExample(BaseModel):
"""Chat request example"""

    model: str = Field(
        default="qwen2.5:14b",
        example="qwen2.5:14b",
description="Model name to use"
    )

    messages: List[dict] = Field(
        example=[
            {
                "role": "system",
"content": "You are a friendly AI assistant."
            },
            {
                "role": "user",
"content": "Please tell me about list comprehensions in Python."
            }
        ],
description="Conversation history"
    )

    temperature: float = Field(
        default=0.7,
        example=0.7,
description="Generation randomness (0.0-2.0)",
        ge=0.0,
        le=2.0
    )

    class Config:
        schema_extra = {
            "example": {
                "model": "qwen2.5:14b",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?"
                    }
                ],
                "temperature": 0.7
            }
        }
```

### 9.5.2 README and Documentation

```markdown
# Local AI Service

Local AI service running on MS-S1 Max (AMD Ryzen AI Max+ 395, 128GB RAM)

## Features

- 🚀 Fast inference (Qwen2.5 14B: 18 tokens/s)
- 💾 Efficient model operation using large memory (128GB)
- 🔒 Works completely locally (no worries about data leakage)
- 🎨 Multimodal support (text, images, audio)
- 📊 RAG System Integration

## System requirements

- OS: Ubuntu 24.04
- CPU: AMD Ryzen AI Max+ 395 (16 cores)
- GPU: Radeon 8060S (RDNA 3.5、16GB VRAM)
- Memory: 128GB LPDDR5X-8000
- Storage: 500GB or more recommended
- ROCm: 6.4.2

## install

### 1. Installing ROCm

```bash
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.4.2 jammy main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev rocm-libs
```

### 2. Installing Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Application setup

```bash
git clone https://github.com/yourusername/local-ai-service
cd local-ai-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Setting environment variables

```bash
cp .env.example .env
# Edit .env file
```

### 5. Database migration

```bash
alembic upgrade head
```

### 6. Starting the service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Starting with Docker

```bash
docker-compose up -d
```

## How to use

### Chat completion

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "model": "qwen2.5:14b",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

print(response.json())
```

### Embedding vector generation

```python
response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "model": "mxbai-embed-large",
        "input": "Sample text for embedding"
    }
)

print(response.json())
```

## performance

| Model | Speed ​​| Memory Usage |
|--------|------|--------------|
| qwen2.5:3b | 45 tokens/s | 2.5GB |
| qwen2.5:7b | 32 tokens/s | 5.8GB |
| qwen2.5:14b | 18 tokens/s | 11.2GB |
| llava:13b | 15 tokens/s | 9.0GB |

## License

MIT License

## support

If you have any issues or questions, please create a [GitHub Issue](https://github.com/yourusername/local-ai-service/issues).
```

## 9.6 Team collaboration

### 9.6.1 Code Review Guidelines

```markdown
# Code review guidelines

## Purpose of review

1. Early detection of bugs
2. Improving code quality
3. Knowledge sharing
4. Dissemination of best practices

## Review perspective

### Security

- [ ] Is input validation performed properly?
- [ ] Are authentication and authorization implemented correctly?
- [ ] Is sensitive information hard-coded?
- [ ] Are SQL injection measures taken?

### performance

- [ ] Are there any N+1 query problems?
- [ ] Are indexes used properly?
- [ ] Is cache being used effectively?
- [ ] Is there a possibility of memory leak?

### Code quality

- [ ] Is the naming convention followed?
- [ ] Is the function/method an appropriate size? (Recommended within 50 lines)
- [ ] Are there any duplicate codes?
- [ ] Is the error handling appropriate?

### Test

- [ ] Are there unit tests?
- [ ] Are edge cases covered?
- [ ] Is the test meaningful?

### Document

- [ ] Is the comment properly written?
- [ ] Has the API documentation been updated?
- [ ] Has the README been updated?
```

## 9.7 Summary

In this chapter, you learned about operations and best practices in a production environment.

**Key points**

1. **Security**
- JWT authentication and permission management
- Input validation and sanitization
- Security header

2. **Performance optimization**
- Advanced caching strategies
- Connection pooling
- Asynchronous batch processing

3. **Scalability**
- Horizontal scaling with Kubernetes
- Load balancing
- Auto scaling

4. **Cost optimization**
- Optimized resource usage
- Optimization of model selection
- Cost tracking

5. **Documentation**
- Comprehensive API documentation
- Clear README
- Code review guidelines

6. **Operations Best Practices**
- Continuous integration
- Automated testing
- Team collaboration

**Utilization points of MS-S1 Max**

- **Large memory (128GB)**: Simultaneous operation of multiple models, large-scale RAG system
- **High performance CPU (16 cores)**: Parallel processing, faster batch processing
- **Integrated GPU (16GB VRAM)**: Local inference, image processing
- **Power saving**: Lower cost of operation compared to cloud

Through this book, I learned the overall picture of local AI application development that takes full advantage of MS-S1 Max. Use this knowledge to build safe, high-performance AI systems.
