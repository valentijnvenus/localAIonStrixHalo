# Chapter 9: Integration and Deployment

In this chapter, you will learn how to integrate ComfyUI into your production system and deploy it to production. We will explain in detail practical integration techniques using MS-S1 Max, such as Dockerization, Web UI construction, collaboration with other tools, and operation in a production environment.

---

## 9.1 Docker Integration

### 9.1.1 Building ComfyUI Docker images

Create MS-S1 Max optimized Docker image:

**Dockerfile:**

```dockerfile
# Dockerfile.mss1max
FROM ubuntu:24.04

# ROCm 6.4.2 installation
RUN apt-get update && apt-get install -y \
    wget gnupg2 software-properties-common && \
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4.2 noble main" | \
    tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update && apt-get install -y \
    rocm-hip-sdk rocm-libs python3 python3-pip git && \
    apt-get clean

# PyTorch ROCm 6.4
RUN pip3 install --no-cache-dir \
    torch==2.6.0+rocm6.4 torchvision==0.21.0+rocm6.4 \
    --index-url https://download.pytorch.org/whl/rocm6.4

# ComfyUI installation
WORKDIR /app
RUN git clone https://github.com/comfyanonymous/ComfyUI && \
    cd ComfyUI && \
    pip3 install --no-cache-dir -r requirements.txt

# MS-S1 Max environment variables
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTORCH_ROCM_ARCH=gfx1100 \
    GPU_MAX_ALLOC_PERCENT=95 \
    PYTORCH_TUNABLEOP_ENABLED=1 \
    MIGRAPHX_MLIR_USE_SPECIFIC_OPS="attention" \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16

WORKDIR /app/ComfyUI
EXPOSE 8188

CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--highvram", "--use-pytorch-cross-attention", "--disable-xformers"]
```

**Build and run:**

```bash
# image build
docker build -t comfyui-mss1max:latest -f Dockerfile.mss1max .

# Run (GPU enabled)
docker run -d \
    --name comfyui \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -p 8188:8188 \
    -v $(pwd)/models:/app/ComfyUI/models \
    -v $(pwd)/output:/app/ComfyUI/output \
    comfyui-mss1max:latest

# Check log
docker logs -f comfyui

# access
# http://localhost:8188
```

### 9.1.2 Integrating multiple services with Docker Compose

ComfyUI + Ollama + database integrated environment:

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  comfyui:
    image: comfyui-mss1max:latest
    container_name: comfyui
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    ports:
      - "8188:8188"
    volumes:
      - ./models:/app/ComfyUI/models
      - ./output:/app/ComfyUI/output
      - ./workflows:/app/ComfyUI/workflows
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
      - PYTORCH_ROCM_ARCH=gfx1100
    restart: unless-stopped
    networks:
      - ai-network

  ollama:
    image: ollama/ollama:rocm
    container_name: ollama
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    ports:
      - "11434:11434"
    volumes:
      - ./ollama_data:/root/.ollama
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
    restart: unless-stopped
    networks:
      - ai-network

  postgres:
    image: postgres:16
    container_name: postgres_db
    environment:
      - POSTGRES_USER=comfyui
      - POSTGRES_PASSWORD=comfyuipass
      - POSTGRES_DB=comfyui_db
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - ai-network

  redis:
    image: redis:7-alpine
    container_name: redis_cache
    ports:
      - "6379:6379"
    volumes:
      - ./redis_data:/data
    restart: unless-stopped
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge

volumes:
  models:
  output:
  workflows:
  ollama_data:
  postgres_data:
  redis_data:
```

**boot:**

```bash
# start all services
docker-compose up -d

# Service confirmation
docker-compose ps

# Check log
docker-compose logs -f comfyui

# Stop
docker-compose down
```

---

## 9.2 Building the Web UI

### 9.2.1 FastAPI backend

FastAPI server wrapping ComfyUI API:

**backend/main.py:**

```python
#!/usr/bin/env python3
# backend/main.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import uuid
import asyncio
from typing import Optional, Dict, List

app = FastAPI(title="ComfyUI Web API")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMFYUI_URL = "http://localhost:8188"

# data model
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry"
    steps: int = 25
    cfg: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str

# In-memory job store (Redis recommended for production environments)
jobs: Dict[str, dict] = {}

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest, background_tasks: BackgroundTasks):
"""Image generation request"""

    job_id = str(uuid.uuid4())

# Load workflow template
    with open("workflow_template.json", "r") as f:
        workflow = json.load(f)

# Parameter settings
    workflow["6"]["inputs"]["text"] = request.prompt
    workflow["7"]["inputs"]["text"] = request.negative_prompt
    workflow["3"]["inputs"]["steps"] = request.steps
    workflow["3"]["inputs"]["cfg"] = request.cfg
    workflow["5"]["inputs"]["width"] = request.width
    workflow["5"]["inputs"]["height"] = request.height

    if request.seed:
        workflow["3"]["inputs"]["seed"] = request.seed

# Job registration
    jobs[job_id] = {
        "status": "queued",
        "workflow": workflow,
        "result": None
    }

# background execution
    background_tasks.add_task(process_generation, job_id, workflow)

    return GenerationResponse(
        job_id=job_id,
        status="queued",
        message="Generation queued successfully"
    )

async def process_generation(job_id: str, workflow: dict):
"""Background generation process"""

    jobs[job_id]["status"] = "processing"

    try:
# Send to ComfyUI
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow}
        )
        prompt_id = response.json()["prompt_id"]

# Wait for completion
        while True:
            history = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
            if prompt_id in history:
# Get result
                outputs = history[prompt_id]["outputs"]
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        jobs[job_id]["status"] = "completed"
                        jobs[job_id]["result"] = node_output["images"][0]
                        return

            await asyncio.sleep(1)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
"""Check job status"""

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error")
    }

@app.get("/queue")
async def get_queue():
"""Check queue status"""

    queued = [jid for jid, job in jobs.items() if job["status"] == "queued"]
    processing = [jid for jid, job in jobs.items() if job["status"] == "processing"]

    return {
        "queued": len(queued),
        "processing": len(processing),
        "total_jobs": len(jobs)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**boot:**

```bash
# Install dependencies
pip install fastapi uvicorn requests

# start server
python backend/main.py

# test
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset", "steps": 20}'
```

---

## 9.3 Cooperation with other tools

### 9.3.1 Use with Stable Diffusion WebUI

Simultaneous operation of ComfyUI and SD WebUI:

**Reverse proxy settings (Nginx):**

```nginx
# /etc/nginx/sites-available/ai-services

upstream comfyui {
    server localhost:8188;
}

upstream sd_webui {
    server localhost:7860;
}

server {
    listen 80;
    server_name ai.local;

    location /comfyui/ {
        proxy_pass http://comfyui/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

# WebSocket compatible
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /sd-webui/ {
        proxy_pass http://sd_webui/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Startup script:**

```bash
#!/bin/bash
# start_all_services.sh

# Start ComfyUI (using GPU 0)
export HIP_VISIBLE_DEVICES=0
cd ~/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --highvram &
COMFYUI_PID=$!

# Start SD WebUI (GPU 0 shared)
cd ~/stable-diffusion-webui
./webui.sh --listen --port 7860 --api &
SDWEBUI_PID=$!

echo "Services started:"
echo "  ComfyUI: http://localhost:8188 (PID: $COMFYUI_PID)"
echo "  SD WebUI: http://localhost:7860 (PID: $SDWEBUI_PID)"

# Stop all services with Ctrl+C
trap "kill $COMFYUI_PID $SDWEBUI_PID" EXIT
wait
```

### 9.3.2 Integration with Ollama (prompt generation)

**Prompt generation service:**

```python
#!/usr/bin/env python3
# prompt_service.py

from fastapi import FastAPI
import requests
import json

app = FastAPI()

OLLAMA_URL = "http://localhost:11434"
COMFYUI_URL = "http://localhost:8188"

@app.post("/generate-with-llm")
async def generate_with_llm(simple_prompt: str):
"""Prompt extension with LLM → ComfyUI generation"""

# Step 1: Expand prompt with Ollama
    llm_response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": f"Expand this Stable Diffusion prompt: {simple_prompt}",
            "stream": False
        }
    )
    enhanced_prompt = llm_response.json()["response"]

# Step 2: Generate image with ComfyUI
    with open("workflow.json") as f:
        workflow = json.load(f)

    workflow["6"]["inputs"]["text"] = enhanced_prompt

    comfyui_response = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow}
    )

    return {
        "original_prompt": simple_prompt,
        "enhanced_prompt": enhanced_prompt,
        "comfyui_job_id": comfyui_response.json()["prompt_id"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

---

## 9.4 Deploying to production environment

### 9.4.1 System service

**systemd service file:**

```ini
# /etc/systemd/system/comfyui.service

[Unit]
Description=ComfyUI Service for MS-S1 Max
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/ComfyUI
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="PYTORCH_ROCM_ARCH=gfx1100"
Environment="GPU_MAX_ALLOC_PERCENT=95"
Environment="PYTORCH_TUNABLEOP_ENABLED=1"
ExecStart=/usr/bin/python3 /home/user/ComfyUI/main.py --listen 0.0.0.0 --port 8188 --highvram --use-pytorch-cross-attention --disable-xformers
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Activation and launch:**

```bash
# Service registration
sudo systemctl daemon-reload
sudo systemctl enable comfyui

# boot
sudo systemctl start comfyui

# Check status
sudo systemctl status comfyui

# Check log
sudo journalctl -u comfyui -f

# restart
sudo systemctl restart comfyui

# Stop
sudo systemctl stop comfyui
```

### 9.4.2 Load balancing and queuing

**Distributed task queue with Celery:**

```python
# tasks.py

from celery import Celery
import requests
import json
import time

app = Celery('comfyui_tasks', broker='redis://localhost:6379/0')

@app.task(bind=True)
def generate_image_task(self, workflow_data):
"""Celery task: ComfyUI image generation"""

    COMFYUI_URL = "http://localhost:8188"

    try:
# status update
        self.update_state(state='PROCESSING', meta={'status': 'Queuing to ComfyUI'})

# Send to ComfyUI
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow_data}
        )
        prompt_id = response.json()["prompt_id"]

# Wait for completion
        while True:
            history = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
            if prompt_id in history:
                outputs = history[prompt_id]["outputs"]
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        return {
                            'status': 'completed',
                            'images': node_output["images"]
                        }
            time.sleep(2)

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

# Start Celery Worker
# celery -A tasks worker --loglevel=info
```

---

## 9.5 Monitoring and Logging

### 9.5.1 Prometheus Metrics

**Metrics exporter:**

```python
# metrics_exporter.py

from prometheus_client import start_http_server, Gauge, Counter, Histogram
import requests
import json
import subprocess
import time

# Metric definition
gpu_utilization = Gauge('comfyui_gpu_utilization', 'GPU Utilization %')
vram_used = Gauge('comfyui_vram_used_gb', 'VRAM Used GB')
queue_length = Gauge('comfyui_queue_length', 'Queue Length')
generation_counter = Counter('comfyui_generations_total', 'Total Generations')
generation_duration = Histogram('comfyui_generation_duration_seconds', 'Generation Duration')

COMFYUI_URL = "http://localhost:8188"

def collect_metrics():
"""Metrics collection"""

    while True:
        try:
# Get GPU information (rocm-smi)
            result = subprocess.run(
                ['rocm-smi', '--showuse', '--json'],
                capture_output=True,
                text=True
            )
            gpu_data = json.loads(result.stdout)

# update metrics
            gpu_utilization.set(gpu_data['card0']['GPU_use'])
            vram_used.set(gpu_data['card0']['VRAM_used'] / 1024)  # MB → GB

# get queue length
            queue_response = requests.get(f"{COMFYUI_URL}/queue")
            queue_data = queue_response.json()
            queue_length.set(queue_data['queue_running'])

        except Exception as e:
            print(f"Metrics collection error: {e}")

time.sleep(15) # every 15 seconds

if __name__ == "__main__":
# Start Prometheus exporter (port 9090)
    start_http_server(9090)
    print("Prometheus metrics exporter started on :9090")

    collect_metrics()
```

**Prometheus configuration (prometheus.yml):**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'comfyui'
    static_configs:
      - targets: ['localhost:9090']
```

---

## 9.6 Security and Access Control

### 9.6.1 Adding an authentication layer

**FastAPI JWT Authentication:**

```python
# auth.py

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

def create_access_token(data: dict):
"""JWT token generation"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
"""Token validation"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

# Usage example
@app.post("/generate")
async def generate_image(
    request: GenerationRequest,
    username: str = Depends(verify_token)
):
# Accessible only to authenticated users
    ...
```

### 9.6.2 Rate Limiting

**Redis + FastAPI rate limit:**

```python
# rate_limiter.py

from fastapi import HTTPException
from redis import Redis

redis_client = Redis(host='localhost', port=6379, db=0)

async def rate_limit(username: str, max_requests: int = 10, window: int = 60):
"""Rate limit check"""

    key = f"rate_limit:{username}"
    current = redis_client.get(key)

    if current is None:
        redis_client.setex(key, window, 1)
        return True

    if int(current) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_requests} requests per {window} seconds."
        )

    redis_client.incr(key)
    return True

# Usage example
@app.post("/generate")
async def generate_image(
    request: GenerationRequest,
    username: str = Depends(verify_token)
):
    await rate_limit(username, max_requests=10, window=60)
# Generation process...
```

---

## 9.7 Summary of this chapter

In this chapter, you learned about ComfyUI integration and deployment.

### Review of learning content

**9.1-9.2: Containerization and Web UI**
- ✅ MS-S1 Max optimized Docker image construction
- ✅ Integration of multiple services with Docker Compose
- ✅ FastAPI backend API construction
- ✅ Asynchronous task processing

**9.3-9.4: Integration and Production Deployment**
- ✅ Collaboration with other tools (SD WebUI, Ollama)
- ✅ Systemd service
- ✅ Distributed task queue with Celery
- ✅ Load balancing

**9.5-9.7: Monitoring/Security/Operations**
- ✅ Prometheus metrics collection
- ✅ Grafana dashboard construction
- ✅ JWT authentication and rate limiting
- ✅ Production environment operation best practices

### MS-S1 Max production environment recommended configuration

```yaml
Hardware:
CPU: AMD Ryzen AI Max+ 395 (16 cores/32 threads)
  RAM: 128GB LPDDR5X-8000
  GPU: Radeon 8060S (16GB VRAM)
  Storage: NVMe SSD 1TB+

Software stack:
  OS: Ubuntu 24.04 LTS
  ROCm: 6.4.2
  PyTorch: 2.6.0+rocm6.4
  ComfyUI: Latest
  Docker: 24.0+
Nginx: Reverse proxy
Redis: cache queue
PostgreSQL: Metadata DB
Prometheus + Grafana: Monitoring

performance:
Number of simultaneous generation: 1 (single GPU)
Queue processing: Celery distributed tasks
Average generation time: 10.2 seconds (1024x1024 SDXL)
Throughput: about 350 images/hour
```

### Part 4 complete

In Part 4, you learned how to fully utilize ComfyUI and Stable Diffusion XL on MS-S1 Max:

- **Chapter 01-03**: ComfyUI basics, installation, SDXL
- **Chapter 04-06**: Workflow, ControlNet, LoRA
- **Chapter 07**: Performance optimization (58% speedup achieved)
- **Chapter 08**: Advanced techniques (animations, APIs, custom nodes)
- **Chapter 09**: Integration/deployment (Docker, Web UI, production operation)

We have mastered the techniques to maximize the strengths of MS-S1 Max's integrated APU (128GB RAM + 16GB VRAM) and build a professional-level image generation system.

---

**Reference materials:**

- Docker: https://docs.docker.com/
- FastAPI: https://fastapi.tiangolo.com/
- Prometheus: https://prometheus.io/
- Grafana: https://grafana.com/
- Celery: https://docs.celeryproject.org/

---
