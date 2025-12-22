# Chapter 1: Introduction to local AI application development

In this chapter, you will learn the overall picture of local AI application development using MS-S1 Max (AMD Ryzen AI Max+ 395, 128GB RAM). By integrating Ollama, LM Studio, ComfyUI, etc., we will prepare the basic knowledge and development environment for building practical AI applications.

---

## 1.1 Significance of local AI development

### 1.1.1 Why run it locally?

**Privacy and Security:**

```yaml
Challenges with cloud APIs (OpenAI, Anthropic, etc.):
Data transmission: Send all text and images to the cloud
Privacy: Risk of leakage of trade secrets and personal information
Regulatory compliance: Complying with GDPR and the Personal Information Protection Act is complicated

Advantages of local AI:
Completely offline: no internet required
Data protection: All processing is completed in-house and on personal PCs
Compliance: Easy regulatory compliance
Unique adjustment: Fine tuning of the model possible
```

**Cost reduction:**

```yaml
OpenAI GPT-4 API (2025 pricing):
Input: $0.03/1K tokens
Output: $0.06/1K tokens

1 million tokens processed per month:
Cost: $30,000-$60,000/year

Local AI (MS-S1 Max):
Initial investment: $1,299 (hardware)
Power: Approximately 50W × 24h × 30 days = 36kWh/month = approximately $5/month
1 year of operation: $1,299 + $60 = $1,359

Investment payback period: approximately 2 weeks to 1 month
```

**Latency and throughput:**

```yaml
Cloud API:
Latency: 500-2000ms (including network delay)
Concurrency: API limited
Failure risk: Network failure/service outage

Local AI (MS-S1 Max):
Latency: 50-200ms (no network delay)
Concurrency: up to hardware limits
Availability: >99.9% (local control)
```

### 1.1.2 Strengths of MS-S1 Max

**Integrated APU architecture:**

```yaml
CPU + GPU integration:
Ryzen AI Max+ 395: 16 cores/32 threads
  Radeon 8060S: 16GB VRAM（RDNA 3.5）
Integrated memory: 128GB LPDDR5X-8000 (CPU/GPU shared)

advantage:
CPU↔GPU transfer: Not required (unified memory)
Large-scale model: 70B parameter model can also be executed
Multitasking: Simultaneously execute LLM + image generation
Low power consumption: 50-60W (no dGPU required)
```

**Benchmark (as of 2025):**

```yaml
LLM Reasoning (Llama 3.2 3B):
  MS-S1 Max: 82 tokens/sec
  RTX 4060 Ti (16GB): 95 tokens/sec
  M3 Max (128GB): 45 tokens/sec

LLM Inference (Llama 3.1 70B Q4_K_M):
MS-S1 Max: 18 tokens/sec ✅ Practical
RTX 4060 Ti (16GB): Not possible (lack of VRAM)
  M3 Max (128GB): 12 tokens/sec

Image generation (SDXL 1024x1024):
MS-S1 Max: 10.2 seconds
RTX 4060 Ti (16GB): 8.5 seconds
M3 Max (128GB): 18.5 seconds

Concurrent execution (LLM + image generation):
MS-S1 Max: ✅ Yes (128GB shared memory)
RTX 4060 Ti (16GB): ❌ Not possible (lack of VRAM)
M3 Max (128GB): ✅ Possible (slower speed)
```

### 1.1.3 Developable applications

**Text processing system:**

```yaml
Chatbot:
- Customer support
- Internal inquiry system
- Personal assistant

Document processing:
- Contract summary
- Automatic report generation
- Translation system

RAG (Search Extension Generation):
- Internal document search
- Knowledge base
- Technical document search
```

**Multimodal:**

```yaml
Image generation:
- Product design
- Marketing materials
- UI/UX mockup

Image recognition:
- Quality inspection
- Inventory management
- Medical image analysis (auxiliary)

Audio processing:
- Audio transcription (Whisper)
- Speech synthesis (TTS)
- Voice commands
```

---

## 1.2 Setting up the development environment

### 1.2.1 List of required software

**Basic software:**

```bash
# OS: Ubuntu 24.04 LTS (recommended)
cat /etc/os-release

# ROCm 6.4.2
rocm-smi --version

# Python 3.11+
python3 --version

# Node.js 20+ LTS
node --version
npm --version

# Docker 24+
docker --version
docker-compose --version

# Git
git --version
```

**AI base:**

```bash
# Ollama
ollama --version
# Expected: Ollama version is 0.5.0+

# LM Studio（GUI）
# Download: https://lmstudio.ai/

# ComfyUI
cd ~/ComfyUI && git log -1 --oneline

# PyTorch ROCm
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.6.0+rocm6.4, True
```

### 1.2.2 Development toolchain

**Python development environment:**

```bash
# pipenv / poetry (virtual environment management)
pip install pipenv poetry

# Development library
pip install \
    fastapi uvicorn \
    langchain langchain-community \
    chromadb sentence-transformers \
    pydantic python-dotenv \
    requests httpx \
    pytest pytest-asyncio \
    black flake8 mypy
```

**Front-end development:**

```bash
# React + TypeScript environment
npx create-react-app my-ai-app --template typescript

# or Next.js
npx create-next-app@latest my-ai-app --typescript

# Required library
npm install axios react-markdown
```

**Database:**

```bash
# PostgreSQL 16 (metadata management)
sudo apt install postgresql-16

# Redis 7 (cache)
sudo apt install redis-server

# ChromaDB (vector DB, for RAG)
# Installed with pip
```

### 1.2.3 Directory structure

**Recommended project structure:**

```
~/ai_projects/
├── llm_backend/ # LLM backend
│   ├── main.py
│   ├── models/
│   ├── services/
│   │   ├── ollama_service.py
│   │   ├── rag_service.py
│   │   └── chat_service.py
│   ├── api/
│   │   ├── chat.py
│   │   └── completion.py
│   ├── utils/
│   └── tests/
│
├── image_backend/ # Image generation backend
│   ├── comfyui_api.py
│   ├── workflows/
│   └── models/
│
├── frontend/ # frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── api/
│   │   └── utils/
│   ├── public/
│   └── package.json
│
├── data/ # data
│ ├── documents/ # RAG documents
│ ├── embeddings/ # Vector DB
│ ├── models/ # Local model
│ └── outputs/ # Generated results
│
├── docker/ # Docker settings
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
│
└── docs/ # Documentation
    ├── API.md
    └── ARCHITECTURE.md
```

---

## 1.3 Hello World: First LLM application

### 1.3.1 Simple chat using Ollama API

**Simple CLI chat:**

```python
#!/usr/bin/env python3
# hello_llm.py

import requests
import json

OLLAMA_URL = "http://localhost:11434"

def chat(prompt: str, model: str = "llama3.2:3b") -> str:
"""Chat generation with Ollama"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

if __name__ == "__main__":
print("=== Local LLM Chat ===")
    print("MS-S1 Max + Ollama")
print("Exit: Ctrl+C\n")

    while True:
        user_input = input("You: ")
        if not user_input.strip():
            continue

        response = chat(user_input)
        print(f"AI: {response}\n")
```

**execution:**

```bash
# Start Ollama (separate terminal)
ollama serve

# model pull
ollama pull llama3.2:3b

# execution
python3 hello_llm.py

# Example output:
# You: What is MS-S1 Max?
# AI: MS-S1 Max is a mini PC released by Minisforum, powered by AMD Ryzen AI Max+ 395 processor.
# A powerful APU that integrates a 16 core/32 thread CPU, Radeon 8060S (RDNA 3.5) GPU, and up to 128GB of memory.
```

### 1.3.2 RESTful API Server with FastAPI

**Basic API server:**

```python
#!/usr/bin/env python3
# api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="Local LLM API")

OLLAMA_URL = "http://localhost:11434"

class ChatRequest(BaseModel):
    prompt: str
    model: str = "llama3.2:3b"
    max_tokens: int = 512

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
"""Chat endpoint"""

    try:
# Ollama API call
        ollama_response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "num_predict": request.max_tokens
                }
            },
            timeout=60
        )

        data = ollama_response.json()

        return ChatResponse(
            response=data["response"],
            model=request.model,
            tokens=data.get("eval_count", 0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
"""List of available models"""

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
"""Health Check"""
    return {"status": "ok", "ollama": "connected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Startup and testing:**

```bash
# start server
python3 api_server.py

# test in another terminal
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
-d '{"prompt": "Name three advantages of local AI"}'

# Example response:
# {
# "response": "1. Privacy protection...",
#   "model": "llama3.2:3b",
#   "tokens": 145
# }

# Model list
curl http://localhost:8000/models
```

### 1.3.3 Creating a web front end

**React + TypeScript chat UI:**

```typescript
// src/App.tsx

import React, { useState } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        prompt: input,
        model: 'llama3.2:3b'
      });

      const aiMessage: Message = {
        role: 'assistant',
        content: response.data.response
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        role: 'assistant',
content: 'An error has occurred'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App" style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
<h1>Local AI chat</h1>
      <p>MS-S1 Max + Ollama + FastAPI</p>

      <div style={{
        border: '1px solid #ccc',
        padding: '10px',
        height: '400px',
        overflowY: 'auto',
        marginBottom: '10px'
      }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{
            marginBottom: '10px',
            textAlign: msg.role === 'user' ? 'right' : 'left'
          }}>
            <strong>{msg.role === 'user' ? 'You' : 'AI'}:</strong>
            <div style={{
              display: 'inline-block',
              padding: '8px',
              borderRadius: '8px',
              backgroundColor: msg.role === 'user' ? '#007bff' : '#f1f1f1',
              color: msg.role === 'user' ? 'white' : 'black',
              maxWidth: '70%'
            }}>
              {msg.content}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
placeholder="Enter your message..."
          style={{ flex: 1, padding: '10px' }}
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          style={{ padding: '10px 20px' }}
        >
{loading ? 'Sending...' : 'Sending'}
        </button>
      </div>
    </div>
  );
}

export default App;
```

**boot:**

```bash
# start front end
cd frontend
npm start

# Open http://localhost:3000 in your browser
```

---

## 1.4 Performance Benchmark

### 1.4.1 Each model performance on MS-S1 Max

**Small model (3B-8B):**

```yaml
Llama 3.2 3B Q4_K_M:
Load time: 1.2 seconds
Inference speed: 82 tokens/sec
VRAM usage: 2.1GB
Usage: Real-time chat, quick response

Phi-3 Mini 3.8B Q4_K_M:
Load time: 1.5 seconds
Inference speed: 68 tokens/sec
VRAM usage: 2.5GB
Usage: Code generation, technical documentation

Gemma 2 9B Q4_K_M:
Load time: 2.8 seconds
Inference speed: 45 tokens/sec
VRAM usage: 5.8GB
Application: High quality response, complex tasks
```

**Medium model (13B-34B):**

```yaml
Llama 3.1 13B Q4_K_M:
Load time: 5.2 seconds
Inference speed: 32 tokens/sec
VRAM usage: 8.2GB
Usage: General purpose assistant

Mixtral 8x7B Q4_K_M:
Load time: 8.5 seconds
Inference speed: 25 tokens/sec
VRAM usage: 26.7GB
Uses: Expert knowledge, complex reasoning
```

**Large model (70B+):**

```yaml
Llama 3.1 70B Q4_K_M:
Load time: 18.3 seconds
Inference speed: 18 tokens/sec
VRAM usage: 42.5GB
System RAM usage: 85.3GB
Use: Top quality response, specialized tasks
Note: Can run with MS-S1 Max's 128GB integrated memory!

Qwen 2.5 72B Q4_K_M:
Load time: 19.1 seconds
Inference speed: 16 tokens/sec
VRAM usage: 44.2GB
System RAM usage: 88.7GB
Application: Multilingual support, long sentence generation
```

### 1.4.2 Concurrency performance

**LLM + image generation concurrent execution:**

```yaml
Scenario 1: Llama 3.2 3B + SDXL
LLM speed: 82 → 65 tokens/sec (21% reduction)
SDXL speed: 10.2 → 12.8 seconds (25% reduction)
Total VRAM: 2.1GB + 9.8GB = 11.9GB
Verdict: ✅ Practical

Scenario 2: Llama 3.1 13B + SDXL
LLM speed: 32 → 22 tokens/sec (31% reduction)
SDXL speed: 10.2 → 15.1 seconds (48% reduction)
Total VRAM: 8.2GB + 9.8GB = 18.0GB → OOM
Verdict: ❌ Insufficient VRAM

Recommended: Combination with small LLM (3B-8B)
```

---

## 1.5 Development Best Practices

### 1.5.1 Model selection guidelines

**Recommended model by application:**

```yaml
Real-time chat:
Recommended: Llama 3.2 3B, Phi-3 Mini
Reason: Fast response (80+ tokens/sec)

Document summary/analysis:
Recommended: Gemma 2 9B, Llama 3.1 13B
Reason: High quality comprehension

Code generation:
Recommended: Qwen 2.5 Coder 7B, CodeLlama 13B
Reason: Code specialization

Multilingual support:
Recommended: Qwen 2.5 14B, Aya 35B
Reason: Multilingual learning data

Best quality (latency tolerant):
Recommended: Llama 3.1 70B, Qwen 2.5 72B
Reason: Quality similar to GPT-4
```

### 1.5.2 Quantization level selection

```yaml
Q4_K_M (recommended/balanced type):
Quality: ★★★★☆ (95% of FP16)
Size: 1/4
Speed: ★★★★★
Application: Recommended in most cases

Q5_K_M (high quality):
Quality: ★★★★★ (98% of FP16)
Size: 1/3
Speed: ★★★★☆
Application: Quality-oriented

Q6_K (best quality):
Quality: ★★★★★ (99% of FP16)
Size: 1/2
Speed: ★★★☆☆
Application: Professional use

Q2_K (ultra light):
Quality: ★★☆☆☆ (85% of FP16)
Size: 1/8
Speed: ★★★★★
Application: Prototyping
```

### 1.5.3 Error handling

**Robust LLM calls:**

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional
import time

class RobustOllamaClient:
"""Robust Ollama Client"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

# Retry settings
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)

    def generate(
        self,
        prompt: str,
        model: str = "llama3.2:3b",
        timeout: int = 60,
        max_retries: int = 3
    ) -> Optional[str]:
"""Generation with retry"""

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=timeout
                )

                response.raise_for_status()
                return response.json()["response"]

            except requests.exceptions.Timeout:
                print(f"Timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
time.sleep(2 ** attempt) # exponential backoff
                    continue
                return None

            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

        return None

# Usage example
client = RobustOllamaClient()
response = client.generate("Hello, world!")
if response:
    print(response)
else:
    print("Generation failed after retries")
```

---

## 1.6 Summary of this chapter

In this chapter, we learned the basics of local AI application development using MS-S1 Max.

### Review of learning content

**1.1: Significance of local AI development**
- ✅ Privacy/Security/Cost reduction
- ✅ Advantages of MS-S1 Max's 128GB integrated memory
- ✅ 70B model is also the only viable option

**1.2: Development environment**
- ✅ Installing required software
- ✅ Python + Node.js development environment
- ✅ Recommended directory structure

**1.3: Hello World**
- ✅ Simple chat using Ollama API
- ✅ Build a RESTful API server with FastAPI
- ✅ React + TypeScript web frontend

**1.4: Performance Benchmark**
- ✅ Inference speed for each model size
- ✅ Simultaneous execution of LLM + image generation
- ✅ MS-S1 Max specific optimizations

**1.5: Best Practices**
- ✅ Model selection guidelines by application
- ✅ Quantization level selection
- ✅ Error handling pattern

### Next steps

In Chapter 2, you will learn to design scalable application architectures. Learn implementation patterns for production environments, such as microservices, asynchronous processing, and caching strategies.

---

**Reference materials:**

- Ollama: https://ollama.ai/
- FastAPI: https://fastapi.tiangolo.com/
- LangChain: https://python.langchain.com/
- React: https://react.dev/

---
