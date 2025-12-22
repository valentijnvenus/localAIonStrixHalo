# Chapter 6: API Utilization and Integration

## 6.1 Ollama REST API Basics

### 6.1.1 API Endpoints

Ollama automatically starts the REST API server on startup (default: http://localhost:11434).

```bash
# API connection confirmation
curl http://localhost:11434/api/version

# output
{"version":"0.5.4"}
```

**Primary endpoint:**

| Endpoint | Method | Description |
|--------------|---------|------|
| `/api/generate` | POST | Text generation |
| `/api/chat` | POST | Chat format |
| `/api/tags` | GET | Model list |
| `/api/show` | POST | Model information |
| `/api/pull` | POST | Model download |
| `/api/push` | POST | Model upload |
| `/api/delete` | DELETE | Model deletion |

### 6.1.2 Basic API calls

#### generate endpoint

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

**response:**
```json
{
  "model": "llama3.1",
  "created_at": "2025-01-15T12:34:56.789Z",
  "response": "The sky appears blue due to Rayleigh scattering...",
  "done": true,
  "total_duration": 2847293847,
  "load_duration": 123456789,
  "prompt_eval_count": 12,
  "prompt_eval_duration": 234567890,
  "eval_count": 87,
  "eval_duration": 2489269158
}
```

#### chat endpoint

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [
{"role": "user", "content": "Hello"}
  ],
  "stream": false
}'
```

### 6.1.3 Streaming

```bash
# Streaming response (real-time generation)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Write a long story",
  "stream": true
}'
```

**Output (NDJSON format):**
```json
{"model":"llama3.1","response":"Once","done":false}
{"model":"llama3.1","response":" upon","done":false}
{"model":"llama3.1","response":" a","done":false}
...
{"model":"llama3.1","response":"","done":true}
```

## 6.2 Utilization in Python

### 6.2.1 Official Python Library

```bash
# install
pip install ollama
```

#### Basic usage

```python
import ollama

# simple generation
response = ollama.generate(
    model='llama3.1',
    prompt='Explain quantum computing'
)
print(response['response'])
```

#### Chat format

```python
import ollama

response = ollama.chat(
    model='qwen2.5:7b',
    messages=[
{'role': 'user', 'content': 'Hello! '}
    ]
)
print(response['message']['content'])
```

#### Streaming

```python
import ollama

for chunk in ollama.generate(
    model='llama3.1',
    prompt='Write a story',
    stream=True
):
    print(chunk['response'], end='', flush=True)
```

### 6.2.2 Chat application example

```python
# simple_chat.py
import ollama

def chat_session(model='qwen2.5:7b'):
"""Interactive chat session"""
    messages = []

    print(f"=== Ollama Chat ({model}) ===")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        messages.append({
            'role': 'user',
            'content': user_input
        })

        print("Assistant: ", end='', flush=True)

        response_content = ""
        for chunk in ollama.chat(
            model=model,
            messages=messages,
            stream=True
        ):
            content = chunk['message']['content']
            print(content, end='', flush=True)
            response_content += content

        print("\n")

        messages.append({
            'role': 'assistant',
            'content': response_content
        })

if __name__ == '__main__':
    chat_session()
```

### 6.2.3 Advanced settings

```python
import ollama

response = ollama.generate(
    model='llama3.1',
    prompt='Explain AI',
    options={
        'temperature': 0.8,
        'top_p': 0.9,
        'top_k': 40,
        'num_predict': 200,
        'repeat_penalty': 1.1,
        'num_ctx': 8192
    }
)
```

### 6.2.4 Asynchronous processing

```python
# async_ollama.py
import asyncio
import ollama

async def async_generate(prompt):
"""Asynchronous generation"""
    response = await ollama.AsyncClient().generate(
        model='qwen2.5:7b',
        prompt=prompt
    )
    return response['response']

async def multi_query():
"""Execute multiple queries in parallel"""
    tasks = [
        async_generate("What is AI?"),
        async_generate("What is ML?"),
        async_generate("What is DL?")
    ]

    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}:\n{result}")

if __name__ == '__main__':
    asyncio.run(multi_query())
```

## 6.3 Integration with LangChain

### 6.3.1 Installation and configuration

```bash
pip install langchain langchain-community
```

### 6.3.2 Basic usage

```python
from langchain_community.llms import Ollama

# LLM initialization
llm = Ollama(
    model="qwen2.5:14b",
    temperature=0.7
)

# execution
response = llm.invoke("Tell me about MS-S1 Max")
print(response)
```

### 6.3.3 Chain construction

```python
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# LLM
llm = Ollama(model="llama3.1")

# prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms:"
)

# chain
chain = LLMChain(llm=llm, prompt=prompt)

# execution
result = chain.invoke({"topic": "quantum computing"})
print(result['text'])
```

### 6.3.4 RAG（Retrieval-Augmented Generation）

```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# load document
loader = TextLoader("ms-s1-max-manual.txt")
documents = loader.load()

# text split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# Embed generation (using Ollama)
embeddings = OllamaEmbeddings(model="llama3.1")

# Vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="qwen2.5:14b"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# question
query = "What is the memory bandwidth of MS-S1 Max?"
result = qa_chain.invoke({"query": query})
print(result['result'])
```

## 6.4 OpenAI API Compatibility Mode

### 6.4.1 Endpoint

Ollama provides an OpenAI API compatible interface with the `/v1` endpoint.

```bash
# OpenAI API compatible endpoint
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 6.4.2 Use with OpenAI Python SDK

```python
from openai import OpenAI

# specify Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1',
api_key='ollama' # dummy (required but value ignored)
)

# Chat Completions
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### 6.4.3 Streaming

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

stream = client.chat.completions.create(
    model="llama3.1",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

### 6.4.4 Integration with existing apps

```python
# Switch OpenAI library with environment variable
import os
os.environ['OPENAI_API_KEY'] = 'ollama'
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'

# Existing OpenAI code works as is
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="llama3.1",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## 6.5 Web Application Integration

### 6.5.1 Flask basic example

```python
# app.py
from flask import Flask, request, jsonify, Response
import ollama
import json

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'qwen2.5:7b')

    response = ollama.generate(model=model, prompt=prompt)
    return jsonify({'response': response['response']})

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'llama3.1')

    def generate():
        for chunk in ollama.generate(model=model, prompt=prompt, stream=True):
            yield f"data: {json.dumps(chunk)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 6.5.2 FastAPI High Performance Edition

```python
# fastapi_app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:7b"
    stream: bool = False

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if request.stream:
            async def generate():
                for chunk in ollama.generate(
                    model=request.model,
                    prompt=request.prompt,
                    stream=True
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            response = ollama.generate(
                model=request.model,
                prompt=request.prompt
            )
            return {"response": response['response']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.5.3 WebSocket Chat

```python
# websocket_server.py
from fastapi import FastAPI, WebSocket
import ollama
import json

app = FastAPI()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    messages = []

    try:
        while True:
# receive message from client
            data = await websocket.receive_text()
            user_message = json.loads(data)

            messages.append({
                'role': 'user',
                'content': user_message['content']
            })

# streaming response
            response_content = ""
            for chunk in ollama.chat(
                model=user_message.get('model', 'qwen2.5:7b'),
                messages=messages,
                stream=True
            ):
                content = chunk['message']['content']
                response_content += content

# Real-time transmission
                await websocket.send_text(json.dumps({
                    'type': 'chunk',
                    'content': content
                }))

# Completion notification
            await websocket.send_text(json.dumps({
                'type': 'done',
                'content': response_content
            }))

            messages.append({
                'role': 'assistant',
                'content': response_content
            })

    except Exception as e:
        await websocket.close()
```

## 6.6 VSCode Integration

### 6.6.1 Continue.dev

```bash
# Install Continue.dev extension
# VSCode: Extensions → "Continue"

# Configuration (~/.continue/config.json)
{
  "models": [
    {
      "title": "Ollama Qwen2.5",
      "provider": "ollama",
      "model": "qwen2.5:14b"
    },
    {
      "title": "Ollama CodeLlama",
      "provider": "ollama",
      "model": "codellama:13b"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Ollama",
    "provider": "ollama",
    "model": "qwen2.5:7b"
  }
}
```

### 6.6.2 Custom VSCode Extension

```javascript
// extension.js
const vscode = require('vscode');
const axios = require('axios');

async function generateWithOllama(prompt) {
    const response = await axios.post('http://localhost:11434/api/generate', {
        model: 'qwen2.5:7b',
        prompt: prompt,
        stream: false
    });
    return response.data.response;
}

function activate(context) {
    let disposable = vscode.commands.registerCommand(
        'ollama.explainCode',
        async function () {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const code = editor.document.getText(selection);

            const prompt = `Explain this code:\n${code}`;
            const explanation = await generateWithOllama(prompt);

            vscode.window.showInformationMessage(explanation);
        }
    );

    context.subscriptions.push(disposable);
}

module.exports = { activate };
```

## 6.7 Operation with Docker

### 6.7.1 Docker Compose settings

```yaml
# docker-compose.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    devices:
      - /dev/kfd
      - /dev/dri
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
      - OLLAMA_HOST=0.0.0.0:11434
    restart: unless-stopped

  web_app:
    build: .
    container_name: ollama_web
    ports:
      - "8000:8000"
    depends_on:
      - ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434

volumes:
  ollama_data:
```

```bash
# boot
docker-compose up -d

# Model download
docker exec -it ollama ollama pull qwen2.5:7b
```

## 6.8 Monitoring and Logging

### 6.8.1 Prometheus Metrics

```python
# metrics_exporter.py
from prometheus_client import start_http_server, Counter, Histogram
import ollama
import time

# Metric definition
request_count = Counter('ollama_requests_total', 'Total requests')
request_duration = Histogram('ollama_request_duration_seconds', 'Request duration')

def monitored_generate(model, prompt):
    request_count.inc()

    with request_duration.time():
        response = ollama.generate(model=model, prompt=prompt)

    return response

if __name__ == '__main__':
# Start Prometheus exporter
    start_http_server(8001)

# run application
    while True:
        time.sleep(1)
```

### 6.8.2 Log aggregation

```python
# logging_wrapper.py
import logging
import ollama
import time
import json

# Log settings
logging.basicConfig(
    filename='ollama_requests.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def logged_generate(model, prompt):
    start_time = time.time()

    response = ollama.generate(model=model, prompt=prompt)

    elapsed = time.time() - start_time

    log_data = {
        'model': model,
        'prompt_length': len(prompt),
        'response_length': len(response['response']),
        'duration': elapsed,
        'tokens_per_second': response.get('eval_count', 0) / elapsed
    }

    logging.info(json.dumps(log_data))

    return response
```

## 6.9 Summary of this chapter

In this chapter, you learned the following contents.

✅ **REST API**
- Basic endpoints
- Streaming

✅ **Python integration**
- Official library
- LangChain, OpenAI compatible

✅ **Web application**
- Flask, FastAPI
- WebSocket

✅ **Development tools integration**
- VSCode (Continue.dev)
- Custom extension

✅ **Operation**
- Docker
- Monitoring and logging

In the next chapter, you will learn in-depth performance optimization for MS-S1 Max.

---

**Previous chapter**: [Chapter 5 Model Management and Customization](chapter05_model_management.md)
**Next Chapter**: [Chapter 7 Performance Optimization for MS-S1 Max](chapter07_performance_optimization.md)
