# Chapter 04: Chatbot development

## 4.1 Chatbot Basics

### 4.1.1 Types of chatbots

Chatbots that can be implemented with MS-S1 Max are classified into the following types.

**1. Simple question-and-answer type**
```python
# simple_qa_bot.py
import ollama

def simple_chat(question: str, model: str = "qwen2.5:14b") -> str:
"""Simple Q&A chatbot"""
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": question}
    ])
    return response['message']['content']

# Usage example
answer = simple_chat("What is a list comprehension in Python?")
print(answer)
```

**2. Context preservation type**
```python
# contextual_bot.py
from typing import List, Dict
import ollama

class ContextualBot:
    def __init__(self, model: str = "qwen2.5:14b", system_prompt: str = ""):
        self.model = model
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str) -> str:
"""Chat with context"""
        self.messages.append({"role": "user", "content": user_message})

        response = ollama.chat(model=self.model, messages=self.messages)
        assistant_message = response['message']['content']

        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def reset(self):
"""Reset conversation history"""
        system_msg = [msg for msg in self.messages if msg['role'] == 'system']
        self.messages = system_msg

# Usage example
bot = ContextualBot(system_prompt="You are a Python expert.")
print(bot.chat("Please tell me about list comprehensions"))
print(bot.chat("Please show me an example using it")) # Understand the previous context
```

**3. RAG integrated type**
```python
# rag_bot.py
from typing import List, Dict
import ollama
import chromadb

class RAGBot:
    def __init__(self, collection_name: str, model: str = "qwen2.5:14b"):
        self.model = model
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection(name=collection_name)
        self.messages: List[Dict[str, str]] = []

    def chat(self, user_message: str, n_results: int = 3) -> str:
"""Chat using RAG"""
# Search related documents
        results = self.collection.query(
            query_texts=[user_message],
            n_results=n_results
        )

# Build context
        context = "\n\n".join(results['documents'][0])

# add to prompt
augmented_message = f"""Answer the questions using the context below.

【context】
{context}

【question】
{user_message}"""

        self.messages.append({"role": "user", "content": augmented_message})

        response = ollama.chat(model=self.model, messages=self.messages)
        assistant_message = response['message']['content']

        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message
```

### 4.1.2 Performance characteristics with MS-S1 Max

**Response speed by model (MS-S1 Max)**

| Model | Number of parameters | Memory usage | Response speed | Recommended use |
|--------|--------------|--------------|----------|----------|
| qwen2.5:3b | 3B | 2.5GB | 45 tokens/s | When fast response is required |
| qwen2.5:7b | 7B | 5.8GB | 32 tokens/s | Balanced |
| qwen2.5:14b | 14B | 11.2GB | 18 tokens/s | High quality response |
| llama3.1:8b | 8B | 6.5GB | 28 tokens/s | General purpose |
| mixtral:8x7b | 47B | 28GB | 8 tokens/s | Specialized tasks |

**Context length and memory usage**

```python
# context_benchmark.py
import ollama
import psutil
import time

def measure_context_impact(context_lengths: List[int]):
"""Measuring the impact of context length"""
    results = []

    for length in context_lengths:
# Generate long context for testing
context = "This is a test." * (length // 10)

        start_mem = psutil.virtual_memory().used / 1024**3
        start_time = time.time()

        response = ollama.chat(
            model="qwen2.5:14b",
messages=[{"role": "user", "content": context + "\nPlease summarize"}]
        )

        end_time = time.time()
        end_mem = psutil.virtual_memory().used / 1024**3

        results.append({
            "context_tokens": length,
            "response_time": end_time - start_time,
            "memory_delta": end_mem - start_mem
        })

    return results

# test execution
results = measure_context_impact([512, 1024, 2048, 4096, 8192])
for r in results:
    print(f"Tokens: {r['context_tokens']}, Time: {r['response_time']:.2f}s, Memory: {r['memory_delta']:.2f}GB")
```

**Actual measurement results with MS-S1 Max**
```
Tokens: 512, Time: 2.3s, Memory: 0.8GB
Tokens: 1024, Time: 3.1s, Memory: 1.2GB
Tokens: 2048, Time: 4.8s, Memory: 1.9GB
Tokens: 4096, Time: 7.5s, Memory: 3.2GB
Tokens: 8192, Time: 13.2s, Memory: 5.8GB
```

## 4.2 Conversation management system

### 4.2.1 Session management

```python
# session_manager.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path

class SessionManager:
    def __init__(self, session_dir: str = "./sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, Dict] = {}

    def create_session(self, user_id: str, system_prompt: str = "") -> str:
"""Create a new session"""
        session_id = str(uuid.uuid4())

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "messages": [],
            "system_prompt": system_prompt
        }

        if system_prompt:
            session_data["messages"].append({
                "role": "system",
                "content": system_prompt,
                "timestamp": datetime.now().isoformat()
            })

        self.active_sessions[session_id] = session_data
        self._save_session(session_id)

        return session_id

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
"""Add message"""
        if session_id not in self.active_sessions:
            self._load_session(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        if metadata:
            message["metadata"] = metadata

        self.active_sessions[session_id]["messages"].append(message)
        self.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()

        self._save_session(session_id)

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
"""Get message history"""
        if session_id not in self.active_sessions:
            self._load_session(session_id)

        messages = self.active_sessions[session_id]["messages"]

        if limit:
            return messages[-limit:]
        return messages

    def get_context_messages(self, session_id: str, max_tokens: int = 4096) -> List[Dict]:
"""Get message within token limit"""
        messages = self.get_messages(session_id)

# Simple token estimation (actually uses tiktoken etc.)
        estimated_tokens = 0
        context_messages = []

# Always include system prompts
        system_messages = [msg for msg in messages if msg['role'] == 'system']
        context_messages.extend(system_messages)
        estimated_tokens += sum(len(msg['content'].split()) * 1.3 for msg in system_messages)

# Add new messages first
        for msg in reversed([m for m in messages if m['role'] != 'system']):
            msg_tokens = len(msg['content'].split()) * 1.3
            if estimated_tokens + msg_tokens > max_tokens:
                break
            context_messages.insert(len(system_messages), msg)
            estimated_tokens += msg_tokens

        return context_messages

    def delete_session(self, session_id: str):
"""Delete session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        session_file = self.session_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

    def cleanup_old_sessions(self, days: int = 7):
"""Delete old sessions"""
        cutoff = datetime.now() - timedelta(days=days)

        for session_file in self.session_dir.glob("*.json"):
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            last_activity = datetime.fromisoformat(session_data['last_activity'])
            if last_activity < cutoff:
                session_file.unlink()

    def _save_session(self, session_id: str):
"""Save session to file"""
        session_file = self.session_dir / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(self.active_sessions[session_id], f, ensure_ascii=False, indent=2)

    def _load_session(self, session_id: str):
"""Load session from file"""
        session_file = self.session_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                self.active_sessions[session_id] = json.load(f)
        else:
            raise ValueError(f"Session {session_id} not found")
```

### 4.2.2 Advanced chatbot implementation

```python
# advanced_chatbot.py
from typing import List, Dict, Optional, AsyncIterator
import ollama
from session_manager import SessionManager
import asyncio
from datetime import datetime

class AdvancedChatbot:
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        session_manager: Optional[SessionManager] = None
    ):
        self.model = model
        self.session_manager = session_manager or SessionManager()

    def create_session(self, user_id: str, bot_config: Optional[Dict] = None) -> str:
"""Create a new chat session"""
        system_prompt = self._build_system_prompt(bot_config)
        return self.session_manager.create_session(user_id, system_prompt)

    def chat(
        self,
        session_id: str,
        user_message: str,
        stream: bool = False
    ):
"""Chat (streaming compatible)"""
# save user message
        self.session_manager.add_message(session_id, "user", user_message)

# get context
        messages = self.session_manager.get_context_messages(session_id)

# convert to Ollama format
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

        if stream:
            return self._stream_chat(session_id, ollama_messages)
        else:
            return self._sync_chat(session_id, ollama_messages)

    def _sync_chat(self, session_id: str, messages: List[Dict]) -> Dict:
"""Sync Chat"""
        start_time = datetime.now()

        response = ollama.chat(model=self.model, messages=messages)

        assistant_message = response['message']['content']

# save with metadata
        metadata = {
            "model": self.model,
            "eval_count": response.get('eval_count', 0),
            "eval_duration": response.get('eval_duration', 0),
            "response_time": (datetime.now() - start_time).total_seconds()
        }

        self.session_manager.add_message(
            session_id,
            "assistant",
            assistant_message,
            metadata=metadata
        )

        return {
            "message": assistant_message,
            "metadata": metadata
        }

    def _stream_chat(self, session_id: str, messages: List[Dict]) -> AsyncIterator[str]:
"""Streaming Chat"""
        full_response = []
        start_time = datetime.now()

        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            content = chunk['message']['content']
            full_response.append(content)
            yield content

# save complete response
        assistant_message = ''.join(full_response)
        metadata = {
            "model": self.model,
            "response_time": (datetime.now() - start_time).total_seconds(),
            "streaming": True
        }

        self.session_manager.add_message(
            session_id,
            "assistant",
            assistant_message,
            metadata=metadata
        )

    def _build_system_prompt(self, config: Optional[Dict]) -> str:
"""Build system prompt"""
        if not config:
            return ""

        parts = []

        if "role" in config:
parts.append(f"You are {config['role']}.")

        if "personality" in config:
parts.append(f"Personality: {config['personality']}")

        if "constraints" in config:
parts.append(f"constraints: {config['constraints']}")

        if "examples" in config:
parts.append(f"Example: {config['examples']}")

        return "\n".join(parts)

    def get_session_summary(self, session_id: str) -> Dict:
"""Get session summary"""
        messages = self.session_manager.get_messages(session_id)

        user_messages = [m for m in messages if m['role'] == 'user']
        assistant_messages = [m for m in messages if m['role'] == 'assistant']

        total_response_time = sum(
            m.get('metadata', {}).get('response_time', 0)
            for m in assistant_messages
        )

        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_response_time": total_response_time,
            "avg_response_time": total_response_time / len(assistant_messages) if assistant_messages else 0
        }
```

## 4.3 Streaming response

### 4.3.1 Streaming implementation with FastAPI

```python
# streaming_api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
from advanced_chatbot import AdvancedChatbot

app = FastAPI()
chatbot = AdvancedChatbot()

class ChatRequest(BaseModel):
    session_id: str
    message: str
    stream: bool = True

class SessionCreate(BaseModel):
    user_id: str
    bot_config: Optional[dict] = None

@app.post("/session/create")
async def create_session(request: SessionCreate):
"""Create a new session"""
    session_id = chatbot.create_session(request.user_id, request.bot_config)
    return {"session_id": session_id}

@app.post("/chat")
async def chat(request: ChatRequest):
"""Chat (streaming compatible)"""
    if request.stream:
        async def generate():
            for chunk in chatbot.chat(request.session_id, request.message, stream=True):
# Server-Sent Events format
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        result = chatbot.chat(request.session_id, request.message, stream=False)
        return result

@app.get("/session/{session_id}/summary")
async def get_summary(session_id: str):
"""Get session summary"""
    try:
        return chatbot.get_session_summary(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/session/{session_id}/messages")
async def get_messages(session_id: str, limit: Optional[int] = None):
"""Get message history"""
    try:
        messages = chatbot.session_manager.get_messages(session_id, limit)
        return {"messages": messages}
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")
```

### 4.3.2 React frontend (streaming compatible)

```javascript
// StreamingChat.jsx
import React, { useState, useEffect, useRef } from 'react';

function StreamingChat() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
// Create session
    fetch('http://localhost:8000/session/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: 'user_' + Date.now(),
        bot_config: {
role: 'AI assistant',
personality: 'kind and polite'
        }
      })
    })
    .then(res => res.json())
    .then(data => setSessionId(data.session_id));
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || !sessionId) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

// streaming response
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        message: userMessage,
        stream: true
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = '';

// Add assistant message placeholder
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            setIsLoading(false);
            break;
          }

          try {
            const parsed = JSON.parse(data);
            assistantMessage += parsed.content;

// update last message
            setMessages(prev => {
              const newMessages = [...prev];
              newMessages[newMessages.length - 1] = {
                role: 'assistant',
                content: assistantMessage
              };
              return newMessages;
            });
          } catch (e) {
// Ignore JSON parse errors
          }
        }
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
<h1>Streaming chat</h1>

      <div style={{
        border: '1px solid #ccc',
        borderRadius: '8px',
        height: '500px',
        overflowY: 'auto',
        padding: '20px',
        marginBottom: '20px',
        backgroundColor: '#f9f9f9'
      }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{
            marginBottom: '15px',
            textAlign: msg.role === 'user' ? 'right' : 'left'
          }}>
            <div style={{
              display: 'inline-block',
              padding: '10px 15px',
              borderRadius: '10px',
              backgroundColor: msg.role === 'user' ? '#007bff' : '#e9ecef',
              color: msg.role === 'user' ? 'white' : 'black',
              maxWidth: '70%',
              textAlign: 'left',
              whiteSpace: 'pre-wrap'
            }}>
              {msg.content}
              {msg.role === 'assistant' && !msg.content && isLoading && (
                <span className="typing-indicator">...</span>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div style={{ display: 'flex', gap: '10px' }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
placeholder="Enter your message... (Enter to send, Shift+Enter to break a line)"
          style={{
            flex: 1,
            padding: '10px',
            borderRadius: '5px',
            border: '1px solid #ccc',
            resize: 'vertical',
            minHeight: '60px'
          }}
          disabled={isLoading || !sessionId}
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || !sessionId}
          style={{
            padding: '10px 20px',
            borderRadius: '5px',
            border: 'none',
            backgroundColor: '#007bff',
            color: 'white',
            cursor: 'pointer'
          }}
        >
send
        </button>
      </div>

      {sessionId && (
        <p style={{ marginTop: '10px', color: '#666', fontSize: '0.9em' }}>
Session ID: {sessionId}
        </p>
      )}
    </div>
  );
}

export default StreamingChat;
```

## 4.4 Multimodal chat

### 4.4.1 Image-enabled chatbot

```python
# multimodal_chatbot.py
from typing import List, Dict, Optional
import ollama
from pathlib import Path
import base64

class MultimodalChatbot:
    def __init__(self, model: str = "llava:13b"):
        """
multimodal chatbot
Recommended models for MS-S1 Max:
- llava:7b: lightweight and fast
- llava:13b: Balanced
- bakllava:7b: More detailed explanation
        """
        self.model = model
        self.messages: List[Dict] = []

    def chat_with_image(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> str:
"""Chat with images"""
        message_content = prompt

# Prepare image
        images = []
        if image_path:
# base64 encode local image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            images.append(image_data)
        elif image_url:
            images.append(image_url)

# construct message
        user_message = {"role": "user", "content": message_content}
        if images:
            user_message["images"] = images

        self.messages.append(user_message)

# Send to Ollama
        response = ollama.chat(
            model=self.model,
            messages=self.messages
        )

        assistant_message = response['message']['content']
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def analyze_image(self, image_path: str, analysis_type: str = "general") -> Dict:
"""Image analysis"""
        prompts = {
"general": "Please tell us more about this image.",
"ocr": "Extract all text contained in this image.",
"objects": "Please list all objects in this image.",
"scene": "Describe the scene (location, time of day, atmosphere, etc.) of this image."
        }

        prompt = prompts.get(analysis_type, prompts["general"])
        result = self.chat_with_image(prompt, image_path=image_path)

        return {
            "analysis_type": analysis_type,
            "result": result,
            "image_path": image_path
        }

# Usage example
bot = MultimodalChatbot(model="llava:13b")

# Image analysis
result = bot.analyze_image("screenshot.png", analysis_type="ocr")
print(result['result'])

# Ongoing conversation
response1 = bot.chat_with_image("What stands out most in this image?", image_path="photo.jpg")
response2 = bot.chat_with_image("What color is that?") # See previous image
```

### 4.4.2 Image support API

```python
# multimodal_api.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from multimodal_chatbot import MultimodalChatbot

app = FastAPI()
chatbot = MultimodalChatbot()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    analysis_type: str = Form("general")
):
"""Upload images and analyze"""
# save image
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

# Run analysis
    result = chatbot.analyze_image(str(file_path), analysis_type)

    return JSONResponse(content=result)

@app.post("/chat-with-image")
async def chat_with_image(
    prompt: str = Form(...),
    file: UploadFile = File(None),
    image_url: str = Form(None)
):
"""Chat with images"""
    image_path = None

    if file:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_path = str(file_path)

    response = chatbot.chat_with_image(
        prompt=prompt,
        image_path=image_path,
        image_url=image_url
    )

    return {"response": response}
```

## 4.5 Chatbot evaluation and improvement

### 4.5.1 Evaluation of response quality

```python
# chatbot_evaluator.py
from typing import List, Dict
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ChatbotEvaluator:
    def __init__(self, embedding_model: str = "mxbai-embed-large"):
        self.embedding_model = embedding_model

    def get_embedding(self, text: str) -> np.ndarray:
"""Get text embedding"""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return np.array(response['embedding'])

    def evaluate_relevance(self, question: str, answer: str) -> float:
"""Assess the relevance of questions and answers"""
        q_emb = self.get_embedding(question)
        a_emb = self.get_embedding(answer)

        similarity = cosine_similarity([q_emb], [a_emb])[0][0]
        return float(similarity)

    def evaluate_consistency(self, answers: List[str]) -> float:
"""Assess consistency of multiple answers"""
        embeddings = [self.get_embedding(ans) for ans in answers]

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def evaluate_coherence(self, conversation: List[Dict]) -> float:
"""Evaluate conversational consistency"""
# Extract user and assistant message pairs
        pairs = []
        for i in range(len(conversation) - 1):
            if conversation[i]['role'] == 'user' and conversation[i+1]['role'] == 'assistant':
                pairs.append({
                    'question': conversation[i]['content'],
                    'answer': conversation[i+1]['content']
                })

        if not pairs:
            return 0.0

        relevance_scores = [
            self.evaluate_relevance(pair['question'], pair['answer'])
            for pair in pairs
        ]

        return float(np.mean(relevance_scores))

    def evaluate_response_quality(
        self,
        question: str,
        answer: str,
        reference_answer: Optional[str] = None
    ) -> Dict:
"""Comprehensive response quality evaluation"""
        metrics = {
            "relevance": self.evaluate_relevance(question, answer)
        }

        if reference_answer:
            metrics["similarity_to_reference"] = self.evaluate_relevance(answer, reference_answer)

# Length-based metrics
        metrics["length"] = len(answer)
        metrics["word_count"] = len(answer.split())

        return metrics

# Usage example
evaluator = ChatbotEvaluator()

# Evaluate a single response
question = "What is a list comprehension in Python?"
answer = "A list comprehension is a Python syntax that concisely creates a new list from an existing list."
metrics = evaluator.evaluate_response_quality(question, answer)
print(f"Relevance score: {metrics['relevance']:.3f}")

# Evaluation of the entire conversation
conversation = [
{"role": "user", "content": "Tell me about Python"},
{"role": "assistant", "content": "Python is a general-purpose programming language..."},
{"role": "user", "content": "What is a list comprehension?"},
{"role": "assistant", "content": "This is a concise syntax for creating lists..."}
]
coherence = evaluator.evaluate_coherence(conversation)
print(f"Conversation coherence: {coherence:.3f}")
```

### 4.5.2 Performance Monitoring

```python
# chatbot_monitor.py
from typing import Dict, List
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

class ChatbotMonitor:
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"

    def log_interaction(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        metadata: Dict
    ):
"""Log interactions"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_message_length": len(user_message),
            "bot_response_length": len(bot_response),
            "response_time": metadata.get("response_time", 0),
            "model": metadata.get("model", ""),
            "tokens": metadata.get("eval_count", 0)
        }

        with self.metrics_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def get_statistics(self, hours: int = 24) -> Dict:
"""Get statistics"""
        cutoff = datetime.now() - timedelta(hours=hours)

        metrics = {
            "total_interactions": 0,
            "unique_sessions": set(),
            "response_times": [],
            "message_lengths": [],
            "tokens_generated": []
        }

        if not self.metrics_file.exists():
            return self._format_statistics(metrics)

        with self.metrics_file.open('r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry['timestamp'])

                if timestamp < cutoff:
                    continue

                metrics["total_interactions"] += 1
                metrics["unique_sessions"].add(entry['session_id'])
                metrics["response_times"].append(entry['response_time'])
                metrics["message_lengths"].append(entry['bot_response_length'])
                metrics["tokens_generated"].append(entry['tokens'])

        return self._format_statistics(metrics)

    def _format_statistics(self, metrics: Dict) -> Dict:
"""Format statistics"""
        return {
            "total_interactions": metrics["total_interactions"],
            "unique_sessions": len(metrics["unique_sessions"]),
            "avg_response_time": np.mean(metrics["response_times"]) if metrics["response_times"] else 0,
            "median_response_time": np.median(metrics["response_times"]) if metrics["response_times"] else 0,
            "p95_response_time": np.percentile(metrics["response_times"], 95) if metrics["response_times"] else 0,
            "avg_message_length": np.mean(metrics["message_lengths"]) if metrics["message_lengths"] else 0,
            "total_tokens_generated": sum(metrics["tokens_generated"])
        }

    def get_performance_report(self) -> str:
"""Generate performance report"""
        stats_24h = self.get_statistics(hours=24)
        stats_1h = self.get_statistics(hours=1)

        report = f"""
# Chatbot performance report

## Last 24 hours
- Total interactions: {stats_24h['total_interactions']}
- Number of unique sessions: {stats_24h['unique_sessions']}
- Average response time: {stats_24h['avg_response_time']:.2f} seconds
- 95th percentile response time: {stats_24h['p95_response_time']:.2f} seconds
- Average message length: {stats_24h['avg_message_length']:.0f} characters
- Total number of generated tokens: {stats_24h['total_tokens_generated']}

## Past hour
- Total interactions: {stats_1h['total_interactions']}
- Number of unique sessions: {stats_1h['unique_sessions']}
- Average response time: {stats_1h['avg_response_time']:.2f} seconds
"""
        return report
```

## 4.6 Summary

In this chapter, we learned about chatbot development using MS-S1 Max.

**Key points**

1. **Types of chatbots**
- From simple Q&A to advanced RAG integration
- Model selection according to usage (qwen2.5:3b-14b, llava)

2. **Session management**
- Persisted session management
- Context length control
- Message management within token limits

3. **Streaming response**
- Server-Sent Events implementation in FastAPI
- Real-time display in React frontend
- Improved user experience

4. **Multimodal compatible**
- Image understanding with LLaVA model
- OCR, object detection, scene analysis
- Implementation of chat with images

5. **Evaluation and Monitoring**
- Quantitative evaluation of response quality
- Collect performance metrics
- Data analysis for continuous improvement

**Practical tips on MS-S1 Max**

- **Memory management**: Utilizing 128GB large capacity memory, multiple models can be operated simultaneously
- **Model selection**: Balanced qwen2.5:14b is optimal in many cases
- **Streaming**: Long responses do not detract from the user experience
- **Batch processing**: Improve efficiency with parallel processing of multiple sessions

In the next chapter, we will learn about developing a Web API that integrates these technologies.
