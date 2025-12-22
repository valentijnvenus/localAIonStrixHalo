# Chapter 1: Introduction - Ollama and the Revolution in Local Large-Scale Language Models

## 1.1 What is Ollama?

**Ollama** is an innovative open source tool that makes it easy to run large language models (LLMs) locally. It is designed around a command line interface and is optimized for developers and power users.

### Ollama's Philosophy

Ollama was developed with the aim of bringing the "simplicity of Docker" to the world of LLM.

```bash
# As easy as Docker
docker pull ubuntu
docker run ubuntu

# Ollama is equally simple
ollama pull llama3.1
ollama run llama3.1
```

This design philosophy makes it easy for anyone to use cutting-edge AI models, even without complex machine learning knowledge.

### Why Ollama is attracting attention

#### 1. **Overwhelming simplicity**

Traditional LLM execution environments are complex.

**Traditional way (Python + Transformers):**

```bash
# A few hours just to build the environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install transformers accelerate bitsandbytes
pip install sentencepiece protobuf

# Need to write code
python inference.py --model meta-llama/Llama-3.1-70B-Instruct --prompt "Hello"
```

**For Ollama:**

```bash
# Installation (1 minute)
curl -fsSL https://ollama.com/install.sh | sh

# Ready to use
ollama run llama3.1 "Hello"
```

#### 2. **Developer Friendly**

- Simple CLI
- REST API included as standard
- OpenAI API compatible
- Multilingual SDK (Python, JavaScript, Go, Rust, etc.)

#### 3. **Performance-focused**

- A high-speed inference engine implemented in C++ (based on llama.cpp)
- Automatic GPU detection and offloading
- Efficient memory management
- Parallel execution of multiple models

#### 4. **Extensive model library**

The official Ollama library contains over 100 models.

```bash
# Major model families
ollama pull llama3.1           # Meta Llama 3.1
ollama pull qwen2.5            # Alibaba Qwen2.5
ollama pull mistral            # Mistral AI
ollama pull gemma2             # Google Gemma 2
ollama pull codellama # coding specialized
ollama pull phi3               # Microsoft Phi-3
```

## 1.2 Differences between LM Studio and Ollama

LM Studio, which we learned about in Part 1, and Ollama, the subject of this book, each have different strengths.

### LM Studio's Strengths

Features | detail
--- | ---
**GUI** | Intuitive graphical interface
**For beginners** | Complete with just a click
**visualization** | Visualization of GPU usage and memory usage
**Chat-focused** | ChatGPT-like interface
**Preset Management** | Easily save settings on the GUI

### Ollama's Strengths

Features | detail
--- | ---
**CLI** | Command line centric design
**For developers** | API, SDK, automation friendly
**lightweight** | Resource efficient
**Scripting** | Easy automation and batch processing
**Server Mode** | Background operation

### Guidelines for proper use

```
[Cases when you should choose LM Studio]
✓ Try local LLM for the first time
✓ Main use is interactive chat
✓ I want to adjust settings with GUI
✓ I want to visualize performance
✓ Windows users

[Cases when you should choose Ollama]
✓ Developer, engineer
✓ Link with other apps via API
✓ I want to automate it with a script
✓ Server use (resident)
✓ Linux/macOS users
✓ Switch and use multiple models
```

### Recommendation for combined use

In fact, a combination of both is most effective.

```
MS-S1 Max (128GB memory)
├── LM Studio (port 1234)
│ └── For interactive chat
└── Ollama (port 11434)
└── API/Development
```

With a massive 128GB of memory, the MS-S1 Max has enough capacity to run both at the same time.

## 1.3 The ultimate combination of MS-S1 Max and Ollama

### The overwhelming advantage of AMD Ryzen AI Max+ 395

The MS-S1 Max's Ryzen AI Max+ 395 is ideal hardware for Ollama.

#### 1. **Large integrated memory (128GB)**

```
Regular PC (16GB):
└── Can only be executed on 7B model

High-end PC (64GB):
└── Up to 34B model

MS-S1 Max (128GB):
├── Runs the 70B model comfortably
├── Run multiple 34B models simultaneously
├── 13B model x 3 + 7B model x 2
└── Extra-long context (128K+) is also possible
```

#### 2. **AMD Radeon 8060S GPU (ROCm compatible)**

Ollama fully supports ROCm and enables fast inference on AMD GPUs.

```bash
# Automatic GPU detection
ollama run llama3.1:70b

# Output example
>>> Inference speed: 15-20 tokens/s (70B model, Q4 quantization)
>>> GPU usage: 85-95%
>>> VRAM usage: 42GB
```

#### 3. **High memory bandwidth (256GB/s)**

Quad-channel LPDDR5X-8000 enables fast loading and inference of large models.

```
Memory bandwidth impact:

DDR4-3200 (51GB/s):
- 70B model load: 90 seconds
- Prompt processing: 350 t/s

LPDDR5X-8000 (256GB/s):
- 70B model load: 18 seconds (5x faster)
- Prompt processing: 1200 t/s (3.4x faster)
```

### Best hardware characteristics for Ollama

The MS-S1 Max is a perfect match for Ollama's:

#### Multi-model parallel execution

```bash
# Terminal 1: Translation task (34B model)
ollama run qwen2.5:32b "Translate to Japanese: Hello World"

# Terminal 2: Coding (13B model)
ollama run codellama:13b "Write a Python function for..."

# Terminal 3: Chat (7B model)
ollama run llama3.1 "What is the meaning of life?"

# All can be executed simultaneously!
# Total memory usage: Approximately 65GB → 128GB is sufficient
```

#### Long-lasting operation

Ollama runs as a background service, and the MS-S1 Max's efficient cooling system ensures stable operation 24/7.

```
MS-S1 Max Balance mode (130W):
- Temperature: 65-75℃ (stable)
- Fan noise: acceptable
- Inference performance: fast

→ Ideal for server applications
```

## 1.4 What you can do with Ollama

### 1.4.1 Interactive Chat

This is the simplest usage.

```bash
# Interactive mode
ollama run qwen2.5:32b

>>> Hello! Please tell me about MS-S1 Max.
MS-S1 Max is a powerful mini PC powered by AMD Ryzen AI Max+ 395.
The main features...

>>> /bye
```

### 1.4.2 One-shot execution

When used from a script.

```bash
# Use in shell script
RESPONSE=$(ollama run llama3.1 "Summarize: $(cat article.txt)")
echo "$RESPONSE" > summary.txt

# Use in pipeline
echo "Translate to English: Hello" | ollama run qwen2.5:14b
```

### 1.4.3 REST API Server

Ollama automatically starts a REST API server on startup.

```bash
# Ollama service starts automatically (default: port 11434)
# Use with curl
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Why is the sky blue?"
}'

# Streaming response (real time)
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:32b",
  "prompt": "Write a long story...",
  "stream": true
}'
```

### 1.4.4 OpenAI API compatibility mode

Existing OpenAI API clients can be used as is.

```python
# Use OpenAI Python SDK as is
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
api_key='ollama' # dummy (required but value ignored)
)

response = client.chat.completions.create(
    model="llama3.1",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

### 1.4.5 Customization with Modelfile

You can create your own model variations.

```dockerfile
# Modelfile
FROM llama3.1

# system prompt
SYSTEM """
You are a kind and knowledgeable Japanese assistant.
As an MS-S1 Max expert, I will explain in detail.
"""

#parameter
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40

# template
TEMPLATE """{{ .System }}

User: {{ .Prompt }}
Assistant:"""
```

```bash
# Create custom model
ollama create my-assistant -f Modelfile

# Use
ollama run my-assistant "Tell me about Ryzen AI Max+ 395"
```

## 1.5 What you will learn in this book

### Part 2: The Complete Guide to Ollama (this book)

This book consists of the following nine chapters:

**Chapter 1 (this chapter)** : Overview of Ollama and the advantages of MS-S1 Max **Chapter 2** : Installation and setup **Chapter 3** : ROCm settings and AMD GPU optimization **Chapter 4** : Basic commands and usage **Chapter 5** : Model management and customization **Chapter 6** : API utilization and integration **Chapter 7** : Performance optimization for MS-S1 Max **Chapter 8** : Multi-model operation and concurrent execution **Chapter 9** : Advanced techniques and troubleshooting

### How to proceed with your studies

#### Beginners (first time LLM student)

1. Understand the basics in Chapters 1 and 2
2. Chapter 3: Perfecting Your GPU Settings
3. Chapter 4: Hands-on experience
4. Try out different models in Chapter 5

#### Intermediate users (those with LM Studio experience)

1. Install in Chapter 2
2. Check the ROCm settings in Chapter 3
3. Chapter 6: Learn how to use APIs
4. Optimization techniques in Chapter 7
5. Chapter 8: Multi-model operation

#### For advanced users (developers)

1. Chapter 2 and Chapter 3: Building the environment
2. Learn more about API integration in Chapter 6
3. Thorough optimization in Chapter 7
4. Chapters 8 and 9 explain practical operation methods
5. Automation scripts, application development

### Prerequisite knowledge

To get the most out of this book, it is best to have knowledge of the following:

```
[Required]
✓ Basic Linux commands (cd, ls, cat, etc.)
✓ Using a text editor (nano, vim, etc.)
✓ Fundamentals of terminal operation

[Recommended]
✓ Fundamentals of shell scripting
✓ REST API concepts
✓ Understanding JSON format

[Convenient to have]
✓ Basics of Python, JavaScript
✓ Docker experience
✓ Experience using Git
```

However, even if you don't have advanced knowledge, the course is structured so that you can fully understand it by studying step by step.

## 1.6 Ollama Ecosystem

Ollama is more than just a tool; it's the center of an ecosystem.

### Official Tools

#### 1. **Ollama CLI**

```bash
# core tools
ollama run, pull, push, list, rm, etc.
```

#### 2. **Ollama Web UI (formerly Ollama WebUI)**

```bash
# Browser-based chat UI
docker run -d -p 3000:8080 ghcr.io/open-webui/open-webui
```

#### 3.Official **SDK**

```bash
# Python
pip install ollama

# JavaScript/TypeScript
npm install ollama

# Go
go get github.com/ollama/ollama/api

# Rust
cargo add ollama-rs
```

### Third-Party Integrations

```
[Popular integration destinations]
✓ LangChain / LangSmith
✓ LlamaIndex
✓ Continue.dev (VSCode extension)
✓ Jan (desktop app)
✓ Open WebUI
✓ Obsidian (note app)
✓ Raycast (macOS launcher)
```

### Community Model

```bash
# Import from Hugging Face
ollama create mymodel -f Modelfile

# Unique fine-tuning model
ollama create my-finetuned -f custom.Modelfile

# share
ollama push myusername/mymodel
```

## 1.7 Ollama Licensing and Business Use

### license

- **Ollama** : MIT License (commercial use available)
- **Model** : Compliant with the license of each model

### Major model licenses

Model | license | commercial use
--- | --- | ---
Llama 3.1 | Meta License | ✓ Possible
Qwen2.5 | Apache 2.0 | ✓ Possible
Mistral | Apache 2.0 | ✓ Possible
Gemma 2 | Gemma License | ✓ Yes (with limitations)
Phi-3 | MIT License | ✓ Possible

**⚠️ Note** : Please make sure to check the license of the model you want to use before using it for your business.

### Benefits of private use

```
[Use in companies]
✓ Data is not sent externally
✓ Securely handle sensitive information
✓ Customizable
✓ Running cost reduction
✓ No rate limits
✓ Can work offline

[Personal use]
✓ Privacy protection
✓ No monthly fee required
✓ Ideal for learning and experimentation
✓ Free customization
```

## 1.8 Ollama Version and Compatibility

### Latest version (as of 2025)

```bash
# Check version
ollama --version

# Output example
ollama version is 0.5.4
```

### Major Milestones

```
v0.1.0 (August 2023)
- Initial release
- Basic model execution functionality

v0.2.0 (November 2023)
- Added REST API
- Multimodal compatible

v0.3.0 (February 2024)
- Modelfile support
- Custom model creation

v0.4.0 (June 2024)
- OpenAI API compatibility mode
- Performance improvements

v0.5.0 (October 2024)
- AMD ROCm full support
- Enhanced multi-GPU support
- Contextual caching

v0.5.4 (as of 2025)
- RDNA 3.5 optimization
- Improved memory management
- Compatible with new model formats
```

### MS-S1 Max compatibility

```bash
# Required version for MS-S1 Max
Ollama: v0.4.0 or later (v0.5.4 recommended)
ROCm: 6.1 or later (6.3 recommended)
Linux Kernel: 6.5 or later
```

## 1.9 Document Conventions

### Command Notation

```bash
# Comment: Explanation
command --option value

# Output example
>>> View results
```

### environmental variables

```bash
# setting
export VARIABLE_NAME=value

# Use
echo $VARIABLE_NAME
```

### File editing

```bash
# File path notation
~/.bashrc
/etc/systemd/system/ollama.service
```

### API example (curl)

```bash
curl http://localhost:11434/api/endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "key": "value"
  }'
```

### Important Information

> **💡 TIP** : Useful techniques and tips

> **⚠️ Caution** : Things to be aware of

> **🚨 WARNING** : SERIOUS WARNING

## 1.10 Summary of this chapter

In this chapter, you learned the following:

✅ **Ollama's features and philosophy**

- Docker-like simplicity
- Developer-friendly design
- High-performance inference engine

✅ **Difference between LM Studio and Ollama**

- GUI vs CLI
- Guidelines for proper use
- Both can be used together

✅ **Compatibility with MS-S1 Max**

- Utilizing 128GB of large memory
- AMD Radeon 8060S + ROCm
- Multi-model parallel execution

✅ **What you can do with Ollama**

- Interactive Chat
- REST API
- OpenAI compatible API
- Custom Model Creation

✅Ecosystem **and Licensing**

- Extensive integration tools
- Community Support
- Available for commercial use

In the next chapter, we will actually install Ollama on the MS-S1 Max and perform the initial setup. We will explain step by step how to configure ROCm, recognize AMD GPUs, and run the first model.

---

**Next Chapter** : [Chapter 2 Installation and Setup](chapter02_installation.md)
