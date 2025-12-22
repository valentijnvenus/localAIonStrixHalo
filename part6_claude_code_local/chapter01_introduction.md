# Chapter 01: Integrating Claude Code with local LLM

## 1.1 What is Claude Code?

Claude Code is an AI-powered coding assistant provided by Anthropic. Typically we use Anthropic's cloud API, but with appropriate configuration it is also possible to use local LLM.

### 1.1.1 Features of Claude Code

**Main features**
- Code generation and editing
- File operations (read, write, search)
- Execute terminal command
- Understanding and analyzing the entire project
- Git operations
- Automatic execution of multi-step tasks

**Normal use (Anthropic API)**
```
[Developer] → [Claude Code CLI] → [Anthropic API] → [Claude 3.5 Sonnet]
                                     ↓
(Cloud/Paid)
```

**Using local LLM (method in this book, latest as of November 2025)**
```
[Developer] → [Claude Code] → [LiteLLM Proxy] → [Ollama] → [Qwen3 Coder 30B Q8_0]
                                ↓                ↓
(Local/Free) (MS-S1 Max: 96GB VRAM)
```

### 1.1.2 Why use local LLM?

**merit**
1. **Cost Savings**: Avoid cloud API charges (completely free)
2. **Privacy**: Code is not sent externally
3. **Offline operation**: No internet connection required
4. **Customization**: Use your own model or fine-tuned model
5. **Response speed**: High-speed response that takes advantage of the high performance of MS-S1 Max
6. **Data Sovereignty**: Protecting your company's confidential information

**Disadvantages (limitations)**
1. Possibility of not being able to use some Claude-specific functions
2. Depends on local model performance
3. Initial setup required
4. Consumes memory and GPU resources

### 1.1.3 Advantages of MS-S1 Max (latest specs as of November 2025)

**Minisforum MS-S1 Max** (AMD Ryzen AI Max+ 395) is the perfect environment for local LLM operations.

**Hardware specs**
- **Processor**: AMD Ryzen AI Max+ 395 (16 cores/32 threads, Zen 5)
- **Integrated Memory**: 128GB LPDDR5x-8000 (quad channel)
- **VRAM Allocation**: **Up to 96GB configurable for GPU** (Unified Memory Architecture)
- **GPU**: Radeon 8060S (40 RDNA 3.5 compute units)
- **AI Performance**: Total 126 TOPS (including NPU 50 TOPS)
- **TDP**: 110W~160W (adjustable in 4 levels)

**Advantage in local LLM operation**
- **Huge VRAM (96GB)**: comfortably runs Qwen3 Coder 30B Q8_0 in 256K contexts
- **Unified Memory**: No data transfer bottleneck between CPU and GPU
- **ROCm 6.4.2 compatible**: Fast inference with AMD GPU optimization
- **Low Latency**: Zero network delay when running locally

**Recommended model (as of November 2025)**

| Model | Parameters | VRAM Usage | Context | Speed ​​(MS-S1 Max) | Application |
|--------|-----------|------------|--------------|-------------------|------|
| **qwen3-coder:30b-a3b-q8_0** | **30B (3.3B active)** | **32GB** | **256K tokens** | **22 tokens/s** | **Recommended/Highest quality** |
| qwen3-coder:14b | 14B MoE | 18GB | 256K tokens | 28 tokens/s | Balanced |
| qwen3-coder:7b | 7B | 8GB | 256K tokens | 42 tokens/s | Fast development |
| deepseek-coder-v2:16b | 16B MoE | 20GB | 128K tokens | 25 tokens/s | Alternatives |

**Recommended structure for this book**
- **Main model**: Qwen3 Coder 30B Q8_0 (32GB VRAM, 256K context)
- **Remaining VRAM**: 64GB (simultaneous execution of multiple models, large-scale context processing)

## 1.2 Architecture Overview

### 1.2.1 Component configuration

```
┌─────────────────────────────────────────────────────────┐
│ Developer │
└───────────────────┬─────────────────────────────────────┘
│ (Command/Question)
                    ↓
┌─────────────────────────────────────────────────────────┐
│                  Claude Code CLI                         │
│ - File operations │
│ - Git integration │
│ - Terminal execution │
└───────────────────┬─────────────────────────────────────┘
│ (OpenAI compatible API)
                    ↓
┌─────────────────────────────────────────────────────────┐
│                 LiteLLM Proxy                            │
│ - API format conversion │
│ - Request Routing │
│ - Log cache │
└───────────────────┬─────────────────────────────────────┘
                    │ (Ollama API)
                    ↓
┌─────────────────────────────────────────────────────────┐
│                    Ollama                                │
│ - Model management │
│ - Inference engine │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────┐
│ MS-S1 Max Hardware (November 2025) │
│ - AMD Ryzen AI Max+ 395 (16 cores/32 threads, Zen 5) │
│  - Radeon 8060S (40 RDNA 3.5 CU)                        │
│ - 128GB LPDDR5x-8000 integrated memory (up to 96GB VRAM allocable) │
│ - ROCm 6.4.2, total 126 TOPS AI performance │
└─────────────────────────────────────────────────────────┘
```

### 1.2.2 Communication flow

**1. User request**
```
Developer: "Fix the bug in this file"
    ↓
Claude Code CLI: Load files and build context
```

**2. API conversion**
```
Claude Code → LiteLLM
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [...],
  "tools": [...]
}
    ↓
LiteLLM → Ollama (after conversion)
{
  "model": "qwen3-coder:30b-a3b-q8_0",
  "messages": [...],
  "stream": true
}
```

**3. Reasoning and response**
```
Ollama (MS-S1 Max) → LiteLLM
{
  "choices": [{
    "message": {
"content": "How to fix: ...",
      "tool_calls": [...]
    }
  }]
}
    ↓
LiteLLM → Claude Code (OpenAI format)
    ↓
Claude Code: Edit files and view results
```

## 1.3 Required prerequisite knowledge

### 1.3.1 Basic Linux Commands

To proceed with this book, you should be familiar with the following Linux commands:

```bash
# Directory operations
cd /path/to/directory # change directory
mkdir my_folder # create directory
ls -la # Display file list

# File operations
cat file.txt # Display file contents
nano file.txt # Edit file
chmod +x script.sh # Grant execution permission

# Process management
ps aux | grep ollama # process search
kill -9 12345 # Kill process
systemctl status ollama # Check service status

# network
curl http://localhost:11434/api/tags # API operation check
netstat -tuln | grep 8000 # Check port usage
```

### 1.3.2 Python Basics

LiteLLM is implemented in Python. It will be easier to understand if you have basic Python knowledge.

```python
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# package installation
pip install litellm

# Easy script execution
python3 script.py
```

### 1.3.3 JSON format

API requests and responses are in JSON format.

```json
{
  "model": "qwen3-coder:30b-a3b-q8_0",
  "messages": [
    {
      "role": "user",
      "content": "Hello, World!"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "num_ctx": 262144
}
```

## 1.4 Environmental requirements

### 1.4.1 Hardware Requirements

**Minimum requirements (for lightweight models)**
- CPU: 8 cores or more
- Memory: 32GB or more
- Storage: 100GB or more free space
- GPU: Optional (works on CPU too, but slower)

**Recommended requirements (for MS-S1 Max - Qwen3 Coder 30B Q8_0)**
- CPU: AMD Ryzen AI Max+ 395 (16 cores/32 threads, Zen 5)
- Integrated memory: 128GB LPDDR5x-8000 (quad channel)
- VRAM allocation: 96GB (Qwen3 Coder 30B Q8_0 uses 32GB, remaining 64GB available)
- GPU: Radeon 8060S (40 RDNA 3.5 compute units)
- Storage: 500GB or more NVMe SSD (approximately 35GB model file)
- Power: 320W built-in PSU, TDP 110-160W

### 1.4.2 Software Requirements

**OS**
- Ubuntu 24.04 LTS (recommended)
- Ubuntu 22.04 LTS
- Other Linux distributions (needs adjustment)

**Required software**
- Python 3.10 or higher
- Node.js 18 or higher (for Claude Code CLI)
- Git 2.30 or higher
- curl、wget

**AMD ROCm (when using GPU)**
- ROCm 6.4.2 (MS-S1 Max optimized version)

## 1.5 Structure of this document

### Chapter 01: Introduction (main chapter)
- What is Claude Code?
- Advantages of local LLM integration
- Architecture overview

### Chapter 02: Basic Setup
-Ollama installation
- Model download
- Operation confirmation

### Chapter 03: LiteLLM Setup
- LiteLLM installation
- Create configuration file
- Proxy launch

### Chapter 04: Claude Code Integration
- Claude Code CLI installation
- Edit configuration file
- Initial startup and operation check

### Chapter 05: Feature Compatibility and Limitations
- Functions that work
- Features that don't work
- Workarounds and alternatives

### Chapter 06: Model selection and optimization
- Comparison of coding specialized models
- Performance tuning on MS-S1 Max
- Memory/GPU optimization

### Chapter 07: Practical examples
- Project creation
- Bug fixes
- Refactoring
- Test generation

### Chapter 08: Troubleshooting
- Common problems and solutions
- How to view logs
- Debugging steps

### Chapter 09: Advanced configuration and best practices
- How to use multiple models
- Cash utilization
- Performance monitoring
- Cost comparison (Claude API vs local)

## 1.6 Overview of setup

Throughout this document, you will build the environment using the following steps.

**Step 1: Ollama Preparation (Chapter 02)**
```bash
#Ollamainstallation
curl -fsSL https://ollama.com/install.sh | sh

# Model download (Qwen3 Coder 30B Q8_0)
ollama pull qwen3-coder:30b-a3b-q8_0
```

**Step 2: LiteLLM settings (Chapter 03)**
```bash
# LiteLLM installation
pip install litellm[proxy]

# Create configuration file
nano litellm_config.yaml

# start proxy
litellm --config litellm_config.yaml
```

**Step 3: Claude Code Settings (Chapter 04)**
```bash
# Claude Code CLI installation
npm install -g @anthropic-ai/claude-code

# Edit configuration file
nano ~/.config/claude-code/config.json

# boot
claude-code
```

**Step 4: Operation check**
```bash
# Ask a question with Claude Code
You: "Hello, can you help me with Python?"
Assistant: "Of course! I'm running on Qwen3-Coder 30B Q8_0 locally on your MS-S1 Max with 96GB VRAM..."
```

## 1.7 Expected Outcomes

By practicing the contents of this book, you will be able to achieve the following:

### 1.7.1 Cost reduction

**When using Claude API (approximate monthly estimate)**
- Light usage (100,000 tokens/month): $3-5
- Moderate usage (1 million tokens/month): $30-50
- Heavy usage (10 million tokens/month): $300-500

**When using local LLM**
- Initial cost: $0 (using existing MS-S1 Max)
- Monthly cost: Electricity bill only (approximately $3-5, assuming 24-hour operation)
- **Annual savings**: $360-$6,000

### 1.7.2 Privacy protection

- **Internal code**: Not sent externally
- **Confidential Information**: Completed locally
- **Intellectual Property**: Fully protected

### 1.7.3 Improving development efficiency

**Actual measurement data (MS-S1 Max + Qwen3 Coder 30B Q8_0)**
- Code generation speed: 22 tokens/s (inference), 160 tokens/s (prompt processing)
- Response time: average 2-5 seconds
- Simultaneous processing: Supports multiple sessions (128GB memory, 96GB VRAM allocation)
- Context: 256K tokens (native), up to 1M tokens (expanded)

**Improve productivity**
- Coding time: 30-50% reduction
- Bug fix: 2-3x speedup
- Document creation: automation

## 1.8 Summary

In this chapter, we learned the significance and overall picture of integrating Claude Code and local LLM.

**Important points**
1. Claude Code can use Ollama via LiteLLM
2. MS-S1 Max is ideal for operating large-scale models
3. High-quality development support is possible while protecting privacy, completely free of charge.
4. Requires proper configuration and understanding

In the next chapter, we will learn the steps to actually install Ollama and download the model. Let's build the environment step by step while moving your hands.
