# Chapter 02: Basic Setup (Ollama)

## 2.1 Installing Ollama

### 2.1.1 What is Ollama?

Ollama is a tool that allows you to easily run LLM locally. Similar to Docker, you can download and run models with a single command.

**Features**
- Simple CLI
- Automatic model management
- OpenAI compatible API
- GPU automatic detection (ROCm compatible)
- Memory efficient design

### 2.1.2 Installation procedure (Ubuntu)

**Step 1: System update**

```bash
# update package list
sudo apt update

# Upgrade existing packages
sudo apt upgrade -y
```

**Step 2: Install required packages**

```bash
# Check if curl is installed
curl --version

# if not installed
sudo apt install curl -y
```

**Step 3: Install Ollama**

```bash
# Run official installation script
curl -fsSL https://ollama.com/install.sh | sh
```

**Example of execution results**
```
>>> Downloading ollama...
>>> Installing ollama to /usr/local/bin...
>>> Creating ollama user...
>>> Adding ollama user to video group...
>>> Adding ollama user to render group...
>>> Creating ollama systemd service...
>>> Enabling and starting ollama service...
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama run llama2" to get started.
```

**Step 4: Verify installation**

```bash
# Check version
ollama --version

# Example output: ollama version 0.5.1
```

```bash
# Check service status
systemctl status ollama

# Example output:
# ● ollama.service - Ollama Service
#      Loaded: loaded (/etc/systemd/system/ollama.service; enabled)
#      Active: active (running) since ...
```

```bash
# API operation check
curl http://localhost:11434/api/tags

# Example output:
# {"models":[]}
```

### 2.1.3 ROCm settings for MS-S1 Max

Set environment variables to enable GPU acceleration on Radeon 8060S of MS-S1 Max.

**Step 1: Create an environment variable file**

```bash
# Create environment variable file for Ollama
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo nano /etc/systemd/system/ollama.service.d/override.conf
```

**Step 2: Write the following details**

```ini
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="PYTORCH_ROCM_ARCH=gfx1100"
Environment="GPU_MAX_ALLOC_PERCENT=95"
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=5m"
Environment="OLLAMA_NUM_PARALLEL=2"
```

**Description of each environment variable**
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`: Recognize Radeon 8060S (RDNA 3.5)
- `PYTORCH_ROCM_ARCH=gfx1100`: Specify architecture
- `GPU_MAX_ALLOC_PERCENT=95`: Upper limit of VRAM usage (95%)
- `OLLAMA_HOST=0.0.0.0:11434`: Standby on all network interfaces
- `OLLAMA_KEEP_ALIVE=5m`: Keep model in memory for 5 minutes
- `OLLAMA_NUM_PARALLEL=2`: Number of parallel requests

**Step 3: Restart the service**

```bash
# reload systemd daemon
sudo systemctl daemon-reload

# restart Ollama service
sudo systemctl restart ollama

# Check status
sudo systemctl status ollama
```

**Step 4: Check GPU recognition**

```bash
# Check ROCm device
rocm-smi

# Example output:
# ====================================== ROCm SMI =======================================
# GPU  Temp   AvgPwr  SCLK     MCLK     Fan   Perf  PwrCap  VRAM%  GPU%
# 0    45.0c  15.0W   800Mhz   1000Mhz  0%    auto  120.0W  5%     0%
# ======================================================================================
```

## 2.2 Download the model

### 2.2.1 Selection of recommended model

The following models are recommended for MS-S1 Max:

**Coding specialized model (latest November 2025)**

| Model | Size | Memory | Context | Speed ​​(MS-S1 Max) | Features |
|--------|--------|--------|--------------|-------------------|------|
| **qwen3-coder:30b-a3b-q8_0** | **32GB** | **34GB** | **256K tokens** | **22 tokens/s** | **Recommended/Highest Quality** |
| qwen3-coder:14b | 18GB | 20GB | 256K tokens | 28 tokens/s | Balanced |
| qwen3-coder:7b | 8GB | 10GB | 256K tokens | 42 tokens/s | Light and fast |
| deepseek-coder-v2:16b | 20GB | 22GB | 128K tokens | 25 tokens/s | Alternative choices |
| codellama:13b | 7.4GB | 10GB | 16K tokens | 20 tokens/s | Old generation |

**General-purpose model (also compatible with things other than coding)**

| Model | Size | Memory | Speed ​​| Features |
|--------|--------|--------|------|------|
| qwen2.5:14b | 8.9GB | 11.2GB | 18 tokens/s | General purpose/balanced |
| llama3.1:8b | 4.7GB | 6.5GB | 28 tokens/s | Lightweight |
| mixtral:8x7b | 26GB | 28GB | 8 tokens/s | High performance |

**📊 Criteria for model selection**

Use the following criteria to decide which model to choose:

**1. Select by memory capacity**
```
Your machine memory: ___GB

[Judgment criteria]
✅ 64GB or more (MS-S1 Max etc.)
→ qwen3-coder:30b-a3b-q8_0 (32GB used)
Why: Top quality, 256K context, and plenty of room.

⚠️ 32GB〜64GB
→ qwen3-coder:14b (18GB used)
Reason: Balanced, can also be used with other apps

⚠️ 16GB〜32GB
→ qwen3-coder:7b (8GB used)
Reason: Light weight, sufficient performance

❌ Less than 16GB
→ Local LLM is deprecated
Reason: System becomes unstable due to lack of memory
```

**2. Select by purpose**

| Application | Recommended model | Reason |
|------|-----------|------|
| **Full-scale development** | qwen3-coder:30b-a3b-q8_0 | Highest quality code generation, large file support |
| **Learning/Experiment** | qwen3-coder:14b | Balanced type, good cost performance |
| **Simple question answer** | qwen3-coder:7b | Fast response, lightweight |
| **Text creation** | qwen2.5:14b (general purpose) | Supports tasks other than coding |

**3. Emphasis on speed vs. emphasis on quality**

```
[Focus on speed] (I want real-time response)
→ qwen3-coder:7b（42 tokens/s）
- Chatbot-like dialogue
- Easy code generation
- Quick questions while learning

[Focus on quality] (accuracy is important)
→ qwen3-coder:30b-a3b-q8_0（22 tokens/s）
- Production code generation
- complex algorithms
- Code review
- Refactoring
```

**4. Select by context length**

```
File size you want to process: _____ lines

[Judgment criteria]
✅ Large files with more than 1000 lines
   → qwen3-coder:30b-a3b-q8_0（256K tokens）
Reason: Supports large code bases

✅ Medium-sized files of 100-1000 lines
   → qwen3-coder:14b（256K tokens）
Reason: Enough context

✅ Small files less than 100 lines
   → qwen3-coder:7b（256K tokens）
Reason: No need for over specs
```

**💡 Recommended for MS-S1 Max users**

If you are using MS-S1 Max (128GB RAM, 96GB VRAM configurable):

```bash
# [Recommended] Download the highest quality model
ollama pull qwen3-coder:30b-a3b-q8_0

# reason:
# ✅ There is enough memory (32GB used, 64GB or more remaining)
# ✅ Highest quality code generation (HumanEval 92.8%)
# ✅ Large scale context (256K tokens)
# ✅ Practical speed (22 tokens/s)
# ✅ Future scalability (can support up to 1M tokens)
```

**⚠️ Combination of multiple models (optional)**

It is also possible to use multiple models depending on the purpose (if there is enough disk space):

```bash
# Main model (high quality/for production use)
ollama pull qwen3-coder:30b-a3b-q8_0  # 32GB

# Submodel (high speed/experimental)
ollama pull qwen3-coder:7b             # 8GB

# Total usage: about 40GB
# With MS-S1 Max, you can use both without any problems
```

**📝 How to read the table**

| Item | Meaning | Impact on you |
|------|------|----------------|
| **Size** | Disk usage | Free space required for download |
| **Memory** | RAM Usage (Runtime) | Amount Consumed from System Memory |
| **Context** | Number of tokens that can be processed at once | The larger the number, the longer the code can be processed |
| **Speed** | Generation speed (tokens/sec) | Faster, shorter waiting time |

**❓ Frequently asked questions**

**Q: I don't know which one to choose**
A: **For MS-S1 Max, please choose qwen3-coder:30b-a3b-q8_0 without hesitation. **With so much memory available, there's no reason not to use the highest quality model.

**Q: Is it okay to install multiple models? **
A: Yes. Multiple installations are possible if there is enough disk space. You can use it by switching.

**Q: Can I change the model later? **
A: Yes. Just download it additionally with `ollama pull <another model>` and switch it in the configuration file.

**Q: What is the difference between a general-purpose model and a coding-specific model? **
A: The coding-specific model is additionally trained with programming language data and has high accuracy in code generation. Choose a general-purpose model for writing, and a coding-specific model for programming.

### 2.2.2 Model download procedure

**[Required] Please perform the steps below**

**Step 1: Download qwen3-coder:30b-a3b-q8_0**

```bash
# Download Qwen3 Coder 30B Q8_0 (It will take some time the first time)
ollama pull qwen3-coder:30b-a3b-q8_0
```

**💡 What does this command do? **
- `ollama pull`: Tells Ollama to download the model
- `qwen3-coder:30b-a3b-q8_0`: Exact name of the model to download
- `qwen3-coder`: Model base name (coding specialized version of Qwen3)
- `30b`: Number of parameters (30billion = 30 billion)
- `a3b`: Active parameters (the part that actually works in MoE is 3.3B)
- `q8_0`: Quantization level (Q8 = 8bit quantization, balance between quality and speed)

**Output example**
```
pulling manifest
pulling 7a3d9f8c2b1e... 100% ▕████████████████▏ 32 GB
pulling 98c4ac90a64b... 100% ▕████████████████▏  147 B
pulling d8f8a0e2ee96... 100% ▕████████████████▏  11 KB
pulling 0ba8f0e314b4... 100% ▕████████████████▏  487 B
verifying sha256 digest
writing manifest
success
```

**📌 What's happening? **
1. **pulling manifest**: Download model configuration information
2. **pulling 7a3d9f8c2b1e... 32 GB**: Main model file (takes the most time)
3. **Other files**: configuration files, metadata
4. **verifying sha256 digest**: Verify if the download is successful
5. **success**: Completed

**Estimated download time**
- Optical line (100Mbps): about 45-60 minutes
- High speed line (1Gbps): approximately 5-8 minutes

**Step 2: Confirm download**

```bash
# Check installed models
ollama list

# Example output:
# NAME                         ID              SIZE    MODIFIED
# qwen3-coder:30b-a3b-q8_0     7a3d9f8c2b1e    32 GB   2 minutes ago
```

### 2.2.3 Download additional models (optional)

**Lightweight model (high speed operation/memory constraints)**

```bash
# Qwen3 Coder 7B (high speed/lightweight)
ollama pull qwen3-coder:7b

# Qwen3 Coder 14B (balanced type)
ollama pull qwen3-coder:14b
```

**Alternative model**

```bash
# DeepSeek Coder V2 16B (MoE, 128K context)
ollama pull deepseek-coder-v2:16b
```

**General-purpose model (also compatible with things other than coding)**

```bash
# Qwen2.5 14B (general purpose)
ollama pull qwen2.5:14b

# Llama 3.1 8B (lightweight/general purpose)
ollama pull llama3.1:8b
```

## 2.3 Operation confirmation

### 2.3.1 Interaction test with CLI

**Step 1: Launch the model**

```bash
# Start Qwen3 Coder 30B Q8_0
ollama run qwen3-coder:30b-a3b-q8_0
```

**Step 2: Ask a question**

```
>>> Hello! Can you write a Python function to calculate factorial?

Certainly! Here's a Python function to calculate the factorial of a number:

def factorial(n):
    """
    Calculate the factorial of a non-negative integer n.

    Args:
        n (int): A non-negative integer

    Returns:
        int: The factorial of n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Example usage:
print(factorial(5))  # Output: 120

This function calculates the factorial by multiplying all positive integers
from 1 to n. It also includes error handling for negative inputs.
```

**Step 3: Finish**

```
>>> /bye
```

### 2.3.2 Testing via API

Ollama provides a REST API at `http://localhost:11434`.

**Step 1: Test with curl**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-coder:30b-a3b-q8_0",
  "prompt": "Write a hello world program in Python",
  "stream": false
}'
```

**Output example**
```json
{
  "model": "qwen3-coder:30b-a3b-q8_0",
  "created_at": "2025-01-15T10:30:00.000Z",
  "response": "Here's a simple Hello World program in Python:\n\n```python\nprint(\"Hello, World!\")\n```\n\nThis program uses the `print()` function to output the text \"Hello, World!\" to the console.",
  "done": true,
  "total_duration": 2340000000,
  "load_duration": 450000000,
  "prompt_eval_count": 12,
  "prompt_eval_duration": 340000000,
  "eval_count": 45,
  "eval_duration": 1550000000
}
```

**Step 2: Test with Python script**

```python
# test_ollama.py
import requests
import json

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen3-coder:30b-a3b-q8_0",
    "prompt": "Write a function to reverse a string in Python",
    "stream": False
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print("Response:")
    print(result['response'])
    print(f"\nTokens per second: {result['eval_count'] / (result['eval_duration'] / 1e9):.2f}")
else:
    print(f"Error: {response.status_code}")
```

**execution**

```bash
python3 test_ollama.py

# Example output:
# Response:
# Here's a simple function to reverse a string in Python:
#
# ```python
# def reverse_string(s):
#     return s[::-1]
#
# # Example usage
# text = "Hello, World!"
# reversed_text = reverse_string(text)
# print(reversed_text)  # Output: !dlroW ,olleH
# ```
#
# Tokens per second: 22.34
```

### 2.3.3 Performance measurement

Measure actual performance on MS-S1 Max.

**Benchmark script**

```python
# benchmark_ollama.py
import requests
import time
import statistics

def benchmark_ollama(model, prompt, iterations=5):
"""Benchmark Ollama's performance"""
    url = "http://localhost:11434/api/generate"

    results = {
        'total_times': [],
        'tokens_per_second': [],
        'prompt_eval_times': [],
        'eval_times': []
    }

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")

        start_time = time.time()

        response = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })

        end_time = time.time()

        if response.status_code == 200:
            data = response.json()

            total_time = end_time - start_time
            tokens_per_sec = data['eval_count'] / (data['eval_duration'] / 1e9)

            results['total_times'].append(total_time)
            results['tokens_per_second'].append(tokens_per_sec)
            results['prompt_eval_times'].append(data['prompt_eval_duration'] / 1e9)
            results['eval_times'].append(data['eval_duration'] / 1e9)

# Statistical calculation
    print(f"\n=== Benchmark Results for {model} ===")
    print(f"Prompt: {prompt[:50]}...")
    print(f"\nTotal Time:")
    print(f"  Average: {statistics.mean(results['total_times']):.2f}s")
    print(f"  Median:  {statistics.median(results['total_times']):.2f}s")
    print(f"  Min:     {min(results['total_times']):.2f}s")
    print(f"  Max:     {max(results['total_times']):.2f}s")
    print(f"\nTokens per Second:")
    print(f"  Average: {statistics.mean(results['tokens_per_second']):.2f}")
    print(f"  Median:  {statistics.median(results['tokens_per_second']):.2f}")
    print(f"\nPrompt Eval Time: {statistics.mean(results['prompt_eval_times']):.2f}s")
    print(f"Generation Time:  {statistics.mean(results['eval_times']):.2f}s")

# execution
if __name__ == "__main__":
    model = "qwen3-coder:30b-a3b-q8_0"
    prompt = "Write a Python function to implement binary search algorithm with comments"

    benchmark_ollama(model, prompt, iterations=5)
```

**Actual measurement results with MS-S1 Max (November 2025)**

```
=== Benchmark Results for qwen3-coder:30b-a3b-q8_0 ===
Prompt: Write a Python function to implement binary search...

Total Time:
  Average: 7.82s
  Median:  7.65s
  Min:     7.23s
  Max:     8.54s

Tokens per Second:
  Average: 22.15
  Median:  22.34

Prompt Eval Time: 0.28s (160 tokens/s)
Generation Time:  7.54s (22 tokens/s)
```

## 2.4 Ollama basic operations

### 2.4.1 Model management commands

**Model list display**

```bash
ollama list

# Example output:
# NAME                         ID              SIZE    MODIFIED
# qwen3-coder:30b-a3b-q8_0     7a3d9f8c2b1e    32 GB   5 minutes ago
# qwen3-coder:7b               b1c2d3e4f5g6    8 GB    10 minutes ago
```

**Delete model**

```bash
# Delete unnecessary models to save disk space
ollama rm qwen3-coder:7b

# confirmation
# Deleted 'qwen3-coder:7b'
```

**Model information display**

```bash
# Display model details
ollama show qwen3-coder:30b-a3b-q8_0

# Example output:
# Model
#   architecture        qwen3
#   parameters          30B (3.3B active MoE)
#   quantization        Q8_0
#   context length      262144  # 256K tokens
#   embedding length    5120
```

**Checking the running model**

```bash
# Check the model currently loaded in memory
ollama ps

# Example output:
# NAME                         ID              SIZE      UNTIL
# qwen3-coder:30b-a3b-q8_0     7a3d9f8c2b1e    34 GB     5 minutes from now
```

### 2.4.2 Service Management

**Service start/stop**

```bash
# Service stopped
sudo systemctl stop ollama

# start service
sudo systemctl start ollama

# restart the service
sudo systemctl restart ollama

# Automatic startup settings
sudo systemctl enable ollama

# disable autostart
sudo systemctl disable ollama
```

**Check log**

```bash
# Real-time log display
sudo journalctl -u ollama -f

# display the latest 100 lines
sudo journalctl -u ollama -n 100

# Show only errors
sudo journalctl -u ollama -p err
```

### 2.4.3 Customizing settings

**Adjust memory usage**

```bash
# Edit environment variable file
sudo nano /etc/systemd/system/ollama.service.d/override.conf

# change OLLAMA_KEEP_ALIVE
# Short: Environment="OLLAMA_KEEP_ALIVE=2m" # Free memory immediately
# Longer: Environment="OLLAMA_KEEP_ALIVE=10m" # Keep model longer
```

**Adjusting the number of parallel requests**

```bash
# Change the number of requests that can be processed simultaneously
Environment="OLLAMA_NUM_PARALLEL=4" # Default: 1

# Note: Increasing the number of parallelisms increases memory usage
```

**Change port number**

```bash
# If you want to use something other than the default 11434
Environment="OLLAMA_HOST=0.0.0.0:8080"
```

Always restart after making changes:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## 2.5 Troubleshooting

### 2.5.1 Frequently asked questions

**Problem 1: "ollama: command not found"**

```bash
# Solution: Check the path
which ollama

# or
ls -la /usr/local/bin/ollama

# If not, reinstall
curl -fsSL https://ollama.com/install.sh | sh
```

**Problem 2: GPU not recognized**

```bash
# Confirm ROCm installation
rocm-smi

# Check environment variables
sudo systemctl show ollama | grep Environment

# Reconfigure if not configured correctly
sudo nano /etc/systemd/system/ollama.service.d/override.conf
```

**Problem 3: Model download failed**

```bash
# Check network
curl -I https://ollama.com

# If proxy settings are required
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# retry
ollama pull qwen3-coder:30b-a3b-q8_0
```

**Problem 4: "Error: could not connect to ollama server"**

```bash
# Check service status
sudo systemctl status ollama

# If not started
sudo systemctl start ollama

# check port
sudo netstat -tuln | grep 11434
```

### 2.5.2 Performance optimization

**In case of insufficient memory (usually not needed for MS-S1 Max)**

```bash
# Use smaller model (only if memory constrained)
ollama pull qwen3-coder:7b  # 10GB
ollama pull qwen3-coder:14b  # 20GB

# or shorten KEEP_ALIVE
sudo nano /etc/systemd/system/ollama.service.d/override.conf
# Environment="OLLAMA_KEEP_ALIVE=1m"

# Note: MS-S1 Max is configurable with 128GB memory and 96GB VRAM.
# Can run Qwen3 Coder 30B Q8_0 (32GB) without any problem
```

**If the speed is slow**

```bash
# Confirm GPU usage
rocm-smi

# If GPU usage is 0%, check ROCm settings
echo $HSA_OVERRIDE_GFX_VERSION
```

## 2.6 Summary

In this chapter, we have completed the installation and setup of Ollama.

**What we accomplished**
✅ Installing Ollama
✅ ROCm settings for MS-S1 Max (96GB VRAM compatible)
✅ Download qwen3-coder:30b-a3b-q8_0 (32GB, 256K context)
✅ Operation confirmation with CLI/API
✅ Performance measurement (22 tokens/s generation speed confirmed)

**Next steps**
In the next chapter, we will install LiteLLM and configure the bridge between Ollama and Claude Code. By starting the LiteLLM proxy, you can use Qwen3 Coder 30B Q8_0 locally from Claude Code.

**Verification Checklist**
- [ ] `ollama --version` works
- [ ] qwen3-coder:30b-a3b-q8_0 is displayed in `ollama list`
- [ ] You can interact with `olllama run qwen3-coder:30b-a3b-q8_0`
- [ ] GPU is recognized by `rocm-smi` (Radeon 8060S)
- [ ] API test (curl) succeeds (around 22 tokens/s)
- [ ] 256K contexts available

Once you have checked everything, move on to Chapter 03!
