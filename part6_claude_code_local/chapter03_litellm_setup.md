# Chapter 03: LiteLLM setup

## 3.1 What is LiteLLM?

### 3.1.1 Role of LiteLLM

LiteLLM is a Python library that can handle various LLM providers (Ollama, OpenAI, Anthropic, Azure OpenAI, etc.) with a unified interface.

**Why you need LiteLLM**

```
Claude Code → Expected API format: OpenAI compatible
    ↓
Ollama → API format provided: Ollama original (partially compatible with OpenAI)
```

LiteLLM bridges this gap:

```
Claude Code → [LiteLLM Proxy] → Ollama
                  ↓
Convert to full OpenAI compatible API
+ Additional features (caching, logging, fallback)
```

**Main features**
1. **API format conversion**: Convert Ollama response to OpenAI format
2. **Proxy Server**: Acts as an independent API server
3. **Load Balancing**: Distributing requests to multiple models
4. **Caching**: Reusing results of the same request
5. **Logging**: Records all requests and responses
6. **Fallback**: If a model fails, try again with another model

### 3.1.2 Architecture

```
┌──────────────────┐
│  Claude Code     │
└────────┬─────────┘
         │ HTTP POST /v1/chat/completions
         │ {model: "claude-3-5-sonnet", messages: [...]}
         ↓
┌──────────────────┐
│  LiteLLM Proxy   │
│  (Port 8000)     │
│                  │
│  - Model Mapping │ ← config.yaml
│  - Cache         │
│  - Logging       │
└────────┬─────────┘
         │ HTTP POST /api/chat
         │ {model: "qwen3-coder:30b-a3b-q8_0", messages: [...]}
         ↓
┌──────────────────┐
│  Ollama          │
│  (Port 11434)    │
└────────┬─────────┘
         │
         ↓
┌──────────────────────────┐
│  Qwen3 Coder 30B Q8_0   │
│  (MS-S1 Max: 96GB VRAM) │
└──────────────────────────┘
```

## 3.2 Installing LiteLLM

### 3.2.1 Check Python version

```bash
# Check Python version (requires 3.10 or higher)
python3 --version

# Example output: Python 3.11.6

# Upgrade if below 3.10
sudo apt install python3.11 python3.11-venv -y
```

### 3.2.2 Creating a virtual environment

```bash
# create working directory
mkdir -p ~/litellm
cd ~/litellm

# create virtual environment
python3 -m venv venv

# enable virtual environment
source venv/bin/activate

# Make sure the prompt changes
# (venv) user@ms-s1-max:~/litellm$
```

### 3.2.3 Installing LiteLLM

```bash
# upgrade pip
pip install --upgrade pip

# Install LiteLLM (with proxy function)
pip install 'litellm[proxy]'

# Confirm installation
litellm --version

# Example output: litellm 1.55.7
```

**Installation time**: About 2-3 minutes

**Required disk space**: Approximately 500MB

## 3.3 Creating configuration file

### 3.3.1 Basic configuration file

LiteLLM's operation is controlled by a configuration file in YAML format.

```bash
# create configuration file
nano ~/litellm/config.yaml
```

**Minimum configuration (config.yaml)**

```yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      api_base: http://localhost:11434
      temperature: 0.7
      max_tokens: 4096
      num_ctx: 262144  # 256K tokens context

general_settings:
master_key: sk-1234 # Any key (simple and OK as it is local)
```

**📖 Detailed explanation of each setting item**

**1. model_list section (required)**

```yaml
model_list: # ← Array that can define multiple models
- model_name: claude-3-5-sonnet-20241022 # ← Name specified by client
```

**💡 What is `model_name`? **
- **Purpose**: Model name specified by Claude Code or Aider in API requests
- **Your environment**: Use as is (no changes required)
- **Customize**: If you use other tools, change to the corresponding model name
- **Example**: `gpt-4` for Cursor, `claude-3-5-sonnet` for Continue.dev

**2. litellm_params section (required)**

```yaml
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
```

**💡 `model` parameter**
- **Purpose**: Specify the Ollama model to actually use
- **Format**: `ollama/<model name>`
- **Your environment**:
- For MS-S1 Max: `ollama/qwen3-coder:30b-a3b-q8_0` (recommended)
- If you have less memory: `ollama/qwen3-coder:14b` or `ollama/qwen3-coder:7b`
- **How ​​to change**: Specify the downloaded model name in Ollama
- **Confirmation command**: Show available models in `ollama list`

```yaml
      api_base: http://localhost:11434
```

**💡 `api_base` parameter**
- **Purpose**: Ollama's API endpoint URL
- **Default**: `http://localhost:11434`
- **Your environment**:
- When running locally: **No changes required**
- When using Ollama on another machine: `http://<IP address>:11434`
- If you have changed the port: `http://localhost:<different port>`

```yaml
      temperature: 0.7
```

**💡 `temperature` parameter (optional/important)**
- **Purpose**: Control the randomness of generated results
- **Range**: 0.0 to 2.0
- **Default**: 0.7
- **Your environment**:
  ```
temperature: 0.1 ← Deterministic (same result every time)
↓ Application: Testing, emphasis on reproducibility
temperature: 0.7 ← Balanced type (recommended)
↓ Usage: Normal coding
temperature: 1.5 ← Creative (various results)
↓ Use: Generating ideas, brainstorming
  ```
- **Recommended changes**:
- Code review: `0.3` (focus on consistency)
- Normal development: `0.7` (as is)
- Idea generation: `1.2` (emphasis on diversity)

```yaml
      max_tokens: 4096
```

**💡 `max_tokens` parameter (optional)**
- **Purpose**: Maximum number of tokens to generate in one response
- **Range**: 1 to 256000 (up to model upper limit)
- **Default**: 4096
- **Your environment**:
  ```
max_tokens: 1024 ← Short response (simple question)
max_tokens: 4096 ← Standard (recommended)
max_tokens: 8192 ← Long response (document generation)
max_tokens: 16384 ← Very long (large scale generation)
  ```
- **Impact**:
- Large value: long response possible but slow
- Small value: Fast but may break midway
- **Recommended**: `4096` (sufficient for most cases)

```yaml
      num_ctx: 262144  # 256K tokens context
```

**💡 `num_ctx` parameter (important/Qwen3 specific)**
- **Purpose**: Context window size that the model can handle
- **Qwen3 Coder 30B Q8_0 upper limit**: 262144 (256K tokens)
- **Your environment**:
  ```yaml
num_ctx: 262144 # ← Qwen3 Coder recommended (256K)
# ✅ Can handle large files
# ✅ Multiple files can be read at the same time
# ⚠️ Increased memory usage (no problem with MS-S1 Max)

num_ctx: 131072 # ← 128K (memory saving)
# ✅ Supports medium-sized files
# ✅ Reduce memory usage

num_ctx: 32768 # ← 32K (lightweight)
# ⚠️ Small files only
# ❌ Not suitable for large code bases
  ```
- **MS-S1 Max users**: **262144 recommended** (memory available)
- **If you need to change**: Make it smaller if you get an out of memory error.

**3. general_settings section (required)**

```yaml
general_settings:
  master_key: sk-1234
```

**💡 `master_key` parameter**
- **Purpose**: Authenticating access to LiteLLM proxy
- **Format**: Any string (usually starts with `sk-`)
- **Your environment**:
- **Local environment**: Simple keys such as `sk-1234` are OK
- **When publishing externally**: Use long random strings
- **Example**: `sk-local-dev-1234`, `sk-test`, `sk-my-secret-key`
- **Important**: Remember this value and use the same value in your client settings (Aider etc.)

**📌 Configuration file template (for copy and paste)**

```yaml
# ~/litellm/config.yaml
# [Minimum configuration] For MS-S1 Max + Qwen3 Coder 30B Q8_0

model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      api_base: http://localhost:11434
temperature: 0.7 # ← Adjust as necessary (0.3~1.2)
max_tokens: 4096 # ← Change to 8192 if you need a long response
num_ctx: 262144 # ← Leave as is for MS-S1 Max

general_settings:
master_key: sk-local-dev-1234 # ← Can be changed to any value
```

**🔧 Customization example to suit your environment**

**Case 1: For machines with less than 64GB of memory**
```yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
model: ollama/qwen3-coder:14b # ← Change to lightweight model
      api_base: http://localhost:11434
      temperature: 0.7
      max_tokens: 4096
num_ctx: 131072 # ← Reduced to 128K
```

**Case 2: When speed is important**
```yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
model: ollama/qwen3-coder:7b # ← High-speed model
      api_base: http://localhost:11434
      temperature: 0.7
max_tokens: 2048 # ← Faster response with shorter response
      num_ctx: 262144
```

**Case 3: Deterministic (emphasis on reproducibility)**
```yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      api_base: http://localhost:11434
temperature: 0.1 # ← Consistent at low temperatures
      max_tokens: 4096
      num_ctx: 262144
```

**❓ Frequently asked questions**

**Q: Do I need to set all parameters? **
A: No. Only three are required: `model_list.model_name`, `litellm_params.model`, and `general_settings.master_key`. Default values ​​are used for the others.

**Q: What happens if I omit temperature or max_tokens? **
A: Default values ​​(temperature: 0.7, max_tokens: unlimited) will be used.

**Q: What happens if I make a mistake in the settings? **
A: An error message will be displayed when LiteLLM starts. If there is a syntax error, it will not start, and if there is a parameter error, an error will occur when making a request.

**Q: Can I change the settings later? **
A: Yes. After editing config.yaml, restart LiteLLM for the changes to take effect.

### 3.3.2 Advanced configuration file (optional)

Configuration with multiple models, caching, and logging enabled:

```yaml
# ~/litellm/config.yaml (full-featured version)

model_list:
# Claude model mapping (highest quality/30B Q8_0)
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      api_base: http://localhost:11434
      temperature: 0.7
      max_tokens: 4096
      num_ctx: 262144  # 256K tokens context

# Mapping of GPT-4 model (also using 30B)
  - model_name: gpt-4
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      api_base: http://localhost:11434
      temperature: 0.7
      num_ctx: 262144

# Lightweight model (high speed/14B)
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: ollama/qwen3-coder:14b
      api_base: http://localhost:11434
      temperature: 0.7
      num_ctx: 262144

# Ultra-light model (fastest/7B)
  - model_name: claude-3-haiku-20240307
    litellm_params:
      model: ollama/qwen3-coder:7b
      api_base: http://localhost:11434
      temperature: 0.7
      num_ctx: 262144

litellm_settings:
# enable debug log
  set_verbose: true

# cache settings
  cache: true
  cache_params:
    type: redis
    host: localhost
    port: 6379
ttl: 3600 # cache for 1 hour

# timeout settings
request_timeout: 600 # 10 minutes

# Streaming settings
  stream: true

general_settings:
# Authentication key
  master_key: sk-local-dev-1234

# Allowed models
  allowed_models:
    - claude-3-5-sonnet-20241022
    - gpt-4
    - gpt-3.5-turbo
    - claude-3-haiku-20240307

# Database (for log storage)
  database_url: sqlite:///litellm.db

# Log settings
  success_callback: ["langfuse"]
  failure_callback: ["langfuse"]

router_settings:
# Load balancing strategy
  routing_strategy: latency-based-routing

# Retry settings
  num_retries: 2
  timeout: 300

# fallback
  fallbacks:
    - claude-3-5-sonnet-20241022: [gpt-4, gpt-3.5-turbo]
```

### 3.3.3 Validating the configuration file

```bash
# Check syntax of configuration file
python3 << EOF
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print("Config loaded successfully!")
    print(f"Number of models: {len(config['model_list'])}")
    for model in config['model_list']:
        print(f"  - {model['model_name']} → {model['litellm_params']['model']}")
EOF
```

**Output example**
```
Config loaded successfully!
Number of models: 4
  - claude-3-5-sonnet-20241022 → ollama/qwen3-coder:30b-a3b-q8_0 (256K context)
  - gpt-4 → ollama/qwen3-coder:30b-a3b-q8_0 (256K context)
  - gpt-3.5-turbo → ollama/qwen3-coder:14b (256K context)
  - claude-3-haiku-20240307 → ollama/qwen3-coder:7b (256K context)
```

## 3.4 Starting LiteLLM Proxy

### 3.4.1 Basic startup method

```bash
# Enable virtual environment (if not already done)
cd ~/litellm
source venv/bin/activate

# start LiteLLM proxy
litellm --config config.yaml --port 8000 --host 0.0.0.0
```

**Startup log example (latest configuration as of November 2025)**
```
INFO: Starting LiteLLM Proxy Server
INFO: Loaded config from config.yaml
INFO: Loaded 4 models
INFO:   - claude-3-5-sonnet-20241022 -> ollama/qwen3-coder:30b-a3b-q8_0 (256K ctx)
INFO:   - gpt-4 -> ollama/qwen3-coder:30b-a3b-q8_0 (256K ctx)
INFO:   - gpt-3.5-turbo -> ollama/qwen3-coder:14b (256K ctx)
INFO:   - claude-3-haiku-20240307 -> ollama/qwen3-coder:7b (256K ctx)
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Proxy server started successfully!
```

### 3.4.2 Background startup

Run the proxy in the background so it doesn't take up your terminal.

**Method 1: Use nohup**

```bash
# start in background
nohup litellm --config config.yaml --port 8000 --host 0.0.0.0 > litellm.log 2>&1 &

# Check process ID
echo $!

# check log
tail -f litellm.log
```

**Method 2: Start as a systemd service**

```bash
# create service file
sudo nano /etc/systemd/system/litellm.service
```

```ini
[Unit]
Description=LiteLLM Proxy Server
After=network.target ollama.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/litellm
Environment="PATH=/home/your_username/litellm/venv/bin:/usr/bin"
ExecStart=/home/your_username/litellm/venv/bin/litellm --config /home/your_username/litellm/config.yaml --port 8000 --host 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Important**: Replace `your_username` with your actual username.

```bash
# enable/start the service
sudo systemctl daemon-reload
sudo systemctl enable litellm
sudo systemctl start litellm

# Check status
sudo systemctl status litellm
```

**Method 3: Use tmux (recommended)**

```bash
# create tmux session
tmux new -s litellm

# Start LiteLLM
litellm --config config.yaml --port 8000 --host 0.0.0.0

# Detach with Ctrl+B → D (background)

# To reconnect later
tmux attach -t litellm
```

### 3.4.3 Confirm startup

**Step 1: Process confirmation**

```bash
# Check LiteLLM process
ps aux | grep litellm

# Example output:
# user  12345  1.5  0.8 456789 102400 ?  Sl  10:30  0:05 /home/user/litellm/venv/bin/python3 /home/user/litellm/venv/bin/litellm ...
```

**Step 2: Verify port**

```bash
# Check if port 8000 is listening
sudo netstat -tuln | grep 8000

# Example output:
# tcp  0  0  0.0.0.0:8000  0.0.0.0:*  LISTEN
```

**Step 3: Endpoint Verification**

```bash
# Health check
curl http://localhost:8000/health

# Example output:
# {"status":"healthy"}
```

**Step 4: Get model list**

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-local-dev-1234"

# Example output:
# {
#   "data": [
#     {"id": "claude-3-5-sonnet-20241022", "object": "model"},
#     {"id": "gpt-4", "object": "model"},
#     {"id": "gpt-3.5-turbo", "object": "model"},
#     {"id": "claude-3-haiku-20240307", "object": "model"}
#   ]
# }
```

## 3.5 Operation test

### 3.5.1 Testing with curl

```bash
# Test Chat Completions API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-dev-1234" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate Fibonacci numbers"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

**Example output when successful**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1705315200,
  "model": "claude-3-5-sonnet-20241022",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here's a Python function to calculate Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    \"\"\"\n    Calculate the nth Fibonacci number.\n    \n    Args:\n        n (int): The position in the Fibonacci sequence\n    \n    Returns:\n        int: The Fibonacci number at position n\n    \"\"\"\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n + 1):\n            a, b = b, a + b\n        return b\n\n# Example usage\nprint(fibonacci(10))  # Output: 55\n```\n\nThis function uses an iterative approach for efficiency."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 142,
    "total_tokens": 157
  }
}
```

### 3.5.2 Testing with Python script

```python
# test_litellm.py
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-local-dev-1234"
}

payload = {
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful coding assistant."
        },
        {
            "role": "user",
            "content": "Explain what a Python decorator is with a simple example."
        }
    ],
    "temperature": 0.7,
    "max_tokens": 1000
}

print("Sending request to LiteLLM proxy...")
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    print("\n=== Response ===")
    print(result['choices'][0]['message']['content'])
    print(f"\n=== Usage ===")
    print(f"Prompt tokens: {result['usage']['prompt_tokens']}")
    print(f"Completion tokens: {result['usage']['completion_tokens']}")
    print(f"Total tokens: {result['usage']['total_tokens']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

**execution**

```bash
python3 test_litellm.py
```

### 3.5.3 Testing streaming

```python
# test_streaming.py
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-local-dev-1234"
}

payload = {
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
        {
            "role": "user",
            "content": "Count from 1 to 10 in English."
        }
    ],
"stream": True # Streaming enabled
}

print("Streaming response:")
print("-" * 50)

response = requests.post(url, headers=headers, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
data_str = line_str[6:] # remove "data: "
            if data_str == '[DONE]':
                break
            try:
                data = json.loads(data_str)
                content = data['choices'][0]['delta'].get('content', '')
                print(content, end='', flush=True)
            except json.JSONDecodeError:
                pass

print("\n" + "-" * 50)
```

## 3.6 Logging and Monitoring

### 3.6.1 Check the log

**Real-time log**

```bash
# For systemd services
sudo journalctl -u litellm -f

# for nohup
tail -f ~/litellm/litellm.log

# For tmux
tmux attach -t litellm
```

**Example of log contents**
```
INFO: Request: POST /v1/chat/completions
INFO: Model: claude-3-5-sonnet-20241022 -> ollama/qwen3-coder:30b-a3b-q8_0
INFO: Context: 256K tokens available
INFO: Prompt tokens: 15, Completion tokens: 142
INFO: Response time: 7.54s (22 tokens/s)
INFO: Status: 200
```

### 3.6.2 Database Log

Specifying `database_url` in the configuration file will save all requests to the database.

```bash
# Check SQLite database
sqlite3 ~/litellm/litellm.db

# Table list
.tables

# show request history
SELECT model, status_code, response_time, created_at
FROM logs
ORDER BY created_at DESC
LIMIT 10;

# end
.quit
```

### 3.6.3 Monitoring Dashboard

LiteLLM has a built-in dashboard.

```bash
# start with dashboard
litellm --config config.yaml --port 8000 --host 0.0.0.0 --ui
```

If you access `http://localhost:8000/ui` in your browser, you will see the following:
- Number of requests
- Response time
- error rate
- Model usage

## 3.7 Troubleshooting

### 3.7.1 Startup error

**Error: "Port 8000 is already in use"**

```bash
# Identify the process using the port
sudo lsof -i :8000

# terminate the process
sudo kill -9 <PID>

# or use another port
litellm --config config.yaml --port 8001
```

**Error: "Could not connect to Ollama"**

```bash
# Check if Ollama is running
sudo systemctl status ollama

# start if not started
sudo systemctl start ollama

# API confirmation
curl http://localhost:11434/api/tags
```

### 3.7.2 Request error

**Error: "Unauthorized"**

```bash
# Check the Authorization header
# Does the Bearer token match the master_key in config.yaml?
```

**Error: "Model not found"**

```bash
# Check if the model name is correct
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-local-dev-1234"

# Check if it exists in model_list of config.yaml
```

## 3.8 Summary

In this chapter, we set up LiteLLM proxy.

**What we accomplished**
✅ Installing LiteLLM
✅ Creating config.yaml
✅ Start proxy server
✅ API operation confirmation (curl, Python)
✅ Streaming test

**Current configuration**
```
Ollama (Port 11434) ← LiteLLM Proxy (Port 8000)
```

**Next steps**
In the next chapter, we will install the Claude Code CLI and connect to the LiteLLM proxy. Now you can use local LLM from Claude Code!

**Verification Checklist**
- [ ] `litellm --version` works
- [ ] config.yaml is created correctly
- [ ] Proxy is running (Port 8000)
- [ ] `/health` endpoint responds
- [ ] Chat completion works with curl
- [ ] Logs are output correctly

Once you have checked everything, move on to Chapter 04!
