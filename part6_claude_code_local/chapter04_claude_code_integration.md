# Chapter 04: Claude Code Integration

## 4.1 Installing Claude Code CLI

### 4.1.1 Installing Node.js

Claude Code CLI is a Node.js tool. First, install Node.js.

**Step 1: Check Node.js version**

```bash
# Check if Node.js is installed
node --version

# Check npm (package manager)
npm --version
```

**Requires Node.js 18 or higher. ** If it is not installed or is outdated, install it using the steps below.

**Step 2: Install via NodeSource (recommended)**

```bash
# Install Node.js 20 LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Confirm installation
node --version  # v20.x.x
npm --version   # 10.x.x
```

### 4.1.2 Installing Claude Code CLI

**Important Note**:
As of now (January 2025), the official Claude Code CLI has not been released yet. This book explains **Using Claude Code with custom settings** or **How ​​to use similar tools** based on provided Zenn and Medium articles.

**Method 1: Use Cursor/Continue.dev (recommended)**

The following tools are available as Claude Code-like AI coding assistants:

```bash
# Install Continue.dev (VSCode extension)
# Open VSCode, search for "Continue" from extensions and install it

# Or download the Cursor editor
wget https://downloader.cursor.sh/linux/appImage/x64 -O cursor.AppImage
chmod +x cursor.AppImage
sudo mv cursor.AppImage /usr/local/bin/cursor
```

**Method 2: Using OpenAI compatible CLI tools**

```bash
# Directly use LiteLLM's CLI mode
pip install openai

# Locally with Python script

Use LLM
```

### 4.1.3 Alternative method: Aider (AI pair programming tool)

We recommend using **Aider**, which has similar functionality to Claude Code.

```bash
# Install Aider
pip install aider-chat

# Confirm installation
aider --version

# Example output: aider 0.60.0
```

**Aider Features**
- Git integration
- File editing function
- Compatible with OpenAI API
- Supports local LLM
- Multiple file operations
- Context management

## 4.2 Aider configuration (Claude Code alternative)

### 4.2.1 Setting environment variables

Connect Aider to local LLM via LiteLLM proxy.

```bash
# Add to ~/.bashrc or ~/.zshrc
nano ~/.bashrc
```

Add the following:

```bash
# LiteLLM API settings
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="sk-local-dev-1234" # master_key of LiteLLM
export OPENAI_API_MODEL="claude-3-5-sonnet-20241022" # Model to use
```

Reflect settings:

```bash
source ~/.bashrc

# confirmation
echo $OPENAI_API_BASE
echo $OPENAI_API_KEY
echo $OPENAI_API_MODEL
```

### 4.2.2 Creating Aider configuration file

**[Required] Create a configuration file exclusively for Aider. **

```bash
# create configuration file in home directory
nano ~/.aider.conf.yml
```

**Basic settings (for copy and paste)**

```yaml
# ~/.aider.conf.yml
# [Basic configuration] For MS-S1 Max + Qwen3 Coder 30B Q8_0

# API endpoint (required)
openai-api-base: http://localhost:8000/v1
openai-api-key: sk-local-dev-1234

# Model settings (required)
model: claude-3-5-sonnet-20241022

# Context settings (important)
map-tokens: 4096
max-chat-history-tokens: 8192

# Function settings (optional)
auto-commits: true
dirty-commits: true
git: true
stream: true

# Editor settings (optional)
editor: nano

# Log settings (optional)
verbose: false
show-diffs: true
```

**📖 Detailed explanation of each setting item**

**1. API connection settings (required)**

```yaml
openai-api-base: http://localhost:8000/v1
```

**💡 What is `openai-api-base`? **
- **Purpose**: API endpoint for LiteLLM proxy
- **Default**: `http://localhost:8000/v1`
- **Your environment**:
- **Local**: Use as is (no modification required)
- **LiteLLM on another machine**: `http://<IP address>:8000/v1`
- **When changing port**: `http://localhost:<different port>/v1`
- **Important**: Don't forget the `/v1` at the end

```yaml
openai-api-key: sk-local-dev-1234
```

**💡 What is `openai-api-key`? **
- **Purpose**: LiteLLM authentication key
- **Your environment**: Use **same value** as `master_key` set in LiteLLM's `config.yaml`
- **example**:
- If `master_key: sk-1234` is used in LiteLLM, this is also `sk-1234`
- If `master_key: sk-local-dev-1234` is used in LiteLLM, `sk-local-dev-1234` is also used here.
- **How ​​to check**: `cat ~/litellm/config.yaml | grep master_key`

**2. Model settings (required)**

```yaml
model: claude-3-5-sonnet-20241022
```

**💡 `model` parameter**
- **Purpose**: Specify the model to use
- **Default**: `claude-3-5-sonnet-20241022`
- **Your environment**:
- **MS-S1 Max (recommended)**: `claude-3-5-sonnet-20241022` (maps to Qwen3 30B Q8_0)
- **Speed-oriented**: `gpt-3.5-turbo` (mapped to Qwen3 14B)
- **Fastest**: `claude-3-haiku-20240307` (maps to Qwen3 7B)
- **Correspondence**: Use the model name defined in LiteLLM's `config.yaml`

```
[Guidelines for model selection]
claude-3-5-sonnet-20241022 ← Highest quality (22 tokens/s)
↓ Use: Production code, review
gpt-3.5-turbo ← Fast (28 tokens/s)
↓ Usage: Normal development, experimentation
claude-3-haiku-20240307 ← Fastest (42 tokens/s)
Uses: quick questions, learning
```

**3. Context settings (important/affects performance)**

```yaml
map-tokens: 4096
```

**💡 `map-tokens` parameter**
- **Purpose**: Number of tokens used for code base map (file list/structure)
- **Default**: 1024
- **Your environment**:
  ```yaml
map-tokens: 2048 # ← Small project (less than 10 files)
map-tokens: 4096 # ← Recommended/medium size (10-50 files)
map-tokens: 8192 # ← Large project (more than 50 files)
map-tokens: 0 # ← Disable map (lightest)
  ```
- **Impact**:
- Large: Easy to understand the entire project, but increases memory consumption
- Small: Saves memory but makes it difficult to understand the whole picture.
- **MS-S1 Max recommended**: `8192` (with plenty of room)

```yaml
max-chat-history-tokens: 8192
```

**💡 `max-chat-history-tokens` parameter**
- **Purpose**: Number of tokens used for conversation history
- **Default**: 2048
- **Your environment**:
  ```yaml
max-chat-history-tokens: 4096 # ← Short-term conversation (5-10 round trips)
max-chat-history-tokens: 8192 # ← Recommended/Normal (10-20 round trips)
max-chat-history-tokens: 16384 # ← Long-term conversation (more than 20 round trips)
max-chat-history-tokens: 32768 # ← Very long (for MS-S1 Max)
  ```
- **Impact**:
- Large: Preserves context for long conversations, increases memory consumption
- Small: saves memory but makes it easy to forget past conversations
- **MS-S1 Max recommended**: `16384` (utilizes 256K context)

**4. Git integration settings (optional/convenient)**

```yaml
auto-commits: true
```

**💡 `auto-commits` parameter**
- **Purpose**: Automatically commit Git when changing code
- **Default**: false
- **Your environment**:
- `true`: **Recommended**・Changes are automatically recorded and history management is easy.
- `false`: Manual commit, control timing yourself
- **Advantages**: Change history is automatically maintained, rollbacks are easy
- **Disadvantages**: More commits (more git logs)

```yaml
dirty-commits: true
```

**💡 `dirty-commits` parameter**
- **Purpose**: Can commit even if there are uncommitted changes
- **Default**: false
- **Your environment**:
- `true`: **Recommended**・Flexible work possible
- `false`: Commit only clean state (strict)
- **Recommended**: If `auto-commits: true`, also enable `dirty-commits: true`

```yaml
git: true
```

**💡 `git` parameters**
- **Purpose**: Enable Git integration functionality
- **Default**: true
- **Your environment**:
- `true`: **Recommended**・Use Git functionality
- `false`: Git not used (if not a Git repository)
- **Prerequisite**: The project has been `git init`

**5. User experience settings (optional)**

```yaml
stream: true
```

**💡 `stream` parameter**
- **Purpose**: Real-time display of responses
- **Default**: true
- **Your environment**:
- `true`: **Recommended**・Typewriter-like display, waiting time feels shorter
- `false`: Batch display after complete generation
- **Recommended**: `true` (improves user experience)

```yaml
editor: nano
```

**💡 `editor` parameter**
- **Purpose**: Editor used with the `/editor` command
- **Default**: System default
- **Your environment**:
- `nano`: For beginners (easy)
- `vim`: For Vim users
- `code`: For VSCode users
- `emacs`: For Emacs users
- **How ​​to change**: Specify your favorite editor name

```yaml
verbose: false
show-diffs: true
```

**💡 `verbose` / `show-diffs` parameters**
- **verbose**: Display debug information
- `false`: **Recommended**/Normal use
- `true`: Only for troubleshooting
- **show-diffs**: Show the difference before and after the change
- `true`: **Recommended**・Easy to check what has changed
- `false`: Hide differences (simple)

**🔧 Customization examples by environment**

**Case 1: Optimization for MS-S1 Max (recommended)**

```yaml
# Settings that take advantage of large memory
openai-api-base: http://localhost:8000/v1
openai-api-key: sk-local-dev-1234
model: claude-3-5-sonnet-20241022

# Increase context
map-tokens: 8192 # ← Supports large-scale projects
max-chat-history-tokens: 16384 # ← Supports long conversations

# Git automation
auto-commits: true
dirty-commits: true
git: true
stream: true
editor: nano
verbose: false
show-diffs: true
```

**Case 2: Memory saving type (64GB or less)**

```yaml
# Lightweight settings
openai-api-base: http://localhost:8000/v1
openai-api-key: sk-local-dev-1234
model: gpt-3.5-turbo # ← Lightweight model

# Reduce context
map-tokens: 2048 # ← Small-scale support
max-chat-history-tokens: 4096 # ← Short-term conversation

# Basic functions only
auto-commits: true
git: true
stream: true
```

**Case 3: Prioritize speed**

```yaml
# Fast setting
openai-api-base: http://localhost:8000/v1
openai-api-key: sk-local-dev-1234
model: claude-3-haiku-20240307 # ← Fastest model

# context minimum
map-tokens: 0 # ← Disable map
max-chat-history-tokens: 2048 # ← minimum

auto-commits: false # ← Speed ​​up with manual control
git: true
stream: true
```

**❓ Frequently asked questions**

**Q: Do I need to write all settings? **
A: No. Only three are required: `openai-api-base`, `openai-api-key`, and `model`. If other values ​​are omitted, default values ​​will be used.

**Q: What happens if I make a mistake in the settings? **
A: An error message will be displayed when starting Aider. Please be careful of typos.

**Q: Can I change the settings later? **
A: Yes. Edit `~/.aider.conf.yml` and restart Aider for it to take effect.

**Q: Can it be overwritten with options at startup? **
A: Yes. Example: You can temporarily change it with `aider --model gpt-3.5-turbo --map-tokens 2048`.

**Q: I can't decide which settings I need**
A: **For MS-S1 Max, please copy and paste the above "Case 1: Optimization" as is. **For other environments, please try the basic settings first and then adjust.

### 4.2.3 Operation confirmation

**Step 1: Launch Aider**

```bash
# move to project directory
mkdir -p ~/test-project
cd ~/test-project

# Initialize Git repository
git init
git config user.name "Your Name"
git config user.email "[email protected]"

# start Aider
aider
```

**Example log at startup**
```
Aider v0.60.0
Model: claude-3-5-sonnet-20241022 using http://localhost:8000/v1
Git repo: /home/user/test-project
Repository map: disabled (no files)

Use /help to see available commands.

>
```

**Step 2: Quick test**

```
> /add test.py

> Create a Python script that prints "Hello, Local LLM!"

# Aider generates code
# test.py is created
```

**Step 3: Confirm**

```bash
# Check the generated file
cat test.py

# execution
python3 test.py

# Output: Hello, Local LLM!
```

## 4.3 Detailed usage instructions

### 4.3.1 File operations

**Add file to context**

```
> /add main.py utils.py

# Add multiple files at once
> /add src/*.py

# add entire directory
> /add src/
```

**Remove file from context**

```
> /drop utils.py
```

**Check current context**

```
> /ls

# Example output:
# Files in chat:
#   main.py (125 lines)
#   utils.py (45 lines)
```

### 4.3.2 Code generation/editing

**Create new file**

```
> Create a FastAPI server with a /hello endpoint that returns "Hello, World!"

# Aider generates main.py
```

**Edit existing file**

```
> /add main.py

> Add error handling to the /hello endpoint

# Aider edits main.py
```

**Check the difference**

```
> /diff

# or displayed automatically (show-diffs: true)
```

### 4.3.3 Git Integration

**Commit changes**

```
> /commit

# enter message
Commit message: Add error handling to hello endpoint
```

**Enable autocommit**

```yaml
# Already configured in ~/.aider.conf.yml
auto-commits: true
dirty-commits: true
```

Once enabled, Aider will automatically commit any changes you make.

**Check commit history**

```bash
git log --oneline

# Example output:
# a1b2c3d Add error handling to hello endpoint
# d4e5f6g Create initial FastAPI server
```

### 4.3.4 Advanced Features

**Batch editing of multiple files**

```
> /add src/*.py

> Refactor all functions to use type hints

# Add type hints to all Python files
```

**Command execution**

```
> /run python3 main.py

# Run the script and check the results
```

**Code review**

```
> /add main.py

> Review this code and suggest improvements

# Aider provides code review
```

**Bug fix**

```
> /add buggy_function.py

> This function crashes when input is empty. Fix the bug.

# Aider identifies and fixes the bug
```

### 4.3.5 Model switching

You can use multiple models.

```
# Switch to lightweight model (faster)
> /model gpt-3.5-turbo

# Switch to high quality model
> /model claude-3-5-sonnet-20241022

# or specified at startup
aider --model gpt-3.5-turbo
```

**Model mapping defined in config.yaml (latest November 2025)**
- `gpt-3.5-turbo` → `qwen3-coder:14b` (high speed/256K context)
- `claude-3-5-sonnet-20241022` → `qwen3-coder:30b-a3b-q8_0` (highest quality/256K context)
- `gpt-4` → `qwen3-coder:30b-a3b-q8_0` (highest quality/256K context)
- `claude-3-haiku-20240307` → `qwen3-coder:7b` (fastest/256K context)

## 4.4 Practical examples

### 4.4.1 Web Application Restructure

**Scenario**: Refactoring an existing Flask app to FastAPI

```bash
# move to project directory
cd ~/my-flask-app

# start Aider
aider

# add file to context
> /add app.py

# Refactor request
> Convert this Flask application to FastAPI. Keep the same endpoints and functionality.

# Aider converts the code
# Check the difference
> /diff

# commit
> /commit
# Commit message: Refactor Flask to FastAPI
```

**Example generated code**

**Before (Flask):**
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/hello')
def hello():
    return jsonify({"message": "Hello, World!"})

if __name__ == '__main__':
    app.run(debug=True)
```

**After (FastAPI):**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HelloResponse(BaseModel):
    message: str

@app.get('/hello', response_model=HelloResponse)
def hello():
    return {"message": "Hello, World!"}

# Run with: uvicorn main:app --reload
```

### 4.4.2 Test generation

```bash
> /add main.py

> Generate pytest tests for all functions in this file

# Aider generates test_main.py
```

**Generated test example**

```python
# test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_hello_endpoint():
    response = client.get('/hello')
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

def test_hello_response_schema():
    response = client.get('/hello')
    assert "message" in response.json()
    assert isinstance(response.json()["message"], str)
```

**Test run**

```bash
> /run pytest test_main.py -v

# Aider displays execution results
```

### 4.4.3 Document generation

```bash
> /add src/*.py

> Generate a README.md file that documents all modules, functions, and their usage

# README.md is generated
```

## 4.5 Performance optimization

### 4.5.1 Adjusting the context size

Take advantage of MS-S1 Max's large memory capacity to use large contexts.

```yaml
# ~/.aider.conf.yml

# Default (standard)
map-tokens: 4096
max-chat-history-tokens: 8192

# Optimized for MS-S1 Max (large capacity)
map-tokens: 8192
max-chat-history-tokens: 16384

# Note: Larger contexts will increase response time
```

### 4.5.2 Utilizing streaming

```yaml
# ~/.aider.conf.yml
stream: true # Display response in real time
```

Enabling streaming allows you to see the code being generated in real time.

### 4.5.3 Utilizing cache

Enabling Redis caching on the LiteLLM side will speed up the response to the same question.

```bash
# Install Redis (if not already done)
sudo apt install redis-server -y

# Start Redis
sudo systemctl start redis-server

# Enable caching in config.yaml (see Chapter 03)
```

**Benefits of caching**
- Same code review: Instant response
- Iterative refactoring: faster
- Document generation: It takes time only the first time.

## 4.6 Troubleshooting

### 4.6.1 Connection error

**Error: "Could not connect to API endpoint"**

```bash
# Check if LiteLLM proxy is running
curl http://localhost:8000/health

# start if not started
cd ~/litellm
source venv/bin/activate
litellm --config config.yaml
```

**Error: "Unauthorized"**

```bash
# Check API key
echo $OPENAI_API_KEY

# Check if it matches master_key in config.yaml
```

### 4.6.2 Model error

**Error: "Model not found"**

```bash
# Check the model list
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-local-dev-1234"

# Check if the model name is correct
echo $OPENAI_API_MODEL

# Check the model: setting in ~/.aider.conf.yml
```

### 4.6.3 Performance issues

**Slow response**

```bash
# Check the status of Ollama
ollama ps

# Check if GPU is used
rocm-smi

# Check ROCm environment variable if GPU usage is 0%
sudo systemctl status ollama
```

**Out of memory**

```bash
# Check system memory
free -h

# Switch to a lighter model (usually not needed with MS-S1 Max)
> /model gpt-3.5-turbo  # qwen3-coder:14b (256K context)
> /model claude-3-haiku-20240307 # qwen3-coder:7b (lightest)
```

## 4.7 Continue.dev (VSCode extension) settings

As an alternative to Claude Code, we also recommend the **Continue** extension for VSCode.

### 4.7.1 Installing Continue

```bash
# start VSCode
code

# Search for "Continue" in extensions and install it
# or
code --install-extension continue.continue
```

### 4.7.2 Continue settings

Ctrl+Shift+P → "Continue: Open config.json"

```json
{
  "models": [
    {
      "title": "Qwen3 Coder 30B Q8_0 (Best - 256K ctx)",
      "provider": "openai",
      "model": "claude-3-5-sonnet-20241022",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "sk-local-dev-1234"
    },
    {
      "title": "Qwen3 Coder 14B (Fast - 256K ctx)",
      "provider": "openai",
      "model": "gpt-3.5-turbo",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "sk-local-dev-1234"
    },
    {
      "title": "Qwen3 Coder 7B (Fastest - 256K ctx)",
      "provider": "openai",
      "model": "claude-3-haiku-20240307",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "sk-local-dev-1234"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Tab Autocomplete",
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "sk-local-dev-1234"
  },
  "embeddingsProvider": {
    "provider": "ollama",
    "model": "mxbai-embed-large",
    "apiBase": "http://localhost:11434"
  }
}
```

### 4.7.3 Using Continue

**1. Chat function**
- Ctrl+L: Open chat
- Select code → Ctrl+L: Ask about selection

**2. Inline editing**
- Ctrl+I: Inline editing
- Type "Add error handling to this function"

**3. Tab completion**
- Automatic suggestions when you start writing code
- Tab: Accept suggestion

## 4.8 Summary

In this chapter, you learned how to use local LLM with Claude Code-like tools.

**What we accomplished**
✅ Install and configure Aider
✅ Connect to LiteLLM proxy
✅ Basic code generation/editing
✅ Git integration
✅ Continue.dev (VSCode) settings

**What we can achieve**
- AI pair programming
- Code review
- Refactoring
- Test generation
- Document creation
- Bug fixes

**Advantages with MS-S1 Max (latest November 2025)**
- 128GB memory (96GB VRAM allocable) → large context (256K tokens)
- Radeon 8060S (40 RDNA 3.5 CU) → Fast inference (22 tokens/s) 
- Qwen3 Coder 30B Q8_0 → Highest quality code generation
- Fully local → Privacy protection
- Cost $0 → Unlimited usage

**Next steps**
The next chapter details the compatibility and limitations of local LLM and cloud LLM. Learn what you can and cannot do, and how to avoid it.

**Verification Checklist**
- [ ] Aider starts
- [ ] Environment variables are set correctly
- [ ] Can connect to local LLM
- [ ] Code generation works
- [ ] Git integration works
- [ ] Continue.dev (optional) is set

Once you have checked everything, move on to Chapter 05!
