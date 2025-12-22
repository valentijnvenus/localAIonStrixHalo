# Chapter 08: Troubleshooting

**📖 Purpose of this chapter**

When an error occurs, it allows you to identify and resolve the cause yourself.

**🎯 Things to consider in this chapter**
```
□ What problems are you currently experiencing?
□ Where is the cause of the problem? (Ollama / LiteLLM / Aider / Settings)
□ Can you solve it yourself or do you need support?
```

**💡 TOP 3 MOST COMMON PROBLEMS**

```
1st place: 🔌 Unable to connect to LiteLLM (50%)
Cause: LiteLLM is not started
Solved: sudo systemctl start litellm

2nd place: 🤖 Model not found (30%)
Cause: Model not installed
Solved: ollama pull qwen3-coder:30b-a3b-q8_0

3rd place: 🐌 Slow response (15%)
Cause: GPU acceleration disabled
Solution: Set ROCm environment variable
```

**🔍Problem isolation flowchart**

```
Error occurred
    ↓
┌───────────────────────────────┐
│ Connection error?                  │
│ (Connection refused, timeout) │
└───────────────────────────────┘
↓ YES → Go to section 8.1 (connection error)
    ↓ NO
┌───────────────────────────────┐
│ Authentication error?                  │
│ (Unauthorized, Invalid key)   │
└───────────────────────────────┘
↓ YES → Go to section 8.1.2 (authentication error)
    ↓ NO
┌───────────────────────────────┐
│ Model error?                │
│ (Model not found)             │
└───────────────────────────────┘
↓ YES → Go to section 8.1.3 (model error)
    ↓ NO
┌───────────────────────────────┐
│ Performance issue?          │
│ (Slow, Out of memory) │
└───────────────────────────────┘
↓ YES → Go to section 8.2 (Performance)
    ↓ NO
┌───────────────────────────────┐
│ Code generation quality problem?        │
└───────────────────────────────┘
↓ YES → Go to section 8.3 (quality problem)
```

---

## 8.1 Common problems and solutions

### 8.1.1 Connection error

**🤔 Does this apply to you? **
```
□ "Connection refused" error occurs
□ Aider says "Could not connect"
□ curl http://localhost:8000/health fails
```
→ Check at least one: **You should read this section**

**Problem 1: "Could not connect to LiteLLM proxy"**

**💡 What's happening? **
Aider is trying to connect to LiteLLM proxy, but LiteLLM is not started.

```bash
# Symptoms
$ aider
Error: Could not connect to http://localhost:8000

# Check the cause
$ curl http://localhost:8000/health
curl: (7) Failed to connect to localhost port 8000: Connection refused
```

**🔧 Solution (Things to try first)**

```bash
# Step 1: Check if LiteLLM is running
$ ps aux | grep litellm
# Nothing is displayed → Not started

# Step 2: Start LiteLLM
$ cd ~/litellm
$ source venv/bin/activate
$ litellm --config config.yaml --port 8000 --host 0.0.0.0

# or as a systemd service
$ sudo systemctl start litellm
$ sudo systemctl status litellm

# Step 3: Verify operation
$ curl http://localhost:8000/health
# OK if {"status": "healthy"} is displayed
```

**✅ Normal startup status**
```bash
$ sudo systemctl status litellm
● litellm.service - LiteLLM Proxy
   Active: active (running) since ...

$ curl http://localhost:8000/health
{"status": "healthy"}
```

**⚠️ If the issue still persists**
```bash
# check if port is in use
$ sudo lsof -i :8000
# If another process is using it, stop that process

# check log
$ sudo journalctl -u litellm -n 50
# Check error message
```

**Problem 2: "Ollama service not responding"**

**💡 What's happening? **
Even if LiteLLM is normal, Ollama behind it is not starting.

```bash
# Symptoms
$ curl http://localhost:11434/api/tags
curl: (7) Failed to connect

# Solution
$ sudo systemctl status ollama
# Active: inactive (dead) → Not activated

$ sudo systemctl start ollama
$ sudo systemctl status ollama
# Active: active (running) → OK

# GPU recognition confirmation
$ rocm-smi
# If GPU 0 is displayed and the usage rate is displayed, it is OK
```

**✅ Normal startup status**
```bash
$ ollama ps
NAME                         ID        SIZE    UNTIL
qwen3-coder:30b-a3b-q8_0     abc123    34 GB   5 minutes from now
```

### 8.1.2 Authentication error

**🤔 Does this apply to you? **
```
□ "Unauthorized" error occurs
□ It says "Invalid API key"
□ Environment variable OPENAI_API_KEY is empty
```
→ Check at least one: **You should read this section**

**Problem: "Unauthorized" or "Invalid API key"**

**💡 What's happening? **
API key is not set or does not match master_key in config.yaml.

```bash
# Check the cause
$ echo $OPENAI_API_KEY
# empty or wrong key

# Solution
$ export OPENAI_API_KEY="sk-local-dev-1234"

# Add to ~/.bashrc (persistent)
$ echo 'export OPENAI_API_KEY="sk-local-dev-1234"' >> ~/.bashrc
$ source ~/.bashrc

# Check if it matches master_key in config.yaml
$ grep master_key ~/litellm/config.yaml
```

**✅ Correct settings**
```bash
# ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-local-dev-1234"

# config.yaml
general_settings:
master_key: "sk-local-dev-1234" # ← Matches
```

### 8.1.3 Model error

**🤔 Does this apply to you? **
```
□ "Model not found" error occurs
□ Model is not displayed in ollama list
□ Model is not pulled
```
→ Check at least one: **You should read this section**

**Problem: "Model not found"**

**💡 What's happening? **
The model required by LiteLLM is not installed in Ollama.

```bash
# Symptoms
Error: Model 'qwen3-coder:30b-a3b-q8_0' not found

# Check the cause
$ ollama list
# qwen3-coder:30b-a3b-q8_0 is not displayed

# Solution
$ ollama pull qwen3-coder:30b-a3b-q8_0

# Check LiteLLM config.yaml
$ cat ~/litellm/config.yaml | grep qwen3-coder
```

**Problem: "Context length exceeded"**

**💡 What's happening? **
The code you are trying to process exceeds the model's context window size.

**🤔 Does this apply to you? **
```
□ You are trying to process a large file (more than 1000 lines)
□ /adding many files at the same time
□ Long conversation history
```

```bash
# Symptoms
Error: This model's maximum context length is 262144 tokens

# 💡 Things to try first (in order of priority)

# Method 1: Exclude unnecessary files
> /drop unnecessary_file.py
> /drop old_code.py

# Method 2: Clear conversation history
> /clear

# Method 3: Reduce map-tokens
$ nano ~/.aider.conf.yml
map-tokens: 4096 # reduced from 8192
max-chat-history-tokens: 8192 # reduced from 16384

# Method 4: Split the file and process it
> /add module_part1.py
# After processing is completed
> /drop module_part1.py
> /add module_part2.py
```

**✅ Precautions**
```bash
# Adjust settings from the beginning for large projects
# ~/.aider.conf.yml
map-tokens: 4096 # Default: 8192
max-chat-history-tokens: 8192 # Default: 16384
```

---

## 8.2 Performance issues

### 8.2.1 Slow response

**🤔 Does this apply to you? **
```
□ It takes more than 10 seconds to respond
□ Low GPU usage (less than 50%)
□ GPU is displayed as 0% in rocm-smi
```
→ Check at least one: **You should read this section**

**Problem: Response takes more than 30 seconds**

**💡 What's happening? **
GPU acceleration is disabled and CPU processing is used.

**diagnosis**

```bash
# Check GPU usage
$ rocm-smi

# Example output (with problems):
# GPU  Temp   AvgPwr  SCLK     MCLK     Fan   Perf  PwrCap  VRAM%  GPU%
# 0    45.0c  15.0W   800Mhz   1000Mhz  0%    auto  120.0W  15%    0%
# ↑ 0% is a problem
```

**Solution**

```bash
# Check ROCm environment variables
$ sudo systemctl show ollama | grep Environment

# If not configured correctly
$ sudo nano /etc/systemd/system/ollama.service.d/override.conf

[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="PYTORCH_ROCM_ARCH=gfx1100"
Environment="GPU_MAX_ALLOC_PERCENT=95"

$ sudo systemctl daemon-reload
$ sudo systemctl restart ollama

# Recheck GPU usage
$ rocm-smi
# OK if GPU% is 80-100%
```

**✅ Normal performance (MS-S1 Max)**
```bash
$ rocm-smi
GPU  Temp  AvgPwr  SCLK    MCLK    Fan  Perf  VRAM%  GPU%
0    65c   80W     2600Mhz 2000Mhz 60%  auto  35%    85%
↑ 80-100% normal
```

**⚠️ Expected response time**
```
Simple task (add comment): 2-4 seconds
Normal task (function modification): 3-8 seconds
Complex task (refactoring): 8-15 seconds

→ If it is slower than this, check the settings.
```

### 8.2.2 Out of memory

**🤔 Does this apply to you? **
```
□ "Out of memory" error occurs
□ Memory usage is over 90%
□ Using multiple large models at the same time
```
→ Check at least one: **You should read this section**

**Problem: "Out of memory" error**

**💡 What's happening? **
The model or context is running out of memory.

```bash
# Check memory usage
$ free -h
              total        used        free
Mem:          128Gi        95Gi        33Gi

# Check memory usage of Ollama
$ ollama ps
NAME                         ID              SIZE      UNTIL
qwen3-coder:30b-a3b-q8_0     abc123          34 GB     5 minutes
qwen3-coder:14b              def456          20 GB     5 minutes
# Total 54GB in use (MS-S1 Max has plenty of space with 96GB VRAM + 128GB total memory)
```

**Solution**

```bash
# Method 1: Unload unnecessary models (usually unnecessary for MS-S1 Max)
$ ollama stop qwen3-coder:14b # unload lightweight model

# Method 2: Shorten KEEP_ALIVE
$ sudo nano /etc/systemd/system/ollama.service.d/override.conf
Environment="OLLAMA_KEEP_ALIVE=2m" # Shortened from 5m

$ sudo systemctl daemon-reload
$ sudo systemctl restart ollama

# Method 3: Use a smaller model (usually not needed for MS-S1 Max)
> /model gpt-3.5-turbo  # qwen3-coder:14b (20GB)
> /model claude-3-haiku-20240307 # qwen3-coder:7b (10GB, lightest)
```

### 8.2.3 Disk I/O issues

**Problem: Model loading is slow**

```bash
# check disk speed
$ sudo hdparm -Tt /dev/nvme0n1

# Move if it is not an SSD or a USB drive
$ sudo mv ~/.ollama /mnt/nvme/ollama
$ sudo ln -s /mnt/nvme/ollama ~/.ollama
$ sudo systemctl restart ollama
```

**✅ Normal memory usage on MS-S1 Max**
```bash
$ free -h
              total        used        free
Mem:          128Gi        54Gi        74Gi
# ← It is normal to use around 50GB (qwen3-coder:30b-a3b-q8_0 is 34GB)
```

**⚠️ Note**: The MS-S1 Max has 128GB RAM and 96GB VRAM allocation, so you won't normally run out of memory. A lightweight model is recommended for machines with 64GB or less.

---

## 8.3 Code generation quality issues

### 8.3.1 Generated code is different than expected

**🤔 Does this apply to you? **
```
□ Code different from the instructions is generated
□ Prompt is too vague
□ Different results are returned each time
```
→ Check at least one: **You should read this section**

**Problem: Generates code that does not follow prompts**

**💡 What's happening? **
The prompt is ambiguous and the LLM does not understand the intent.

**Bad example**

```
> make it better
```

**Good example**

```
> Improve this function by:
> 1. Adding type hints to all parameters and return value
> 2. Adding comprehensive docstring with Args, Returns, and Examples
> 3. Adding input validation for edge cases (None, empty list, negative numbers)
> 4. Using more descriptive variable names
> 5. Optimizing the algorithm for better time complexity
```

### 8.3.2 Inconsistent output

**Problem: Same question, different answer every time**

**Solution: Adjust Temperature**

```yaml
# config.yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      num_ctx: 262144  # 256K context
temperature: 0.3 # reduced from 0.7 (more deterministic)
```

or

```
> /model claude-3-5-sonnet-20241022
# Specify temperature with Aider
aider --model claude-3-5-sonnet-20241022 --temperature 0.2
```

### 8.3.3 Code doesn't work

**Problem: There is a bug in the generated code**

**Debugging steps**

```bash
# Step 1: Run the code
> /run python3 generated_code.py

# Step 2: Feedback error message
> The code fails with this error:
> Traceback (most recent call last):
>   File "generated_code.py", line 10, in <module>
>     result = process_data(None)
>   File "generated_code.py", line 5, in process_data
>     return data.split(',')
> AttributeError: 'NoneType' object has no attribute 'split'
>
> Fix this bug by adding proper input validation

# Step 3: Request a review
> Review the fixed code and suggest additional improvements
```

## 8.4 Git integration issues

### 8.4.1 Commit error

**Problem: "Git user not configured"**

```bash
$ git config user.name
# nothing is displayed

# Solution
$ git config --global user.name "Your Name"
$ git config --global user.email "[email protected]"
```

**Problem: "Uncommitted changes"**

```bash
# Aider doesn't commit changes

# Check the cause
$ cat ~/.aider.conf.yml | grep commit
auto-commits: false # disabled

# Solution
$ nano ~/.aider.conf.yml
auto-commits: true
dirty-commits: true

# or commit manually
> /commit
Commit message: Add error handling
```

### 8.4.2 Handling large files

**Problem: "File too large"**

```bash
# Error when trying to add large files (>10MB)

# Solution 1: Add to .gitignore
$ echo "large_data.csv" >> .gitignore
$ git add .gitignore
$ git commit -m "Ignore large data files"

# Solution 2: Use Git LFS
$ git lfs install
$ git lfs track "*.csv"
$ git add .gitattributes
$ git commit -m "Track CSV files with LFS"
```

## 8.5 Diagnostic Tools

### 8.5.1 Comprehensive Health Check

```bash
#!/bin/bash
# health_check.sh

echo "=== System Health Check ==="

# 1. Ollama
echo -e "\n[1] Ollama Status:"
systemctl is-active ollama && echo "✓ Running" || echo "✗ Not running"
curl -s http://localhost:11434/api/tags > /dev/null && echo "✓ API responding" || echo "✗ API not responding"

# 2. LiteLLM
echo -e "\n[2] LiteLLM Status:"
curl -s http://localhost:8000/health > /dev/null && echo "✓ Running" || echo "✗ Not running"

# 3. GPU
echo -e "\n[3] GPU Status:"
rocm-smi --showuse | grep "GPU use" || echo "✗ ROCm not available"

# 4. Memory
echo -e "\n[4] Memory:"
free -h | grep Mem

# 5. Models
echo -e "\n[5] Loaded Models:"
ollama ps

# 6. Disk Space
echo -e "\n[6] Disk Space:"
df -h ~ | tail -1

echo -e "\n=== Health Check Complete ==="
```

execution:

```bash
$ chmod +x health_check.sh
$ ./health_check.sh
```

### 8.5.2 Detailed log collection

```bash
#!/bin/bash
# collect_logs.sh

LOG_DIR="./debug_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Collecting logs to $LOG_DIR..."

# Ollama logs
sudo journalctl -u ollama -n 100 > "$LOG_DIR/ollama.log"

# LiteLLM logs
if [ -f ~/litellm/litellm.log ]; then
    cp ~/litellm/litellm.log "$LOG_DIR/"
fi

# System info
uname -a > "$LOG_DIR/system_info.txt"
free -h >> "$LOG_DIR/system_info.txt"
df -h >> "$LOG_DIR/system_info.txt"

# ROCm info
rocm-smi > "$LOG_DIR/rocm_info.txt" 2>&1

# Ollama models
ollama list > "$LOG_DIR/ollama_models.txt"

# LiteLLM config
cp ~/litellm/config.yaml "$LOG_DIR/" 2>/dev/null

# Aider config
cp ~/.aider.conf.yml "$LOG_DIR/" 2>/dev/null

echo "Logs collected in $LOG_DIR"
echo "You can share this directory for troubleshooting"
```

## 8.6 Performance Benchmark

### 8.6.1 End-to-end benchmark

```python
# e2e_benchmark.py
import time
import requests

def benchmark_e2e():
"""End-to-end performance testing"""
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-local-dev-1234",
        "Content-Type": "application/json"
    }

    test_cases = [
        "Write a function to reverse a string",
        "Explain what a Python decorator is",
        "Fix this bug: def f(l): return l[10]"
    ]

    results = []

    for i, prompt in enumerate(test_cases, 1):
        print(f"Test {i}/3: {prompt[:50]}...")

        start = time.time()
        response = requests.post(url, headers=headers, json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": prompt}]
        })
        end = time.time()

        if response.status_code == 200:
            data = response.json()
            results.append({
                "prompt": prompt,
                "time": end - start,
                "tokens": data['usage']['total_tokens']
            })
            print(f"  ✓ {end - start:.2f}s")
        else:
            print(f"  ✗ Error: {response.status_code}")

    # Summary
    print(f"\n=== Summary ===")
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average response time: {avg_time:.2f}s")
    print(f"Total tokens: {sum(r['tokens'] for r in results)}")

if __name__ == "__main__":
    benchmark_e2e()
```

### 8.6.2 Expected Results (MS-S1 Max)

```
Test 1/3: Write a function to reverse a string...
  ✓ 3.42s
Test 2/3: Explain what a Python decorator is...
  ✓ 5.78s
Test 3/3: Fix this bug: def f(l): return l[10]...
  ✓ 4.21s

=== Summary ===
Average response time: 4.47s
Total tokens: 387
```

**Normal range**: 3-8 seconds
**Requires investigation**: 10 seconds or more

## 8.7 Support Resources

### 8.7.1 Community

- **Ollama GitHub**: https://github.com/ollama/ollama
- **LiteLLM GitHub**: https://github.com/BerriAI/litellm
- **Aider GitHub**: https://github.com/paul-gauthier/aider

### 8.7.2 Providing debug information

Information to include when reporting an issue:

```bash
# System information
uname -a
lsb_release -a

# Version information
ollama --version
litellm --version
aider --version

# GPU information
rocm-smi
rocminfo | grep "Name:"

# log
sudo journalctl -u ollama -n 50
tail -50 ~/litellm/litellm.log

# setting
cat ~/litellm/config.yaml
cat ~/.aider.conf.yml
```

---

## 8.8 Summary

In this chapter, you learned how to troubleshoot common problems.

**Important checkpoints**
✅ Is the service running?
✅ Are the environment variables set correctly?
✅ Is your GPU recognized?
✅ Is there enough memory?
✅ Are the prompts clear?

**Basic steps for debugging**
```
1. Check the error message
→ Identify which component has the error

2. Check the log
   → journalctl -u ollama / journalctl -u litellm

3. Check settings
→ config.yaml, .aider.conf.yml, environment variables

4. Restart the service
   → systemctl restart ollama litellm

5. Verification with benchmarks
   → rocm-smi, free -h, ollama ps
```

**❓ Frequently Asked Questions (FAQ)**

**Q: I checked all the checks but the problem persists**
A: Please run `health_check.sh` and `collect_logs.sh` to collect the logs and report them in a GitHub Issue.

**Q: I don't understand the error message in English**
A: Paste the error message into Aider/LLM and ask "What does this error mean and how can I resolve it?"

**Q: Which log should I check? **
A:
- Ollama: `sudo journalctl -u ollama -n 50`
- LiteLLM: `tail -50 ~/litellm/litellm.log`
- Aider: Terminal output

**Q: What if a problem occurs in the production environment? **
A:
1. First restart the service
2. Collect logs
3. Restore from backup (see Chapter 09)

**Q: Performance suddenly worsened**
A:
- GPU driver (ROCm) may have been updated
- Double check environment variables
- Reinstall Ollama

**💡Troubleshooting Tips**
1. **Dividing without haste**: Use the flowchart (at the beginning of this chapter)
2. **Habit of reading logs**: Errors are always logged.
3. **Gradual Revert**: Revert the last changed settings
4. **Keep records**: Make a note of the problem and solution (use it next time)
5. **Leverage the community**: Ask a question in a GitHub Issue

**📊 Average time to resolve issues**
```
Connection error: 2-5 minutes (service startup)
Authentication error: 1-3 minutes (environment variable settings)
Model error: 5-15 minutes (model download)
Performance issues: 10-30 minutes (configuration adjustment)
Quality issue: 3-10 minutes (prompt improvement)
```

**Next steps**
In the final chapter, you will learn advanced configuration and best practices.

**Verification Checklist**
- [ ] health_check.sh can be executed
- [ ] Can collect logs
- [ ] Can measure performance
- [ ] Can solve basic problems
- [ ] Understand the problem isolation flowchart

Once you have checked everything, move on to Chapter 09!
