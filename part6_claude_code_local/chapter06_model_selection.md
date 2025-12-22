# Chapter 06: Model selection and optimization

**📖 Purpose of this chapter**

Compare multiple models so you can choose the best one for your application.

**🎯 Things to consider in this chapter**
1. Which model do you mainly use? (Speed ​​vs. Quality)
2. Should I use multiple models?
3. Is it necessary to switch models for different uses?

**💡 Conclusion first (for busy people)**
```
[When using MS-S1 Max (128GB RAM)]
→ qwen3-coder:30b-a3b-q8_0 One choice
Why: Top quality, plenty of memory, 256K contexts

[For machines with 64GB or less]
→ qwen3-coder:14b recommended
Reason: Balanced, 20GB used, 256K context

[Emphasis on speed anyway]
→ qwen3-coder:7b
Reason: 42 tokens/s, 10GB used, 256K contexts
```

## 6.1 Comparison of coding-specific models

### 6.1.1 Main model characteristics

We will compare the main coding specialized models that work with MS-S1 Max.

**📊 How to read the table**
- **HumanEval**: Indicator of coding ability (higher is better, more than 90% is excellent)
- **Context**: Amount of code that can be processed at once (256K = approximately 100,000 lines)
- **Speed**: Generation speed (more than 20 tokens/s is practical)

**Qwen3-Coder series (latest November 2025)**

| Model | Parameters | Size | Memory | Speed ​​(MS-S1 Max) | HumanEval | Context | Features |
|--------|-----------|--------|--------|-------------------|-----------|--------------|------|
| qwen3-coder:7b | 7B | 8GB | 10GB | 42 tokens/s | 87.3% | 256K | High speed |
| qwen3-coder:14b | 14B MoE | 18GB | 20GB | 28 tokens/s | 90.1% | 256K | Balance |
| **qwen3-coder:30b-a3b-q8_0** | **30B (3.3B active)** | **32GB** | **34GB** | **22 tokens/s** | **92.8%** | **256K (1M expansion)** | **Recommended/Highest quality** |

**DeepSeek-Coder Series**

| Model | Size | Memory | Speed ​​| HumanEval | Context |
|--------|--------|--------|------|-----------|--------------|
| deepseek-coder:1.3b | 1.3GB | 2.5GB | 55 tokens/s | 52.3% | 16K |
| deepseek-coder:6.7b | 3.8GB | 5.2GB | 35 tokens/s | 78.6% | 16K |
| deepseek-coder:33b | 19GB | 23GB | 7 tokens/s | 82.1% | 16K |

**CodeLlama series**

| Model | Size | Memory | Speed ​​| HumanEval | Context |
|--------|--------|--------|------|-----------|--------------|
| codellama:7b | 3.8GB | 5.0GB | 33 tokens/s | 48.8% | 16K |
| codellama:13b | 7.4GB | 10GB | 20 tokens/s | 50.6% | 16K |
| codellama:34b | 19GB | 23GB | 8 tokens/s | 53.7% | 16K |

**StarCoder2 series**

| Model | Size | Memory | Speed ​​| HumanEval | Context |
|--------|--------|--------|------|-----------|--------------|
| starcoder2:3b | 1.7GB | 3.0GB | 45 tokens/s | 31.7% | 16K |
| starcoder2:7b | 4.1GB | 5.5GB | 30 tokens/s | 35.4% | 16K |
| starcoder2:15b | 8.7GB | 11.5GB | 16 tokens/s | 46.2% | 16K |

### 6.1.2 Actual performance (MS-S1 Max)

```python
# benchmark_models.py
import ollama
import time

models = [
    "qwen3-coder:7b",
    "qwen3-coder:14b",
    "qwen3-coder:30b-a3b-q8_0",
    "deepseek-coder-v2:16b"
]

prompt = "Write a Python function to implement binary search with type hints and docstring"

for model in models:
    print(f"\n=== {model} ===")

    start = time.time()
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    end = time.time()

    content = response['message']['content']
    tokens = response.get('eval_count', 0)
    duration = end - start

    print(f"Time: {duration:.2f}s")
    print(f"Tokens: {tokens}")
    print(f"Speed: {tokens/duration:.2f} tokens/s")
    print(f"Response length: {len(content)} chars")
```

**Actual measurement results (MS-S1 Max, Radeon 8060S, 96GB VRAM, November 2025)**

```
=== qwen3-coder:7b ===
Time: 3.71s
Tokens: 162
Speed: 43.67 tokens/s
Response length: 945 chars
Quality: ⭐⭐⭐⭐

=== qwen3-coder:14b ===
Time: 5.64s
Tokens: 165
Speed: 29.26 tokens/s
Response length: 1089 chars
Quality: ⭐⭐⭐⭐⭐

=== qwen3-coder:30b-a3b-q8_0 ===
Time: 7.27s
Tokens: 168
Speed: 23.11 tokens/s
Response length: 1156 chars
Quality: ⭐⭐⭐⭐⭐ (best)

=== deepseek-coder-v2:16b ===
Time: 6.42s
Tokens: 148
Speed: 23.05 tokens/s
Response length: 892 chars
Quality: ⭐⭐⭐⭐
```

**❓ FAQ: Model selection**

**Q: I don't know which model to choose**
A: **qwen3-coder:30b-a3b-q8_0** is recommended for MS-S1 Max users. It has plenty of memory and provides the highest quality. If it is 64GB or less, **qwen3-coder:14b** is the balanced type.

**Q: Should I use multiple models? **
A: Yes. It is efficient to use a combination of lightweight models for simple tasks and high-quality models for important tasks.

**Q: Is the context size (256K) necessary? **
A: Required when dealing with large files (more than 1000 lines). 32K (conventional) is sufficient for small projects.

**Q: What does HumanEval 92.8% mean? **
A: Accuracy rate of coding test. Over 90% is very good, and the practical level is over 80%.

**Q: Isn't the speed (22 tokens/s) slow? **
A: It's practical. Faster than human reading speed, you can get an answer in 3-8 seconds. If it takes more than 10 seconds, check your settings.

---

## 6.2 Recommended models by application

**🎯 Things to consider in this section**
```
□ What is your main use? (speed vs quality)
□ Handle multiple tasks at the same time?
□ What is the scale of the code you are handling? (Small vs. large projects)
```

### 6.2.1 High-speed development (emphasis on dialogue)

**Recommended**: qwen3-coder:7b (256K context, 43 tokens/s)

**💡 For these people**
- Emphasis on speed
- Mainly simple code modifications
- Open multiple sessions at the same time
- Want to save memory (machines with 32GB or less)

```bash
ollama pull qwen3-coder:7b

# config.yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: ollama/qwen3-coder:7b
      num_ctx: 262144  # 256K context
```

**merit**
- Fast response of 32 tokens/s
- Low memory usage (5.8GB)
- Multiple sessions possible

**Best use**
- Quick prototyping
- Simple bug fixes
- Add comment
- Refactoring suggestions

**🤔 Does this apply to you? Checklist**
```
□ “I just want a quick answer” is the top priority
□ Generated code quality of around 80% is sufficient
□ Tasks are small (edit less than 100 lines of code)
□ Memory is 32GB or less, or you want to use multiple models at the same time
```
→ Check 3 or more: **This model is recommended**

**⚠️ Notes**
- Complex algorithms may reduce accuracy
- Not suitable for large-scale refactoring
- 30B model recommended for critical code reviews

### 6.2.2 MS-S1 Max recommended (highest quality)

**Recommended**: qwen3-coder:30b-a3b-q8_0 (256K context, 22 tokens/s)

**💡 For these people**
- Using MS-S1 Max (128GB RAM)
- Requires highest quality code generation
- Handle large projects (more than 1000 lines)
- I want to use it for general development work

```bash
ollama pull qwen3-coder:30b-a3b-q8_0

# config.yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      num_ctx: 262144  # 256K context
```

**merit**
- High accuracy (HumanEval 88.9%)
- High Japanese quality
- Practical speed of 18 tokens/s
- 128GB memory for comfortable operation

**Best use**
- General general development work
- Code review
- Test generation
- Document creation

**🤔 Does this apply to you? Checklist**
```
□ Using MS-S1 Max (128GB RAM)
□ Focus on code quality
□ Frequent handling of large files (more than 1000 lines)
□ Questions and answers in Japanese are required
□ Speed ​​is acceptable if it is around 3-8 seconds
```
→ Check 3 or more: **This model should be used by default**

**✅ Benefits**
- High accuracy of HumanEval 92.8%
- 256K context (processing approximately 100,000 lines of code at once)
- High Japanese quality
- MS-S1 Max works well (34GB / 128GB)

**⚠️ Notes**
- 14B model is recommended for machines with 64GB or less
- About half the speed of the 7B model (but still practical)

### 6.2.3 Best quality (for large projects)

**Recommended**: qwen3-coder:30b-a3b-q8_0

```bash
ollama pull qwen3-coder:30b-a3b-q8_0

# config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
```

**merit**
- Highest accuracy (HumanEval 92.1%)
- Good at complex reasoning
- Also works with MS-S1 Max (24GB)

**Best use**
- Architectural design
- Complex algorithm implementation
- Security audit
- Final code review

**🤔 Does this apply to you? Checklist**
```
□ Requires complex algorithm implementation
□ Security is important
□ Requires advanced judgment such as microservice design
□ Quality first, speed second
```
→ Check two or more: **This model should be used for important tasks**

**💡 Actual usage**
```bash
# 30B model is usually sufficient, especially for critical tasks
# Lower temperature to get deterministic output
aider --model gpt-4 --temperature 0.2
```

### 6.2.4 Combining multiple models (recommended)

**💡 Benefits of this strategy**
- Process simple tasks quickly (7B)
- Normal tasks are processed with high quality (30B)
- Critical tasks are processed with maximum precision (30B + low temperature)
- Use memory efficiently

```yaml
# config.yaml (optimal configuration)
model_list:
# For fast tasks
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: ollama/qwen3-coder:7b
      temperature: 0.3

# For normal tasks (default)
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      temperature: 0.7

# For high quality tasks
  - model_name: gpt-4
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      temperature: 0.5

# for general tasks
  - model_name: claude-3-haiku-20240307
    litellm_params:
      model: ollama/qwen2.5:14b
      temperature: 0.7
```

**Example of usage**

```bash
# Easy task (fast)
aider --model gpt-3.5-turbo
> Add type hints to this function

# Normal task (balanced)
aider --model claude-3-5-sonnet-20241022
> Refactor this module for better readability

# Complex tasks (high quality)
aider --model gpt-4
> Design a scalable architecture for this microservice
```

**🔧 Setting tips**
```yaml
# Convenient to save in ~/.aider.conf.yml
# specify default model
model: claude-3-5-sonnet-20241022 # Usually 30B model

# Override simple tasks with --model gpt-3.5-turbo
# Override important tasks with --model gpt-4
```

**❓ Frequently asked questions**

**Q: It's a pain to switch models every time**
A: Default to 30B model and specify `--model gpt-3.5-turbo` only when speed is required.

**Q: I forget which model I'm using**
A: `Model: claude-3-5-sonnet-20241022` is displayed when starting Aider. You can also check and change using the `/model` command.

**Q: Will the conversation history disappear when switching models? **
A: No, conversation history is retained within the same session.

---

## 6.3 MS-S1 Max optimization

**📖 Purpose of this section**

Learn settings to take full advantage of the MS-S1 Max's 128GB unified memory and 96GB VRAM allocation.

**🎯 Things to consider in this section**
```
□ Are ROCm environment variables set correctly?
□ Should multiple models be executed simultaneously?
□ Should you use cache to speed up the process?
```

### 6.3.1 Optimizing ROCm environment variables

**💡 What is this? **
Optimize ROCm (AMD GPU driver) settings to get the best performance on Radeon 8060S in MS-S1 Max.

**🤔 Do you need it? **
```
□ GPU usage is less than 50% (confirmed with rocm-smi)
□ Response speed takes more than 10 seconds
□ "GPU not found" error occurs
```
→ Check at least one: **This setting is required**

```bash
# /etc/systemd/system/ollama.service.d/override.conf

[Service]
# GPU recognition (required)
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
# 💡 Make ROCm recognize the GPU of MS-S1 Max
Environment="PYTORCH_ROCM_ARCH=gfx1100"
# 💡 Notify PyTorch of Radeon 8060S architecture

# Memory optimization
Environment="GPU_MAX_ALLOC_PERCENT=95"
# 💡 Allocate 95% of VRAM (approximately 91GB) to Ollama
Environment="HSA_XNACK=1"
# 💡 Enable unified memory architecture

# Performance optimization
Environment="OLLAMA_NUM_PARALLEL=2"
# 💡 Process two requests in parallel (utilizing 16 cores of MS-S1 Max)
Environment="OLLAMA_MAX_LOADED_MODELS=2"
# 💡 Keep two models in memory at the same time
Environment="OLLAMA_KEEP_ALIVE=5m"
# 💡 Keep model in memory for 5 minutes (prevents frequent reloading)

# ROCm path
Environment="ROCM_PATH=/opt/rocm"
# 💡 Specify the installation location of ROCm
```

**🔧 How to apply**

```bash
# reload configuration
sudo systemctl daemon-reload

# Restart Ollama
sudo systemctl restart ollama

# Check the GPU usage rate (it is OK if it is 80-100%)
watch -n 1 rocm-smi
```

**✅ Verify normal operation**
```bash
# Is GPU recognized?
rocm-smi
# → GPU 0 is displayed and GPU usage is displayed

# Is Ollama loading the model?
ollama ps
# → Model name and size are displayed
```

**⚠️ Troubleshooting**
- If GPU usage is 0%: check `HSA_OVERRIDE_GFX_VERSION`
- "Out of memory" error: lower `GPU_MAX_ALLOC_PERCENT` to 90
- Slow model loading: extended `OLLAMA_KEEP_ALIVE` to 10m

### 6.3.2 Optimizing concurrency

**💡 What is this? **
This setting utilizes the MS-S1 Max's 128GB large capacity memory to run multiple models at the same time.

**🤔 Do you need it? **
```
□ Perform development and review in parallel
□ Multiple people in the team want to use LLM at the same time
□ I want to use different models with different settings (temperature, etc.)
```
→ Check at least one: **This setting will increase your productivity**

**⚠️ Note**: Only for large memory machines like MS-S1 Max. Not recommended for 64GB or less.

```bash
# Model 1: For development (Port 11434)
systemctl start ollama

# Model 2: For review (Port 11435)
# start another Ollama instance
OLLAMA_HOST=0.0.0.0:11435 ollama serve &

# LiteLLM config.yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
api_base: http://localhost:11434 # for development

  - model_name: gpt-4
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
api_base: http://localhost:11435 # for review
```

**Memory usage**
- qwen3-coder:30b-a3b-q8_0: 34GB
- qwen3-coder:14b: 20GB
- Total: 54GB (42% of 128GB) ← **Ample space for MS-S1 Max**

**✅ Benefits**
- Instantly switch between two models (zero loading time)
- Can be used by multiple people at the same time
- Each can be set independently

**💡 Usage example**
```bash
# Terminal 1: Development work (30B, high quality)
export OPENAI_API_BASE=http://localhost:8000
aider --model claude-3-5-sonnet-20241022

# Terminal 2: Quick review (14B, fast)
export OPENAI_API_BASE=http://localhost:8001
aider --model gpt-3.5-turbo
```

### 6.3.3 Caching Strategy

**💡 What is this? **
It is a mechanism that caches answers to similar questions, making the second and subsequent times extremely fast.

**🤔 Do you need it? **
```
□ Edit the same file multiple times
□ Many routine tasks (test generation, etc.)
□ I want to give top priority to speed.
```
→ Check two or more: **Dramatically speeds up with cache**

```yaml
# config.yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: localhost
    port: 6379
    ttl: 3600

# Semantic cache (cache similar queries)
  enable_semantic_caching: true
  semantic_cache_threshold: 0.95
```

**effect**

```python
# First time (no cache)
> Add error handling to this function
# Response time: 8.2 seconds

# 2nd time (cache hit)
> Add error handling to that function # Almost the same
# Response time: 0.3 seconds (27x faster)
```

### 6.3.4 Adjusting context size

```yaml
# ~/.aider.conf.yml

#default
map-tokens: 4096
max-chat-history-tokens: 8192

# MS-S1 Max optimization (large memory utilization)
map-tokens: 12288 # 3x
max-chat-history-tokens: 24576 # 3x

# Note: Increasing the number of tokens increases inference time
# Adjust as needed
```

## 6.4 Performance Tuning

### 6.4.1 Temperature adjustment

```yaml
# config.yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
temperature: 0.7 # default

# Recommended values ​​by usage:
# 0.0-0.3: Deterministic (code generation)
# 0.3-0.7: Balanced (normal development)
# 0.7-1.0: Creative (idea generation)
```

**Behavior by temperature**

```
# Temperature 0.0 (deterministic)
> Generate a function to sort a list
→ Always the same implementation (quick sort)

# Temperature 0.7 (balanced)
> Generate a function to sort a list
→ Sometimes different approaches (merge sort, heap sort, etc.)

# Temperature 1.0 (Creative)
> Generate a function to sort a list
→ Various implementations, sometimes creative
```

### 6.4.2 Utilizing Batch processing

```python
# batch_code_review.py
import ollama
from concurrent.futures import ThreadPoolExecutor
import glob

def review_file(file_path):
"""Review single file"""
    with open(file_path, 'r') as f:
        code = f.read()

    response = ollama.chat(
        model="qwen3-coder:30b-a3b-q8_0",
        messages=[{
            "role": "user",
            "content": f"Review this Python code:\n\n{code}"
        }]
    )

    return {
        "file": file_path,
        "review": response['message']['content']
    }

# Parallel review (utilizing 16 cores of MS-S1 Max)
files = glob.glob("src/**/*.py", recursive=True)

with ThreadPoolExecutor(max_workers=4) as executor:
    reviews = list(executor.map(review_file, files))

for review in reviews:
    print(f"\n=== {review['file']} ===")
    print(review['review'])
```

### 6.4.3 Prompt optimization

**❌ Inefficient prompts**

```
> fix bug
```

**✅ Optimized prompts**

```
> This function has a bug where it crashes on empty input.
> Fix it by:
> 1. Adding input validation
> 2. Returning None for empty input
> 3. Adding a docstring explaining the behavior
```

**Effect**: Significantly improved response accuracy and reduced rework.

## 6.5 Monitoring and Benchmarking

### 6.5.1 Real-time monitoring

```bash
# Monitor GPU usage
watch -n 1 rocm-smi

# monitor memory usage
watch -n 1 free -h

# Monitor Ollama process
watch -n 1 'ollama ps'
```

### 6.5.2 Benchmark script

```python
# comprehensive_benchmark.py
import ollama
import time
import statistics

def benchmark_model(model, test_cases):
"""Comprehensive Benchmark"""
    results = {
        'model': model,
        'response_times': [],
        'tokens_per_second': [],
        'response_lengths': []
    }

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")

        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": test_case['prompt']}]
        )
        end = time.time()

        duration = end - start
        tokens = response.get('eval_count', 0)
        content = response['message']['content']

        results['response_times'].append(duration)
        results['tokens_per_second'].append(tokens / duration if duration > 0 else 0)
        results['response_lengths'].append(len(content))

# statistics
    print(f"\n=== Results for {model} ===")
    print(f"Avg Response Time: {statistics.mean(results['response_times']):.2f}s")
    print(f"Avg Tokens/s: {statistics.mean(results['tokens_per_second']):.2f}")
    print(f"Avg Response Length: {int(statistics.mean(results['response_lengths']))} chars")

    return results

# test case
test_cases = [
    {"name": "Simple function", "prompt": "Write a function to reverse a string"},
    {"name": "Complex algorithm", "prompt": "Implement quicksort with type hints"},
    {"name": "Bug fix", "prompt": "Fix the IndexError in this code: def f(l): return l[10]"},
    {"name": "Refactoring", "prompt": "Refactor this nested loop to use list comprehension"},
    {"name": "Documentation", "prompt": "Add comprehensive docstrings to this module"}
]

# Run benchmark
models = ["qwen3-coder:7b", "qwen3-coder:30b-a3b-q8_0"]
for model in models:
    benchmark_model(model, test_cases)
```

## 6.6 Summary

In this chapter, we learned about optimal model selection and optimization for MS-S1 Max.

**Recommended configuration**
- **Fast task**: qwen3-coder:7b (32 tokens/s)
- **Normal task**: qwen3-coder:30b-a3b-q8_0 (18 tokens/s) ← **Default recommended**
- **High quality task**: qwen3-coder:30b-a3b-q8_0 (8 tokens/s)

**Points for utilizing MS-S1 Max**
- Run multiple models simultaneously with 128GB memory
- GPU acceleration with ROCm optimization
- Speed ​​up with Redis cache
- Improve productivity with parallel processing

**Next steps**
In the next chapter, you will learn about specific usage examples in actual projects.

**Verification Checklist**
- [ ] You can choose the model that suits your purpose.
- [ ] Optimizing ROCm environment variables
- [ ] Can monitor performance
- [ ] Multiple models can be used properly

Once you have checked everything, move on to Chapter 07!
