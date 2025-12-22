# Chapter 7: Performance optimization for MS-S1 Max

On Linux, you want to run 6.17.x kernel as it introduces some important optimizations. ROCm will give you much better prefill than Vulkan, and just a slightly lower tg. Use llama.cpp - either compile from the source, or get ROCm build from Lemonade SDK: https://github.com/lemonade-sdk/llamacpp-rocm

## 7.1 Maximum utilization of hardware resources

### 7.1.1 Strategic use of 128GB memory

Make full use of MS-S1 Max's 128GB large capacity memory.

```bash
# System-wide memory allocation strategy
┌─────────────────────────────────────────┐
│ Total memory: 128GB                     │
├─────────────────────────────────────────┤
│ OS + System: 8GB                        │
│ Ollama service: 2GB                     │
│ Model weight: 40-80GB (variable)        │
│ Context cache: 10-30GB                  │
│ Working buffer: 10-20GB                 │
│ Spare: 10-20GB                          │
└─────────────────────────────────────────┘
```

### 7.1.2 Simultaneous execution of multiple models

```bash
# Environment variable settings (up to 3 models can be loaded simultaneously)
sudo nano /etc/systemd/system/ollama.service.d/performance.conf
```

```ini
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_KEEP_ALIVE=30m"
```

**Recommended configuration example:**
```bash
# Combinations that can be executed simultaneously
1. qwen2.5:32b (20GB) + qwen2.5:14b (9GB) + qwen2.5:7b (5GB) = 34GB
2. llama3.1:70b (42GB) + qwen2.5:14b (9GB) = 51GB
3. qwen2.5:7b × 3 + codellama:13b = 23GB
```

### 7.1.3 Context Cache Optimization

```bash
# Long run settings
OLLAMA_KEEP_ALIVE=60m # Keep model in memory for 60 minutes
```

```python
# Configuration in Python
import ollama

# maintain long context
ollama.generate(
    model='qwen2.5:32b',
    prompt='...',
keep_alive='3600s' # 1 hour
)
```

## 7.2 GPU optimization

### 7.2.1 ROCm Memory Management

```bash
# add to ~/.bashrc
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

### 7.2.2 Adjusting batch size

```dockerfile
# Batch size optimization in Modelfile
FROM llama3.1:70b

PARAMETER num_batch 512 # 512-1024 is optimal for MS-S1 Max
PARAMETER num_thread 32 # Utilize all cores
PARAMETER num_gpu 1
```

### 7.2.3 Utilizing Flash Attention

```bash
# Enable Flash Attention 2 (improves memory efficiency by 30%)
export OLLAMA_FLASH_ATTENTION=1
```

**Effect measurement:**
```bash
# No Flash Attention
time ollama run llama3.1:70b "Write 500 words"
# → 45 seconds

# With Flash Attention
OLLAMA_FLASH_ATTENTION=1 time ollama run llama3.1:70b "Write 500 words"
# → 32 seconds (29% faster)
```

## 7.3 Optimizing prompt handling

### 7.3.1 Properly Setting Context Length

```python
# Benchmark: Context length versus speed
import ollama
import time

context_lengths = [2048, 4096, 8192, 16384, 32768]

for ctx_len in context_lengths:
    start = time.time()

    ollama.generate(
        model='qwen2.5:14b',
        prompt='Hello' * 100,
        options={'num_ctx': ctx_len}
    )

    elapsed = time.time() - start
    print(f"Context: {ctx_len}, Time: {elapsed:.2f}s")
```

**Recommended value for MS-S1 Max:**
| Model | Recommended num_ctx | Memory Usage | Speed ​​|
|--------|-------------|-----------|------|
| 7B | 32768 | ~12GB | High speed |
| 14B | 16384 | ~15GB | High speed |
| 32B | 8192 | ~25GB | Medium speed |
| 70B | 8192 | ~50GB | Medium speed |

### 7.3.2 Prompt Caching

```python
# Cache frequently used prompts
import ollama
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate(model, prompt):
    return ollama.generate(model=model, prompt=prompt)

# Use
result1 = cached_generate('qwen2.5:7b', 'What is AI?') # Execute
result2 = cached_generate('qwen2.5:7b', 'What is AI?') # Get from cache
```

## 7.4 Optimization of quantization level

### 7.4.1 Recommended settings for MS-S1 Max

```bash
# 7B-14B model: Q5_K_M (quality emphasis)
ollama pull qwen2.5:7b-instruct-q5_K_M   # 5.8GB
ollama pull qwen2.5:14b-instruct-q5_K_M  # 11GB

# 32B-34B models: Q4_K_M (balanced)
ollama pull qwen2.5:32b-instruct-q4_K_M  # 19GB

# 70B model: Q4_K_M (practical)
ollama pull llama3.1:70b-instruct-q4_K_M # 41GB

# By usage
# - Production environment: Q5_K_M (high quality)
# - Development/Test: Q4_K_M (balanced)
# - Experiment: Q3_K_M (small size)
```

### 7.4.2 Speed ​​vs. Quality Benchmarking

```bash
#!/bin/bash
# benchmark_quant.sh

MODEL="qwen2.5:7b"
PROMPT="Explain quantum computing in detail."
QUANTS=("q8_0" "q6_K" "q5_K_M" "q4_K_M" "q3_K_M")

echo "Quantization Benchmark"
echo "======================"

for quant in "${QUANTS[@]}"; do
    model_name="${MODEL}-instruct-${quant}"
    echo "Testing: $model_name"

# Download
    ollama pull $model_name 2>/dev/null

# Speed ​​measurement
    start=$(date +%s%N)
    response=$(ollama run $model_name "$PROMPT" 2>/dev/null)
    end=$(date +%s%N)

    elapsed=$((($end - $start) / 1000000))  # ms
    length=${#response}

    echo "  Time: ${elapsed}ms"
    echo "  Length: $length chars"
    echo "  Speed: $((length * 1000 / elapsed)) chars/s"
    echo ""
done
```

## 7.5 Optimizing concurrency

### 7.5.1 Multithread configuration

```python
# concurrent_ollama.py
from concurrent.futures import ThreadPoolExecutor
import ollama

def process_query(query):
    return ollama.generate(model='qwen2.5:7b', prompt=query)

queries = [
    "What is AI?",
    "What is ML?",
    "What is DL?",
    "What is NLP?",
    "What is CV?"
]

# Up to 4 parallel executions (for MS-S1 Max)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_query, queries))

for i, result in enumerate(results):
    print(f"Query {i+1}: {result['response'][:100]}...")
```

### 7.5.2 Asynchronous processing

```python
# async_batch.py
import asyncio
import ollama

async def async_generate(prompt):
    client = ollama.AsyncClient()
    response = await client.generate(
        model='qwen2.5:7b',
        prompt=prompt
    )
    return response['response']

async def batch_process(prompts):
    tasks = [async_generate(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

if __name__ == '__main__':
    prompts = [f"Tell me about topic {i}" for i in range(10)]
    results = asyncio.run(batch_process(prompts))
```

## 7.6 Network optimization

### 7.6.1 Optimizing local connections

```bash
# Use UNIX sockets (faster than TCP)
export OLLAMA_HOST=unix:///tmp/ollama.sock
```

### 7.6.2 Compressing API responses

```python
import requests
import gzip
import json

# enable gzip compression
headers = {
    'Content-Type': 'application/json',
    'Accept-Encoding': 'gzip'
}

response = requests.post(
    'http://localhost:11434/api/generate',
    json={'model': 'llama3.1', 'prompt': 'Hello'},
    headers=headers
)
```

## 7.7 System-level optimization

### 7.7.1 CPU governor settings

```bash
# performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Persistence
sudo apt install cpufrequtils
sudo nano /etc/default/cpufrequtils

# addition
GOVERNOR="performance"

sudo systemctl restart cpufrequtils
```

### 7.7.2 NUMA optimization

```bash
# Check NUMA information
numactl --hardware

# Run Ollama on specific NUMA nodes
numactl --cpunodebind=0 --membind=0 ollama serve
```

### 7.7.3 IRQ Affinity

```bash
# Fix GPU IRQ to specific CPU
# Check GPU IRQ number with /proc/interrupts
cat /proc/interrupts | grep amdgpu

# IRQ affinity setting
echo "4" | sudo tee /proc/irq/[IRQ number]/smp_affinity_list
```

## 7.8 Benchmark Tool

### 7.8.1 Comprehensive benchmark

```python
# comprehensive_benchmark.py
import ollama
import time
import statistics
import json

def benchmark_model(model, test_cases, num_runs=3):
    results = {
        'model': model,
        'tests': {}
    }

    for test_name, prompt in test_cases.items():
        print(f"Testing {model} - {test_name}...")

        speeds = []
        for i in range(num_runs):
            start = time.time()

            response = ollama.generate(
                model=model,
                prompt=prompt
            )

            elapsed = time.time() - start
            tokens = response.get('eval_count', 0)
            speed = tokens / elapsed if elapsed > 0 else 0
            speeds.append(speed)

        results['tests'][test_name] = {
            'avg_speed': statistics.mean(speeds),
            'std_dev': statistics.stdev(speeds) if len(speeds) > 1 else 0,
            'min_speed': min(speeds),
            'max_speed': max(speeds)
        }

    return results

# test case
test_cases = {
    'short': 'Hello',
    'medium': 'Explain quantum computing in 100 words.',
    'long': 'Write a detailed 500-word essay on artificial intelligence.',
    'code': 'Write a Python function to sort a list of numbers.'
}

models = ['qwen2.5:7b', 'qwen2.5:14b', 'qwen2.5:32b']

all_results = []
for model in models:
    result = benchmark_model(model, test_cases)
    all_results.append(result)

# save result
with open('benchmark_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nBenchmark complete! Results saved to benchmark_results.json")
```

### 7.8.2 Memory Profiling

```python
# memory_profile.py
import ollama
import psutil
import time

def profile_memory(model, prompt):
    process = psutil.Process()

# memory at start
    mem_before = process.memory_info().rss / 1024**3  # GB

    start = time.time()

    response = ollama.generate(model=model, prompt=prompt)

    elapsed = time.time() - start

# memory at exit
    mem_after = process.memory_info().rss / 1024**3  # GB
    mem_used = mem_after - mem_before

    return {
        'model': model,
        'time': elapsed,
        'memory_used_gb': mem_used,
        'tokens': response.get('eval_count', 0)
    }

# run profile
models = ['qwen2.5:7b', 'qwen2.5:14b', 'qwen2.5:32b']
prompt = "Write a 200-word essay."

for model in models:
    result = profile_memory(model, prompt)
    print(f"{model}:")
    print(f"  Time: {result['time']:.2f}s")
    print(f"  Memory: {result['memory_used_gb']:.2f}GB")
    print(f"  Speed: {result['tokens']/result['time']:.1f} tokens/s")
    print()
```

## 7.9 Real-time monitoring

### 7.9.1 Grafana + Prometheus

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ollama'
    static_configs:
      - targets: ['localhost:11434']
```

### 7.9.2 Custom Dashboard

```python
# metrics_exporter.py
from prometheus_client import start_http_server, Gauge, Counter
import ollama
import time
import threading

# Metric definition
requests_total = Counter('ollama_requests_total', 'Total requests')
request_duration = Gauge('ollama_request_duration_seconds', 'Request duration')
tokens_per_second = Gauge('ollama_tokens_per_second', 'Generation speed')
memory_usage = Gauge('ollama_memory_usage_bytes', 'Memory usage')

def collect_metrics():
    while True:
# Get memory usage
        import psutil
        process = psutil.Process()
        memory_usage.set(process.memory_info().rss)

        time.sleep(5)

# metrics collection thread
threading.Thread(target=collect_metrics, daemon=True).start()

# Start Prometheus exporter
start_http_server(8001)
print("Metrics available at http://localhost:8001/metrics")

# application loop
while True:
    time.sleep(1)
```

## 7.10 Summary of this chapter

In this chapter, you learned the following contents.

✅ **Memory optimization**
- 128GB utilization strategy
- Simultaneous execution of multiple models

✅ **GPU optimization**
- ROCm settings
- Flash Attention

✅ **Quantization optimization**
- Recommended settings for MS-S1 Max

✅ **Parallel processing**
- Multi-threaded
- Asynchronous processing

✅ **System optimization**
- CPU governor
- NUMA settings

✅ **Monitoring and benchmarking**
- Comprehensive benchmark
- Real-time monitoring

In the next chapter, you will learn practical techniques for multi-model operation and concurrent execution.

---

**Go to previous chapter**: [Chapter 6 API Utilization and Integration](chapter06_api_integration.md)
**Next chapter**: [Chapter 8 Multi-model operation and simultaneous execution] (chapter08_multi_model.md)
