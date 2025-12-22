# Chapter 3: ROCm settings and ExLlamaV2 optimization

## 3.1 What is ExLlamaV2?

ExLlamaV2 is an ultra-fast inference engine dedicated to GPTQ/EXL2 quantization models.

### Advantages with MS-S1 Max

```
Speed ​​comparison (70B model, Q4 quantization):
- Transformers: 2-3 tokens/s
- llama.cpp: 4-6 tokens/s
- ExLlamaV2: 8-12 tokens/s (2-3 times faster!)
```

## 3.2 ROCm optimization

```bash
# environmental variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_HOME=/opt/rocm
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

## 3.3 Installing ExLlamaV2

```bash
cd ~/text-generation-webui
source venv/bin/activate

pip install exllamav2 --no-build-isolation --extra-index-url https://download.pytorch.org/whl/rocm6.2
```

## 3.4 Settings.yaml optimization

```yaml
# ~/text-generation-webui/settings.yaml
loader: exllamav2
gpu_memory:
  - 96  # MS-S1 Max: 96GB
cpu_memory: 32

# ExLlamaV2 specific
max_seq_len: 32768
compress_pos_emb: 1.0
alpha_value: 1.0
rope_freq_base: 0
cache_8bit: False # Disable because there is 128GB
```

## 3.5 Performance Test

```python
# test_performance.py
import time

def benchmark():
# Use WebUI API
    import requests
    
    start = time.time()
    response = requests.post('http://localhost:5000/api/v1/generate', json={
        'prompt': 'Write a 200 word essay',
        'max_new_tokens': 200
    })
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {200/elapsed:.1f} tokens/s")

benchmark()
```

## 3.6 Troubleshooting

### GPU not recognized
```bash
# Confirm ROCm
rocm-smi

# Check environment variables
env | grep HSA
```

### Memory error
```yaml
# adjust settings.yaml
gpu_memory:
- 80 # reduced from 96
```

## 3.7 Summary of this chapter

✅ Features and advantages of ExLlamaV2
✅ ROCm optimization settings
✅ Performance test
✅ Troubleshooting

---

**Go to previous chapter**: [Chapter 2 Installation and environment construction](chapter02_installation.md)
**Next Chapter**: [Chapter 4 Basic Operation and Interface](chapter04_basic_usage.md)
