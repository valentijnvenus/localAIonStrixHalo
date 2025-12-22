# Chapter 5: Model Loader and Format

## 5.1 Loader types

### Supported loaders

```python
loaders = {
'Transformers': 'Standard HuggingFace',
'llama.cpp': 'GGUF format',
'ExLlamaV2': 'GPTQ/EXL2 (fastest)',
'AutoGPTQ': 'GPTQ quantization',
'AutoAWQ': 'AWQ quantization',
'HQQ': 'High quality quantization'
}
```

## 5.2 Model format selection

### MS-S1 Max recommended

```
7B-13B models:
├── ExLlamaV2 (EXL2) - Fastest
└── llama.cpp (GGUF Q5) - Balance

32B-70B models:
├── ExLlamaV2 (EXL2 4.0bpw) - Recommended
└── llama.cpp (GGUF Q4_K_M) - Alternative
```

## 5.3 Quantization level

### GGUF Quantization

```
Q8_0: 8bit - highest quality
Q6_K: 6bit - High quality
Q5_K_M: 5bit - recommended
Q4_K_M: 4bit - Balanced
Q3_K_M: 3bit - compact
```

### EXL2 Quantization

```
8.0 bpw: Top quality
6.5 bpw: high quality
5.0 bpw: Recommended (70B possible)
4.0 bpw: Compact (70B recommended)
3.0 bpw: ultra compact
```

## 5.4 Model download

### Recommended model

```bash
# 7B General purpose (Japanese)
python download-model.py elyza/ELYZA-japanese-Llama-2-7b-fast-instruct

# 13B Balance
python download-model.py TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF

# 32B High Performance
python download-model.py TheBloke/Yi-34B-200K-GGUF

# 70B Highest performance
python download-model.py TheBloke/Llama-2-70B-chat-GGUF
```

## 5.5 Model switching

### Web UI

```
1. "Model" tab
2. Select model from dropdown
3. Click "Load" button
4. Loader selection (auto detection)
5. Wait for loading to complete
```

### Command line

```bash
# Specify at startup
python server.py --model llama-2-70b-chat.Q4_K_M.gguf --loader llama.cpp
```

## 5.6 Optimization by loader

### ExLlamaV2 settings

```yaml
loader: exllamav2
max_seq_len: 32768
cache_8bit: False
flash_attn: True
```

### llama.cpp settings

```yaml
loader: llama.cpp
n_ctx: 8192
n_batch: 512
n_gpu_layers: -1 # All layers GPU
```

## 5.7 Summary of this chapter

✅ Loader types and features
✅ Quantization level selection
✅ MS-S1 Max recommended settings
✅ Model download and management

---

**Go to previous chapter**: [Chapter 4 Basic operations and interface](chapter04_basic_usage.md)
**Next Chapter**: [Chapter 6 Advanced Parameter Settings](chapter06_advanced_parameters.md)
