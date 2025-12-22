# Chapter 5: Model Management and Customization

## 5.1 Modelfile basics

### 5.1.1 What is Modelfile?

**Modelfile** is a configuration file for defining a custom model. Uses a Dockerfile-like syntax.

```dockerfile
# Basic Modelfile
FROM llama3.1

SYSTEM """
You are a helpful AI assistant.
"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
```

### 5.1.2 Modelfile syntax

#### FROM instruction

```dockerfile
# Specify base model
FROM llama3.1
FROM qwen2.5:14b
FROM mistral:latest

# local GGUF file
FROM ./models/mymodel.gguf

# Specify quantization level
FROM llama3.1:8b-instruct-q4_K_M
```

#### SYSTEM instruction

```dockerfile
# System prompt (define AI role)
SYSTEM """
You are an experienced software engineer.
Perform code reviews and provide improvement suggestions.
"""
```

#### PARAMETER instruction

```dockerfile
# Inference parameter settings
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048
PARAMETER stop "<|end|>"
```

#### TEMPLATE instruction

```dockerfile
# prompt template
TEMPLATE """{{ .System }}

User: {{ .Prompt }}
Assistant: """
```

#### MESSAGE instruction

```dockerfile
# Presetting conversation history
MESSAGE user "Hello"
MESSAGE assistant "Hello! Is there anything I can help you with?"
```

### 5.1.3 Complete Modelfile example

```dockerfile
# Custom assistant for MS-S1 Max
FROM qwen2.5:14b

# system prompt
SYSTEM """
You are a kind and knowledgeable Japanese AI assistant.
It has the following characteristics:
- Provide accurate and detailed information
- Clear and detailed explanation
- Strong in technical questions
- MS-S1 Max Expert
"""

#parameter
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER num_predict -1

# stop sequence
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# template
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

## 5.2 Creating a custom model

### 5.2.1 Basic creation procedure

```bash
# 1. Create Modelfile
nano my-assistant.Modelfile

# 2. Model creation
ollama create my-assistant -f my-assistant.Modelfile

# Output example
parsing modelfile
creating model layers
writing manifest
success

# 3. Confirm
ollama list

# 4. Execute
ollama run my-assistant
```

### 5.2.2 Example of creating a specialization model

#### Specialized in Japanese translation

```bash
nano translator-jp.Modelfile
```

```dockerfile
FROM qwen2.5:14b

SYSTEM """
You are a highly accurate translation AI.
Please translate according to the following rules:
- Natural and easy to read Japanese
- Translate technical terms appropriately
- Contextual translation
- Add explanation if anything is unclear
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER num_ctx 16384
```

```bash
# Create and use
ollama create translator-jp -f translator-jp.Modelfile
ollama run translator-jp "Translate to Japanese: The quick brown fox..."
```

#### Specialized in code review

```bash
nano code-reviewer.Modelfile
```

```dockerfile
FROM qwen2.5:32b

SYSTEM """
You are an experienced senior engineer.
When performing a code review:
1. Point out bugs and potential issues
2. Performance improvement suggestions
3. Advice on improving readability
4. Applying best practices
5. Security concerns

Reviews should be constructive and specific.
"""

PARAMETER temperature 0.5
PARAMETER top_p 0.9
PARAMETER num_ctx 16384
```

```bash
ollama create code-reviewer -f code-reviewer.Modelfile
```

#### MS-S1 Max Expert

```bash
nano ms-s1-expert.Modelfile
```

```dockerfile
FROM llama3.1:8b

SYSTEM """
You are an MS-S1 Max expert.
I am familiar with the following information:

[Hardware]
- AMD Ryzen AI Max+ 395 (16 cores/32 threads)
- 128GB LPDDR5X-8000
- Radeon 8060S (RDNA 3.5)
- 256GB/s memory bandwidth

【optimization】
- ROCm settings
- Performance tuning
- Cooling management

We provide specific, practical advice for your questions.
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

```bash
ollama create ms-s1-expert -f ms-s1-expert.Modelfile
```

## 5.3 Advanced parameter settings

### 5.3.1 Temperature (creativity)

```dockerfile
# Low temperature: Deterministic, accurate (0.1-0.5)
PARAMETER temperature 0.3 # Great for translation, summarization and code

# Medium temperature: Balance (0.6-0.8)
PARAMETER temperature 0.7 # General conversation

# High temperature: creative (0.9-1.5)
PARAMETER temperature 1.2 # Creation, brainstorming
```

### 5.3.2 Top-P (probability threshold)

```dockerfile
# Conservative: only high probability tokens
PARAMETER top_p 0.7

# balance
PARAMETER top_p 0.9

# Focus on diversity
PARAMETER top_p 0.95
```

### 5.3.3 Context Length

```dockerfile
# Shorter context (faster)
PARAMETER num_ctx 2048

# Standard
PARAMETER num_ctx 8192

# Long context (possible with MS-S1 Max)
PARAMETER num_ctx 32768

# Ultra-long context (uses large memory)
PARAMETER num_ctx 131072
```

**Recommended settings for MS-S1 Max:**

| Model size | Recommended num_ctx | Memory usage |
|------------|-------------|-------------|
| 7B | 32768 | ~12GB |
| 14B | 16384 | ~15GB |
| 32B | 8192 | ~25GB |
| 70B | 8192 | ~45GB |

### 5.3.4 Repeat Penalty

```dockerfile
# allow repetition
PARAMETER repeat_penalty 1.0

# standard suppression
PARAMETER repeat_penalty 1.1

# strong suppression
PARAMETER repeat_penalty 1.3
```

### 5.3.5 Complete parameter list

```dockerfile
PARAMETER num_predict 2048 # Maximum number of generated tokens
PARAMETER temperature 0.8 # Creativity (0.0-2.0)
PARAMETER top_p 0.9 # Nuclear sampling (0.0-1.0)
PARAMETER top_k 40 # Top-K sampling
PARAMETER repeat_penalty 1.1 # Repeat penalty
PARAMETER repeat_last_n 64 # Penalty application range
PARAMETER num_ctx 8192 # context window
PARAMETER num_batch 512 # Batch size
PARAMETER num_gpu 1 # Number of GPUs
PARAMETER num_thread 16 # Number of CPU threads
PARAMETER stop "<|end|>" # stop sequence
PARAMETER mirostat 0 # Mirostat sampling (0=disabled)
PARAMETER mirostat_eta 0.1 # Mirostat learning rate
PARAMETER mirostat_tau 5.0 # Mirostat target entropy
```

## 5.4 Customizing the template

### 5.4.1 Basic template variables

```dockerfile
{{ .System }} # System prompt
{{ .Prompt }} # User input
{{ .Response }} # Assistant response (for message history)
```

### 5.4.2 Chat ML format

```dockerfile
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

### 5.4.3 Llama 3 format

```dockerfile
TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
```

### 5.4.4 Custom format

```dockerfile
# Markdown style
TEMPLATE """# System
{{ .System }}

## User
{{ .Prompt }}

## Assistant
"""

# Structured format
TEMPLATE """[SYSTEM]
{{ .System }}
[/SYSTEM]

[USER]
{{ .Prompt }}
[/USER]

[ASSISTANT]
"""
```

## 5.5 Multimodal model

### 5.5.1 Creating a vision model

```bash
# LLaVA base (image + text)
nano vision-assistant.Modelfile
```

```dockerfile
FROM llava:13b

SYSTEM """
You are an AI assistant who can understand images and explain them in detail.
"""

PARAMETER temperature 0.7
PARAMETER num_ctx 4096
```

```bash
# create
ollama create vision-assistant -f vision-assistant.Modelfile

# Use (pass image)
ollama run vision-assistant "Describe this image" image.jpg
```

### 5.5.2 Code understanding specialization

```bash
nano code-understanding.Modelfile
```

```dockerfile
FROM qwen2.5-coder:7b

SYSTEM """
Analyze your code and provide:
1. Function description
2. Complexity analysis
3. Improvement suggestions
4. Document generation
"""

PARAMETER temperature 0.5
PARAMETER num_ctx 16384
```

## 5.6 Model version control

### 5.6.1 Tagging System

```bash
# Create versioned model
ollama create my-assistant:v1.0 -f assistant-v1.Modelfile
ollama create my-assistant:v1.1 -f assistant-v1.1.Modelfile
ollama create my-assistant:latest -f assistant-latest.Modelfile

# execute version specification
ollama run my-assistant:v1.0
ollama run my-assistant:latest
```

### 5.6.2 Exporting and importing models

```bash
# Export Modelfile
ollama show --modelfile my-assistant > my-assistant.exported.Modelfile

# import on another machine
ollama create my-assistant -f my-assistant.exported.Modelfile
```

### 5.6.3 Sharing the model

```bash
# Push to Ollama Hub (account required)
ollama push username/my-assistant:latest

# Used by other users
ollama pull username/my-assistant
```

## 5.7 Quantization level selection

### 5.7.1 Fundamentals of Quantization

Quantization is a tradeoff between model size and accuracy.

```
Q8_0: 8-bit quantization (highest quality, large size)
Q6_K: 6-bit quantization (high quality)
Q5_K_M: 5-bit quantization (balanced)
Q4_K_M: 4-bit quantization (recommended, good balance)
Q3_K_M: 3-bit quantization (small size, reduced quality)
Q2_K: 2-bit quantization (minimum size, significant quality loss)
```

### 5.7.2 Recommended settings for MS-S1 Max

```bash
# 7B-14B model: Q4_K_M or Q5_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M   # 4.9GB
ollama pull qwen2.5:7b-instruct-q5_K_M   # 5.8GB

# 32B-34B model: Q4_K_M
ollama pull qwen2.5:32b-instruct-q4_K_M  # 19GB

# 70B model: Q4_K_M (comfortable with 128GB memory)
ollama pull llama3.1:70b-instruct-q4_K_M # 41GB
```

### 5.7.3 Comparison of quantization levels

```bash
# comparison script
#!/bin/bash
MODEL_BASE="qwen2.5:7b"
QUANTS=("q8_0" "q5_K_M" "q4_K_M" "q3_K_M")
PROMPT="Write a detailed explanation of quantum computing."

for quant in "${QUANTS[@]}"; do
    model="${MODEL_BASE}-instruct-${quant}"
    echo "Testing: $model"

    ollama pull $model
    time ollama run $model "$PROMPT"
    echo "---"
done
```

## 5.8 Model Fine Tuning

### 5.8.1 Using LoRA adapter

```dockerfile
# Base model + LoRA adapter
FROM llama3.1:8b

# LoRA adapter (learned separately)
ADAPTER ./adapters/japanese-qa-lora.bin

SYSTEM """
This is a model specialized for Japanese Q&A.
"""
```

### 5.8.2 Integrating a custom GGUF model

```bash
# 1. Place GGUF model
mkdir -p ~/.ollama/models/custom
cp my-finetuned-model.gguf ~/.ollama/models/custom/

# 2. Create Modelfile
nano custom-model.Modelfile
```

```dockerfile
FROM ~/.ollama/models/custom/my-finetuned-model.gguf

SYSTEM """
Custom fine tuning model
"""

PARAMETER temperature 0.7
```

```bash
# 3. Import
ollama create custom-model -f custom-model.Modelfile
```

## 5.9 Troubleshooting

### 5.9.1 Model creation error

```bash
# Error example
Error: failed to parse modelfile

# Solution: Check syntax
# - Does the FROM line come first?
# - Are quotes closed correctly?
# - Is the parameter name correct?
```

### 5.9.2 Out of memory

```bash
# error
Error: failed to allocate memory

# Solution 1: Use smaller quantization
FROM llama3.1:8b-q4_K_M # instead of q8_0

# Solution 2: Reduce context length
PARAMETER num_ctx 4096 # instead of 16384
```

### 5.9.3 Performance degradation

```bash
# Symptom: Custom model is slow

# Check 1: Quantization level of base model
ollama show --modelfile my-model

# Check 2: Parameter settings
# Check if num_ctx is not too large
```

## 5.10 Summary of this chapter

In this chapter, you learned the following contents.

✅ **Modelfile basics**
- Syntax and instructions
- Custom model creation

✅ **Parameter optimization**
- Temperature, Top-P, Context Length
- Recommended settings for MS-S1 Max

✅ **Template customization**
- Chat ML format
- Custom format

✅ **Quantization level**
- Q4_K_M vs Q8_0
- Size vs. quality trade-off

✅ **Model management**
- Version control
- Export/Import

In the next chapter, you will learn how to leverage the Ollama API and integrate it with other applications.

---

**Go to previous chapter**: [Chapter 4 Basic Commands and How to Use](chapter04_basic_commands.md)
**Next Chapter**: [Chapter 6 API Utilization and Integration](chapter06_api_integration.md)
