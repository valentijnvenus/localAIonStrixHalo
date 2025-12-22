# Chapter 5: Downloading and managing models

## 5.1 Basic knowledge of LLM model

### 5.1.1 What is a model family?

A **model family** is a set of models that share the same base architecture.

#### Main model families (2025)

**1. Llama（Meta）**
```
Developer: Meta AI
Features: Versatile, open source
Version: Llama 3.1, Llama 3.2
Available sizes: 1B, 3B, 8B, 70B, 405B
License: Llama 3 Community License
Japanese language proficiency: Intermediate to high (improved with Fine Tune version)
```

**2. Qwen（Alibaba）**
```
Developer: Alibaba Cloud
Features: Multilingual support, strong in Japanese
Version: Qwen2.5
Available sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
License: Apache 2.0
Japanese proficiency: Very high
Recommended: Best for Japanese users ⭐
```

**3. Mistral（Mistral AI）**
```
Developer: Mistral AI (France)
Features: High performance, efficient
Version: Mistral 7B, Mixtral 8x7B, Mixtral 8x22B
Size range: 7B, 8x7B (MoE), 8x22B (MoE)
License: Apache 2.0
Japanese proficiency: Intermediate
```

**4. Gemma（Google）**
```
Developer: Google DeepMind
Features: Light weight, high efficiency
Version: Gemma 2
Available sizes: 2B, 9B, 27B
License: Gemma Terms of Use
Japanese proficiency: Intermediate
```

**5. DeepSeek（DeepSeek AI）**
```
Developer: DeepSeek (China)
Features: Specialized in coding
Version: DeepSeek-Coder V2
Size range: 16B, 236B
License: DeepSeek License
Japanese proficiency: middle to high
Recommended: Programming use ⭐
```

### 5.1.2 What is Quantization?

**Quantization** is a technique that reduces the data precision and size of model weights.

#### Comparison of quantization levels

| Quantization | Number of Bits | Size Ratio | Quality | Recommended Uses |
|--------|---------|---------|------|----------|
| F32 | 32 bit | 100% | Highest | For research (not normally used) |
| F16 | 16 bit | 50% | Best | If you want full quality |
| Q8_0 | 8 bit | 25% | Very High | High Quality Inference |
| Q6_K | 6 bit | 19% | High | Balanced (high quality) |
| Q5_K_M | 5 bits | 16% | High | Balanced |
| **Q4_K_M** | 4 bits | **12.5%** | **Good** | **Most Recommended** ⭐ |
| Q4_K_S | 4 bits | 12% | Good | Size priority |
| Q3_K_M | 3 bits | 9% | Medium | When memory is insufficient |
| Q2_K | 2 bit | 6% | Low | Not recommended |

**K-Quants**

The `K` suffix indicates a more advanced quantization technique.

```
_M (Medium): Balanced (recommended)
_S (Small): Size priority
_L (Large): Prioritize quality
```

**💡 TIP**: For MS-S1 Max's 128GB memory, **Q4_K_M** is the best balance. You can reduce size while maintaining quality.

### 5.1.3 GGUF format

**GGUF (GPT-Generated Unified Format)** is the standard file format for LLM.

**Features:**
- Single file
- Metadata embedding
- Fast loading
- Cross-platform compatible

**File naming convention:**
```
qwen2.5-7b-instruct-q4_k_m.gguf
│    │  │  │        │
│ │ │ │ └─ Quantization level
│ │ │ └────────── Model type (instruct = for chat)
│ │ └────────────── Number of parameters (7B = 7 billion)
│ └──────────────── Version
└──────────────────── Model family
```

## 5.2 Searching and downloading models

### 5.2.1 Search within LM Studio

**Basic search procedure**

1. **Open Search tab**
   ```
Launch LM Studio → Click “🔍 Search” from the left tab
   ```

2. **Type in search field**
   ```
Example: "qwen2.5 7b q4"
   ```

3. **Filtering**
   ```
Size: 1B-10B / 10B-30B / 30B+
Quantization: Q4 / Q5 / Q6 / Q8
Model type: Instruct / Base / Chat
   ```

4. **Sort**
   ```
Sort by popularity / Sort by new arrivals / Sort by size / Sort by name
   ```

### 5.2.2 Recommended model catalog (for MS-S1 Max)

#### For beginners (3B-7B)

**Qwen2.5 7B Instruct**
```
Model Name: Qwen/Qwen2.5-7B-Instruct-GGUF
File: qwen2.5-7b-instruct-q4_k_m.gguf
Size: 4.8GB
Memory: Approximately 6GB
Speed: 35-45t/s
Features:
✓ Very good at Japanese
✓ Fast response
✓ Highly versatile
Recommended use: Daily chat, question answering, simple text generation
```

**Llama 3.2 3B Instruct**
```
Model name: meta-llama/Llama-3.2-3B-Instruct-GGUF
File: llama-3.2-3b-instruct-q4_k_m.gguf
Size: 2.0GB
Memory: Approximately 3GB
Speed: 50-60t/s
Features:
✓ Super fast
✓ Light weight
✓ Modern architecture
Recommended use: Applications that require fast response, resource saving
```

#### Intermediate (13B-34B)

**Qwen2.5 14B Instruct**
```
Model Name: Qwen/Qwen2.5-14B-Instruct-GGUF
File: qwen2.5-14b-instruct-q4_k_m.gguf
Size: 9.0GB
Memory: Approximately 12GB
Speed: 20-25t/s
Features:
✓ Higher quality than 7B
✓ Comprehension of complex instructions
✓ Excellent understanding of Japanese context
Recommended uses: Professional text generation, translation, summarization
```

**Qwen2.5 32B Instruct**
```
Model Name: Qwen/Qwen2.5-32B-Instruct-GGUF
File: qwen2.5-32b-instruct-q4_k_m.gguf
Size: 20GB
Memory: Approximately 24GB
Speed: 8-12t/s
Features:
✓ Very high level of understanding
✓ Capable of complex inferences
✓ Creative sentence generation
Recommended use: Professional writing, advanced question answering
```

#### For advanced users (70B+)

**Llama 3.1 70B Instruct**
```
Model name: meta-llama/Meta-Llama-3.1-70B-Instruct-GGUF
File: meta-llama-3.1-70b-instruct-q4_k_m.gguf
Size: 42GB
Memory: Approximately 48GB
Speed: 3-5t/s
Features:
✓ Top class performance
✓ GPT-4 level capability
✓ Complex reasoning and analysis
Recommended use: Highest quality applications, research and professional analysis.
```

**Qwen2.5 72B Instruct**
```
Model name: Qwen/Qwen2.5-72B-Instruct-GGUF
File: qwen2.5-72b-instruct-q4_k_m.gguf
Size: 44GB
Memory: Approximately 50GB
Speed: 3-5t/s
Features:
✓ Qwen's largest model
✓ Best-in-class performance in Japanese
✓ Supports 128K contexts
Recommended use: Highest quality inference in Japanese, long text processing
```

#### Professional use

**DeepSeek-Coder-V2 16B**
```
Model name: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF
File: deepseek-coder-v2-lite-instruct-q4_k_m.gguf
Size: 10GB
Memory: Approximately 13GB
Speed: 18-22t/s
Features:
✓ Coding specialized
✓ Supports multilingual programming
✓ Code generation/debugging
Recommended use: programming assistance, code review, debugging
```

### 5.2.3 Download instructions

**Standard download**

1. Select model
2. Select quantization level (Q4_K_M recommended)
3. Click the "Download" button
4. Check download progress
   ```
   Downloading... 25% (1.2 GB / 4.8 GB)
   Speed: 15 MB/s
   ETA: 4 minutes
   ```
5. Wait for completion

**💡 TIP**: LM Studio can be used while downloading. You can perform inference on other models.

**Download multiple models simultaneously**

LM Studio can download up to three models in parallel (default setting).

```
Settings → Network → Concurrent Downloads: 3
```

**Pause/resume download**

```
Pause: “⏸” button on download bar
Resume: "▶" button
Cancel: “✕” button
```

### 5.2.4 Hugging Face Direct Download

**Manual download from Hugging Face**

Models that do not appear in LM Studio can be downloaded directly from Hugging Face.

**procedure:**

1. **Access Hugging Face in your browser**
   ```
   https://huggingface.co/
   ```

2. **Search for model**
   ```
Enter "qwen2.5 gguf" etc. in the search field
   ```

3. **Download the GGUF file**
   ```
Model page → Files and versions → Click on the .gguf file
   ```

4. **Place in the model directory of LM Studio**
   ```
Windows: C:\Users\<username>\.cache\lm-studio\models\
   Linux: ~/.cache/lm-studio/models/

or a custom directory you have configured
   ```

5. **Check recognition with LM Studio**
   ```
Open the My Models tab → Manually placed models will be displayed
   ```

**Downloading large models using Git LFS (Linux)**

```bash
# Install Git LFS
sudo apt install git-lfs
git lfs install

# Clone model
cd ~/.cache/lm-studio/models/
git clone https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF

# Download only specific files
git lfs pull --include="qwen2.5-72b-instruct-q4_k_m.gguf"
```

## 5.3 Managing models

### 5.3.1 My Models Tab

**Display model list**

```
LM Studio → 📁 My Models
```

**Display information:**
```
┌──────────────────────────────────────────────────────────┐
│ Model Name              | Size  | Quantization | Added   │
├──────────────────────────────────────────────────────────┤
│ qwen2.5-7b-instruct | 4.8GB | Q4_K_M | 2 days ago │
│ llama-3.1-70b-instruct | 42GB | Q4_K_M | 1 week ago │
│ deepseek-coder-v2 | 10GB | Q4_K_M | Yesterday │
└──────────────────────────────────────────────────────────┘
```

**Sort and filter**

```
Sort:
- Sort by name (A-Z/Z-A)
- Size order (large → small / small → large)
- Recently used (New → Old / Old → New)
- Added date and time

Filter:
- Size range (0-10GB / 10-50GB / 50GB+)
- Quantization level (Q4/Q5/Q6)
- model family
```

### 5.3.2 Model details

**Model information display**

Right-click the model → "Model Info"

```
Model information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: qwen2.5-7b-instruct-q4_k_m.gguf
Size: 4.8GB
Quantization: Q4_K_M
Architecture: Llama (Qwen2.5)
Number of parameters: 7.62B
Context length: 32768 tokens
Vocabulary size: 151,936 tokens
Added date: 2025-10-15 14:23:45
Last used: 2025-10-28 09:15:32
Number of uses: 47 times
File path: ~/.cache/lm-studio/models/qwen2.5-7b-instruct-q4_k_m.gguf
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.3.3 Delete model

**Individual deletion**

```
My Models → Right click on model → Delete
Confirmation dialog → Click “Delete”
```

**Batch deletion**

```
My Models → Select multiple models with Ctrl+click (Windows/Linux)
Right click → Delete Selected
```

**⚠️ Note**: Deleted models cannot be restored. You will need to re-download it to use it again.

**Check storage usage**

```
Settings → Storage → Storage Usage

Display contents:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Models: 156.3 GB (23 models)
Cache: 4.2 GB
Logs: 128 MB
Total: 160.6 GB

Free disk space: 1.2 TB / 2 TB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.3.4 Exporting and importing models

**Export model**

If you want to share the model with other machines or users:

```bash
# copy model file
# Windows
copy "%USERPROFILE%\.cache\lm-studio\models\qwen2.5-7b-instruct-q4_k_m.gguf" D:\Shared\

# Linux
cp ~/.cache/lm-studio/models/qwen2.5-7b-instruct-q4_k_m.gguf /mnt/shared/
```

**Import the model**

```
Method 1: Copy directly from file manager
→ Place the .gguf file in the model directory

Method 2: Import within LM Studio
My Models → ⚙ → Import Model → Select file
```

### 5.3.5 Model organization strategy

**Recommended directory structure**

```
models/
├── daily_use/ # Daily use (7B-14B)
│   ├── qwen2.5-7b-instruct-q4_k_m.gguf
│   └── llama-3.2-3b-instruct-q4_k_m.gguf
├── professional/ # Professional use (32B-72B)
│   ├── qwen2.5-32b-instruct-q4_k_m.gguf
│   └── llama-3.1-70b-instruct-q4_k_m.gguf
├── specialized/ # Special use
│   ├── deepseek-coder-v2-lite-q4_k_m.gguf
│   └── mistral-7b-instruct-v0.3-q4_k_m.gguf
└── experimental/ # For experiments and tests
    └── new-model-test-q4_k_m.gguf
```

**💡 TIP**: LM Studio does not currently support subdirectories, but you can organize them by prefixing file names.

```
example:
daily_qwen2.5-7b-instruct-q4_k_m.gguf
pro_llama-3.1-70b-instruct-q4_k_m.gguf
code_deepseek-coder-v2-q4_k_m.gguf
```

## 5.4 Model Updates and Tracking Latest Versions

### 5.4.1 Model versioning

**Model update frequency**

```
Models updated frequently:
- Qwen: 1-2 times a month (minor updates)
- Llama: Quarterly (major updates)
- Mistral: Irregular (2-3 months)

Updated content:
- Bug fixes
- Improved performance
- Added new features
- Improved safety
```

**Setting update notifications**

```
Settings → Updates
[✓] Check out new versions of models
[✓] New model family notification
```

### 5.4.2 Comparing models

**Comparison of different versions**

```
My Models → Select two models you want to compare
Right click → Compare Models

Display contents:
┌────────────────────────────────────────────────────┐
│ Feature        │ Qwen2.5 7B  │ Llama 3.2 3B    │
├────────────────────────────────────────────────────┤
│ Parameters     │ 7.62B       │ 3.21B           │
│ Context Length │ 32K         │ 128K            │
│ Quantization   │ Q4_K_M      │ Q4_K_M          │
│ Size           │ 4.8GB       │ 2.0GB           │
│ Speed (Est.)   │ 35-45 t/s   │ 50-60 t/s       │
│ Japanese       │ ★★★★★      │ ★★★☆☆         │
│ English        │ ★★★★☆      │ ★★★★★         │
│ Coding         │ ★★★★☆      │ ★★★☆☆         │
└────────────────────────────────────────────────────┘
```

### 5.4.3 Recommended model update strategy

**Recommended set for MS-S1 Max (total 80GB)**

```
Minimum configuration (20GB):
✓ Qwen2.5 7B Q4_K_M (4.8GB) - Main
✓ Llama 3.2 3B Q4_K_M (2.0GB) - for high speed
✓ DeepSeek-Coder V2 16B Q4_K_M (10GB) - for coding
Remaining: 108GB

Standard configuration (50GB):
✓ Qwen2.5 14B Q4_K_M (9GB) - Main
✓ Llama 3.1 8B Q4_K_M (5GB) - for English
✓ Qwen2.5 32B Q4_K_M (20GB) - For high quality
✓ DeepSeek-Coder V2 16B Q4_K_M (10GB) - for coding
✓ Llama 3.2 3B Q4_K_M (2GB) - for high speed
Remaining: 78GB

Pro configuration (100GB):
✓ All above models
✓ Llama 3.1 70B Q4_K_M (42GB) - for highest quality
or
✓ Qwen2.5 72B Q4_K_M (44GB) - Japanese highest quality
Remaining: 28GB
```

## 5.5 Troubleshooting

### 5.5.1 Download failure

**Symptom: Download stops midway**

```
Cause 1: Network connectivity issue
Solution: Pause download → resume download

Cause 2: Temporary issue with Hugging Face server
Solution: Retry after 30 minutes

Cause 3: Insufficient disk space
Solution: Delete unnecessary files to free up space
```

**Symptom: Corrupted files**

```
Error message: "Corrupted model file" or "Invalid GGUF format"

Solution steps:
1. Delete model
2. Clear browser cache
3. Re-download
```

### 5.5.2 Model not recognized

**Symptom: Downloaded models do not appear in My Models**

```
Cause 1: File is not in the correct directory
Check: Settings → Storage → Models Directory
Solution: Move to correct directory

Cause 2: File is not completely downloaded
Check: Compare file size with Hugging Face display
Solution: Re-download

Cause 3: LM Studio needs to be reloaded
Solution: Restart LM Studio
```

### 5.5.3 Model load failure

**Symptom: "Failed to load model"**

```
Cause 1: Insufficient memory
Check: Check memory usage with task manager/htop
Solved:
- Close other applications
- Select smaller quantization level
- Reduce GPU layers

Cause 2: Corrupted model file
Solved:
- Delete model
- Re-download

Cause 3: Compatibility issue
Check: LM Studio version
Solution: Update LM Studio to the latest version
```

## 5.6 Summary of this chapter

In this chapter, you learned about downloading and managing models.

✅ **LLM model basics**
- Major model families (Llama, Qwen, Mistral, etc.)
- Quantization level (Q4_K_M recommended)
- Understanding the GGUF format

✅ **Search and download models**
- Search within LM Studio
- Recommended model catalog for MS-S1 Max
- Direct download from Hugging Face

✅ **Model management**
- Utilize the My Models tab
- Organization strategy
- Export/Import

✅ **Optimal configuration**
- Minimum configuration (20GB)
- Standard configuration (50GB)
- Pro configuration (100GB)

In the next chapter, you will learn more about inference settings and understand how each parameter affects the inference results.

---

**Previous chapter**: [Chapter 4 Complete Guide to AMD GPU Settings](chapter04_amd_gpu_settings.md)
**Go to next chapter**: [Chapter 6 Complete explanation of inference settings](chapter06_inference_settings.md)
