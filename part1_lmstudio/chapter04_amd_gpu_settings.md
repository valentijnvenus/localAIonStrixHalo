# Chapter 4: Complete Guide to AMD GPU Configuration

## 4.1 Fundamentals of GPU Offload

### 4.1.1 What is GPU offload?

**GPU offload** is a technology that significantly speeds up LLM inference calculations by moving them from the CPU to the GPU.

#### Working principle

LLM is made up of many "layers". Each layer performs the following operations:

```
Input → Matrix multiplication → Activation function → Normalization → Output
```

GPU offloading benefits from parallel processing by running these layers on the GPU.

#### Comparison of CPU execution vs GPU execution

**CPU execution (GPU Layers: 0)**
```
Speed: 2-5 tokens/s (70B model)
Memory: Uses system RAM
Advantage: Can run with less VRAM
Disadvantages: very slow
```

**Full GPU execution (GPU Layers: All)**
```
Speed: 15-60 tokens/s (depending on model size)
Memory: Uses GPU VRAM (unified memory on MS-S1 Max)
Advantage: maximum speed
Disadvantages: Requires sufficient VRAM
```

**Hybrid execution (GPU Layers: some)**
```
Speed: Moderate (proportional to number of layers)
Memory: Uses both RAM and VRAM
Advantage: Compromise when memory is low
Disadvantage: Data transfer overhead
```

**💡 TIP**: MS-S1 Max has 128GB of integrated memory, so full GPU offload is best in most cases.

### 4.1.2 GPU Layers Settings

In LM Studio, adjust the number of layers assigned to the GPU with the "GPU Layers" slider.

#### Setting method

1. **Before loading the model**:
   ```
Chat screen → Model selection → ⚙️ (gear icon) → GPU Settings
   ```

2. **GPU Layers slider**:
   ```
Minimum value: 0 (CPU only)
Maximum: Number of layers in the model (e.g. 35, 80, etc.)
Recommended value: Maximum value (offload all layers to GPU)
   ```

3. **Automatic configuration**:
   ```
Click the [Auto] button
→ Automatically calculated by LM Studio based on available VRAM
   ```

#### Number of layers by model

| Model | Number of layers | Recommended GPU Layers (MS-S1 Max) |
|--------|-----------|----------------------------|
| Qwen2.5 7B | 32 | 32 (all) |
| Qwen2.5 14B | 40 | 40 (all) |
| Llama 3.1 8B | 32 | 32 (all) |
| Llama 3.1 70B | 80 | 80 (all) |
| Mistral 7B | 32 | 32 (all) |
| Mixtral 8x7B | 32 | 32 (all) |

**⚠️ Note**: The number of layers may vary depending on the model variation.

### 4.1.3 Understanding memory allocation

#### Memory allocation in MS-S1 Max

```
Total memory: 128GB LPDDR5X

Allocation example (when running 70B Q4 model):
┌─────────────────────────────────────────┐
│ OS reservation: 4GB │
│ System process: 4GB │
│ LM Studio main unit: 2GB │
│ Model weight: 40GB │
│ Context cache: 8GB (32K context) │
│ KV cache: 10GB │
│ Working memory: 10GB │
│ ─────────────────────────────────────   │
│ Total usage: 78GB │
│ Remaining available: 50GB │
└─────────────────────────────────────────┘
```

**Dynamic memory allocation**

With AMD Variable Graphics Memory technology, the GPU dynamically allocates VRAM from system memory.

```bash
# Linux: Check current memory usage status
rocm-smi --showmeminfo

# Example output:
# GPU[0]: Memory Total: 96GB (dynamic allocation)
# GPU[0]: Memory Used: 48GB
# GPU[0]: Memory Free: 48GB
```

## 4.2 Advanced GPU Settings (Hardware Settings)

### 4.2.1 Open the Hardware Settings screen

**How ​​to access**

- **Keyboard shortcut**: `Ctrl+Shift+H` (Windows/Linux)
- **From the menu**: Settings → Advanced → Hardware Settings

### 4.2.2 GPU Enablement Settings

#### Enable GPU Acceleration

```
[✓] Enable GPU Acceleration

Feature: Enable inference using GPU
Recommended setting: ON (checked)
When to disable: Only during debugging and CPU performance testing
```

**Check the effect**

```
When enabled:
- "🎮 GPU: Active" is displayed on the status bar
- Significantly improved inference speed
- GPU usage increased to 70-95%

When disabled:
- "💻CPU Only" display
- Inference speed reduced to less than 1/10
- CPU usage increased to 100%
```

### 4.2.3 GPU selection settings (multi-GPU environment)

MS-S1 Max only has an integrated GPU, but this is a configuration if you add an external GPU to the PCIe slot in the future.

#### GPU Device Selection

```
Available GPUs:
[✓] GPU 0: AMD Radeon 8060S (integrated)
[✓] GPU 1: NVIDIA RTX 4060 (PCIe) ← If added

Allocation strategy:
◉ Even Distribution
○ Priority Order
○ Manual (manual assignment)
```

**Allocation Strategy Description**

1. **Even Distribution**
   ```
Assign layers equally to each GPU
Example: 80 layer model, 2 GPUs
→ GPU 0: 40 layers, GPU 1: 40 layers
   ```

2. **Priority Order**
   ```
Assign to GPUs in order of priority
Example: Use GPU 0 until full, then GPU 1
   ```

3. **Manual**
   ```
Manually specify the number of layers for each GPU
For advanced tuning
   ```

**💡 TIP**: These settings are not required in a single GPU environment (standard MS-S1 Max).

### 4.2.4 Memory limit settings

#### VRAM Limit

```
Setting item: Maximum VRAM Usage
Setting value: 4GB ~ 96GB
Default: Auto
Recommended value: 80GB (MS-S1 Max)
```

**Setting intent**

This is a limit to leave memory for other applications (browsers, IDEs, etc.).

```
Setting example:
80GB limit = 80GB reserved for model and 48GB reserved for system
```

#### GPU Memory Type

```
◉ Unified Memory ← MS-S1 Max default
○ Dedicated Only (dedicated memory only)
```

**Unified Memory (recommended)**
- CPU and GPU share memory
- Standard configuration of MS-S1 Max
- Ideal for running large models

**Dedicated Only**
- Uses only dedicated VRAM (dGPU)
- Not normally used with MS-S1 Max

### 4.2.5 Flash Attention Settings

**Flash Attention** is a technique that speeds up the calculation of attention mechanisms.

#### Flash Attention v2

```
[✓] Enable Flash Attention 2

Feature: Memory-efficient attention calculation
effect:
- Memory usage reduced by 20-30%
- Inference speed improved by 10-20%
- Particularly useful in long contexts (32K+)

Recommended setting: ON (checked)
```

**Compatible models**

- Llama 3 series
- Qwen2 series
- Mistral type
- Most of the other latest models

**Non-compatible models**

- Old GPT-2 base model
- Some custom architectures

**⚠️ Note**: Automatically turns off on non-compatible models.

## 4.3 Optimizing ROCm environment (Linux)

### 4.3.1 ROCm environment variables

On Linux, you can adjust ROCm's behavior with environment variables.

#### Basic environment variables

```bash
# Add to ~/.bashrc or ~/.zshrc

# Specify GPU target (recognize RDNA 3.5 as gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# ROCm path
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# HIP settings
export HIP_VISIBLE_DEVICES=0 # GPU ID to use

# Device memory limit (optional, unit: bytes)
# export HSA_XNACK=1 # Enable page fault handling
```

**Reflection of settings**

```bash
source ~/.bashrc # or ~/.zshrc
```

### 4.3.2 ROCm Performance Profiling

#### Monitor GPU activity

```bash
# Real-time GPU usage monitoring
watch -n 1 rocm-smi

# Example output:
# ========================ROCm System Management Interface========================
# GPU  Temp   AvgPwr  SCLK     MCLK     Fan   Perf  PwrCap  VRAM%  GPU%
# 0    68.0c  95.0W   2900Mhz  1000Mhz  65%   auto  120.0W  45%    92%
```

#### Memory Bandwidth Test

```bash
# ROCm memory bandwidth test tool
/opt/rocm/bin/rocm_bandwidth_test

# Expected results (MS-S1 Max):
# Unidirectional copy peak bandwidth GB/s: ~212 GB/s
```

#### HIP Profiling

```bash
# Enable profiling when starting LM Studio
ROCM_PROFILE=1 /usr/local/bin/lmstudio

# Profile data is saved in ~/.rocm_profile/
```

### 4.3.3 Kernel parameter optimization

#### AMDGPU kernel options

```bash
# Edit /etc/modprobe.d/amdgpu.conf
sudo nano /etc/modprobe.d/amdgpu.conf

# Add the following:
options amdgpu ppfeaturemask=0xffffffff
options amdgpu gpu_recovery=1
options amdgpu noretry=0
```

**Parameter description**

- `ppfeaturemask=0xffffffff`: Enable all power management features
- `gpu_recovery=1`: Enable automatic recovery when GPU hangs
- `noretry=0`: Enable memory retry (for large models)

**How ​​to apply**

```bash
# reload kernel module
sudo update-initramfs -u
sudo reboot
```

### 4.3.4 Configuring Performance Mode

#### GPU clock profile

```bash
# Check current performance mode
sudo rocm-smi --showprofile

# Set to high performance mode (recommended for inference)
sudo rocm-smi --setperflevel high

# return to automatic mode
sudo rocm-smi --setperflevel auto
```

#### Adjusting power limits

```bash
# Check current power limit
sudo rocm-smi --showpower

# Set power limit to 120W (default)
sudo rocm-smi --setpoweroverdrive 120

# ⚠️ Caution: Do not exceed the TDP setting (BIOS) of MS-S1 Max
```

## 4.4 Temperature management and thermal throttling

### 4.4.1 Temperature monitoring

#### Real-time temperature monitoring

**Linux:**
```bash
# Monitor temperature and fan speed
watch -n 1 "rocm-smi | grep -E 'Temp|Fan'"

# sensors command (lm-sensors package)
sensors | grep -A 5 amdgpu
```

**Windows:**
```
AMD Software: Adrenalin Edition
→ Performance → Metrics
```

#### Temperature threshold

```
Temperature characteristics of MS-S1 Max (Balance mode):

Idle: 35-40℃
Light load (7B inference): 50-60℃
Medium load (34B inference): 60-75℃
High load (70B inference): 75-85℃

⚠️ Warning temperature: 90℃
🚨 Critical: 95℃ (Thermal throttling starts)
🛑 Shutdown: 105℃
```

### 4.4.2 Cooling optimization

#### Performance mode selection in BIOS

MS-S1 Max BIOS settings (Press DEL or F2 key at startup)

```
Advanced → Power Management → Performance Mode

Options and characteristics:

1. Performance（160W）
Temperature: High (75-85℃ commonly used)
Speed: Max
Fan sound: loud
Recommended use: Highest performance inference in short time

2. Balance (130W) ← Recommended
Temperature: Moderate (65-75℃ regular use)
Speed: High
Fan noise: acceptable
Recommended use: Regular reasoning tasks

3. Quiet（110W）
Temperature: Low (55-65℃ commonly used)
Speed: Slightly low
Fan sound: quiet
Recommended use: Working in quiet environments

4. Rack（140W）
Temperature: High (commonly used at 70-80℃)
Speed: High
Fan noise: constant (high)
Recommended use: Server rack environment
```

#### Customizing the fan curve (Linux)

```bash
# fancontrol settings
sudo apt install lm-sensors fancontrol

# sensor detection
sudo sensors-detect

# Fan curve setting
sudo pwmconfig

# start fancontrol
sudo systemctl enable fancontrol
sudo systemctl start fancontrol
```

**Custom fan curve example:**
```
Below 30℃: 30% (minimum rotation speed)
30-50℃: 30-45% (slow rise)
50-70℃: 45-70% (moderate)
70-85℃: 70-90% (high speed rotation)
85℃ or higher: 100% (maximum rotation)
```

### 4.4.3 Thermal throttling measures

#### Thermal throttling detection

**Linux:**
```bash
# Check throttling events with dmesg
sudo dmesg | grep -i "thermal"
sudo dmesg | grep -i "throttle"

# Example output (when throttling occurs):
# [12345.678] amdgpu 0000:01:00.0: GPU thermal throttling activated
```

**Windows:**
```
Event Viewer → Windows Logs → System
Filter: source "amdgpu" or "thermal"
```

#### Countermeasures

**1. Environmental improvement**
```
- Ensure ventilation around the main unit (more than 10 cm from front to back and left to right)
- Lower the room temperature (air conditioner, 25℃ or less recommended)
- Install the main unit in a high position (cold air collects at the bottom)
```

**2. Performance mode change**
```
Change Performance → Balance in BIOS settings
```

**3. GPU Layers reduction (last resort)**
```
If you are offloading all layers to the GPU:
Reduce GPU Layers to 75-80%
Example: 80 layers → 60 layers

effect:
- Temperature decreased by 5-10℃
- 10-20% speed reduction (tradeoff)
```

## 4.5 Benchmarking and performance measurement

### 4.5.1 LM Studio built-in benchmark

**Running the benchmark**

1. Load the model
2. "⋮" menu at the top right of the Chat screen
3. Select “Run Benchmark”
4. Test parameter settings:
   ```
   Prompt length: 512 tokens
   Generation length: 128 tokens
   Runs: 3
   ```
5. 「Start Benchmark」

**How ​​to read the results**

```
Benchmark results:

Prompt Processing:
  - Speed: 1250 tokens/s
  - Time: 0.41s

Text Generation:
  - Speed: 15.3 tokens/s
  - Time: 8.37s

Total Time: 8.78s
Peak VRAM: 42.5 GB
```

### 4.5.2 Performance test by model

#### Test model and expected values ​​(MS-S1 Max, Balance mode)

| Model | Quantization | Prompt Speed ​​| Generation Speed ​​| VRAM Usage |
|--------|--------|---------------|---------|----------|
| Qwen2.5 3B | Q4_K_M | 2000+ t/s | 50-60 t/s | 2.5GB |
| Qwen2.5 7B | Q4_K_M | 1500+ t/s | 35-45 t/s | 4.8GB |
| Qwen2.5 14B | Q4_K_M | 1000+ t/s | 20-25 t/s | 9GB |
| Qwen2.5 32B | Q4_K_M | 600+ t/s | 8-12 t/s | 20GB |
| Llama 3.1 70B | Q4_K_M | 300+ t/s | 3-5 t/s | 42GB |

**💡 TIP**: Prompt processing speed indicates the speed of context understanding. Generation rate is the actual response generation rate.

### 4.5.3 Optimization Checklist

Check the following to see if your settings are optimized.

```
✅ GPU Settings
[✓] GPU Acceleration: Enabled
[✓] GPU Layers: Maximum value (all layers)
[✓] Flash Attention 2: Enabled
[✓] VRAM Limit: 80GB or more

✅ System Settings
[✓] Performance mode: Balance (or Performance if required)
[✓] Temperature: Maintain below 85℃
[✓] Background apps: minimized

✅ Driver & Runtime（Linux）
[✓] ROCm 6.2 or later
[✓] HSA_OVERRIDE_GFX_VERSION=11.0.0 set
[✓] Belongs to the render/video group

✅ Performance Indicators
[✓] GPU usage: 70-95% (during inference)
[✓] Generation speed: within ±20% of expected value
[✓] Throttling: Not occurring
```

## 4.6 Troubleshooting

### 4.6.1 Common problems and solutions

#### Problem 1: GPU not recognized

**Symptoms:**
```
"CPU Only" displayed on status bar
GPU Layers slider is grayed out
```

**Solution (Windows):**
```
1. Update AMD Software to the latest version
2. Update LM Studio to the latest version
3. Reinstall GPU driver in Device Manager
4. Restart Windows
```

**Solution (Linux):**
```bash
# Reinstall ROCm
sudo apt remove --purge rocm-*
sudo apt autoremove
# Reinstall ROCm using the steps in Chapter 3

# Check environment variables
echo $HSA_OVERRIDE_GFX_VERSION # Should display 11.0.0

# Check user privileges
groups | grep render # render should be included
```

#### Problem 2: Inference speed is extremely slow

**Symptoms:**
```
5 tokens/s or less for 7B model
GPU usage is below 10%
```

**Cause and solution:**

**Cause A: Few GPU layers**
```
Verify: Check the value of GPU Layers
Solution: Set slider to maximum value
```

**Cause B: Operating in CPU mode**
```
Confirm: Check the status bar
Solution: Enable GPU Acceleration in Hardware Settings
```

**Cause C: Thermal throttling**
```
Confirm: GPU temperature is above 90℃
Solved:
- Reduce environmental temperature
- Change performance mode to Balance
- Ensure ventilation of the main unit
```

**Cause D: Background process**
```
confirmation:
  # Windows
Check memory usage in task manager

  # Linux
Check memory with htop or top

Solution: Close unnecessary applications
```

#### Problem 3: Out of memory error

**Symptoms:**
```
Error message: "Out of memory"
Failed to load model
```

**Solution:**

```
1. Select a smaller quantization level
   Q8 → Q6 → Q5 → Q4

2. Reduce context length
   128K → 32K → 8K

3. Close other applications
Browsers, IDEs, etc.

4. Reduce model size
   70B → 34B → 13B
```

#### Problem 4: Abnormal termination/crash

**Symptoms:**
```
LM Studio quits unexpectedly during inference
Response stops midway
```

**Solution:**

**Windows:**
```
1. Check the error log in event viewer
2. Run GPU diagnostics with AMD Software
3. Check the LM Studio log:
   %APPDATA%\LM Studio\logs\
4. Clean installation
```

**Linux:**
```bash
# Check system log
sudo dmesg | tail -50
journalctl -xe | grep lmstudio

# Check GPU hang
sudo dmesg | grep "GPU hang"

# Check LM Studio log
~/.config/LM Studio/logs/

# GPU reset
sudo systemctl restart display-manager
```

### 4.6.2 Enabling detailed logging

**Get debug information**

```bash
# Linux: Start LM Studio in verbose log mode
LMSTUDIO_LOG_LEVEL=debug /usr/local/bin/lmstudio

#Log output destination
tail -f ~/.config/"LM Studio"/logs/main.log
```

**Windows:**
```
LM Studio Settings → Advanced → Enable Debug Logging
After reboot, logs are saved to:
%APPDATA%\LM Studio\logs\debug.log
```

## 4.7 Summary of this chapter

In this chapter, you learned more about AMD GPU settings.

✅ **GPU Offload Basics**
- Layer-based offload mechanism
- Full layer offload is optimal for MS-S1 Max

✅ **Advanced GPU settings**
- Enable GPU Acceleration
- Utilization of Flash Attention 2
- Memory limit settings

✅ **ROCm optimization (Linux)**
- Setting environment variables
- Performance profiling
- Kernel parameter adjustment

✅ **Temperature control**
- Performance mode selection (Balance recommended)
- Measures against thermal throttling
- Customize fan curve

✅ **Benchmarking and troubleshooting**
- Performance measurement method
- Solutions to common problems

In the next chapter, you will learn how to actually download and manage models.

---

**Go to previous chapter**: [Chapter 3 LM Studio installation and initial settings](chapter03_installation.md)
**Next Chapter**: [Chapter 5 Model Download and Management](chapter05_model_management.md)
