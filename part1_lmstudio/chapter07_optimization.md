# Chapter 7: Optimization settings for MS-S1 Max

## 7.1 Take advantage of the characteristics of MS-S1 Max

### 7.1.1 Strategic use of 128GB memory

The MS-S1 Max's biggest weapon is its 128GB large-capacity integrated memory. In this chapter, you will learn how to make the most of this resource.

#### Four strategies for memory utilization

**Strategy 1: Run large models**
```
Traditional 16GB GPU: up to 7B Q4
MS-S1 Max: 70B Q4 or higher is comfortable

Specific example:
  Llama 3.1 70B Q4_K_M
Memory usage: Approximately 48GB
Production speed: 3-5 t/s
Remaining memory: 80GB
→ Browser, IDE, and other apps can be used simultaneously
```

**Strategy 2: Using high quality quantization**
```
Conventional: Q4 is the limit
MS-S1 Max: Q6 and Q8 are also available

Example: Qwen2.5 32B
Q4_K_M: 20GB → speed 8-12 t/s
Q5_K_M: 24GB → speed 7-10 t/s (quality improvement)
Q6_K: 28GB → speed 6-9 t/s (even higher quality)

If you have 128GB, you can afford even Q6_K.
```

**Strategy 3: Utilize ultra-long contexts**
```
Conventional: 4K-8K is the limit
MS-S1 Max: 32K-64K is practical

Example: Qwen2.5 14B + 64K context
Model: 9GB
Context cache: approximately 70GB
Total: Approximately 79GB
→ It is possible to analyze one book.
```

**Strategy 4: Multi-model concurrent execution**
```
Load multiple models into memory at the same time

Example configuration:
1. Qwen2.5 7B Q4 (5GB): High-speed chat
2. DeepSeek-Coder V2 16B Q4 (10GB): Coding
3. Llama 3.1 8B Q4 (5GB): English only
Total: 20GB
Remaining: 108GB

advantage:
✓ Instant model switching
✓ Use the optimal model according to the application
✓ Zero loading time
```

### 7.1.2 AMD Radeon 8060S Optimization

#### Understanding GPU characteristics

```
AMD Radeon 8060S (RDNA 3.5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture: RDNA 3.5 (gfx1100)
Compute unit: 40 CU
Stream Processor: 2560 SP
FP16 performance: 29.6 TFLOPS
INT8 performance: 59.2 TOPS
Memory bandwidth: 212 GB/s (measured)
Infinity Cache: 64MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best for:
✓ 7B-14B models: Fully accelerated execution on GPU
✓ 34B model: GPU acceleration enabled
✓ 70B model: Memory bandwidth is the bottleneck
```

#### Optimizing GPU Layers settings

**Recommended settings (by model):**

```
3B-7B models:
GPU Layers: Maximum value (all layers)
VRAM allocation: 10GB
Expected speed: 35-60 t/s
GPU usage: 85-95%

13B-14B models:
GPU Layers: Maximum value (all layers)
VRAM allocation: 20GB
Expected speed: 18-25 t/s
GPU usage: 80-90%

32B-34B models:
GPU Layers: Maximum value (all layers)
VRAM allocation: 35GB
Expected speed: 8-12 t/s
GPU usage: 75-85%

70B model:
GPU Layers: Maximum value (all layers)
VRAM allocation: 60GB
Expected speed: 3-5 t/s
GPU usage: 65-75%
Note: Memory bandwidth is the main bottleneck
```

**💡 TIP**: On MS-S1 Max, it is best to offload all layers to the GPU for almost all models.

## 7.2 Recommended settings for each performance mode

### 7.2.1 BIOS performance mode selection

There are four performance modes to choose from in the MS-S1 Max's BIOS. Use them depending on the purpose.

#### Performance mode (160W)

```
TDP: 160W (peak)
CPU maximum clock: 5.1 GHz
GPU maximum clock: 2.9 GHz

temperature:
Idle: 40-45℃
7B inference: 65-75℃
70B inference: 80-90℃

Fan noise: 45-50 dBA (high)

Recommended use:
✓ Short-duration tasks requiring maximum performance
✓ Benchmark
✓ Demonstration
✓ High load inference (70B Q8 etc.)

Recommended LM Studio settings:
GPU Layers: Max
  Context: 32K-64K
  Flash Attention: ON

Performance improvement: +15-20% compared to standard
```

#### Balance mode (130W) ← Recommended

```
TDP: 130W (sustainable)
CPU maximum clock: 4.8 GHz
GPU maximum clock: 2.7 GHz

temperature:
Idle: 35-40℃
7B inference: 55-65℃
70B inference: 70-80℃

Fan noise: 38-42 dBA (acceptable range)

Recommended use:
✓ Daily reasoning tasks ← Optimal
✓ Long-term use
✓ Most applications
✓ Best balance

Recommended LM Studio settings:
GPU Layers: Max
  Context: 16K-32K
  Flash Attention: ON

Performance: Baseline (100%)
```

#### Quiet mode (110W)

```
TDP: 110W
CPU maximum clock: 4.5 GHz
GPU maximum clock: 2.5 GHz

temperature:
Idle: 30-35℃
7B inference: 45-55℃
70B inference: 60-70℃

Fan noise: 32-36 dBA (quiet)

Recommended use:
✓ Quiet environment (library, office)
✓ Night work
✓ Lightweight model (3B-14B)
✓ Recording audio/video

Recommended LM Studio settings:
GPU Layers: Max
  Context: 8K-16K
Model: 3B-14B recommended

Performance: -10-15% compared to standard
```

#### Rack mode (140W)

```
TDP: 140W
CPU maximum clock: 4.9 GHz
GPU Max Clock: 2.8 GHz

Temperature: Same as Balance
Fan noise: 42-46 dBA (fixed high speed rotation)

Recommended use:
✓ Server rack environment
✓ 24 hour operation
✓ When predictable noise levels are required

Recommended LM Studio settings:
Same as Performance mode

Performance: +8-12% compared to standard
```

### 7.2.2 Optimization by OS

#### Windows 11 optimization

**Power plan settings**

```powershell
# Create a high performance power plan
powercfg -duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# set active
powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Or set via GUI
Control Panel → Power Options
→ Select high performance
```

**Virtual memory optimization**

```
System properties → Advanced settings → Performance settings
→ Advanced tab → Virtual memory

Recommended settings:
Initial size: 16384 MB (16GB)
Maximum size: 32768 MB (32GB)

Reason: 128GB physical memory, so minimal virtual memory is OK
```

**Disable unnecessary background apps**

```
Settings → Privacy & Security
→ Background app
→ Turn off unnecessary apps

Especially recommended to disable:
✓ OneDrive (if not needed)
  ✓ Cortana
✓ Windows Search (Indexing)
✓ Sysmain (formerly Superfetch)
```

**AMD Radeon Settings**

```
Open AMD Software: Adrenalin Edition

Graphics → Advanced settings:
✓ Radeon Anti-Lag: Off (not required in LM Studio)
✓ Radeon Boost: Off
✓ Radeon Image Sharpening: Off
✓ GPU Scaling: Off

Performance → Tuning:
◉ Default (Auto)
or
◉ Manual → Power Limit: +10% (as required)
```

#### Ubuntu 24.04 optimization

**Adding kernel parameters**

```bash
# edit /etc/default/grub
sudo nano /etc/default/grub

# Add to GRUB_CMDLINE_LINUX_DEFAULT line:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.ppfeaturemask=0xffffffff amd_iommu=on iommu=pt"

# apply settings
sudo update-grub
sudo reboot
```

**Swap optimization**

```bash
# Check current swap usage trends
cat /proc/sys/vm/swappiness
# Default: 60

# Minimize swap usage (since we have 128GB memory)
sudo sysctl vm.swappiness=10

# Persistence
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
```

**CPU Governor Settings**

```bash
# set to performance governor
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Automatic configuration at startup
sudo apt install cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

**ROCm optimization**

```bash
# Add to ~/.bashrc
cat >> ~/.bashrc << 'EOF'
# ROCm optimization
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_ENABLE_SDMA=0
EOF

source ~/.bashrc
```

## 7.3 Optimal configuration for each workflow

### 7.3.1 Daily chat environment

**Purpose**: Fast and comfortable chat experience

```yaml
Configuration name: Daily Chat Setup

Hardware:
Performance mode: Balance (130W)
OS: Windows 11 or Ubuntu 24.04

Model configuration:
Main model: Qwen2.5 7B Q4_K_M
Submodel: Llama 3.2 3B Q4_K_M (for super high speed)

LM Studio settings:
GPU Layers: Max
  Context Length: 16384
  Temperature: 0.7
  Top P: 0.95
  Max Tokens: 2048

Expected performance:
Response start: Immediate (< 0.5 seconds)
Production speed: 35-45 t/s
Memory usage: Approximately 8GB
Remaining memory: 120GB (can be used for other tasks)

Recommended use:
✓ Daily question and answer
✓ Create email
✓ Idea generation
✓ Easy sentence generation
```

### 7.3.2 Professional writing environment

**Purpose**: High-quality long text generation

```yaml
Configuration name: Professional Writing Setup

Hardware:
Performance mode: Balance (130W)
OS: Windows 11 or Ubuntu 24.04

Model configuration:
Main model: Qwen2.5 32B Q5_K_M
Sub model: Qwen2.5 14B Q4_K_M (for draft)

LM Studio settings:
GPU Layers: Max
  Context Length: 32768
  Temperature: 0.75
  Top P: 0.92
  Repeat Penalty: 1.15
  Max Tokens: 8192

Expected performance:
Generation rate: 7-10 t/s (32B Q5)
Memory usage: Approximately 60GB (32K context included)
Remaining memory: 68GB

Recommended use:
✓ Blog article writing
✓ Technical document creation
✓ Report creation
✓ Book writing
```

### 7.3.3 Development/coding environment

**Purpose**: Efficient code generation and review

```yaml
Configuration name: Coding Assistant Setup

Hardware:
Performance mode: Balance (130W)
OS: Ubuntu 24.04 recommended (development environment)

Model configuration:
Main model: DeepSeek-Coder-V2 16B Q4_K_M
Submodel: Qwen2.5 7B Q4_K_M (for document generation)

LM Studio settings:
GPU Layers: Max
  Context Length: 16384
Temperature: 0.2 (focus on accuracy)
  Top P: 0.90
  Max Tokens: 4096

Integration:
VS Code + Continue extension
Using LM Studio Local Server mode

Expected performance:
Generation rate: 18-22 t/s
Code completion latency: < 1 second
Memory usage: Approximately 15GB
Remaining memory: 113GB

Recommended use:
✓ Code generation
✓ Code review
✓ Bug fix suggestions
✓ Refactoring
✓ Automatic document generation
```

### 7.3.4 Research and analysis environment

**Purpose**: Highest quality reasoning and long text analysis

```yaml
Configuration name: Research & Analysis Setup

Hardware:
Performance mode: Performance (160W)
Cooling: Optimized (room temperature below 25℃ recommended)
OS: Ubuntu 24.04 recommended

Model configuration:
Main model: Llama 3.1 70B Q4_K_M
or Qwen2.5 72B Q4_K_M (emphasis on Japanese)

LM Studio settings:
GPU Layers: Max
  Context Length: 65536（64K）
Temperature: 0.5 (balanced)
  Top P: 0.92
  Max Tokens: 8192

Expected performance:
Production speed: 3-5 t/s
Memory usage: Approximately 100GB (64K context included)
Remaining memory: 28GB

Recommended use:
✓ Analysis of academic papers
✓ Summary of the entire book
✓ Complex reasoning tasks
✓ Multi-step problem solving
```

## 7.4 Advanced memory management techniques

### 7.4.1 Understanding Context Cache

LM Studio maintains conversation history as a context cache.

**How ​​the cache works:**

```
Conversation flow:

User: "Hello" (5 tokens)
AI: "Hello!..." (50 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cash: 55 tokens

User: "What's the weather like today?" (8 tokens)
AI: "I'm sorry..." (45 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cash: 108 tokens

...as the conversation continues...

Cache reaches Context Length
→ Removed from old conversations
```

**Memory impact:**

```
Example: Qwen2.5 32B, 32K context

At the beginning of the conversation:
Model: 20GB
Cache: 1GB (almost empty)
Total: 21GB

After 15 minutes (15K token conversation):
Model: 20GB
Cache: 35GB
Total: 55GB

After 30 minutes (32K tokens, max reached):
Model: 20GB
Cache: 72GB (max)
Total: 92GB
```

**Optimization techniques:**

```
Technique 1: Resetting the conversation
Reset the conversation with "New Chat" after using it for a long time
→ Cache is cleared and memory is released

Technique 2: Dynamically adjusting context length
Short conversation: 8K settings
Requires a long conversation: Change to 32K setting

Technique 3: Managing multiple chats
Separate chats by purpose
Delete unnecessary chats
```

### 7.4.2 Efficient switching between multiple models

**Method 1: Unload the model**

```
Current model: Llama 3.1 70B (48GB in use)
↓
Chat screen → Model selection → "Unload Model"
↓
Memory free: 48GB
↓
Load new model: Qwen2.5 7B (5GB)
```

**Method 2: Preload (MS-S1 Max recommended)**

```
If you have enough memory, you can load multiple models at the same time.

example:
Model 1: Qwen2.5 7B (5GB) - Chat A
Model 2: DeepSeek-Coder 16B (10GB) - Chat B
Model 3: Llama 3.2 3B (2GB) - Chat C
Total: 17GB
Remaining: 111GB

switching:
Just switch between Chat A, B, and C tabs
Zero loading time
```

**Method 3: Script automation (advanced)**

```bash
# Automatic switching via LM Studio API
curl -X POST http://localhost:1234/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b-instruct-q4_k_m"}'
```

## 7.5 Performance Monitoring

### 7.5.1 Real-time monitoring tools

#### Windows

**Task Manager Enhanced Version**

```
Ctrl+Shift+Esc → Performance tab

Check items:
CPU usage: 15-30% during inference
Memory: monitor usage
GPU: AMD Radeon Graphics usage
```

**HWiNFO64 (recommended)**

```
Download: https://www.hwinfo.com/

Monitoring items:
✓ CPU temperature (each core)
✓ GPU temperature
✓ Memory usage
✓ GPU usage
✓ Power consumption
✓ Clock frequency

setting:
Sensors → Create custom layout
Show in system tray
```

#### Linux

**Command line tools**

```bash
# CPU, memory monitoring
htop

# GPU monitoring
watch -n 1 rocm-smi

# Integrated monitoring (recommended)
sudo apt install nvtop # Also supports AMD GPU
nvtop
```

**Integrated monitoring script**

```bash
#!/bin/bash
# ms-s1-max-monitor.sh

while true; do
  clear
  echo "=== MS-S1 Max Performance Monitor ==="
  echo ""
  echo "--- CPU ---"
  top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "CPU Usage: " 100 - $1"%"}'

  echo ""
  echo "--- Memory ---"
  free -h | awk '/^Mem:/ {print "Used: " $3 " / " $2 " (" $3/$2*100 "%)"}'

  echo ""
  echo "--- GPU ---"
  rocm-smi --showuse | grep "GPU use" || echo "ROCm not available"

  echo ""
  echo "--- Temperature ---"
  sensors | grep -A 0 'edge' | head -1

  sleep 2
done
```

### 7.5.2 Bottleneck diagnosis

**Diagnosis chart by symptom:**

```
Symptom: Inference speed is slow (less than 50% of expected value)

→ Check GPU usage rate

GPU usage < 50%:
→ CPU/memory bottleneck
→ Solution:
1. Close background apps
2. Check GPU Layers settings
3. Restart LM Studio

GPU usage > 80%:
→ GPU performance limit or memory bandwidth
→ Solution:
1. Try a smaller model
2. Try lighter quantization (Q6→Q4)
3. Change performance mode to Performance

Temperature > 85℃:
→ Thermal throttling
→ Solution:
1. Lower the room temperature
2. Ensure ventilation of the main unit
3. Change performance mode to Balance
```

## 7.6 Summary of this chapter

In this chapter, you learned about optimizations specific to MS-S1 Max.

✅ **128GB memory utilization strategy**
- Large model execution (70B Q4)
- High quality quantization (Q6, Q8)
- Very long context (32K-64K)
- Multi-model simultaneous execution

✅ **Performance mode selection**
- Balance (130W): Highly recommended, daily use
- Performance (160W): Maximum performance
- Quiet (110W): Quiet environment

✅ **Optimal configuration for each workflow**
- Daily chat: Qwen2.5 7B + 16K
- Written by: Qwen2.5 32B + 32K
- Coding: DeepSeek-Coder 16B
- Research: Llama 70B + 64K

✅ **Performance Monitoring**
- Real-time monitoring tools
- Bottleneck diagnosis

In the next chapter, you will learn how to use it practically.

---

**Go to previous chapter**: [Chapter 6 Complete explanation of inference settings](chapter06_inference_settings.md)
**Next Chapter**: [Chapter 8 Practical Usage](chapter08_practical_usage.md)
