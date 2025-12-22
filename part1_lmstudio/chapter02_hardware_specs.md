# Chapter 2: Hardware Specifications and System Requirements

## 2.1 AMD Ryzen AI Max+ 395 Detailed Specifications

### 2.1.1 Processor Architecture

The AMD Ryzen AI Max+ 395 is an innovative APU (Accelerated Processing Unit) based on AMD's latest Zen 5 architecture. Developed under the codename " **Strix Halo** ," it is designed to be optimized for AI inference.

#### Core Configuration

```
Architecture: Zen5
Number of cores: 16 cores
Thread count: 32 threads
Base clock: 2.9 GHz
Boost Clock: Up to 5.1 GHz
L2 cache: 16MB (1MB x 16 cores)
L3 cache: 64MB (shared)
Manufacturing process: 4nm (TSMC)
TDP: 45W~55W (standard), maximum 120W (at cTDP setting)
```

#### Zen5 Architecture Features

1. **AI instruction set enhancements**

    - AVX-512 instruction support
    - VNNI (Vector Neural Network Instructions)
    - INT8/FP16 high-speed calculation

2. **IPC (instructions executed per clock) improvement**

    - Approximately 10-15% performance improvement compared to Zen4
    - More efficient branch prediction
    - Extended Micro-Operation Fusion

3. **Improved power efficiency**

    - Advanced Power Gating
    - Dynamic frequency and voltage control
    - Reduced power consumption during idle

### 2.1.2 Integrated GPU Specifications

**Radeon 8060S (RDNA 3.5)**

```
GPU architecture: RDNA 3.5
Compute Units (CU): 40
Stream Processors: 2560
Ray tracing units: 40
AI accelerators: 80
GPU Clock: Up to 2.9 GHz
Computing performance:
- FP32: Approximately 14.8 TFLOPS
- FP16: Approximately 29.6 TFLOPS
- INT8: Approximately 59.2 TOPS
Graphics memory: Shared with system memory (maximum 96GB can be allocated)
```

#### RDNA 3.5 Improvements

1. **AI Matrix Computing Unit**

    - Accelerating INT8/INT4 operations
    - Optimized for LLM inference

2. **Memory access optimization**

    - Larger L2 cache (16MB)
    - Infinity Cache: 64MB
    - Efficient Memory Controller

3. **Fully compatible with ROCm 6.x**

    - Native support in LM Studio
    - Supports PyTorch, TensorFlow, and ONNX Runtime
    - HIP (Heterogeneous-compute Interface for Portability)

### 2.1.3 NPU (Neural Processing Unit)

**Ryzen AI NPU (XDNA 2 architecture)**

```
Generation: 2nd Gen XDNA
Computing performance: 50 TOPS (INT8)
Dedicated accelerator: Yes
Supported frameworks:
- ONNX Runtime
- DirectML
- OpenVINO
- AMD Ryzen AI SDK
```

#### The role of NPU

NPUs are specialized for tasks such as:

1. **Lightweight AI Tasks**

    - Voice Recognition
    - Real-time translation
    - Image Classification
    - Object Detection

2. **Power saving inference**

    - Background AI Tasks
    - Always-on AI capabilities
    - CPU/GPU load reduction

**Note** : Currently (2025) LM Studio mainly uses GPU, NPU support will be implemented in future versions. NPU support is only available with Lemonade atm (Dec, 2025), see https://github.com/lemonade-sdk/lemonade/issues/5

### 2.1.4 Memory Subsystem

The biggest feature of the Ryzen AI Max+ 395 is its powerful memory subsystem.

#### Memory Specifications

```
Memory type: LPDDR5X
Operating frequency: 8000 MT/s (megatransfers per second)
Channel configuration: Quad channel (256-bit width)
Maximum capacity: 128GB
Theoretical bandwidth: 256 GB/s
Actual measured bandwidth: Approximately 212 GB/s (measured by rocm_bandwidth_test)
Latency: Approximately 110ns (equivalent to CL88)
```

#### Unified Memory Architecture (UMA)

The Ryzen AI Max+ 395 uses **a unified memory architecture** where the CPU, GPU, and NPU share the same memory pool.

**merit:**

- No data copying required (zero copy operation)
- Zero overhead for transfers between CPU and GPU
- Flexible memory allocation

**AMD Variable Graphics Memory:**

- Dynamically adjust VRAM allocation
- Up to 96GB can be dedicated to the GPU
- Can be balanced with system memory

#### 128GB memory usage scenario

**Example 1: Running the 70B model**

```
Model size: 70B Q4_K_M (approx. 40GB)
Context cache: 20GB (128K context)
System Reserve: 16GB
GPU VRAM Allocation: 60GB
Remaining available space: 52GB → Can be used for other applications
```

**Example 2: Multi-model execution**

```
Model 1: 34B Q5_K_M (approx. 25GB) - for chat
Model 2: 13B Q6_K (approx. 12GB) - for coding
Model 3: 7B Q8_0 (approximately 8GB) - For high-speed response
System Reserve: 16GB
Total usage: 61GB
Available remaining: 67GB
```

### 2.1.5 Overall AI performance

```
Total AI calculation performance: 126 TOPS

breakdown:
- CPU (Zen5): Approximately 16 TOPS (when using AVX-512/VNNI)
- GPU (RDNA 3.5): Approximately 60 TOPS (INT8 operations)
- NPU (XDNA 2): 50 TOPS

targets for comparison:
- Intel Core Ultra 9 288V: Approximately 47 TOPS
- Apple M3 Max: Approximately 40 TOPS (estimated)
- NVIDIA RTX 4070 Laptop: Approximately 321 TOPS (when using Tensor)
```

**💡 TIP** : Inference in LM Studio mainly uses 60 TOPS of the GPU and CPU computing power. NPU will be utilized in future updates.

## 2.2 Minisforum MS-S1 Max complete specifications

### 2.2.1 System Configuration

#### Basic specifications

```
Model number: Minisforum MS-S1 MAX
Processor: AMD Ryzen AI Max+ 395
Memory: 128GB LPDDR5X-8000 (onboard)
Storage: 2TB NVMe SSD (dual M.2 configuration)
Graphics: Radeon 8060S (integrated)
Supported OS: Windows 11 Pro, Linux (Ubuntu 22.04/24.04 recommended)
```

### 2.2.2 Detailed Hardware Specifications

#### Memory and Storage

**Memory:**

- Capacity: 128GB LPDDR5X-8000
- Configuration: On-board (not expandable)
- Channel: Quad Channel
- ECC: Not supported

**Storage:**

- Slot 1: M.2 2280 NVMe PCIe 4.0 x4 (up to 8TB)
- Slot 2: M.2 2280 NVMe PCIe 4.0 x1 (up to 8TB)
- RAID Support: RAID 0, RAID 1 (BIOS setting)

**💡 TIP** :

- Slot 1 is a high-speed PCIe 4.0 x4 connection, recommended for storing model and datasest.
- Slot 2 is only PCIe 4.0 x1 connection, recommended for installing the OS and LM Studio (system/OS files).

#### Expansion slots

**PCIe expansion slots:**

```
Slot type: PCIe x16 full length
Actual wiring: PCIe 4.0 x4
Compatible cards:
- Graphics card (models that do not require auxiliary power supply)
- NVMe SSD adapter
- 10GbE/25GbE/100GBe/200GBe network cards
- Capture Card
Maximum power supply: 75W (through slot)
```

**⚠️ Note** : When adding an external GPU, please take into consideration the MS-S1 Max's power capacity (320W). High-end GPUs that require auxiliary power cannot be used.

### 2.2.3 Power and Cooling Systems

#### Power Specifications

```
PSU Type: Built-in power supply
Rated power: 320W
Efficiency standard: 80 PLUS compliant
Input voltage: AC 100-240V (wide range)
Connector: Standard 3-pin AC power cord
```

#### TDP Settings and Performance Modes

The MS-S1 Max has four performance modes selectable in the BIOS.

**1. Performance Mode**

```
TDP: 160W (peak)
CPU Max Clock: 5.1 GHz
GPU Max Clock: 2.9 GHz
Recommended for: Inference tasks requiring maximum performance
Noise level: High (approximately 45-50 dBA)
```

**2. Balance Mode**

```
TDP: 130W (sustained)
CPU Max Clock: 4.8 GHz
GPU Max Clock: 2.7 GHz
Recommended Use: Regular use, most LLM tasks
Noise level: Moderate (approximately 38-42 dBA)
```

**3. Quiet Mode**

```
TDP: 110W
CPU Max Clock: 4.5 GHz
GPU Max Clock: 2.5 GHz
Recommended use: Quiet operation, lightweight model
Noise level: Low (approximately 32-36 dBA)
```

**4. Rack Mode**

```
TDP: 140W
CPU Max Clock: 4.9 GHz
GPU Max Clock: 2.8 GHz
Recommended use: Data center/rack environment
Noise level: High (fan speed fixed)
```

**💡 TIP** : **Balance mode** is the best for inference in LM Studio. It offers a good balance of performance and quietness, and is stable even when used for long periods of time.

#### Cooling system

```
Fan Configuration: Dual Fans
Heat pipes: 6
Heat sink: Large aluminum
Thermal Design: Integrated CPU and GPU cooling
Temperature sensors: placed in multiple locations
PWM Control: Yes (Automatic and Manual Adjustable)
```

**Temperature characteristics (Balance mode, room temperature 25℃):**

- Idle: 35-40°C
- Light load (7B model): 50-60°C
- Under heavy load (70B model): 70-80°C
- Maximum temperature: 95°C (thermal throttling begins)

### 2.2.4 Input/Output Interface

#### Front Panel

```
- USB 3.2 Gen2 Type-A × 2
- USB 3.2 Gen2 Type-C × 1
- 3.5mm audio jack (headset)
- Power button
- Status LED
```

#### Rear Panel

```
- USB4 V2 (40Gbps compatible) x 2
- Supports DisplayPort Alt Mode
- Power Delivery compatible (maximum 100W power supply possible)
- Supports data transfer, video output, and charging
- HDMI 2.1 x 1 (supports 8K@60Hz / 4K@120Hz)
- 2 x 10GbE RJ45 LAN ports (dual 10 Gigabit Ethernet)
- USB 3.2 Gen2 Type-A × 2
- DC power input
- Kensington lock slot
```

#### Wireless connection

```
Wi-Fi: Wi-Fi 7 (802.11be) compatible
- Maximum speed: 5.8 Gbps (theoretical value)
- 2.4GHz/5GHz/6GHz tri-band

Bluetooth: Bluetooth 5.4
- Low latency audio support
- Multi-point connection support
```

### 2.2.5 Physical Specifications and Design

```
Size (W x D x H): Approx. 198mm x 207mm x 67.9mm
Weight: Approx. 1.95 kg (main unit only)
Housing material: Aluminum alloy (CNC machined)
Color: Space Gray
VESA mount: Compatible (75mm x 75mm / 100mm x 100mm)
```

## 2.3 LM Studio System Requirements

### 2.3.1 Official Minimum Requirements

#### Windows

```
OS: Windows 10 (64-bit) or later
CPU: AVX2 instruction set compatible processor
RAM: 8GB or more (16GB recommended)
Storage: 10GB or more free space
GPU: Optional (NVIDIA GeForce GTX 1060 or higher, or AMD Radeon RX 5700 or higher)
```

#### Linux

```
OS: Ubuntu 22.04, Fedora 38, and other major distributions
CPU: AVX2 instruction set compatible processor
RAM: 8GB or more (16GB recommended)
Storage: 10GB or more free space
GPU: Optional
- NVIDIA: CUDA 12.x, driver 535 or later
- AMD: ROCm 6.1.2 or later, compatible GPUs (gfx1100, gfx1101, gfx1102, etc.)
```

#### macOS

```
OS: macOS 13.0 (Ventura) or later
CPU: Apple Silicon (M1 or later) or Intel (AVX2 compatible)
RAM: 8GB or more (16GB recommended)
Storage: 10GB or more free space
```

### 2.3.2 Recommended configuration for MS-S1 Max environment

The Minisforum MS-S1 Max significantly exceeds the above requirements, so we recommend the following optimal configuration:

#### OS Selection

**Windows 11 Pro (recommended)**

- Get the latest LM Studio features quickly
- Excellent driver support
- Easy to use even for beginners

**Ubuntu 24.04 LTS (Advanced)**

- High compatibility with ROCm
- More detailed performance tuning is possible
- Ideal for server applications

#### Storage configuration recommendations

**Single drive configuration (beginner)**

```
Slot 1 (PCIe 4.0 x4): 2TB NVMe SSD
- Partition 1: Windows/Linux (100GB)
- Partition 2: LM Studio and Models (remaining capacity)
Slot 2: Unused or additional storage
```

**Dual-drive configuration (advanced)**

```
Slot 1 (PCIe 4.0 x4): 4TB NVMe SSD (high speed, large capacity)
- For loading and saving models, Corpus, fine-tuning data
Slot 2 (PCIe 4.0 x1): 1TB NVMe SSD (normal speed, small capacity)
- For OS, applications

```

### 2.3.3 Model Size and Memory Requirements

Below are some general model sizes and the required memory:

#### Quantization Level Description

- **Q2** : 2-bit quantization (lightest, large quality loss)
- **Q3** : 3-bit quantization (lightweight, reducing quality)
- **Q4** : 4-bit quantization (balanced, recommended)
- **Q5** : 5-bit quantization (high quality)
- **Q6** : 6-bit quantization (very high quality)
- **Q8** : 8-bit quantization (almost original quality)
- **F16** : 16-bit floating point (full quality)

#### Memory Requirements Table

Model size | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | F16 | Context 8K
--- | --- | --- | --- | --- | --- | ---
7B | 4.4GB | 5.3GB | 6.1GB | 7.7GB | 14GB | +0.5GB
13B | 7.9GB | 9.5GB | 11GB | 14GB | 26GB | +1GB
34B | 20GB | 24GB | 28GB | 36GB | 68GB | +2.5GB
70B | 40GB | 48GB | 56GB | 74GB | 140GB | +5GB
123B | 70GB | 84GB | 98GB | 130GB | 246GB | +9GB

**Note** : Increasing the context length requires additional memory.

- 8K → 16K: Approximately double
- 8K → 32K: Approximately 4 times more
- 8K → 128K: Approximately 16 times

#### Models compatible with MS-S1 Max (128GB)

**When running a single model:**

- 70B Q8_0 + 128K context: ✅ Possible (approximately 90GB used)
- 123B Q6_K + 32K context: ✅ Possible (approximately 110GB used)
- 70B F16 + 8K context: ❌ Impossible (requires 145GB)

**Multiple Model Concurrent Execution:**

- 34B Q5_K_M × 2 + System: ✅ Possible
- 13B Q6_K × 4 + System: ✅ Possible
- 7B Q8_0 × 8 + System: ✅ Possible

### 2.3.4 AMD ROCm Version Requirements

To use AMD GPUs with LM Studio, you need the appropriate ROCm version.

#### ROCm requirements by LM Studio version

```
LM Studio 0.3.5 and earlier: ROCm 5.7.x
LM Studio 0.3.6~0.3.8: ROCm 6.0.x
LM Studio 0.3.9 or later: ROCm 6.1.2 or later (6.2.x recommended)
LM Studio 0.3.19 (latest): ROCm 6.2.x and later
```

#### Radeon 8060S (RDNA 3.5) support status

```
Architecture target: gfx1100 (RDNA 3 series)
Official ROCm support: ROCm 6.1 and later
LM Studio support: v0.3.19 or later (released July 2025)
Recommended ROCm version: ROCm 6.2.0 or later
```

**💡 TIP** : The Radeon 8060S in the Ryzen AI Max+ 395 is RDNA 3.5 generation, but is recognized as gfx1100 (RDNA 3) by ROCm.

### 2.3.5 Network Requirements

#### Model Download

- **Bandwidth** : 10Mbps or more recommended (100Mbps or more is ideal)
- **Data volume** : Depends on model size (several GB to over 100 GB)

#### Inference execution

- Local inference: no network required
- API server mode: Local network only (offline allowed)

**💡 TIP** : Take advantage of the MS-S1 Max's dual 10GbE connections to quickly load models from your NAS or server. Even better, use a Mellanox 100Gbe or higher PCIe card.

## 2.4 Performance Predictions and Benchmarks

### 2.4.1 Theoretical performance

#### Token Generation Rate Calculation

The inference speed of LLM is mainly determined by the following factors:

1. Memory Bandwidth
2. Computing performance
3. Model size and quantization level

**Simple calculation formula:**

```
Tokens/sec ≈ Memory Bandwidth (GB/s) / (Model Size (GB) × 2)
```

#### Ryzen AI Max+ 395 theoretical value

```
Memory bandwidth: 212 GB/s (measured)

7B Q4 model (4.4GB):
Theoretical value = 212 / (4.4 × 2) = approximately 24 tokens/second

13B Q4 model (7.9GB):
Theoretical value = 212 / (7.9 × 2) = approximately 13 tokens/second

34B Q4 model (20GB):
Theoretical value = 212 / (20 × 2) = approx. 5.3 tokens/second

70B Q4 model (40GB):
Theoretical value = 212 / (40 × 2) = approx. 2.7 tokens/second
```

**Note** : This is a theoretical maximum. Actual speed may vary depending on GPU performance, LM Studio optimizations, system overhead, etc.

### 2.4.2 Predicting measured performance

In actual use, you can expect performance of about 70-85% of the theoretical value.

#### Expected Token Generation Rate

Model | quantization | size | Expected speed (t/s) | Purpose
--- | --- | --- | --- | ---
Qwen2.5 7B | Q4_K_M | 4.4GB | 17-20 | High-speed chat
Qwen2.5 14B | Q4_K_M | 8.4GB | 10-12 | Balanced
Qwen2.5 32B | Q4_K_M | 19GB | 4-5 | High-quality dialogue
Llama 3.1 70B | Q4_K_M | 40GB | 2-2.5 | Highest Quality
DeepSeek-V3 | Q4_K_M | 22GB | 4-5 | coding

**💡 TIP** : For practical use, a speed of 2 tokens/second or more is sufficient. Even the 70B model of the MS-S1 Max can operate at a practical speed.

### 2.4.3 Comparison with the competitive environment

#### PC with high-end GPU

```
Configuration example: RTX 4090 (24GB VRAM) + 64GB RAM
Memory bandwidth: 1,008 GB/s (GDDR6X)
Maximum model size: 70B Q2 (VRAM limited)
Advantages: Overwhelmingly fast for models under 70B at 3.500-4.500 USD.

MS-S1 Max Advantages:
- High quality models of 70B Q4 and above can be executed
- Power efficient (320W vs 850W+)
- Low initial cost
```

#### Apple Mac Studio (M2 Ultra)

```
Configuration example: M2 Ultra + 192GB integrated memory
Memory bandwidth: 800 GB/s
Maximum model size: 123B Q6
Advantages: Larger model execution, higher bandwidth

MS-S1 Max Advantages:
- Cost-effective (approximately 1/3 the price)
- Expandability (PCIe slot)
- Native Windows support
```

#### Intel Lunar Lake (Core Ultra 9 288V)

```
Configuration example: Core Ultra 9 288V + 32GB
AI performance: 47 TOPS
Memory: Up to 32GB
Advantages: Low power consumption, mobile compatibility

MS-S1 Max Advantages:
- Approximately 2.7x AI performance (126 TOPS)
- 4x more memory
- Faster memory bandwidth
```

## 2.5 Summary of this chapter

In this chapter, you learned the following key points:

✅ AMD Ryzen AI Max+ 395 detailed architecture

- Zen5 CPU (16 cores/32 threads)
- RDNA 3.5 GPU (2560 SP, 60 TOPS)
- 50 TOPS NPU

✅ The power of 128GB LPDDR5X-8000 memory

- 256GB/s theoretical bandwidth
- 70B model can also be run comfortably
- Multi-Model Concurrency

✅ Full specifications of Minisforum MS-S1 Max

- 4 performance modes to choose from
- Dual 10GbE and USB4 V2
- Superior Cooling System

✅ LM Studio system requirements

- ROCm 6.2.x recommended
- Proper Storage Configuration
- Understanding model size and memory requirements

✅ Performance prediction

- 70B Q4 model: Approximately 2-2.5 tokens/second
- 34B Q4 model: Approximately 4-5 tokens/second
- 7B Q4 model: approx. 17-20 tokens/second

In the next chapter, we will finally get started with installing and configuring LM Studio.

---

**Previous Chapter** : [Chapter 1 Introduction](chapter01_introduction.md) **Next Chapter** : [Chapter 3 LM Studio Installation and Initial Settings](chapter03_installation.md)
