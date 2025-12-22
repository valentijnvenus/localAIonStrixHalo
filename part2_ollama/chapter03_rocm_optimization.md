# Chapter 3: ROCm Settings and AMD GPU Optimization

## 3.1 Basic knowledge of ROCm

### 3.1.1 What is ROCm?

**ROCm (Radeon Open Compute)** is an open source GPU computing platform developed by AMD.

```
ROCm consists of:

┌────────────────────────────────────────┐
│ Application Layer │
│ (Ollama, PyTorch, TensorFlow) │
├────────────────────────────────────────┤
│ Library layer │
│ (rocBLAS, MIOpen, hipBLAS) │
├────────────────────────────────────────┤
│ Runtime Layer │
│ (HIP, HSA) │
├────────────────────────────────────────┤
│ Driver Layer │
│ (amdgpu, amdkfd) │
├────────────────────────────────────────┤
│ Hardware │
│ (Radeon 8060S - RDNA 3.5) │
└────────────────────────────────────────┘
```

### 3.1.2 ROCm requirements for MS-S1 Max

```bash
# Recommended version
ROCm: 6.1.0 or later (6.3.0 recommended)
Kernel: 6.5 or later
Ubuntu: 22.04 LTS / 24.04 LTS
```

### 3.1.3 Checking the current environment

```bash
# ROCm version
rocm-smi --version

# Kernel version
uname -r

# GPU driver version
modinfo amdgpu | grep ^version
```

## 3.2 ROCm complete installation

### 3.2.1 Deleting existing ROCm (clean install)

```bash
# Completely delete the existing ROCm
sudo apt purge -y rocm-* hip-* miopen-*
sudo apt autoremove -y
sudo apt autoclean

# Delete the configuration file too
sudo rm -rf /opt/rocm*
sudo rm -rf ~/.cache/hip
```

### 3.2.2 Installing ROCm 6.3

```bash
# Add AMD's GPG key
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Add ROCm repository (for Ubuntu 22.04)
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3 jammy main" \
| sudo tee /etc/apt/sources.list.d/rocm.list

# For Ubuntu 24.04
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3 noble main" \
| sudo tee /etc/apt/sources.list.d/rocm.list

# Priority setting
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
| sudo tee /etc/apt/preferences.d/rocm-pin-600

# Update repository
sudo apt update
```

```bash
# Install the full version of ROCm
sudo apt install -y rocm-hip-sdk rocm-libs

# Minimum required packages for Ollama
sudo apt install -y \
rocm-hip-runtime \
rocm-smi-lib \
hip-runtime-amd \
rocm-core

# Development tools (optional)
sudo apt install -y \
rocm-dev \
rocm-utils \
rocminfo \
rocm-bandwidth-test
```

### 3.2.3 User permission settings

```bash
# Add to render and video groups
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# confirmation
groups $USER

# Example output
username : username adm cdrom sudo dip plugdev render video
```

**⚠️ Important:** After adding a group, you will need to log in again or restart your device.

```bash
# Re-login
exit
# Log in again

# or reboot
sudo reboot
```

## 3.3 Optimizing AMD GPU settings

### 3.3.1 Setting environment variables

```bash
# Add to ~/.bashrc
nano ~/.bashrc

# Add the following to the end
# ========== ROCm Configuration ==========
# ROCm Path
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# GPU Target (RDNA 3.5 → gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# HIP Settings
export HIP_VISIBLE_DEVICES=0
export HIP_LAUNCH_BLOCKING=0

# Performance
export ROCM_HOME=/opt/rocm
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100

# Ollama Specific
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=2
# ========================================
```

```bash
# Apply settings
source ~/.bashrc

# confirmation
echo $HSA_OVERRIDE_GFX_VERSION # should show 11.0.0
```

### 3.3.2 System-wide environment variables

Set the system-wide environment variables for the Ollama service.

```bash
# Add to /etc/environment
sudo nano /etc/environment

# Add (existing lines remain)
HSA_OVERRIDE_GFX_VERSION=11.0.0
ROCM_HOME=/opt/rocm
```

### 3.3.3 Ollama service environment variables

```bash
# Environment variables for Ollama service
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo nano /etc/systemd/system/ollama.service.d/rocm.conf
```

**Contents of rocm.conf:**

```ini
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="ROCM_HOME=/opt/rocm"
Environment="PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin"
Environment="LD_LIBRARY_PATH=/opt/rocm/lib"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="GPU_MAX_HEAP_SIZE=100"
Environment="GPU_MAX_ALLOC_PERCENT=100"
Environment="OLLAMA_FLASH_ATTENTION=1"
```

```bash
# Apply settings
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## 3.4 Checking GPU recognition

### 3.4.1 Checking with the ROCm tool

```bash
# ROCm system information
rocminfo | grep -A 10 "Agent"

# Expected output
Agent 2
  Name:                    gfx1100
  Uuid:                    GPU-XXXXXXXXXXXX
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Max Queue Size:          0(0x0)
  Queue Min Size:          0(0x0)
  Queue Type:              MULTI
```

```bash
# GPU list
rocm-smi --showproductname

# Example output
GPU[0] : Card series: AMD Radeon Graphics
GPU[0] : Card model: 0x1900
GPU[0] : Card vendor: Advanced Micro Devices, Inc. [AMD/ATI]
```

### 3.4.2 HIP operation check

```bash
# HIP version
hipconfig --version

# HIP Platform
hipconfig --platform

# Output: amd
```

**Simple test program:**

```bash
# Create a test file
nano hip_test.cpp
```

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Number of HIP devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    }
    return 0;
}
```

```bash
# Compile and run
hipcc hip_test.cpp -o hip_test
./hip_test

# Expected output
Number of HIP devices: 1
Device 0: AMD Radeon Graphics
Compute Capability: 11.0
Total Memory: 96 GB
```

### 3.4.3 GPU recognition check in Ollama

```bash
# Check GPU detection in Ollama log
sudo journalctl -u ollama | grep -i gpu

# Expected output
Detected GPU: AMD Radeon Graphics (gfx1100)
Using ROCm backend
GPU Memory: 96GB available
```

```bash
# Ollama runtime log
ollama run qwen2.5:7b --verbose
```

## 3.5 Performance Profiling

### 3.5.1 Real-time monitoring of GPU usage

```bash
# Monitoring with rocm-smi
watch -n 1 rocm-smi

# Display detailed information
watch -n 1 "rocm-smi --showmeminfo vram --showuse"
```

**Expected value during inference (7B model):**

```
GPU  Temp   AvgPwr  SCLK     MCLK     Fan   Perf  PwrCap  VRAM%  GPU%
0    65.0c  85.0W   2800Mhz  1000Mhz  55%   auto  120.0W  35%    88%
```

### 3.5.2 Memory Bandwidth Test

```bash
# ROCm bandwidth test
/opt/rocm/bin/rocm_bandwidth_test

# Important output
Unidirectional copy peak bandwidth GB/s:
Host to Device: 212.45
Device to Host: 215.32
Device to Device: 1024.78
```

**Expected values for MS-S1 Max:**

- Host↔Device: 200-220 GB/s (close to the theoretical value of 256 GB/s for LPDDR5X-8000)
- Device internal: 1000+ GB/s

### 3.5.3 Inference Benchmarks

```python
# benchmark_detailed.py
import ollama
import time
import statistics

def benchmark_model(model_name, prompt, num_runs=5):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    speeds = []

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)

        start = time.time()
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={"num_predict": 100}
        )
        elapsed = time.time() - start

        speed = 100 / elapsed
        speeds.append(speed)
        print(f"{speed:.1f} tokens/s")

    avg_speed = statistics.mean(speeds)
    std_dev = statistics.stdev(speeds) if len(speeds) > 1 else 0

    print(f"\nResults:")
    print(f"  Average: {avg_speed:.1f} tokens/s")
    print(f"  Std Dev: {std_dev:.1f} tokens/s")
    print(f"  Min: {min(speeds):.1f} tokens/s")
    print(f"  Max: {max(speeds):.1f} tokens/s")

if __name__ == "__main__":
    models = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"]
    prompt = "Explain quantum computing in simple terms."

    for model in models:
        try:
            benchmark_model(model, prompt)
        except Exception as e:
            print(f"Error with {model}: {e}")
```

```bash
# execution
python3 benchmark_detailed.py
```

**Expected values for MS-S1 Max:**

```
Benchmarking: qwen2.5:7b
Run 1/5... 42.3 tokens/s
Run 2/5... 43.1 tokens/s
Run 3/5... 42.8 tokens/s
Run 4/5... 42.5 tokens/s
Run 5/5... 43.0 tokens/s

Results:
  Average: 42.7 tokens/s
  Std Dev: 0.3 tokens/s
```

## 3.6 ROCm Optimization Settings

### 3.6.1 Kernel parameters

```bash
# amdgpu kernel module configuration
sudo nano /etc/modprobe.d/amdgpu.conf
```

**Recommended settings for amdgpu.conf (MS-S1 Max):**

```conf
# Basic settings
options amdgpu ppfeaturemask=0xffffffff
options amdgpu dpm=1
options amdgpu gpu_recovery=1

# Memory settings
options amdgpu noretry=0
options amdgpu tmz=0

# performance
options amdgpu aspm=0
options amdgpu runpm=0
```

**Parameter description:**

Parameters | explanation | Recommended value
--- | --- | ---
ppfeaturemask | PowerPlay Feature Mask | 0xffffffff (all functions)
dpm | Dynamic Power Management | 1 (enabled)
gpu_recovery | GPU hang recovery | 1 (enabled)
noretry | Memory Access Retry | 0 (retry enabled)
aspm | Active State Power Management | 0 (disabled, prioritizes stability)

```bash
# Reflect the settings (rebuild initramfs)
sudo update-initramfs -u -k all

# restart
sudo reboot
```

### 3.6.2 GPU power and clock settings

```bash
# Check current power settings
sudo rocm-smi --showpower
sudo rocm-smi --showclocks

# Performance level setting
sudo rocm-smi --setperflevel high

# Power limit setting (default 120W)
sudo rocm-smi --setpoweroverdrive 120
```

**MS-S1 Max power modes:**

```
Performance (160W): Highest performance, high temperature
Balance (130W):     Recommended, balanced
Quiet (110W):       Quiet, slightly slow
```

### 3.6.3 Optimizing memory management

```bash
# System swap settings (minimized due to large memory)
sudo sysctl vm.swappiness=10

# Persistence
sudo nano /etc/sysctl.conf

# addition
vm.swappiness=10
vm.vfs_cache_pressure=50
```

## 3.7 Flash Attention Optimization

### 3.7.1 What is Flash Attention?

**Flash Attention** is a technology that speeds up attention calculations and reduces memory usage.

```
Traditional Attention:
- Memory usage: O(N²)
- Speed: Slow

Flash Attention:
- Memory usage: O(N)
- Speed: 2-4x faster
- Particularly useful in long contexts
```

### 3.7.2 Enabling Flash Attention in Ollama

```bash
# Enable with environment variables
export OLLAMA_FLASH_ATTENTION=1

# Persistent setting for Ollama service
sudo nano /etc/systemd/system/ollama.service.d/rocm.conf

# addition
Environment="OLLAMA_FLASH_ATTENTION=1"
```

```bash
# restart
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 3.7.3 Confirming the effect

```bash
# Execution time measurement with Flash Attention disabled
OLLAMA_FLASH_ATTENTION=0 time ollama run llama3.1:70b "Write a 500 word essay."

# Execution time measurement with Flash Attention enabled
OLLAMA_FLASH_ATTENTION=1 time ollama run llama3.1:70b "Write a 500 word essay."
```

**Expected improvements (70B model, 32K context):**

```
Disabled: 45 seconds
When enabled: 32 seconds (approximately 30% faster)
```

## 3.8 Troubleshooting

### 3.8.1 GPU not recognized

**Symptoms:**

```bash
ollama run llama3.1
# CPU only is displayed
```

**Diagnostic Procedure:**

```bash
# 1. Check ROCm installation
dpkg -l | grep rocm-core

# 2. Check environment variables
echo $HSA_OVERRIDE_GFX_VERSION # should be 11.0.0

# 3. Check your GPU device
ls -l /dev/kfd /dev/dri/render*

# 4. Permission check
groups | grep -E 'render|video'

# 5. ROCm operation check
rocminfo | grep -i gfx
```

**Solution:**

```bash
# Reset permissions
sudo usermod -a -G render,video $USER

# Persisting environment variables
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' | sudo tee -a /etc/environment

# restart
sudo reboot
```

### 3.8.2 Low GPU usage

**Symptoms:**

```
rocm-smi shows only 10-20% GPU usage
```

**Cause and Solution:**

**Cause 1: CPU fallback**

```bash
# Check logs
journalctl -u ollama | grep -i "fallback\|cpu"

# Solution: Reinstall ROCm
```

**Cause 2: Environment variables not set**

```bash
# Check the environment variables for the Ollama service
sudo systemctl show ollama | grep Environment

# If not set, see 3.3.3 to set it
```

**Cause 3: Model is too small**

```bash
# Smaller models such as the 3B model have faster CPU processing, so the GPU may go unused.
# Confirmed on models 7B and above
ollama run qwen2.5:14b
```

### 3.8.3 Memory Errors

**Symptoms:**

```
Error: failed to allocate memory
```

**Solution:**

```bash
# 1. Check available memory
free -h

# 2. Check other processes
htop

# 3. Restart Ollama
sudo systemctl restart ollama

# 4. Use smaller quantization
ollama pull llama3.1:70b-q4_K_M # Q4 instead of Q8
```

## 3.9 Summary of this chapter

In this chapter, you learned the following:

✅ **ROCm complete installation**

- Setting up ROCm 6.3
- User permission settings

✅ **Optimizing environment variables**

- HSA_OVERRIDE_GFX_VERSION setting
- System-wide/service settings

✅ **GPU recognition check**

- Check with rocminfo and rocm-smi
- HIP operation test

✅ **Performance optimization**

- Kernel parameter adjustment
- Enable Flash Attention
- Power and clock settings

✅Troubleshooting

- Diagnosing and resolving common problems

In the next chapter, we will learn basic commands and practical usage of Ollama.

---

**Previous Chapter** : [Chapter 2 Installation and Setup](chapter02_installation.md) **Next Chapter** : [Chapter 4 Basic Commands and Usage](chapter04_basic_commands.md)
