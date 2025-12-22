# Chapter 2: ComfyUI installation and ROCm configuration

## 2.1 Check system requirements

### 2.1.1 Prerequisites for MS-S1 Max environment

Review the requirements for running ComfyUI on MS-S1 Max.

**Hardware requirements**
```
Required:
- AMD Ryzen AI Max+ 395
- Radeon 8060S (RDNA 3.5)
- Memory: 128GB LPDDR5X-8000
- Storage: 100GB or more free space

Recommended:
- SSD/NVMe: Faster model loading
- Cooling: Adequate airflow (TDP 130-150W)
```

**Software requirements**
```
OS: Ubuntu 22.04 LTS or 24.04 LTS
Kernel: 5.15 or later (6.x recommended)
Python: 3.10, 3.11, 3.12
ROCm: 6.2 or later (6.4 recommended, 7.0 compatible)
Git: for version control
```

### 2.1.2 Supported ROCm version

**ROCm version selection guide**
```
ROCm 6.2.0:
- Stability: ★★★★☆
- Performance: ★★★★☆
- Recommendation level: ★★★☆☆
- Notes: Stable version, widely tested

ROCm 6.4.2:
- Stability: ★★★★★
- Performance: ★★★★★
- Recommendation level: ★★★★★
- Notes: Most recommended (as of January 2025)

ROCm 7.0:
- Stability: ★★★☆☆
- Performance: ★★★★★
- Recommendation level: ★★★☆☆
- Notes: Latest version, with experimental features
```

**MS-S1 Max recommended configuration**
```bash
# Recommended version
OS: Ubuntu 24.04 LTS
ROCm: 6.4.2
PyTorch: 2.6.0+rocm6.4
Python: 3.11

# reason
- Ubuntu 24.04: Kernel 6.x, latest driver supported
- ROCm 6.4.2: Balancing stability and performance
- PyTorch 2.6.0: Fully compatible with ROCm 6.4
```

## 2.2 Installing ROCm

### 2.2.1 System preparation

**Delete existing driver**
```bash
# Check AMD GPU driver
lspci | grep VGA
# Example output:
# 00:02.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 1900

# Delete existing ROCm package (during clean install)
sudo apt autoremove --purge rocm-* amdgpu-*
sudo apt autoremove --purge hip-* hsa-*

# system update
sudo apt update && sudo apt upgrade -y
```

**Installing required dependencies**
```bash
# Build tools and utilities
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    software-properties-common \
    python3-pip \
    python3-venv \
    libstdc++-12-dev \
    libnuma-dev

# Kernel header (for driver build)
sudo apt install -y linux-headers-$(uname -r)
```

### 2.2.2 Installing ROCm 6.4.2 (recommended)

**Add AMD repository**
```bash
# For Ubuntu 24.04 (Noble)
wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb

# For Ubuntu 22.04 (Jammy)
# wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/jammy/amdgpu-install_6.4.60402-1_all.deb

# Install package
sudo dpkg -i amdgpu-install_6.4.60402-1_all.deb
sudo apt update
```

**ROCm full stack installation**
```bash
# Install ROCm development environment
sudo amdgpu-install --usecase=rocm,graphics \
    --vulkan=pro \
    --opencl=rocr,legacy

# Confirm installation
rocm-smi --showproductname
# Output example: GPU[0] : Card series: Radeon Graphics
# Output example: GPU[0] : Card model: 0x1900

# Check ROCm version
apt list --installed | grep rocm
```

**Setting environment variables**
```bash
# Add to ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# ROCm environment variables
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# MS-S1 Max (RDNA 3.5) dedicated settings
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export ROC_ENABLE_PRE_VEGA=0
export HSA_ENABLE_SDMA=1

# GPU optimization
export GPU_MAX_ALLOC_PERCENT=95
export AMD_DIRECT_RENDERING=1

# Vulkan optimization (for image generation)
export RADV_PERFTEST=gpl,nggc
export ACO_DEBUG=validateir,validatera

EOF

# Reflect settings
source ~/.bashrc
```

### 2.2.3 Configuring user privileges

**Add to render group**
```bash
# Add current user to render and video groups
sudo usermod -a -G render,video $USER

# Confirm group
groups
# Example output: username adm cdrom sudo dip plugdev render video

# Note: Re-login required for group changes to take effect
# or use the following command to temporarily reflect
newgrp render
```

**Verify device access**
```bash
# Check GPU device
ls -l /dev/kfd /dev/dri/render*

# Example output:
# crw-rw----+ 1 root render 510, 0 Jan  1 09:00 /dev/kfd
# crw-rw----+ 1 root render 226, 128 Jan  1 09:00 /dev/dri/renderD128

# Display GPU information with rocm-smi
rocm-smi
```

## 2.3 Installing PyTorch (ROCm version)

### 2.3.1 Creating a Python virtual environment

**Building a venv environment**
```bash
# Create a directory for ComfyUI
mkdir -p ~/ai-tools
cd ~/ai-tools

# Create a Python 3.11 virtual environment
python3.11 -m venv comfyui_env

# Enabling virtual environment
source comfyui_env/bin/activate

# Update pip, setuptools, wheel
pip install --upgrade pip setuptools wheel
```

### 2.3.2 Installing PyTorch ROCm version

**PyTorch installation for ROCm 6.4 (recommended)**
```bash
# Install from AMD official repository (recommended)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.4

# Check version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Example output: PyTorch: 2.6.0+rocm6.4

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Example output: CUDA Available: True (ROCm emulates CUDA API)

python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
# Example output: Device: AMD Radeon Graphics
```

**PyTorch operation confirmation script**
```python
# test_pytorch_rocm.py
import torch

print("=" * 50)
print("PyTorch ROCm environment test")
print("=" * 50)

# Version information
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# memory information
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Simple math test
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU Computation Test: {'Success' if z.shape == (1000, 1000) else 'Failed'}")

# HIP information
    print(f"HIP Version: {torch.version.hip}")
else:
print("⚠️ GPU not detected!")
```

**Running and Troubleshooting**
```bash
# Run test script
python test_pytorch_rocm.py

# Expected output:
# ==================================================
# PyTorch ROCm environment test
# ==================================================
# PyTorch Version: 2.6.0+rocm6.4
# ROCm Available: True
# GPU Count: 1
# Current Device: 0
# Device Name: AMD Radeon Graphics
# Total Memory: 24.00 GB (allocated from unified memory)
# GPU Computation Test: Success
# HIP Version: 6.4.60402
```

### 2.3.3 Alternative installation method

**When using ROCm 7.0 (latest version)**
```bash
# PyTorch Nightly version (ROCm 7.0)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.0/
```

**When using ROCm 6.2 (stable version)**
```bash
# PyTorch ROCm 6.2 version
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2
```

## 2.4 Installing ComfyUI

### 2.4.1 ComfyUI Clone

**Clone from GitHub**
```bash
# Make sure the virtual environment is enabled
source ~/ai-tools/comfyui_env/bin/activate

# Clone ComfyUI
cd ~/ai-tools
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Check the latest stable tag (optional)
git tag --list
git checkout tags/v0.2.8 # Example: When using a specific version
```

### 2.4.2 Installing dependencies

**Edit requirements.txt**
```bash
# Back up the original requirements.txt
cp requirements.txt requirements.txt.bak

# Comment out PyTorch related packages
# (Because ROCm version of PyTorch is already installed)
sed -i 's/^torch/#torch/g' requirements.txt
sed -i 's/^torchvision/#torchvision/g' requirements.txt
sed -i 's/^torchaudio/#torchaudio/g' requirements.txt
sed -i 's/^torchsde/#torchsde/g' requirements.txt

# Check the edited requirements.txt
cat requirements.txt
```

**Installing dependent packages**
```bash
# install remaining dependencies
pip install -r requirements.txt

# Install torchsde separately (ROCm compatible version)
pip install torchsde

# Additional useful packages
pip install \
    opencv-python \
    opencv-contrib-python \
    scikit-image \
    scipy \
    numba \
    matplotlib
```

### 2.4.3 Check directory structure

**ComfyUI directory structure**
```
ComfyUI/
├── comfy/ # Core library
├── custom_nodes/ # Custom nodes
├── input/ # input image
├── models/ # Model storage directory
│ ├── checkpoints/ # SDXL model (.safetensors)
│ ├── clip/ # CLIP model
│ ├── clip_vision/ # CLIP Vision model
│ ├── controlnet/ # ControlNet model
│ ├── embeddings/ # Text embeddings
│ ├── loras/ # LoRA model
│ ├── upscale_models/ # Upscale models
│ └── vae/ # VAE model
├── output/ # Generated image output destination
├── web/ # Web interface
├── main.py # main script
└── requirements.txt # Dependency list
```

**Prepare model directory**
```bash
cd ~/ai-tools/ComfyUI

# check that the required directories exist
ls -la models/

# Check permissions on output directory
chmod 755 output/
```

## 2.5 SDXL Model Download

### 2.5.1 Download from Hugging Face

**SDXL Base 1.0 model**
```bash
cd ~/ai-tools/ComfyUI/models/checkpoints

# Install Hugging Face CLI
pip install huggingface_hub

# Download SDXL Base 1.0 (approx. 6.6GB)
huggingface-cli download \
    stabilityai/stable-diffusion-xl-base-1.0 \
    sd_xl_base_1.0.safetensors \
    --local-dir . \
    --local-dir-use-symlinks False

# or download directly with wget
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

**SDXL Refiner 1.0 model (optional)**
```bash
# SDXL Refiner (approx. 6.1GB)
huggingface-cli download \
    stabilityai/stable-diffusion-xl-refiner-1.0 \
    sd_xl_refiner_1.0.safetensors \
    --local-dir . \
    --local-dir-use-symlinks False
```

### 2.5.2 Recommended model list

**Required model**
```
checkpoints/sd_xl_base_1.0.safetensors (6.6GB)
- SDXL standard model
- 1024x1024 native generation

vae/sdxl_vae.safetensors (335MB) (optional)
- SDXL dedicated VAE
- Color improvement
- URL: https://huggingface.co/stabilityai/sdxl-vae
```

**Recommended additional models**
```
upscale_models/RealESRGAN_x4plus_anime_6B.pth
- Anime upscale
- 4x magnification

upscale_models/RealESRGAN_x4plus.pth
- Realistic upscale
- 4x magnification
```

### 2.5.3 Model download script

**Auto download script**
```bash
#!/bin/bash
# download_models.sh

set -e

COMFYUI_DIR=~/ai-tools/ComfyUI
CHECKPOINTS_DIR=$COMFYUI_DIR/models/checkpoints
VAE_DIR=$COMFYUI_DIR/models/vae
UPSCALE_DIR=$COMFYUI_DIR/models/upscale_models

echo "Download ComfyUI basic model"
echo "================================"

# Checkpoints directory
cd $CHECKPOINTS_DIR
echo "Downloading SDXL Base 1.0..."
if [ ! -f "sd_xl_base_1.0.safetensors" ]; then
    wget -q --show-progress \
        https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
echo "✓ SDXL Base completed"
else
echo "✓ SDXL Base existing"
fi

# VAE directory
cd $VAE_DIR
echo "Downloading SDXL VAE..."
if [ ! -f "sdxl_vae.safetensors" ]; then
    wget -q --show-progress \
        https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
echo "✓ SDXL VAE completed"
else
echo "✓ SDXL VAE existing"
fi

# Upscale model
cd $UPSCALE_DIR
echo "Downloading RealESRGAN x4..."
if [ ! -f "RealESRGAN_x4plus.pth" ]; then
    wget -q --show-progress \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
echo "✓ RealESRGAN completed"
else
echo "✓ RealESRGAN existing"
fi

echo ""
echo "Download complete!"
echo "Total capacity: approximately 7GB"
```

**execution**
```bash
chmod +x download_models.sh
./download_models.sh
```

## 2.6 Initial startup and operation check

### 2.6.1 Starting ComfyUI

**Basic startup command**
```bash
# Enabling virtual environment
source ~/ai-tools/comfyui_env/bin/activate

# Move to ComfyUI directory
cd ~/ai-tools/ComfyUI

# boot
python main.py

# Expected output:
# Total VRAM 24576 MB, total RAM 131072 MB
# Set vram state to: NORMAL_VRAM
# Device: cuda:0 AMD Radeon Graphics : native
# VAE dtype: torch.float16
# Using pytorch cross attention
# Starting server
# To see the GUI go to: http://127.0.0.1:8188
```

**Launch options**
```bash
# VRAM optimization (low memory)
python main.py --lowvram

# High precision mode (utilizes 128GB memory)
python main.py --normalvram --highvram

# change port
python main.py --port 8189

# Access permission within LAN
python main.py --listen 0.0.0.0

# MS-S1 Max recommended startup command
python main.py --normalvram --listen 127.0.0.1 --port 8188
```

### 2.6.2 Accessing the web interface

**Access with browser**
```
URL: http://127.0.0.1:8188

Recommended browser:
- Chrome/Chromium (most stable)
- Firefox
- Edge

On first access:
1. Default workflow is displayed
2. Select sd_xl_base_1.0 in the Load Checkpoint node
3. Start generation with the Queue button
```

### 2.6.3 Initial generation test

**Simple Text-to-Image**
```
1. Use default workflow
2. Enter CLIP Text Encode (Positive):
   "a beautiful mountain landscape, 4k, highly detailed"
3. Enter CLIP Text Encode (Negative):
   "low quality, blurry, distorted"
4. Empty Latent Image:
   - width: 1024
   - height: 1024
   - batch_size: 1
5. KSampler settings:
   - seed: 42
   - steps: 25
   - cfg: 7.5
   - sampler_name: dpmpp_2m_karras
   - scheduler: karras
6. Click the "Queue Prompt" button
7. Generation completed in 10-15 seconds
```

## 2.7 MS-S1 Max optimization settings

### 2.7.1 Optimizing environment variables

**~/ai-tools/comfyui_env.sh created**
```bash
#!/bin/bash
# comfyui_env.sh - MS-S1 Max optimization environment variables

# ROCm basic settings
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# RDNA 3.5 (gfx1100) settings
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100

# GPU allocation optimization (utilizes 128GB memory)
export GPU_MAX_ALLOC_PERCENT=95
export GPU_SINGLE_ALLOC_PERCENT=90

# HIP/ROCm optimization
export HSA_ENABLE_SDMA=1
export ROC_ENABLE_PRE_VEGA=0
export AMD_DIRECT_RENDERING=1

# PyTorch optimization
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ComfyUI dedicated optimization
export COMFYUI_VRAM_MODE=normalvram
export COMFYUI_FORCE_FP16=1

# Vulkan optimization
export RADV_PERFTEST=gpl,nggc
export ACO_DEBUG=validateir,validatera

# Log level (when debugging)
# export ROCM_LOG_LEVEL=3
# export AMD_LOG_LEVEL=3

echo "MS-S1 Max environment variable settings completed"
```

**How ​​to use**
```bash
# Execute permission for script
chmod +x ~/ai-tools/comfyui_env.sh

# Load before starting ComfyUI
source ~/ai-tools/comfyui_env.sh
source ~/ai-tools/comfyui_env/bin/activate
cd ~/ai-tools/ComfyUI
python main.py
```

### 2.7.2 Creating a startup script

**~/ai-tools/start_comfyui.sh**
```bash
#!/bin/bash
# start_comfyui.sh - ComfyUI startup script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$SCRIPT_DIR/ComfyUI"
VENV_DIR="$SCRIPT_DIR/comfyui_env"

echo "ComfyUI startup script (MS-S1 Max optimization)"
echo "========================================"

# Read environment variables
if [ -f "$SCRIPT_DIR/comfyui_env.sh" ]; then
    source "$SCRIPT_DIR/comfyui_env.sh"
else
echo "Warning: comfyui_env.sh not found"
fi

# Enabling virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
echo "✓ Enable virtual environment"
else
echo "Error: Virtual environment not found"
    exit 1
fi

# Move to ComfyUI directory
cd "$COMFYUI_DIR"

# Check GPU
echo ""
echo "GPU information:"
rocm-smi --showproductname 2>/dev/null || echo "rocm-smi not detected"

# Check PyTorch
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  GPU Available: {torch.cuda.is_available()}')"

echo ""
echo "Starting ComfyUI..."
echo "URL: http://127.0.0.1:8188"
echo ""

# Start ComfyUI (MS-S1 Max optimization settings)
exec python main.py \
    --normalvram \
    --listen 127.0.0.1 \
    --port 8188 \
    "$@"
```

**How ​​to use**
```bash
chmod +x ~/ai-tools/start_comfyui.sh

# boot
~/ai-tools/start_comfyui.sh

# Example of startup with options
~/ai-tools/start_comfyui.sh --preview-method auto
```

## 2.8 Installing a custom node

### 2.8.1 Installing ComfyUI Manager

ComfyUI Manager is a tool that simplifies the management of custom nodes.

```bash
cd ~/ai-tools/ComfyUI/custom_nodes

# Clone ComfyUI-Manager
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Install dependencies
cd ComfyUI-Manager
pip install -r requirements.txt

# After restarting ComfyUI, "Manager" button is displayed on WebUI
```

### 2.8.2 Recommended custom nodes

**Image quality improvement system**
```bash
cd ~/ai-tools/ComfyUI/custom_nodes

# ComfyUI-Impact-Pack (Advanced Post-Processing)
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
cd ComfyUI-Impact-Pack && pip install -r requirements.txt && cd ..

# Ultimate SD Upscale (tiled upscale)
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
```

**Workflow extension system**
```bash
# rgthree's ComfyUI Nodes (UI extension)
git clone https://github.com/rgthree/rgthree-comfy.git
cd rgthree-comfy && pip install -r requirements.txt && cd ..

# Efficiency Nodes
git clone https://github.com/jags111/efficiency-nodes-comfyui.git
```

## 2.9 Troubleshooting

### 2.9.1 Common problems and solutions

**Problem 1: GPU not recognized**
```bash
# Symptoms
# "CUDA Available: False" or "Using CPU"

# Check 1: ROCm installation status
rocm-smi

# Check 2: User group
groups | grep render

# Solution: add render group and log in again
sudo usermod -a -G render,video $USER
# Then log out → log in

# Check 3: HSA_OVERRIDE_GFX_VERSION setting
echo $HSA_OVERRIDE_GFX_VERSION
# Output: Should be 11.0.0
```

**Problem 2: Out of memory error**
```bash
# Symptoms
# "RuntimeError: HIP out of memory"

# Solution 1: lowvram mode
python main.py --lowvram

# Solution 2: GPU allocation adjustment
export GPU_MAX_ALLOC_PERCENT=80
python main.py

# Solution 3: Reduce batch size
# Set workflow batch_size to 1
```

**Problem 3: Generation is very slow**
```bash
# Cause check 1: Check if it is not running on the CPU
# Check the log when starting ComfyUI
# "Device: cpu" → GPU not used
# "Device: cuda:0" → GPU in use (normal)

# Cause check 2: Check environment variables
env | grep HSA_OVERRIDE_GFX_VERSION

# Solution: Reset environment variables
source ~/ai-tools/comfyui_env.sh
```

### 2.9.2 Performance Benchmark

**Benchmark script**
```python
# benchmark.py
import torch
import time

print("ComfyUI Performance Test")
print("=" * 50)

# GPU information
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Matrix multiplication test
sizes = [1000, 2000, 4000, 8000]
for size in sizes:
    x = torch.rand(size, size).cuda()
    y = torch.rand(size, size).cuda()

# warm up
    _ = torch.matmul(x, y)
    torch.cuda.synchronize()

# Measurement
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    gflops = (2 * size**3) / elapsed / 1e9
    print(f"{size}x{size}: {elapsed*1000:.1f}ms ({gflops:.1f} GFLOPS)")

print("\nBenchmark completed")
```

**Expected results (MS-S1 Max)**
```
Device: AMD Radeon Graphics
Memory: 24.0 GB
1000x1000: 2.3ms (869 GFLOPS)
2000x2000: 8.1ms (1975 GFLOPS)
4000x4000: 45.2ms (2832 GFLOPS)
8000x8000: 312.5ms (3276 GFLOPS)
```

## 2.10 Summary of this chapter

What we did in this chapter:

**Environment construction**
- Installing Ubuntu 24.04 + ROCm 6.4.2
- PyTorch 2.6.0 (ROCm version) setup
- Setting environment variables for MS-S1 Max

**ComfyUI installation**
- Clone from GitHub
- Installing dependencies
- Download SDXL Base/Refiner model

**Optimization settings**
- HSA_OVERRIDE_GFX_VERSION=11.0.0 (RDNA 3.5 compatible)
- GPU allocation optimization
- Create startup script

**Operation confirmation**
- First launch and WebUI confirmation
- Text-to-Image generation test
- Performance benchmarks

In the next chapter, you will learn more about SDXL basics and prompting techniques.

---

**Reference resources**
- ROCm official documentation: https://rocm.docs.amd.com/
- ComfyUI GitHub: https://github.com/comfyanonymous/ComfyUI
- PyTorch ROCm: https://pytorch.org/get-started/locally/
- AMD GPUOpen: https://gpuopen.com/

