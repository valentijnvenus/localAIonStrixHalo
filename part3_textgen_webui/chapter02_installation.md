# Chapter 2: Installation and environment construction

## 2.1 System Requirements

### MS-S1 Max recommended configuration
```
OS: Ubuntu 22.04 LTS / 24.04 LTS
Python: 3.10 / 3.11
CUDA/ROCm: ROCm 6.1+
Memory: 128GB (MS-S1 Max)
Storage: 100GB+ free space
```

## 2.2 Installing dependencies

```bash
# system package
sudo apt update
sudo apt install -y python3-pip python3-venv git build-essential

# ROCm (see part 2)
# Skip if already installed
```

## 2.3 Installing Text Generation WebUI

### Recommended method (automatic installation)

```bash
# clone repository
cd ~
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui

# Run installation script for AMD GPU
./start_linux.sh

# Automatically install dependencies on first run
# select ROCm support
```

### Manual installation (recommended/MS-S1 Max optimization)

```bash
cd ~/text-generation-webui

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# PyTorch (ROCm version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Basic dependencies
pip install -r requirements_amd.txt

# ExLlamaV2 (ROCm version)
pip install exllamav2 --no-build-isolation

# Gradio and dependencies
pip install gradio==3.50.2
```

## 2.4 Environment variable settings

```bash
nano ~/.bashrc

# addition
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_HOME=/opt/rocm
export PYTORCH_ROCM_ARCH="gfx1100"

source ~/.bashrc
```

## 2.5 First start

```bash
cd ~/text-generation-webui
source venv/bin/activate

# Start WebUI
python server.py --listen --api

# Access with browser
# http://localhost:7860
```

## 2.6 Download Model

### Via Web UI

```
1. Open http://localhost:7860 in your browser
2. Click on the "Model" tab
3. "Download model or LoRA" section
4. Enter the model name (e.g. TheBloke/Llama-2-7B-GGUF)
5. Select file and download
```

### Command line

```bash
cd ~/text-generation-webui
python download-model.py TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf
```

## 2.7 Boot options

```bash
# Basic startup
python server.py

# remote access permission
python server.py --listen

# API enable
python server.py --api

# launch on specific model
python server.py --model llama-2-7b-chat.Q4_K_M

# ExLlamaV2 loader specification
python server.py --loader exllamav2

# multiple options
python server.py --listen --api --loader exllamav2 --gpu-memory 96
```

## 2.8 systemd service

```bash
sudo nano /etc/systemd/system/textgen.service
```

```ini
[Unit]
Description=Text Generation WebUI
After=network.target

[Service]
Type=simple
User=username
WorkingDirectory=/home/username/text-generation-webui
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
ExecStart=/home/username/text-generation-webui/venv/bin/python server.py --listen --api
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable textgen
sudo systemctl start textgen
```

## 2.9 Summary of this chapter

✅ Check system requirements
✅ Dependency installation
✅ Text Generation WebUI Setup
✅ Environment variable settings
✅ First launch and model download

---

**Go to previous chapter**: [Chapter 1 Introduction](chapter01_introduction.md)
**Next chapter**: [Chapter 3 ROCm settings and ExLlamaV2 optimization](chapter03_rocm_exllama.md)
