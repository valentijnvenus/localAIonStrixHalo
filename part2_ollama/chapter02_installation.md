# Chapter 2: Installation and Setup

## 2.1 Checking System Requirements

### 2.1.1 Checking the specs of the MS-S1 Max

Before installing Ollama, verify your system details.

```bash
# CPU information
lscpu | grep -E "Model name|CPU\(s\)|Thread"

# Output example
Model name: AMD Ryzen AI Max+ 395
CPU(s): 32
Thread(s) per core: 2

# memory information
free -h

# Output example
              total        used        free      shared  buff/cache   available
Mem:          125Gi       8.2Gi       110Gi       1.5Gi       6.8Gi       115Gi

# GPU information
lspci | grep VGA

# Output example
0000:01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 1900
```

### 2.1.2 Required Software

#### Ubuntu/Debian

```bash
# Update system to latest version
sudo apt update && sudo apt upgrade -y

# Required packages
sudo apt install -y \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release
```

### 2.1.3 Checking Disk Space

```bash
# Disk usage status
df -h /

# Check space for saving model
# Recommended: At least 100GB free space
```

Ollama models are stored in the following directory:

```
~/.ollama/models/
```

**Approximate capacity:**

- 7B model (Q4 quantization): Approximately 4GB
- 13B model (Q4 quantization): Approximately 8GB
- 34B model (Q4 quantization): Approximately 20GB
- 70B model (Q4 quantization): Approximately 40GB

## 2.2 Installing Ollama

### 2.2.1 Official installation script

The easiest way is to use the official installation script.

```bash
# Run official installation script
curl -fsSL https://ollama.com/install.sh | sh
```

**Installation includes:**

```
Installing Ollama...
✓ Downloaded Ollama binary
✓ Created systemd service
✓ Added to PATH
✓ Started Ollama service

Ollama has been installed successfully!
```

### 2.2.2 Verifying the Installation

```bash
# Check version
ollama --version

# Output example
ollama version is 0.5.4

# Check service status
systemctl status ollama

# Output example
● ollama.service - Ollama Service
     Loaded: loaded (/etc/systemd/system/ollama.service; enabled)
     Active: active (running) since...
```

### 2.2.3 Manual Installation (Optional)

If you want more control, you can install it manually.

```bash
# Download Ollama binaries
sudo curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama

# grant execution permission
sudo chmod +x /usr/local/bin/ollama

# create ollama user and group
sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G render ollama
sudo usermod -a -G video ollama

# create systemd service file
sudo nano /etc/systemd/system/ollama.service
```

**Contents of the service file:**

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="OLLAMA_HOST=0.0.0.0:11434"

[Install]
WantedBy=default.target
```

```bash
# enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
```

## 2.3 Check AMD GPU recognition

### 2.3.1 Checking the ROCm installation status

For Ollama to utilize AMD GPUs, ROCm must be installed correctly.

```bash
# Check ROCm version
rocm-smi --version

# Output example (expected value)
ROCm System Management Interface version: 6.3.0

# Display GPU information
rocm-smi

# Output example
========================ROCm System Management Interface========================
GPU  Temp   AvgPwr  SCLK     MCLK     Fan   Perf  PwrCap  VRAM%  GPU%
0    42.0c  15.0W   500Mhz   96Mhz    0%    auto  120.0W  2%     0%
```

### 2.3.2 Setting environment variables

Set the environment variables to correctly recognize the AMD Radeon 8060S (RDNA 3.5).

```bash
# Add to ~/.bashrc
nano ~/.bashrc

# Add the following to the end
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export OLLAMA_DEBUG=1 # Enable debug log (only during initial setup)
```

```bash
# Reflect settings
source ~/.bashrc
```

### 2.3.3 GPU recognition test

```bash
# Restart Ollama and apply environment variables
sudo systemctl restart ollama

# Check GPU recognition in log
journalctl -u ollama -f
```

**Example of successful output:**

```
Starting Ollama Service...
Detected GPU: AMD Radeon Graphics (gfx1100)
Using ROCm backend
GPU Memory: 96GB available
Ollama server started on http://0.0.0.0:11434
```

## 2.4 Downloading the initial model

### 2.4.1 Testing with a small model

First, check the operation with a small model.

```bash
# Download 7B model
ollama pull qwen2.5:7b

# Progress is displayed
pulling manifest
pulling 8cf58c9acf79... 100% ▕████████████████████████▏ 4.7 GB
pulling 8ab4849b038c... 100% ▕████████████████████████▏  249 B
pulling 23e0f4461c0c... 100% ▕████████████████████████▏  11 KB
pulling df4c5cf440f3... 100% ▕████████████████████████▏  485 B
verifying sha256 digest
writing manifest
success
```

### 2.4.2 Operation check

```bash
# run the model
ollama run qwen2.5:7b

# prompt will be displayed
>>> Hello! Please tell me about MS-S1 Max.

# Response (example)
The MS-S1 Max is from Minisforum featuring an AMD Ryzen AI Max+ 395 processor.
A high-performance mini PC. With 128GB large memory and powerful integrated GPU,
Provides the perfect environment for running large language models locally...

>>> /bye
```

## 2.5 Basic Ollama Configuration

### 2.5.1 Advanced environment variable settings

Ollama's behavior can be controlled by environment variables.

```bash
# create systemd environment variable file
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo nano /etc/systemd/system/ollama.service.d/environment.conf
```

**Contents of environment.conf:**

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="OLLAMA_FLASH_ATTENTION=1"
```

**Explanation of environment variables:**

variable | explanation | Defaults | Recommended value (MS-S1 Max)
--- | --- | --- | ---
OLLAMA_HOST | API server address | 127.0.0.1:11434 | 0.0.0.0:11434
OLLAMA_ORIGINS | CORS allowed origins | localhost | *
OLLAMA_NUM_PARALLEL | Number of parallel requests | 1 | 2-4
OLLAMA_MAX_LOADED_MODELS | Number of simultaneously loaded models | 1 | 2-3
HSA_OVERRIDE_GFX_VERSION | AMD GPU compatibility settings | - | 11.0.0
OLLAMA_FLASH_ATTENTION | Enable Flash Attention | 0 | 1

```bash
# Reflect settings
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 2.5.2 Network Settings

#### Access from the local network

```bash
# Firewall settings (if using UFW)
sudo ufw allow 11434/tcp
sudo ufw reload

# check port
sudo netstat -tlnp | grep 11434

# Output example
tcp  0  0  0.0.0.0:11434  0.0.0.0:*  LISTEN  12345/ollama
```

#### Test access from another machine

```bash
# From another PC/smartphone
curl http://<IP address of MS-S1-Max>:11434/api/version

# Output example
{"version":"0.5.4"}
```

### 2.5.3 Log Settings

```bash
# Real-time log display
journalctl -u ollama -f

# Check past logs
journalctl -u ollama --since "1 hour ago"

# Log level setting (debug mode)
sudo nano /etc/systemd/system/ollama.service.d/environment.conf

# addition
Environment="OLLAMA_DEBUG=1"
```

## 2.6 Performance Check

### 2.6.1 Monitoring GPU usage

```bash
# Monitor GPU in another terminal
watch -n 1 rocm-smi

# Output example during inference execution
GPU  Temp   AvgPwr  SCLK     MCLK     Fan   Perf  PwrCap  VRAM%  GPU%
0    68.0c  95.0W   2900Mhz  1000Mhz  65%   auto  120.0W  45%    92%
```

### 2.6.2 Benchmark Testing

```bash
# Benchmark with Python script
nano benchmark_ollama.py
```

**benchmark_ollama.py:**

```python
import ollama
import time

models = ["qwen2.5:7b", "qwen2.5:14b"]

for model in models:
    print(f"\n=== Testing {model} ===")

# warm up
    ollama.generate(model=model, prompt="Hello")

# Benchmark
    start = time.time()
    response = ollama.generate(
        model=model,
        prompt="Write a detailed explanation of quantum computing in 100 words.",
        options={"num_predict": 100}
    )
    elapsed = time.time() - start

    tokens = 100
    speed = tokens / elapsed
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {speed:.1f} tokens/s")
```

```bash
# execution
python3 benchmark_ollama.py
```

**Expected Results (MS-S1 Max):**

```
=== Testing qwen2.5:7b ===
Time: 2.50s
Speed: 40.0 tokens/s

=== Testing qwen2.5:14b ===
Time: 4.55s
Speed: 22.0 tokens/s
```

## 2.7 Managing Multiple Versions

### 2.7.1 Model tagging

```bash
# Download specific version
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q8_0

# Display model list
ollama list

# Output example
NAME                              ID              SIZE    MODIFIED
llama3.1:8b-instruct-q4_K_M      abcd1234        4.9GB   2 minutes ago
llama3.1:8b-instruct-q8_0        efgh5678        8.5GB   1 minute ago
qwen2.5:7b                        ijkl9012        4.7GB   10 minutes ago
```

### 2.7.2 Creating an Alias

```bash
# Copy model with custom tag
ollama cp qwen2.5:7b my-assistant

# Use
ollama run my-assistant
```

## 2.8 Troubleshooting

### 2.8.1 Service does not start

```bash
# Check error log
sudo journalctl -u ollama --no-pager

# Common problems
```

**Problem 1: Port already in use**

```bash
# Check port usage
sudo lsof -i :11434

# Solution: change port
sudo nano /etc/systemd/system/ollama.service.d/environment.conf
# Change to OLLAMA_HOST=0.0.0.0:11435
```

**Issue 2: GPU Access Permissions**

```bash
# add to render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Login again
exit
# SSH/login again
```

### 2.8.2 GPU not recognized

```bash
# Confirm ROCm
/opt/rocm/bin/rocminfo | grep "Name:"

# Check environment variables
echo $HSA_OVERRIDE_GFX_VERSION # Should be 11.0.0

# Check Ollama log
journalctl -u ollama | grep -i gpu
```

**Solution:**

```bash
# Persist environment variables
sudo nano /etc/environment

# addition
HSA_OVERRIDE_GFX_VERSION=11.0.0

# system restart
sudo reboot
```

### 2.8.3 Model download fails

```bash
# Check network
curl -I https://ollama.com

# Proxy settings (if required)
sudo nano /etc/systemd/system/ollama.service.d/environment.conf

# addition
Environment="HTTP_PROXY=http://proxy.example.com:8080"
Environment="HTTPS_PROXY=http://proxy.example.com:8080"
```

## 2.9 Building an Ollama environment using Docker

### 2.9.1 Integration with Docker Desktop

**Docker Desktop Models**

Docker Desktop (Windows/Mac version) now includes an integrated feature called **Models** , which allows you to easily manage multiple AI model execution environments, including Ollama.

```
Docker Desktop → Left sidebar → Models
→ Choose from Ollama, OpenLLM, vLLM, etc.
→ Install and start with one click
```

**Features:**

- Easy setup from GUI
- Centralized model management
- Container resource allocation settings
- Integrated log display

### 2.9.2 Ollama setup with Docker Compose

**Basic docker-compose.yml**

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - OLLAMA_ORIGINS=*
      - OLLAMA_NUM_PARALLEL=2
      - OLLAMA_MAX_LOADED_MODELS=3
    restart: unless-stopped

volumes:
  ollama_data:
    driver: local
```

**How to start:**

```bash
# start container
docker-compose up -d

# Check log
docker-compose logs -f ollama

# run model
docker exec -it ollama ollama run qwen2.5:7b
```

### 2.9.3 GPU and CPU Distribution Strategy

**Why separate GPU/CPU?**

When utilizing the MS-S1 Max's 128GB memory and Radeon 8060S, GPU/CPU balancing is effective in the following scenarios:

```
Scenario 1: Simultaneous execution of multiple models
→ GPU: Main model (7B-14B)
→ CPU: Auxiliary lightweight model (3B or less)

Scenario 2: Tasks with different priorities
→ GPU: Real-time interaction (low latency)
→ CPU: Background processing (batch processing)

Scenario 3: Maximize resource efficiency
→ Dedicate GPU to main task
→ Parallel processing of lightweight tasks on CPU
```

### 2.9.4 Configuring a GPU-only container

**AMD GPU compatible docker-compose.yml**

```yaml
version: '3.8'

services:
  ollama-gpu:
    image: ollama/ollama:latest
    container_name: ollama-gpu
    ports:
      - "11434:11434"
    volumes:
      - ollama_gpu_data:/root/.ollama
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
      - OLLAMA_NUM_PARALLEL=2
      - OLLAMA_MAX_LOADED_MODELS=2
      - OLLAMA_FLASH_ATTENTION=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: amd
              capabilities: [gpu]
              count: all
    restart: unless-stopped

volumes:
  ollama_gpu_data:
    driver: local
```

**Key Takeaways:**

```yaml
devices:
- /dev/kfd # AMD GPU kernel driver
  - /dev/dri      # Direct Rendering Infrastructure

group_add:
- video # video group permissions
- render # render group permissions

environment:
- HSA_OVERRIDE_GFX_VERSION=11.0.0 # Compatible with RDNA 3.5
```

### 2.9.5 Configuring a CPU-only Container

**CPU-optimized docker-compose.yml**

```yaml
version: '3.8'

services:
  ollama-cpu:
    image: ollama/ollama:latest
    container_name: ollama-cpu
    ports:
- "11435:11434" # different port number
    volumes:
      - ollama_cpu_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - OLLAMA_NUM_PARALLEL=4
      - OLLAMA_MAX_LOADED_MODELS=2
    deploy:
      resources:
        limits:
cpus: '8.0' # limited to 8 cores
memory: 32G # limited to 32GB
        reservations:
cpus: '4.0' # Ensure at least 4 cores
memory: 16G # Ensure at least 16GB
    restart: unless-stopped

volumes:
  ollama_cpu_data:
    driver: local
```

### 2.9.6 Multi-container configuration (GPU + CPU)

**Integrated docker-compose.yml**

```yaml
version: '3.8'

services:
# GPU container - for main inference
  ollama-gpu:
    image: ollama/ollama:latest
    container_name: ollama-gpu
    ports:
      - "11434:11434"
    volumes:
      - ollama_gpu_data:/root/.ollama
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
      - OLLAMA_NUM_PARALLEL=2
      - OLLAMA_MAX_LOADED_MODELS=2
      - OLLAMA_FLASH_ATTENTION=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: amd
              capabilities: [gpu]
              count: all
    restart: unless-stopped
    networks:
      - ollama-network

# CPU container - for lightweight model/background
  ollama-cpu:
    image: ollama/ollama:latest
    container_name: ollama-cpu
    ports:
      - "11435:11434"
    volumes:
      - ollama_cpu_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - OLLAMA_NUM_PARALLEL=4
      - OLLAMA_MAX_LOADED_MODELS=3
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 32G
        reservations:
          cpus: '4.0'
          memory: 16G
    restart: unless-stopped
    networks:
      - ollama-network

# Open WebUI - Web interface
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URLS=http://ollama-gpu:11434;http://ollama-cpu:11434
    depends_on:
      - ollama-gpu
      - ollama-cpu
    restart: unless-stopped
    networks:
      - ollama-network

networks:
  ollama-network:
    driver: bridge

volumes:
  ollama_gpu_data:
  ollama_cpu_data:
  open_webui_data:
```

**Start and check:**

```bash
# start all containers
docker-compose up -d

# Check status
docker-compose ps

# Output example
NAME                IMAGE                              STATUS
ollama-gpu          ollama/ollama:latest               Up 2 minutes
ollama-cpu          ollama/ollama:latest               Up 2 minutes
open-webui          ghcr.io/open-webui/open-webui:main Up 2 minutes

# Run model in GPU container
docker exec -it ollama-gpu ollama run qwen2.5:14b

# Run model on CPU container
docker exec -it ollama-cpu ollama run qwen2.5:3b
```

### 2.9.7 Best Practices

#### 1. Model placement strategy (for MS-S1 Max)

**GPU Container (High performance and low latency):**

```yaml
Recommended model:
- qwen2.5:7b-instruct (4.8GB) - Main chat
- qwen2.5:14b-instruct (9GB) - High quality response
- deepseek-coder:16b (10GB) - Coding

Total: Approximately 24GB
GPU usage: 70-90%
Response: Instant (< 0.5 seconds)
```

**CPU Container (lightweight/background):**

```yaml
Recommended model:
- qwen2.5:3b-instruct (2GB) - Fast response
- phi3:3.8b (2.3GB) - Simple tasks
- gemma2:2b (1.6GB) - Classification task

Total: Approximately 6GB
CPU usage: 40-60% (when using 8 cores)
Response: 1-2 seconds
```

#### 2. Optimizing resource allocation

**Recommended configuration:**

```yaml
# GPU container
OLLAMA_NUM_PARALLEL=2 # Number of simultaneous requests
OLLAMA_MAX_LOADED_MODELS=2 # Number of loaded models
→ Memory usage: 20-30GB

# CPU container
OLLAMA_NUM_PARALLEL=4 # CPU has high degree of parallelism
OLLAMA_MAX_LOADED_MODELS=3 # Many lightweight models
CPU limit: 8 cores
Memory limit: 32GB
→ Memory usage: 10-20GB

Remaining resources:
- CPU: 8 cores (for system)
- Memory: 70-98GB (for other apps)
```

#### 3. Load Balancing Strategy

**Load balancing with Nginx:**

```nginx
# nginx.conf
upstream ollama_backend {
# GPU container - for heavy requests (high weighting)
    server ollama-gpu:11434 weight=3 max_fails=3 fail_timeout=30s;

# CPU container - for light requests
    server ollama-cpu:11434 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 11436;

    location / {
        proxy_pass http://ollama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

# Timeout settings (supports long-term inference)
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

**Add to your docker-compose.yml:**

```yaml
  nginx:
    image: nginx:alpine
    container_name: ollama-loadbalancer
    ports:
      - "11436:11436"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - ollama-gpu
      - ollama-cpu
    restart: unless-stopped
    networks:
      - ollama-network
```

#### 4. Performance Measurement

**Performance comparison script for each container:**

```python
# benchmark_containers.py
import requests
import time
import json

containers = {
    "GPU": "http://localhost:11434",
    "CPU": "http://localhost:11435"
}

prompt = "Explain quantum computing in 50 words."

for name, url in containers.items():
    print(f"\n=== Testing {name} Container ===")

# API call
    start = time.time()
    response = requests.post(
        f"{url}/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False
        }
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        data = response.json()
        tokens = data.get("eval_count", 0)
        speed = tokens / elapsed if elapsed > 0 else 0

        print(f"Time: {elapsed:.2f}s")
        print(f"Tokens: {tokens}")
        print(f"Speed: {speed:.1f} tokens/s")
    else:
        print(f"Error: {response.status_code}")
```

**Expected Results (MS-S1 Max):**

```
=== Testing GPU Container ===
Time: 1.8s
Tokens: 50
Speed: 27.8 tokens/s

=== Testing CPU Container ===
Time: 6.2s
Tokens: 50
Speed: 8.1 tokens/s

→ GPU is about 3.4 times faster than CPU
```

### 2.9.8 Troubleshooting the Docker environment

#### Issue 1: AMD GPU not recognized

```bash
# Check GPU in container
docker exec -it ollama-gpu rocm-smi

# If an error occurs
# 1. Check ROCm installation on host
rocm-smi

# 2. Check device file permissions
ls -la /dev/kfd /dev/dri

# 3. Confirm user group
groups

# Solution: Add to render/video group
sudo usermod -aG render,video $USER
# Re-login or reboot
```

#### Problem 2: Duplicate model downloads between containers

**Solution: Use a shared volume**

```yaml
volumes:
# Shared model storage
  shared_models:
    driver: local

services:
  ollama-gpu:
    volumes:
      - shared_models:/root/.ollama

  ollama-cpu:
    volumes:
      - shared_models:/root/.ollama
```

**Note** : Downloading different models at the same time may cause conflicts, so we recommend downloading them sequentially.

#### Problem 3: Out of memory

```bash
# Check Docker Desktop resource settings
# Settings → Resources → Advanced
# Memory: Set to 100GB or more (for MS-S1 Max)

# or adjust the limit with docker-compose.yml
deploy:
  resources:
    limits:
memory: 28G # reduce
```

### 2.9.9 Summary of this section

You learned about running Ollama in a Docker environment:

✅ **Docker Desktop integration**

- Easy setup with Models function
- Management from the GUI

✅ **GPU/CPU distributed placement**

- GPU: High performance model (7B-14B)
- CPU: Lightweight model (3B or less)
- Maximizing resource efficiency

✅Multi **-container configuration**

- Dedicated containers for role sharing
- Load Balancing
- Strategic use of 128GB memory

✅ **Best Practices**

- Model Placement Strategy
- Resource Allocation Optimization
- Performance Measurement

**💡 Recommended Configuration (MS-S1 Max):**

```
GPU container: 14B + 7B model (24GB used)
CPU container: 3B × 2-3 model (10GB used)
Remaining memory: 94GB (can be used for other apps)
→ Build a flexible multi-model environment
```

## 2.10 Summary of this chapter

In this chapter, you learned the following:

✅ **Check system requirements**

- How to check the specs of the MS-S1 Max
- Disk space requirements

✅ **Installing Ollama**

- Easy installation using the official script
- Manual installation method
- Building with Docker

✅ **AMD GPU recognition**

- ROCm confirmation
- Environment variable setting (HSA_OVERRIDE_GFX_VERSION)

✅Initial **settings**

- Advanced environment variable settings
- Network Settings
- Log Settings

✅ **Docker environment**

- Docker Desktop integration
- GPU/CPU distribution strategy
- Multi-container configuration

✅ **Operation check**

- First model download
- Performance Testing

In the next chapter, we will dig deeper into ROCm's advanced settings and AMD GPU optimizations.

---

**Previous Chapter** : [Chapter 1 Introduction](chapter01_introduction.md) **Next Chapter** : [Chapter 3 ROCm Settings and AMD GPU Optimization](chapter03_rocm_optimization.md)
