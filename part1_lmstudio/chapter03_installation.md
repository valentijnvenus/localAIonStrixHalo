# Chapter 3: Installing and Initializing LM Studio

## 3.1 Pre-installation preparation

### 3.1.1 Checking System Requirements

Before installing LM Studio, make sure your system meets the requirements.

#### Check procedure for Windows 11

**1. Check system information**

```powershell
# Start PowerShell with administrator privileges and run
systeminfo
```

Check items:

- OS Version: Windows 10/11 64-bit
- Physical memory: 128GB (MS-S1 Max)
- Processor: AMD Ryzen AI Max+ 395

**2. Check your AMD GPU**

```powershell
# Check the GPU in Device Manager
devmgmt.msc
```

Make sure "AMD Radeon Graphics" or "Radeon 8060S" is listed under "Display adapters."

**3. Check the available storage space**

```powershell
# Check the free space on the drive
Get-PSDrive -PSProvider FileSystem
```

Please ensure you have at least 10GB of free space, and we recommend 100GB or more (for saving models).

#### Verification procedure for Linux

**1. Check system information**

```bash
# Distribution information
cat /etc/os-release

# Check CPU
lscpu | grep "Model name"

# Check memory
free -h

# Check storage
df -h
```

**2. Check your AMD GPU**

```bash
# Check PCI device
lspci | grep -i vga
lspci | grep -i amd

# Example output:
# 01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Radeon Graphics
```

### 3.1.2 Installing AMD GPU drivers

To use your AMD GPU with LM Studio, you need the appropriate drivers.

#### For Windows 11

**Method 1: AMD Software (Recommended)**

1. **Download from the AMD official website**

    - URL: https://www.amd.com/ja/support
    - Select "Graphics" → "AMD Radeon Graphics" → "Ryzen with Radeon Graphics"
    - Download the latest "AMD Software: Adrenalin Edition"

2. **Installation Instructions**

    ```
    1. Run the downloaded installer
    2. Select &quot;Express Install&quot;
    3. After the installation is complete, reboot the computer.
    ```

3. **Checking the driver version**

    - Right-click on the desktop → "AMD Software: Adrenalin Edition"
    - Click the gear icon in the top right corner → System tab
    - Check the "Driver Version" (24.xx or later recommended)

**Method 2: Windows Update**

```
1. Settings → Windows Update
2. "Check for updates"
3. If there is "AMD Graphics" in the optional updates, install it.
```

**⚠️ Note** : LM Studio for Windows works with standard AMD graphics drivers, no ROCm required.

#### For Ubuntu 24.04

**Installing ROCm (required)**

```bash
# Update the system to the latest version
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y wget gnupg2

# Add ROCm APT repository (ROCm 6.2)
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2 noble main" \
| sudo tee /etc/apt/sources.list.d/rocm.list

echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
| sudo tee /etc/apt/preferences.d/rocm-pin-600

# Update package list
sudo apt update

# Install ROCm
sudo apt install -y rocm-hip-sdk rocm-libs

# Add user to the render group
sudo usermod -a -G render,video $USER

# Restart (important)
sudo reboot
```

**Installation Verification**

After rebooting, check with the following command:

```bash
# Check ROCm version
rocminfo

# Check GPU recognition
rocm-smi

# Expected output example:
# ========================ROCm System Management Interface========================
# GPU Temp AvgPwr SCLK MCLK Fan Perf PwrCap VRAM% GPU%
# 0 45.0c 15.0W 800Mhz 1000Mhz 0.0% auto 120.0W 0% 0%
```

**Setting environment variables**

```bash
# Add to ~/.bashrc
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

**💡 TIP** : `HSA_OVERRIDE_GFX_VERSION=11.0.0` is a setting to recognize Radeon 8060S (RDNA 3.5) as gfx1100.

## 3.2 Download and install LM Studio

### 3.2.1 Download

**Official website** : https://lmstudio.ai/

1. Access the LM Studio official website in your browser.
2. Click the "Download" button
3. Select the installer for your OS
    - Windows: `LM-Studio-Setup-xxxexe`
    - Linux: `LM_Studio-xxxAppImage`
    - macOS: `LM-Studio-xxxdmg`

**💡 TIP** : As of October 2025, the latest version is 0.3.19 or later. Please be sure to download the latest version to fully support the AMD Ryzen AI Max+ 395.

### 3.2.2 Installing on Windows

**Standard Installation Procedure**

1. **Run the installer**

    ```
    Double-click the downloaded LM-Studio-Setup-xxxexe.
    ```

2. **Security Warning**

    - If you see the message "Windows protected your PC"
    - Click "More information" → "Run"

3. **Select installation location**

    ```
    Recommended: C:\Users\<username>\AppData\Local\Programs\LM Studio
    ```

4. **Installation options**

    - [✓] Create a shortcut on the desktop
    - [✓] Add to Start Menu
    - [✓] Check for automatic updates at startup

5. **Installation complete**

    - Check "Launch LM Studio" and click "Finish"

**Custom Installation (Advanced)**

```powershell
# Run with PowerShell (silent installation)
Start-Process -FilePath "LM-Studio-Setup-x.x.x.exe" -ArgumentList "/S" -Wait
```

### 3.2.3 Installation on Linux

**Using AppImage (recommended)**

```bash
# Change to the download directory
cd ~/Downloads

# Grant execution permission
chmod +x LM_Studio-xxxAppImage

# boot
./LM_Studio-xxxAppImage

# Optional: Place in /usr/local/bin (system wide)
sudo mv LM_Studio-xxxAppImage /usr/local/bin/lmstudio
```

**Creating a Desktop Entry**

```bash
# Create ~/.local/share/applications/lmstudio.desktop
cat > ~/.local/share/applications/lmstudio.desktop << 'EOF'
[Desktop Entry]
Name=LM Studio
Comment=Run LLMs locally
Exec=/usr/local/bin/lmstudio
Icon=lmstudio
Terminal=false
Type=Application
Categories=Development;
EOF

# Permission settings
chmod +x ~/.local/share/applications/lmstudio.desktop
```

**⚠️ NOTE** : If you move the AppImage, please adjust the Exec path appropriately.

### 3.2.4 Verifying the Installation

#### Windows

```powershell
# Check installation directory
dir "$env:LOCALAPPDATA\Programs\LM Studio"

# Check the version (after starting LM Studio)
# Help → About
```

#### Linux

```bash
# Check if it's executable
lmstudio --version # or ./LM_Studio-xxxAppImage --version

# Check the process
ps aux | grep lmstudio
```

## 3.3 First startup and basic settings

### 3.3.1 First boot

**Windows/Linux**

1. Launch LM Studio
2. The "Welcome to LM Studio" screen appears.
3. Check the terms of use and click "Agree"

**Dialog box at first startup**

```
□ Send anonymous usage statistics (optional)
□ Check for updates at startup (recommended)
```

**💡 TIP** : Sending usage statistics is optional. Uncheck it if you value your privacy.

### 3.3.2 Interface Overview

The LM Studio main window consists of the following:

```
┌──────────────────────────────────────────────────────────┐
│ [Search] [💬] [⚙️] [📚] [🌐] │ ← Tab bar
├────────────────────────────────────────────────────────┤
│ │ Main content area │
│ │
│ │
├────────────────────────────────────────────────────────┤
│ Status bar: GPU information, memory usage, etc. │
└──────────────────────────────────────────────────────────┘
```

**Tab Description**

icon | Tab Name | explanation
--- | --- | ---
🔍 | Search | Find and download models
💬 | Chat | Chat interface (inference execution)
📁 | My Models | Managing downloaded models
🌐 | Local Server | API Server Mode
⚙️ | Settings | Application Settings

### 3.3.3 Localization settings

LM Studio supports various custom, local language interfaces:

**Localization procedure**

1. Click the gear icon (⚙️) at the bottom right → "Settings"
2. Select "General" from the left menu
3. Find the "Language" section
4. Select "&lt;yourLanguage&gt; (Beta)" from the drop-down menu
5. A "Restart Required" dialog appears
6. Click "Restart now"

**⚠️ Note** : As this is a beta version, some parts may still be in English.

### 3.3.4 Recommended Basic Settings

#### General Settings

```
Language: English (Beta)
Theme: Auto (follows system settings)
Startup behavior:
□ Start LM Studio at system startup
✓ Check for updates on startup
□ Continue running in the background
```

#### Model Storage Settings

**Default model save location**

- **Windows** : `C:\Users\<username>\.cache\lm-studio\models`
- **Linux** : `~/.cache/lm-studio/models`

**Setting a custom path (recommended)**

If you are downloading many large models, set up a dedicated directory.

1. Settings → Storage
2. Click "Change" for "Models Directory"
3. Select a location to save the file (e.g. `D:\LM_Studio\models` or `/data/lmstudio/models` )
4. Click "Save"

**💡 TIP** : If you have a dual storage configuration for the MS-S1 Max, it is useful to specify the larger capacity drive as the model save destination.

#### Network Settings

```
Download Settings:
Number of parallel downloads: 3 (default)
Maximum download speed: Unlimited
Proxy: Not set (set as needed)

Hugging Face Settings:
□ Use Hugging Face tokens (for private models)
```

## 3.4 Check AMD GPU recognition

### 3.4.1 Checking GPU detection status

Check if LM Studio correctly recognizes your AMD GPU.

**Verification Procedure**

1. Launch LM Studio
2. Check the status bar at the bottom of the screen
3. Check that the following is displayed:

**Windows:**

```
🎮 GPU: AMD Radeon Graphics (RDNA 3.5) | VRAM: Available
```

**Linux:**

```
🎮 GPU: AMD ROCm (gfx1100) | VRAM: Available
```

### 3.4.2 Checking GPU details

**Display advanced GPU information**

1. Press the keyboard shortcut `Ctrl+Shift+H` (Windows/Linux)
2. The "Hardware Settings" dialog opens.
3. Check the following information

```
GPU Information:
Name: AMD Radeon Graphics / AMD ROCm
Architecture: RDNA 3.5 / gfx1100
Available memory: Approximately 100GB (128GB minus reserved memory)
Compute units: 40
ROCm version: 6.2.x (Linux)

situation:
✓ GPU detected
✓ GPU Offload is available
```

### 3.4.3 Troubleshooting

#### If Windows does not recognize your GPU

**Reason 1: Outdated drivers**

```powershell
# Open Device Manager
devmgmt.msc

# Right-click "Display adapters" → "AMD Radeon Graphics"
# → "Update Driver"
```

**Cause 2: The version of LM Studio is outdated**

```
1. Completely close LM Studio
2. Download the latest version from the official website
3. Reinstall
```

#### If your GPU isn't recognized in Linux

**Cause 1: ROCm is not installed**

```bash
# Check ROCm installation
dpkg -l | grep rocm

# If it is not installed, install it using the procedure in 3.1.2
```

**Cause 2: Missing environment variables**

```bash
# Check if the following is set in ~/.bashrc
echo $PATH | grep rocm
echo $LD_LIBRARY_PATH | grep rocm
echo $HSA_OVERRIDE_GFX_VERSION

# If not set
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc

# Restart LM Studio
```

**Cause 3: User permission issue**

```bash
# Check if you belong to the render group
groups

# If render is not included
sudo usermod -a -G render,video $USER

# Logout → Login or reboot
```

**Cause 4: GPU is not recognized correctly**

```bash
# Check GPU recognition
rocm-smi

# If an error occurs
sudo dmesg | grep amdgpu
sudo dmesg | grep -i error

# Reload kernel modules
sudo modprobe -r amdgpu
sudo modprobe amdgpu
```

## 3.5 Conducting Performance Tests

### 3.5.1 Downloading the simple test model

Once the initial setup is complete, test it on a small model.

**Recommended test model: Qwen2.5 3B**

1. Open the "Search" tab
2. Enter "qwen2.5 3b q4" in the search field
3. Find "Qwen/Qwen2.5-3B-Instruct-GGUF"
4. Select "qwen2.5-3b-instruct-q4_k_m.gguf"
5. Click "Download"

**Download size** : Approximately 2GB. Download **time** : 1-5 minutes (depending on internet speed)

### 3.5.2 Initial Inference Test

**Test Procedure**

1. Once the download is complete, go to the "Chat" tab
2. Select "qwen2.5-3b-instruct-q4_k_m" from "Select a model" at the top
3. Click "Load Model"

**Check items when loading**

```
Status display:
Loading model... (0%)
Loading model... (50%)
Loading model... (100%)
✓ Model loaded successfully

GPU Information:
GPU Layers: 32/32 (all layers offloaded to GPU)
VRAM usage: Approximately 2.5GB
```

**Test Prompt**

```
Hello! Who are you?
```

**Expected behavior**

- Response time: Starts generating within 0.5 seconds
- Generation speed: 30-50 tokens/second (3B model)
- Smooth response, no delay

**⚠️ Note** : The first time you load it, it may take some time to generate the model cache. After that, it will be faster.

### 3.5.3 Reviewing performance indicators

LM Studio displays performance information during inference.

**Chat window bottom display**

```
⚡ 45.3 tokens/s | 🎯 Prompt: 12 tokens | 📝 Generated: 89 tokens | 🕐 2.0s
```

index | explanation | MS-S1 Max guideline (3B model)
--- | --- | ---
tokens/s | Generation speed | 40-60 t/s
Prompt tokens | Number of input tokens | Depends on prompt
Generated tokens | Number of generated tokens | Depends on the length of the response
Time | Total Processing Time | Response Length/Speed

**Checking GPU usage**

```
Right side of the status bar:
💻 CPU: 15% | 🎮 GPU: 85% | 💾 RAM: 8.5GB/128GB
```

**💡 TIP** : If your GPU is showing high utilization (over 70%), it means that GPU acceleration is working properly.

## 3.6 Backup and Restore Settings

### 3.6.1 Location of configuration files

LM Studio settings are saved in the following location:

**Windows:**

```
C:\Users\<username>\AppData\Roaming\LM Studio\
├── config.json # Application settings
├── models.json # Model information
└── presets/ # custom presets
```

**Linux:**

```
~/.config/LM Studio/
├── config.json
├── models.json
└── presets/
```

### 3.6.2 Creating a Backup

**Manual Backup**

```bash
# Windows (PowerShell)
Copy-Item "$env:APPDATA\LM Studio" -Destination "D:\Backup\LM_Studio_backup" -Recurse

# Linux
cp -r ~/.config/"LM Studio" ~/backup/lmstudio_config_backup
```

**Export settings (presets only)**

1. Settings → Advanced → Export Settings
2. Select a save location
3. Save as `lmstudio_settings_YYYYMMDD.json`

### 3.6.3 Restore

**Restoring the Settings**

```bash
# Windows (PowerShell)
Copy-Item "D:\Backup\LM_Studio_backup\*" -Destination "$env:APPDATA\LM Studio" -Recurse -Force

# Linux
cp -r ~/backup/lmstudio_config_backup/* ~/.config/"LM Studio"/
```

Restart LM Studio to apply the restored settings.

## 3.7 Summary of this chapter

In this chapter, we have done the following:

✅ Check system requirements

- AMD Ryzen AI Max+ 395 Recognition
- Check memory and storage

✅ Installing AMD drivers and ROCm

- Windows: AMD Software Adrenalin Edition
- Linux: ROCm 6.2 Setup

✅ Installing LM Studio

- Windows: Run the installer
- Linux: Configuring AppImage

✅ Performing initial settings

- Japanese localization
- Setting the model save location
- Checking GPU recognition

✅ Performance testing

- Operation check on 3B model
- GPU Acceleration Verification

In the next chapter, we'll dig deeper into AMD GPU settings and learn how to get optimal performance.

---

**Previous Chapter** : [Chapter 2: Hardware Specifications and System Requirements](chapter02_hardware_specs.md) **Next Chapter** : [Chapter 4: The Complete Guide to Configuring AMD GPUs](chapter04_amd_gpu_settings.md)
