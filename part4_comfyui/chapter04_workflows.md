# Chapter 4: ComfyUI workflow creation and custom nodes

## 4.1 Basic concepts of workflow

### 4.1.1 What is Workflow?

A workflow in ComfyUI is a graphical interface with multiple connected nodes that describes the entire AI image generation process.

**Workflow features**
```
Visual programming:
- Build processing flow with nodes and edges
- Visually check data flow
- Manage complex pipelines

Reusability:
- Save and share your workflows
- Export in JSON format
- Embed as metadata in images

Flexibility:
- Extend functionality with custom nodes
- Conditional branching and iteration processing
- Combination of multiple models
```

### 4.1.2 Saving and loading workflows

**How ​​to save**
```javascript
// Method 1: Save from UI
// 1. Right click → Save Workflow
// 2. Save as workflow.json

// Method 2: Restore from generated image
// 1. Drag and drop the image generated with ComfyUI
// 2. Automatically restore workflow from metadata

// Method 3: Save in API format
// Save complete workflow definition in JSON file
```

**MS-S1 Max recommended workflow management**
```bash
# Workflow directory structure
~/ai-tools/ComfyUI/workflows/
├── basic/
│   ├── text-to-image-simple.json
│   ├── image-to-image-basic.json
│   └── upscale-basic.json
├── advanced/
│   ├── controlnet-depth.json
│   ├── lora-mixing.json
│   └── refiner-workflow.json
└── production/
    ├── batch-generation.json
    ├── multi-model.json
    └── automated-pipeline.json
```

## 4.2 Building a basic workflow

### 4.2.1 Text-to-Image Workflow

**Minimum configuration (6 nodes)**
```
┌─────────────────┐
│Load Checkpoint  │
│ sd_xl_base_1.0  │
└────┬───┬───┬────┘
     │   │   │
     │   │   └──→ MODEL → [KSampler]
     │   │
     │   └──→ CLIP → [CLIP Text Encode (Positive)]
     │                      ↓ CONDITIONING
     │                      ↓
     └──→ VAE         [KSampler] ←─ [Empty Latent Image]
                            ↓
                            ↓ LATENT
                      [VAE Decode]
                            ↓ IMAGE
                      [Save Image]
```

**Advanced settings**
```yaml
Load Checkpoint:
  ckpt_name: "sd_xl_base_1.0.safetensors"

CLIP Text Encode (Positive):
  text: "Your detailed prompt here"

CLIP Text Encode (Negative):
  text: "low quality, blurry"

Empty Latent Image:
  width: 1024
  height: 1024
  batch_size: 1

KSampler:
  seed: 42
  steps: 25
  cfg: 7.5
  sampler_name: "dpmpp_2m_karras"
  scheduler: "karras"
  denoise: 1.0

Save Image:
  filename_prefix: "ComfyUI"
```

### 4.2.2 Image-to-Image Workflow

**Configuration (8 nodes)**
```
[Load Image] → IMAGE → [VAE Encode]
                              ↓ LATENT
                              ↓
[Load Checkpoint] → MODEL → [KSampler] ← denoise: 0.75
                   ↓ CLIP            ↑
                   ↓                 └─ [CLIP Text Encode]
                   ↓ VAE
                   ↓
            [VAE Decode] ← LATENT ← [KSampler]
                   ↓ IMAGE
            [Save Image]
```

**Important parameter: denoise**
```
denoise = 1.0:
- Completely new generation
- Minimal influence on original image
- Equivalent to Text-to-Image

denoise = 0.75:
- Good balance (recommended)
- Maintain original image composition
- Change details

denoise = 0.5:
- Keep the original image strong
- Perfect for fine-tuning
- style change

denoise = 0.25:
- minimal changes
- Color adjustment level
- Almost maintains composition
```

### 4.2.3 SDXL Refiner Workflow

**2-stage generation (10 nodes)**
```
[Load Checkpoint: Base] → MODEL → [KSampler: Base]
                                      ↓ steps: 25
                                      ↓ denoise: 1.0
                                      ↓ LATENT
                                      ↓
[Load Checkpoint: Refiner] → MODEL → [KSampler: Refiner]
                                      ↓ steps: 15
                                      ↓ denoise: 0.3
                                      ↓ LATENT
                              [VAE Decode]
                                      ↓ IMAGE
                              [Save Image]
```

**MS-S1 Max optimal settings**
```python
# Generate Base
base_steps = 25
base_denoise = 1.0
estimated_time = 11 # seconds

# Apply Refiner
refiner_steps = 15
refiner_denoise = 0.3 # Regenerate 30% of Base
estimated_time_refiner = 5 # seconds

# total
total_time = 16 # seconds

# memory usage
base_model_mem = 8   # GB
refiner_model_mem = 8  # GB
vram_usage = 16 # GB (both loaded)
```

## 4.3 Introducing custom nodes

### 4.3.1 ComfyUI Manager (required)

ComfyUI Manager is the foundation for all custom node management.

**install**
```bash
cd ~/ai-tools/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ComfyUI-Manager
pip install -r requirements.txt

# Restart ComfyUI
cd ~/ai-tools/ComfyUI
python main.py
```

**Main features**
```
Node management:
✅ Search and install custom nodes
✅ Automatic dependency resolution
✅ One-click updates
✅ Automatic detection of missing nodes

Workflow management:
✅ Import from workflow sharing site
✅ Bulk installation of missing nodes
✅ Resolve "Red Box Hell" (Error)

Model management:
✅ Download from Hugging Face
✅ CivitAI model search
✅ Automatic model placement
```

### 4.3.2 Required custom nodes (2025 version)

**1. Efficiency Nodes for ComfyUI**
```bash
# install
git clone https://github.com/jags111/efficiency-nodes-comfyui.git

# function
- Merge multiple nodes into one
- Integrate KSampler + VAE Decode + Save
- Simplify prompt management
- Improved batch processing efficiency

# MS-S1 Max Benefits
- Improved workflow readability
- Improve memory efficiency by reducing the number of nodes
- Easy to change settings
```

**2. ComfyUI Impact Pack**
```bash
# install
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
cd ComfyUI-Impact-Pack
pip install -r requirements.txt

# function
- Advanced post-processing
- Face detection and Detailer
- Segmentation
- Automatic mask generation

# Usage
- Improved portrait quality
- Facial detail enhancement
- auto-correct
```

**3. WAS Node Suite**
```bash
# install
git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
cd was-node-suite-comfyui
pip install -r requirements.txt

# function
- Image processing utilities
- Text manipulation
- Math operations
- File system operations

# Usage
- Complex image processing
- Batch processing automation
- Custom pipeline
```

**4. ComfyUI Ultimate SD Upscale**
```bash
# install
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git

# function
- Tile upscale
- Memory efficient expansion
- Supports 4x and 8x

# Advantages with MS-S1 Max
- Large-scale image processing with 128GB memory
- 2048x2048 → 8192x8192 possible
- Efficient utilization of GPU and RAM
```

**5. rgthree's ComfyUI Nodes**
```bash
# install
git clone https://github.com/rgthree/rgthree-comfy.git
cd rgthree-comfy
pip install -r requirements.txt

# function
- Power Prompt: Advanced prompt management
- Quick Nodes: Adding nodes more efficiently
- Context Switch: Conditional branch
- Display Any: Debug display

# Usage
- Streamline workflow development
- Debugging work
- Complex condition handling
```

## 4.4 Advanced workflow construction

### 4.4.1 Batch generation workflow

**Multiple seed generation**
```
Purpose: Multiple variations of the same prompt

composition:
[Load Checkpoint]
         ↓
[CLIP Text Encode (Positive)]
[CLIP Text Encode (Negative)]
         ↓
[Empty Latent Image]
batch_size: 4 ← Important
         ↓
[KSampler]
seed: random ← change every time
  control_after_generate: "randomize"
         ↓
[VAE Decode]
         ↓
[Save Image]

Result: 4 pieces generated in one run
MS-S1 Max time: 35-40 seconds (@1024x1024)
```

**Grid generation**
```python
# Batch generation with different parameters

# prompt variations
prompts = [
    "photo of a cat, outdoor",
    "photo of a cat, indoor",
    "photo of a cat, studio lighting",
    "photo of a cat, natural lighting"
]

# CFG variation
cfg_values = [6.0, 7.0, 7.5, 8.0]

# Combination: 4 prompts × 4 CFGs = 16 images
# MS-S1 Max time: about 3 minutes
```

### 4.4.2 Automatic Img2Img pipeline

**Iterative improvement workflow**
```
[Load Image (original image)]
         ↓
[VAE Encode] → LATENT
         ↓
  ┌──→ [KSampler] denoise: 0.5
  │        ↓ LATENT
  │   [VAE Decode]
  │        ↓ IMAGE
  │   [Preview Image]
  │        ↓
  │   [VAE Encode]
  │        ↓
└────── LATENT (Loop)

Usage:
- Gradual quality improvement
- Tweak style adjustments
- Automation of trial and error
```

### 4.4.3 Conditional branching workflow

**Processing by resolution**
```python
# Use rgthree Context Switch

if resolution == "1024x1024":
    steps = 25
    cfg = 7.5
    use_refiner = False
elif resolution == "1536x1536":
    steps = 30
    cfg = 7.0
    use_refiner = True
elif resolution == "2048x2048":
    steps = 35
    cfg = 6.5
    use_refiner = True
# Use tile upscale
```

## 4.5 MS-S1 Max optimization workflow

### 4.5.1 Memory-efficient large-scale generation

**8K generation workflow**
```
Strategy: Tile expression generation + upscaling

Step 1: Base generation (1024x1024)
[KSampler] → IMAGE (1024x1024)
         ↓
Time: 11 seconds
VRAM: 8GB

Step 2: Ultimate SD Upscale (2x)
[Ultimate SD Upscale] → IMAGE (2048x2048)
  tile_size: 512
  overlap: 64
         ↓
Time: 35 seconds
VRAM: 12GB (peak)

Step 3: Ultimate SD Upscale (2x again)
[Ultimate SD Upscale] → IMAGE (4096x4096)
  tile_size: 512
  overlap: 64
         ↓
Time: 140 seconds
VRAM: 12GB (peak)

Total time: about 3 minutes
Final resolution: 4096x4096
Total VRAM: 12GB (Improved efficiency with tile processing)
```

### 4.5.2 Parallel model execution

**128GB memory utilization**
```
Simultaneous load model:

1. SDXL Base (8GB)
2. SDXL Refiner (8GB)
3. SD 1.5 Anime Model (4GB)
4. ControlNet Models (6GB)
5. LoRA Models (2GB)
6. VAE Models (1GB)
7. Upscale Models (1GB)

Total: 30GB
Remaining memory: 98GB (for system)

advantage:
- Zero model switching time
- Simultaneous execution of multiple workflows
- Background processing
```

## 4.6 Practical workflow example

### 4.6.1 Production Quality Portraits

**Complete workflow (15 nodes)**
```yaml
Phase 1: Base generation
- Load Checkpoint: SDXL Base
- CLIP Text Encode: Advanced prompt
- Empty Latent Image: 832x1216
- KSampler: steps=25, cfg=7.5
- Duration: 11 seconds

Phase 2: Apply Refiner
- Load Checkpoint: SDXL Refiner
- KSampler: steps=15, denoise=0.3
- Time: 5 seconds

Phase 3: Face detection and enhancement
- Impact Pack Face Detailer
- Upscale: 1.5x
- Detailed enhancement
- Time: 8 seconds

Phase 4: Final adjustments
- Color Correction
- Sharpen
- Save Image
- Time: 2 seconds

Total time: 26 seconds
Final quality: Professional
```

### 4.6.2 Batch Concept Art

**Mass generation workflow**
```python
setting:
- Batch Size: 8
- Resolution: 1024x1024
- Steps: 20 (speed priority)
- Sampler: euler_a
- CFG: 7.0
- Seeds: Random

Processing flow:
1. Prompt preparation (10 variations)
2. Generate 8 images for each prompt
3. Total 80 images
4. Total time: Approximately 15 minutes (MS-S1 Max)

output:
- 80 concept art
- Automatically divided into folders
- with metadata
```

### 4.6.3 Style Transfer Workflow

**Image-to-Image + LoRA**
```
[Load Image (original image)]
         ↓
[VAE Encode]
         ↓
[Load Checkpoint + LoRA]
  LoRA: anime_style.safetensors
  strength: 0.8
         ↓
[KSampler]
  denoise: 0.7
  prompt: "anime style, {original description}"
         ↓
[VAE Decode]
         ↓
[Save Image]

Usage:
- Animate your photos
- Live action → Illustration
- Batch style conversion
```

## 4.7 Debugging Workflow

### 4.7.1 Common errors and resolution

**Error 1: Red Node**
```
Cause:
- Custom node not installed
- Missing dependencies
- Model file not found

Solution:
1. Start ComfyUI Manager
2. Click "Install Missing Custom Nodes"
3. Automatic installation execution
4. Restart ComfyUI
```

**Error 2: Connection type mismatch**
```
Symptoms:
- Unable to connect between nodes
- The line turns red

Cause:
- Output and input types mismatch
Example: IMAGE → LATENT (not allowed)

Solution:
- Correct transformation node insertion
  IMAGE → [VAE Encode] → LATENT
  LATENT → [VAE Decode] → IMAGE
```

**Error 3: Out of memory**
```
Symptoms:
- "HIP out of memory"
- Generation failed midway

Solution (MS-S1 Max):
1. Reduce batch size
2. Lower resolution
3. Launch with --lowvram flag
4. Use tiling

# Usually does not occur with 128GB
# Check GPU allocation if this occurs
export GPU_MAX_ALLOC_PERCENT=90
```

### 4.7.2 Performance optimization

**Workflow optimization checklist**
```yaml
□ Delete unnecessary Preview Image nodes
- Delete after debugging
- Each Preview: +0.5 seconds

□ Minimize VAE Decode
- Only final output VAE Decode
- Leave the middle as LATENT

□ Reduce model load
- Avoid loading the same model multiple times
- Share one Load Checkpoint

□ Utilize Batch processing
- More efficient than single generation
- Batch size 4 recommended

□ Appropriate number of Steps
- 20-25 is often sufficient
- 50+ usually not required
```

**MS-S1 Max Benchmark**
```python
# Workflow efficiency measurement

# Non-optimized workflow
nodes = 25
preview_nodes = 5
multiple_vae_decodes = 4
time = 35 # seconds

# Optimization workflow
nodes = 12 # use integration node
preview_nodes = 1
vae_decodes = 1
time = 18 # seconds

# Improvement rate: 48% faster
```

## 4.8 Workflow sharing and community

### 4.8.1 Workflow sharing site

**Major Platforms**
```
OpenArt.ai/workflows:
- Large workflow collection
- Search by category
- Display number of downloads
- Comment function

ComfyWorkflows.com:
- Professional workflow site
- Tag-based search
- Difficulty level display

GitHub repositories:
- comfyanonymous/ComfyUI_examples
- Official example collection
- Regular updates
```

### 4.8.2 Export/Import Workflow

**Export steps**
```
Method 1: JSON save
1. Workflow completed
2. Right click → Save Workflow
3. Save as descriptive_name.json
4. Share on GitHub or Drive

Method 2: Image embedding
1. Save the generated image
2. Automatic workflow embedding in images
3. Share images
4. Recipient can restore by drag and drop
```

**Import steps**
```
Method 1: JSON import
1. Click the Load Workflow button
2. JSON file selection
3. Install missing nodes with Manager

Method 2: Image drag
1. Drag image to ComfyUI UI
2. Automatically restore workflow
3. Dependency resolution with "Install Missing Nodes"
```

## 4.9 Summary of this chapter

What you learned in this chapter:

**Workflow basics**
- Node-based visual programming
- Basic configuration (Text-to-Image, Image-to-Image)
- How to save and share

**Custom node**
- ComfyUI Manager (required)
- 5 essential nodes for 2025
- Utilize efficiency nodes

**Advanced construction**
- Batch generation workflow
- Conditional branching and automation
- Large scale image generation

**MS-S1 Max optimization**
- Utilize 128GB memory
- Parallel model loading
- Performance tuning

**Practice and debugging**
- Production quality workflow
- Common errors and solutions
- Community resources

In the next chapter, you will learn about detailed control using ControlNet.

---

**Reference resources**
- ComfyUI Manager: https://github.com/ltdrdata/ComfyUI-Manager
- OpenArt Workflows: https://openart.ai/workflows
- ComfyUI Wiki: https://comfyui-wiki.com/
- Awesome ComfyUI: https://github.com/ComfyUI-Workflow/awesome-comfyui

