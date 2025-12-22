# Chapter 5: Advanced Control with ControlNet

## 5.1 ControlNet Basics

### 5.1.1 What is ControlNet?

ControlNet is a technology that provides additional control signals to Stable Diffusion, allowing precise control over composition, pose, edges, and more.

**Basic principles**
```
Traditional Text-to-Image:
Text prompt → image generation
↓
Control is vague, composition is unstable

When using ControlNet:
Text prompt + control image → image generation
↓
Precise composition control and consistency
```

**Major control types**
```
Canny (edge ​​detection):
- Preserve contour lines
- Coloring from line drawing
- Maintaining the shape of the building

Depth:
- Preserves 3D structure
- Preserve depth information
- Control of spatial arrangement

OpenPose (skeleton):
- Person pose control
- Character placement
- Animation preparation

Scribble (rough sketch):
- Generated from hand-drawn sketches
- Control with rough instructions
- Concept art creation

Lineart:
- Generated from clean line drawings
- Illustration production
- For manga/anime

Normal (normal map):
- Control surface orientation
- Generated from 3D model
- Realistic shadows

Seg (segmentation):
- Control by area
- Complex composition
- Multi-object
```

### 5.1.2 SDXL and ControlNet

**SD 1.5 vs SDXL ControlNet**
```
SD 1.5 ControlNet:
- Lots of official models
- 512x512 optimization
- Active community

SDXL ControlNet:
- No official model
- Third party
- 1024x1024 compatible
- Union version recommended (2025)
```

**ControlNet Union for SDXL**
```
Features:
- Integrate multiple control types into one model
- Includes Canny, Depth, Openpose, etc.
- Memory efficient
- MS-S1 Max recommended

Compatible control:
✅ Canny
✅ Openpose
✅ Depth
✅ LineArt
✅ MLSD (Line Line Detection)
✅ Scribble
✅ HED (Holistic Edge)
✅ Normal
✅ Segmentation

Model size: approx. 2.5GB
```

## 5.2 Installing ControlNet

### 5.2.1 Download the model

**SDXL ControlNet Union model**
```bash
cd ~/ai-tools/ComfyUI/models/controlnet

# ControlNet Union SDXL (recommended)
wget https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors

# or specific control type
# Depth
wget https://huggingface.co/SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe/resolve/main/depth-zoe-xl-v1.0-controlnet.safetensors

# Canny
wget https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors

# OpenPose
wget https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/control-lora-openposeXL2-rank256.safetensors
```

**Model placement confirmation**
```bash
ls -lh ~/ai-tools/ComfyUI/models/controlnet/

# Example output:
# diffusion_pytorch_model_promax.safetensors (2.5GB) - Union
# depth-zoe-xl-v1.0-controlnet.safetensors (2.5GB)
# diffusion_pytorch_model.safetensors (2.5GB) - Canny
```

### 5.2.2 Preprocessor installation

**ComfyUI ControlNet Preprocessors**
```bash
cd ~/ai-tools/ComfyUI/custom_nodes

# Install ControlNet Preprocessors
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd comfyui_controlnet_aux
pip install -r requirements.txt

# Restart ComfyUI
cd ~/ai-tools/ComfyUI
python main.py
```

**Available preprocessors**
```
Image → Canny Edge:
- edge detection
- Threshold adjustable

Image → Depth Map:
- MiDaS, ZoeDepth
- Monocular depth estimation

Image → OpenPose:
- Human skeleton detection
- Detailed support for hands and faces

Image → LineArt:
- Line drawing extraction
- Compatible with both anime and real life

Image → Normal Map:
- Normal map generation
- Using BAE Normalizer

Image → Segmentation:
- Using OneFormer
- Region division
```

## 5.3 Basic ControlNet Workflow

### 5.3.1 Canny Edge Control Workflow

**Configuration (10 nodes)**
```
[Load Image] → IMAGE
         ↓
[Canny Edge Preprocessor]
  low_threshold: 100
  high_threshold: 200
↓ IMAGE (edge ​​image)
         ↓
[Load ControlNet Model]
  control_net_name: "diffusion_pytorch_model_promax.safetensors"
         ↓ CONTROL_NET
         ↓
[Apply ControlNet]
  strength: 0.8
         ↓ CONDITIONING
         ↓
[Load Checkpoint: SDXL Base]
         ↓
[CLIP Text Encode] → CONDITIONING
         ↓
[KSampler]
         ↓
[VAE Decode] → [Save Image]
```

**Setting details**
```yaml
Canny Edge Preprocessor:
low_threshold: 100 # Edge detection lower limit
high_threshold: 200 # Edge detection upper limit
# Lower value → more edges
# High value → only major edges

Apply ControlNet:
strength: 0.8 # control strength
# 0.0 = no control
# 0.5 = medium control
# 1.0 = maximum control

KSampler:
steps: 30 # more when using ControlNet
  cfg: 7.5
  sampler: dpmpp_2m_karras
```

**MS-S1 Max performance**
```
Processing time:
- Preprocessor (Canny): 0.5 seconds
- Generation (1024x1024): 11 seconds
- Total: 11.5 seconds

Memory usage:
- SDXL Base: 8GB
- ControlNet: 2.5GB
- Total: 10.5GB VRAM
```

### 5.3.2 Depth control workflow

**Depth map generation**
```
[Load Image]
         ↓
[MiDaS Depth Preprocessor]
  model_type: "DPT_Large"
# or ZoeDepth (more precise)
↓ IMAGE (depth map)
         ↓
[Apply ControlNet]
  control_net_name: "depth-zoe-xl-v1.0"
  strength: 0.7
         ↓
[SDXL Base] → [KSampler]
         ↓
[VAE Decode] → [Save Image]
```

**Usage and settings**
```yaml
Maintaining the composition of landscape photos:
  strength: 0.6-0.7
  prompt: "fantasy landscape, magical atmosphere"
Effect: Change style while maintaining depth

Building reconstruction:
  strength: 0.8-0.9
  prompt: "futuristic building, cyberpunk style"
Effect: Accurately maintains 3D structure

Controlling person placement:
  strength: 0.5-0.7
  prompt: "portrait in different lighting"
Effect: Change while preserving spatial arrangement
```

### 5.3.3 OpenPose skeleton control

**Person pose control**
```
[Load Image (Reference person)]
         ↓
[OpenPose Preprocessor]
detect_hand: True # Hand detection
detect_face: True # Face detection
detect_body: True # Body detection
↓ IMAGE (skeletal diagram)
         ↓
[Apply ControlNet]
  control_net_name: "control-lora-openposeXL2"
  strength: 0.8
         ↓
[SDXL Base + Prompt]
  prompt: "anime character, magical girl outfit"
         ↓
[KSampler] → [VAE Decode]
```

**Practical example**
```python
# Case 1: Photo → Anime character conversion
input_image = "photo_of_person_pose.jpg"
preprocessor = "OpenPose"
strength = 0.85
prompt = "anime style, character design, colorful outfit"

# Case 2: Placement of multiple people
input_image = "group_photo.jpg"
preprocessor = "OpenPose"
strength = 0.75
prompt = "fantasy characters, RPG party"

# Case 3: Animation frame
input_images = ["frame_001.jpg", "frame_002.jpg", ...]
# Consistent character generation every frame
```

## 5.4 Advanced ControlNet Technology

### 5.4.1 Multi ControlNet

**Use multiple controls at the same time**
```
[Load Image]
    ↓
    ├→ [Canny Edge] → ControlNet 1 (strength: 0.6)
    ↓
    ├→ [Depth Map] → ControlNet 2 (strength: 0.5)
    ↓
└→ [Combine both] → [KSampler]

effect:
- Control both edge and depth
- More accurate composition reproduction
- Effective for complex scenes
```

**strength adjustment strategy**
```yaml
High priority control:
  controlnet_1:
    type: Depth
strength: 0.8 # main control
  controlnet_2:
    type: Canny
strength: 0.4 # auxiliary control

Balance type:
  controlnet_1:
    type: OpenPose
    strength: 0.6
  controlnet_2:
    type: LineArt
    strength: 0.6

Fine adjustment type:
  controlnet_1:
    type: Depth
    strength: 0.9
  controlnet_2:
    type: Normal
strength: 0.3 # Minor adjustments only
```

### 5.4.2 ControlNet and LoRA combination

**Style + composition control**
```
[Load Checkpoint: SDXL Base]
         ↓
[Load LoRA]
  lora_name: "anime_style_v2.safetensors"
  strength_model: 0.8
         ↓
[Apply ControlNet (Depth)]
  strength: 0.7
         ↓
[KSampler]
  prompt: "anime landscape, studio ghibli style"
         ↓
[VAE Decode]

result:
- Apply styles with LoRA
- Composition control using ControlNet
- Get the best of both worlds
```

### 5.4.3 IP-Adapter + ControlNet

**Style reference + composition control**
```
[Load Image: Style Reference]
         ↓
[IP-Adapter]
  weight: 0.7
         ↓
[Load Image: Composition Reference]
         ↓
[ControlNet Depth]
  strength: 0.8
         ↓
[KSampler]

Usage:
- Style transfer + composition maintenance
- Reinterpretation of artwork
- Consistent series creation
```

## 5.5 ControlNet Workflow by Application

### 5.5.1 Architectural perspective creation

**3D model → Photoreal**
```yaml
Step 1: Render from 3D model
- Basic shapes with Blender/SketchUp
- Simple materials
- Camera angle confirmed

Step 2: Depth + Normal extraction
- Depth Map: Z-Buffer
- Normal Map: Render pass

Step 3: Apply ControlNet
preprocessor: none (already depth image)
controlnet_1: Depth (strength: 0.9)
controlnet_2: Normal (strength: 0.5)
prompt: "modern architecture, glass and concrete,
         professional photography, golden hour"
steps: 35
cfg: 8.0

Generation time (MS-S1 Max): 18 seconds
Quality: Professional perspective
```

### 5.5.2 Character Design

**Create pose variations**
```python
# Preparation for base pose
base_pose_images = [
    "standing_pose.jpg",
    "action_pose.jpg",
    "sitting_pose.jpg"
]

# Generate character for each pose
for pose_img in base_pose_images:
    workflow = {
        "preprocessor": "OpenPose",
        "controlnet": "openpose-sdxl",
        "strength": 0.85,
        "prompt": "anime character, warrior outfit,
                   detailed armor, fantasy style",
        "steps": 30,
        "cfg": 7.0,
"batch_size": 4 # 4 variations
    }

# Result: 3 poses x 4 variations = 12 images
# MS-S1 Max time: about 2 minutes
```

### 5.5.3 Replacing the background of product photos

**Maintain composition + change background**
```
[Load Image (product photo)]
         ↓
[Remove Background] # Custom node
↓ Products only
         ↓
[Canny Edge Preprocessor]
         ↓
[ControlNet Canny (strength: 0.9)]
         ↓
[KSampler]
  prompt: "product photography, luxury background,
           marble surface, studio lighting"
         ↓
[VAE Decode]

Usage:
- Image for EC site
-Catalog creation
- Variation generation

MS-S1 Max efficiency:
- Multiple backgrounds with batch size 8
- Generation time: 35 seconds/8 images
```

## 5.6 Preprocessor advanced settings

### 5.6.1 Canny edge adjustment

**Influence of parameters**
```python
# Simple outline only
canny_simple = {
    "low_threshold": 150,
    "high_threshold": 250,
"result": "Detect only major edges"
}

# Detailed edges
canny_detailed = {
    "low_threshold": 50,
    "high_threshold": 150,
"result": "Detect even the smallest details"
}

# Balanced (recommended)
canny_balanced = {
    "low_threshold": 100,
    "high_threshold": 200,
"result": "moderate detail"
}
```

**Settings by use**
```yaml
Building:
  low: 120
  high: 220
Why: Clean lines are important

person:
  low: 80
  high: 180
Reason: Needs soft contours

Landscape:
  low: 100
  high: 200
Reason: Focus on balance

Line art:
  low: 150
  high: 250
Reason: Clean lines only
```

### 5.6.2 Depth accuracy adjustment

**Depth estimation model comparison**
```
MiDaS DPT_Large:
- Accuracy: ★★★★☆
- Speed: ★★★☆☆
- Application: General depth estimation
- Processing time: 1.5 seconds @ 1024x1024 (MS-S1 Max)

ZoeDepth:
- Accuracy: ★★★★★
- Speed: ★★☆☆☆
- Application: Cases where high precision is required
- Processing time: 3.0 seconds @ 1024x1024 (MS-S1 Max)

MiDaS Small:
- Accuracy: ★★★☆☆
- Speed: ★★★★★
- Application: Prototyping
- Processing time: 0.8 seconds @ 1024x1024 (MS-S1 Max)
```

**MS-S1 Max recommended**
```
Normal work: MiDaS DPT_Large
- Good balance
- sufficient accuracy

Final output: ZoeDepth
- Top quality
- It's okay even if it takes a long time.

Mass generation: MiDaS Small
- Speed ​​priority
- For batch processing
```

### 5.6.3 OpenPose precision settings

**Detection options**
```python
openpose_config = {
"detect_body": True, # required
"detect_hand": True, # Hand detail
"detect_face": True, # Face orientation
"resolution": 512, # detection resolution
}

# Settings by usage

# Full body portrait
config_fullbody = {
    "detect_body": True,
"detect_hand": True, # Important
    "detect_face": True,
    "resolution": 512,
}

# Focus on face
config_face = {
    "detect_body": True,
"detect_hand": False, # not required
"detect_face": True, # most important
"resolution": 768, # High resolution
}

# action pose
config_action = {
"detect_body": True, # most important
    "detect_hand": True,
"detect_face": False, # low priority
    "resolution": 512,
}
```

## 5.7 Troubleshooting

### 5.7.1 General issues

**Problem 1: Loss of control**
```
Symptoms:
- The composition does not change even after applying ControlNet
- Generated with prompt only

Cause and solution:
1.Strength is too low
Solution: Increase to 0.7-0.9

2. CFG is too high
Solution: Adjust CFG to below 7.5

3.Steps missing
Solution: Increase to 30-35 steps

4. ControlNet model not loaded
Solution: Apply ControlNet Node Verification
```

**Problem 2: Too much control**
```
Symptoms:
- Simply traced the original image
- No creativity

Cause and solution:
1.Strength is too high
Solution: lower to 0.5-0.7

2. denoise is too low
Solution: set denoise 0.9-1.0

3. Not enough prompts
Solution: Added detailed prompts
```

**Problem 3: Out of memory (rare on MS-S1 Max)**
```
Symptoms:
- "HIP out of memory"
- When using multiple ControlNet

Solution:
1. Reduce the number of ControlNets
limited to 2

2. Lower resolution
   1024x1024 → 896x896

3. Check GPU allocation
   export GPU_MAX_ALLOC_PERCENT=95
```

### 5.7.2 Quality optimization

**Checklist**
```yaml
□ Adjust preprocessor settings
- Canny threshold
- Depth accuracy
- OpenPose detection options

□ Fine-tune ControlNet strength
- starting from 0.5
- Adjust by 0.1
- find the optimal value

□ Appropriate number of Steps
- When using ControlNet: 30-35
- Quality priority: 35-40
- Speed ​​priority: 25-30

□ CFG value adjustment
- When using ControlNet: 7.0-8.0
- Complex composition: 7.5-8.5
- Simple composition: 6.5-7.5

□ Sampler selection
- Recommended: dpmpp_2m_karras
- High quality: dpmpp_sde_karras
```

## 5.8 MS-S1 Max optimization strategy

### 5.8.1 Batch ControlNet Processing

**Batch processing of multiple images**
```python
# Workflow design
input_images = [
    "ref_001.jpg",
    "ref_002.jpg",
    "ref_003.jpg",
    "ref_004.jpg"
]

# Batch processing workflow
for image in input_images:
# Apply Preprocessor (can be executed in parallel)
    depth_map = apply_preprocessor(image, "MiDaS")

# Generate ControlNet
    result = generate_with_controlnet(
        control_image=depth_map,
        prompt="fantasy landscape, dramatic lighting",
        strength=0.75,
        steps=30,
batch_size=2 # 2 variations for each image
    )

# Total: 4 images x 2 variations = 8 images
# MS-S1 Max time: about 1.5 minutes
```

### 5.8.2 Parallel preprocessor execution

**128GB memory utilization**
```bash
# Simultaneous execution of multiple preprocessors

# Terminal 1: Depth processing
python preprocess_batch.py --type depth --input_dir ./images

# Terminal 2: Canny processing (parallel)
python preprocess_batch.py --type canny --input_dir ./images

# Terminal 3: OpenPose processing (parallel)
python preprocess_batch.py --type openpose --input_dir ./images

# Memory usage:
# Each preprocessor: 4-8GB
# Total: 12-24GB
# Remaining: 100GB or more (with plenty of room)
```

## 5.9 Summary of this chapter

What you learned in this chapter:

**ControlNet Basics**
- Understanding control types (Canny, Depth, OpenPose, etc.)
- SDXL compatible model (Union recommended)
- Role of preprocessor

**Basic workflow**
- Canny Edge control
- Depth Map control
- OpenPose skeletal control

**Advanced technology**
- Multi-ControlNet
- Combination with LoRA/IP-Adapter
- Practical workflow examples

**MS-S1 Max optimization**
- Batch processing strategy
- Parallel preprocessor execution
- Memory efficient operation

**troubleshooting**
- Common problems and solutions
- Quality optimization checklist

In the next chapter, we will learn more about using and creating LoRA models.

---

**Reference resources**
- ControlNet paper: https://arxiv.org/abs/2302.05543
- ControlNet Union SDXL: https://huggingface.co/xinsir/controlnet-union-sdxl-1.0
- ComfyUI ControlNet Aux: https://github.com/Fannovel16/comfyui_controlnet_aux
- Stable Diffusion Art ControlNet Guide: https://stable-diffusion-art.com/controlnet/

