# Chapter 6: Using and Creating LoRA Models

## 6.1 LoRA Basics

### 6.1.1 What is LoRA?

LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large-scale models.

**Differences from conventional fine tuning**
```
Full fine tuning:
- Learn all parameters
- Model size: 6.6GB (SDXL)
- Learning time: long
- VRAM requirement: 24GB or more
- Storage capacity: 6.6GB per model

LoRA:
- Learn only a few parameters
- LoRA size: 50-200MB
- Learning time: short
- VRAM requirement: 12-16GB
- Storage capacity: 50-200MB per model
- Possible to combine multiple LoRAs
```

**How ​​LoRA works**
```python
# LoRA mathematical concepts
W' = W + ∆W
∆W = B × A

# W: Original model parameters
# B: Low-rank matrix B (rank × out_features)
# A: Low-rank matrix A (in_features × rank)
# rank: usually 8, 16, 32, 64, 128
# ∆W: Variation to learn

# Memory efficiency
Full fine-tune: in_features × out_features
LoRA: (in_features + out_features) × rank
# For Rank 32, the parameter is approximately 1/100th
```

### 6.1.2 LoRA types

**Classification by use**
```
Character LoRA:
- Specific character appearance
- Anime, game characters
- Consistent character generation

Style LoRA:
- specific art style
- Artist imitation
- Apply painting style

Concept LoRA:
- specific concepts
- Pose, composition
- special effects

Object LoRA:
- specific object
- Products, items
- Architectural style
```

**Classification by parameter size**
```
Rank 8-16:
- Size: 20-50MB
- Learning speed: Fastest
- Expressiveness: Basic
- Usage: Simple style

Rank 32-64:
- Size: 50-150MB
- Learning speed: Standard
- Expressiveness: Good (recommended)
- Usage: General use

Rank 128+:
- Size: 150-300MB
- Learning speed: Slightly slow
- Expressive power: best
- Uses: complex styles, detailed control
```

## 6.2 How to use LoRA

### 6.2.1 Basic Workflow

**Single LoRA application**
```
[Load Checkpoint: SDXL Base]
         ↓ MODEL, CLIP
         ↓
[Load LoRA]
  lora_name: "anime_style_v2.safetensors"
strength_model: 0.8 # Impact on model
strength_clip: 1.0 # Impact on prompt comprehension
         ↓ MODEL, CLIP
         ↓
[CLIP Text Encode]
  prompt: "anime character, magical girl"
         ↓
[KSampler] → [VAE Decode] → [Save Image]
```

**strength_model and strength_clip**
```yaml
strength_model:
Range: 0.0 - 2.0 (typically 0.5-1.0)
0.0: LoRA disabled
0.5: conservative application
0.8: Balanced (recommended)
1.0: Fully applicable
1.5: Strong application (be careful of overfitting)

strength_clip:
Range: 0.0 - 1.0
0.0: No effect on prompt comprehension
0.5: partial impact
1.0: Full impact (recommended)
```

**MS-S1 Max performance**
```
LoRA loading time:
- 50MB LoRA: 0.3 seconds
- 150MB LoRA: 0.8 seconds

Generation time (SDXL + LoRA):
- 1024x1024: 11-12 seconds
- LoRA overhead: approximately +5%

Memory usage:
- SDXL Base: 8GB
- LoRA: 0.5-1GB
- Total: 8.5-9GB VRAM
```

### 6.2.2 Combining multiple LoRAs

**Multi-LoRA Workflow**
```
[Load Checkpoint: SDXL Base]
         ↓
[Load LoRA 1]
  lora: "anime_style.safetensors"
  strength_model: 0.7
  strength_clip: 1.0
         ↓
[Load LoRA 2]
  lora: "character_a.safetensors"
  strength_model: 0.8
  strength_clip: 1.0
         ↓
[Load LoRA 3]
  lora: "lighting_style.safetensors"
  strength_model: 0.5
  strength_clip: 0.5
         ↓
[KSampler]

result:
- anime style
- Specific character
- Special lighting
- 3 effects integrated
```

**Combination strategy**
```yaml
# Balanced type (recommended)
lora_1:
  type: style
  strength: 0.7

lora_2:
  type: character
  strength: 0.8

lora_3:
  type: concept
  strength: 0.5

# Emphasis type
main_lora:
  type: style
  strength: 1.0

support_lora_1:
  type: concept
  strength: 0.3

support_lora_2:
  type: lighting
  strength: 0.2
```

**Notes**
```
Avoid excessive combinations:
✅ 2-3 LoRA: Effective
⚠️ 4-5 LoRA: Potential conflicts
❌ 6 or more: Risk of quality deterioration

Strength total guidelines:
- Total strength of all LoRA: 2.5 or less recommended
- Main LoRA: 0.7-1.0
- Auxiliary LoRA: 0.3-0.5
```

## 6.3 Obtaining and installing LoRA

### 6.3.1 Main distribution sites

**CivitAI**
```
URL: https://civitai.com

Features:
✅ Largest LoRA community
✅ Rich preview images
✅ User reviews
✅ Example prompts included
✅ Version control

category:
- Character
- Style
- Concept
- Poses
-Objects

download:
1. Open the model page
2. Check SDXL compatibility
3. Click the download button
4. Get .safetensors file
```

**Hugging Face**
```
URL: https://huggingface.co/models

Features:
✅ Official/Research LoRA
✅ Download via API
✅ Can be managed with git clone

search:
filter: "lora" AND "sdxl"

download:
# Using CLI tool
huggingface-cli download \
    username/lora-name \
    lora_model.safetensors \
    --local-dir ~/ai-tools/ComfyUI/models/loras
```

### 6.3.2 LoRA placement

**Directory structure**
```bash
~/ai-tools/ComfyUI/models/loras/
├── characters/
│   ├── character_a_v2.safetensors
│   ├── character_b_sdxl.safetensors
│   └── game_character_pack.safetensors
├── styles/
│   ├── anime_style_v3.safetensors
│   ├── watercolor_sdxl.safetensors
│   └── pixel_art_lora.safetensors
├── concepts/
│   ├── dynamic_poses.safetensors
│   ├── cinematic_lighting.safetensors
│   └── fantasy_effects.safetensors
└── test/
    └── experimental_loras/
```

**Permission settings**
```bash
# Check permissions of LoRA directory
ls -la ~/ai-tools/ComfyUI/models/loras/

# Set as required
chmod 755 ~/ai-tools/ComfyUI/models/loras/
chmod 644 ~/ai-tools/ComfyUI/models/loras/*.safetensors
```

## 6.4 Creating and learning LoRA

### 6.4.1 Preparation for learning

**Dataset preparation**
```
Image requirements:
- Number of sheets: 20-100 sheets (ideally 50 sheets)
- Resolution: 1024x1024 recommended
- Format: JPG, PNG
- Quality: High quality, consistent
- Variety: various angles, poses

Recommended number of sheets (by application):
Character: 40-60 sheets
Style: 30-50 sheets
Object: 20-40 pieces
Concept: 50-100 pieces
```

**Create caption**
```bash
# Automatic caption generation

# Method 1: Using BLIP2
cd ~/ai-tools
git clone https://github.com/pharmapsychotic/clip-interrogator.git
cd clip-interrogator
pip install -r requirements.txt

python caption_images.py \
    --input_dir ./training_images \
    --output_dir ./captions

# Method 2: Manual caption
# Create .txt file corresponding to each image
image_001.jpg → image_001.txt
Contents: "anime character, blue hair, red eyes, school uniform, smiling"

# directory structure
training_data/
├── image_001.jpg
├── image_001.txt
├── image_002.jpg
├── image_002.txt
└── ...
```

### 6.4.2 Using Lora-Training-in-Comfy

**Custom node installation**
```bash
cd ~/ai-tools/ComfyUI/custom_nodes
git clone https://github.com/LarryJane491/Lora-Training-in-Comfy.git
cd Lora-Training-in-Comfy
pip install -r requirements.txt

# Restart ComfyUI
cd ~/ai-tools/ComfyUI
python main.py
```

**Learning workflow construction**
```
[Init SDXL LoRA Training]
  model_path: "sd_xl_base_1.0.safetensors"
  dataset_path: "./training_data"
  output_name: "my_lora_v1"
rank: 32 # Rank of LoRA
alpha: 32 # learning rate adjustment
batch_size: 2 # MS-S1 Max recommended
max_train_steps: 1000 # Number of training steps
learning_rate: 1e-4 # learning rate
save_every_n_steps: 250 # Save interval
         ↓
[Train LoRA]
         ↓
[Save LoRA Model]
  output_dir: "~/ai-tools/ComfyUI/models/loras/"
```

**MS-S1 Max optimal settings**
```yaml
# Learning settings (for MS-S1 Max)

Hardware utilization:
batch_size: 4 # 128GB memory utilization
  gradient_accumulation: 2
  mixed_precision: "fp16"
  use_8bit_adam: True

Learning parameters:
rank: 32 # Good balance
alpha: 32 # Same recommendation as rank
learning_rate: 1e-4 # standard
  max_train_steps: 1000-2000

Data extension:
  random_crop: True
  random_flip: True
  color_jitter: 0.1

Learning time (MS-S1 Max):
- 50 images, 1000 steps: about 20-30 minutes
- 100 images, 2000 steps: about 50-70 minutes
- VRAM usage: 12-16GB
```

### 6.4.3 Kohya ss-sdxl-lora-trainer

**More advanced learning environment**
```bash
# Install Kohya Trainer
cd ~/ai-tools
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Check PyTorch ROCm version
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.4
```

**Learning script**
```bash
#!/bin/bash
# train_lora.sh

# ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100

# Execute learning
accelerate launch --num_cpu_threads_per_process=8 \
    sdxl_train_network.py \
    --pretrained_model_name_or_path="sd_xl_base_1.0.safetensors" \
    --train_data_dir="./training_data" \
    --output_dir="./output_loras" \
    --output_name="my_character_lora" \
    --network_module="networks.lora" \
    --network_dim=32 \
    --network_alpha=32 \
    --learning_rate=1e-4 \
    --max_train_steps=2000 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --mixed_precision="fp16" \
    --save_every_n_steps=250 \
    --save_model_as="safetensors" \
    --clip_skip=2 \
    --seed=42 \
    --enable_bucket \
    --bucket_reso_steps=64 \
    --bucket_no_upscale \
    --min_bucket_reso=512 \
    --max_bucket_reso=2048

# MS-S1 Max: Completed in about 30 minutes (50 images, 2000 steps)
```

**Advanced parameters**
```yaml
Data processing:
enable_bucket: True # Resolution bucket
bucket_reso_steps: 64 # resolution steps
  min_bucket_reso: 512
  max_bucket_reso: 2048
  random_crop: True

optimization:
optimizer_type: "AdamW8bit" # Memory efficient
lr_scheduler: "cosine" # Learning rate schedule
  lr_warmup_steps: 100
  gradient_checkpointing: True

Regularization:
noise_offset: 0.05 # Noise offset
  adaptive_noise_scale: 0.00357
clip_skip: 2 # CLIP layer skip
```

## 6.5 LoRA Evaluation and Testing

### 6.5.1 Quality evaluation

**Checkpoint test**
```python
# Test each checkpoint during learning

checkpoints = [
    "my_lora_step_250.safetensors",
    "my_lora_step_500.safetensors",
    "my_lora_step_750.safetensors",
    "my_lora_step_1000.safetensors"
]

test_prompts = [
    "anime character, standing pose, white background",
    "anime character, action pose, outdoor scene",
    "anime character, close-up portrait, detailed face"
]

for checkpoint in checkpoints:
    for prompt in test_prompts:
# Workflow execution
        generate_image(
            lora=checkpoint,
            prompt=prompt,
            strength=0.8,
seed=42 # fixed for consistency
        )

# Compare results and select optimal checkpoint
```

**Overfitting detection**
```
Symptoms:
- Too similar to the training image
- Broke with new prompts
- Lack of diversity

How to check:
1. Test with prompts you didn't use for learning
2. Generated with various strength values
3. Combination test with other LoRAs

countermeasure:
- reduce max_train_steps
- lower learning_rate
- Increase dataset
- Enhanced regularization
```

### 6.5.2 LoRA Merging

**Multiple LoRA integration**
```
Usage:
- Combine multiple LoRA effects into one
- Storage savings
- Improved efficiency during inference

method:
[Load LoRA 1]
  strength: 0.7
     ↓
[Load LoRA 2]
  strength: 0.5
     ↓
[Merge LoRAs]
     ↓
[Save Merged LoRA]
  output: "merged_lora.safetensors"

Note:
- Compatible LoRA only
- Difficult to predict effects
- Test required
```

## 6.6 Practical LoRA usage

### 6.6.1 Maintaining character consistency

**Create a series of works**
```yaml
Purpose: Create multiple scenes with the same character

Workflow:
Step 1: Apply Character LoRA
  lora: "my_character.safetensors"
  strength: 0.9

Step 2: Prompts for each scene
  scene_1: "{character trigger}, in forest, daytime"
  scene_2: "{character trigger}, in city, night"
  scene_3: "{character trigger}, at beach, sunset"

Step 3: Fixed parameters
seed: change every time (variation)
  steps: 30
  cfg: 7.5
  sampler: dpmpp_2m_karras

result:
- Consistent character appearance
- Diverse backgrounds and poses
- Series work completed

MS-S1 Max:
- 10 scene generation: about 2 minutes
- Batch size 4: even more efficient
```

### 6.6.2 Style Mixing

**Multiple styles fused**
```python
# Anime + Watercolor + Fantasy

workflow = {
    "base_model": "sd_xl_base_1.0.safetensors",
    "loras": [
        {
            "name": "anime_style.safetensors",
            "strength_model": 0.6,
            "strength_clip": 1.0
        },
        {
            "name": "watercolor_painting.safetensors",
            "strength_model": 0.5,
            "strength_clip": 0.8
        },
        {
            "name": "fantasy_concept.safetensors",
            "strength_model": 0.4,
            "strength_clip": 0.6
        }
    ],
    "prompt": "magical forest, glowing mushrooms, fairy lights",
    "negative": "low quality, blurry",
    "steps": 30,
    "cfg": 7.5
}

# Result: Anime Feng Watercolor Fantasy Art
```

### 6.6.3 Notes on commercial use

**License confirmation**
```
Things to check:
□ LoRA license (check with CivitAI)
□ Original model license
□ Rights to learning data
□ Possibility of commercial use
□ Whether credit is required

Recommended:
- Prefer official LoRA
- Record license
- Avoid use if unsure
```

## 6.7 Troubleshooting

### 6.7.1 Problems during learning

**Problem 1: Out of memory**
```
Symptoms:
- "HIP out of memory"
- Learning stopped midway

Solution (MS-S1 Max):
1. Reduce batch_size
   4 → 2 → 1

2. Completion with gradient_accumulation
   batch_size: 2
   gradient_accumulation: 2
# Equivalent to batch_size 4

3. Use mixed_precision
   mixed_precision: "fp16"

4. Enable gradient_checkpointing
   gradient_checkpointing: True
```

**Problem 2: Learning is not progressing**
```
Symptoms:
- Loss value does not decrease
- Generated results do not improve

Cause and solution:
1. learning_rate is inappropriate
Solution: Adjusted from 1e-4 to 5e-5

2. Poor dataset quality
Solution: Replace with higher quality image

3. Inappropriate captions
Solution: Create detailed captions

4.Steps missing
Solution: Increase to 2000 steps or more
```

### 6.7.2 Problems when using

**Problem 1: LoRA doesn't work**
```
Symptoms:
- No change even after applying LoRA

Cause and solution:
1.Strength is too low
Solution: Increase to 0.8-1.0

2. Missing trigger word in prompt
Solution: Check/add LoRA trigger word

3. CFG is inappropriate
Solution: Adjust to CFG 7-8

4. Model compatibility
Solution: Check if LoRA for SDXL
```

**Problem 2: Poor quality**
```
Symptoms:
- Artifact generation
- Collapsed image

Cause and solution:
1.Strength is too high
Solution: lower to 0.5-0.7

2. Multiple LoRA conflicts
Solution: Reduce the number of LoRAs

3. Overtrained LoRA
Solution: different checkpoint attempts
```

## 6.8 MS-S1 Max optimization strategy

### 6.8.1 Efficient learning

**Parallel learning**
```bash
# Parallel learning using 128GB memory

# Terminal 1: Character LoRA learning
cd ~/ai-tools/sd-scripts
source venv/bin/activate
./train_character.sh

# Terminal 2: Style LoRA learning (parallel)
cd ~/ai-tools/sd-scripts
source venv/bin/activate
./train_style.sh

# Resource allocation:
# Each learning: 16GB VRAM, 20GB RAM
# Total: 32GB VRAM, 40GB RAM
# Remaining: 88GB (extra)
```

### 6.8.2 LoRA Library Management

**Metadata management**
```bash
# lora_metadata.json
{
    "loras": [
        {
            "name": "anime_style_v3.safetensors",
            "type": "style",
            "rank": 32,
            "trained_on": "SDXL Base 1.0",
            "trigger_words": ["anime style", "cel shaded"],
            "recommended_strength": 0.7,
            "tags": ["anime", "style", "general"],
            "created": "2025-01-15",
            "source": "civitai",
            "license": "CreativeML Open RAIL-M"
        },
        {
            "name": "character_miku.safetensors",
            "type": "character",
            "rank": 64,
            "trained_on": "SDXL Base 1.0",
            "trigger_words": ["miku", "twin tails", "blue hair"],
            "recommended_strength": 0.9,
            "tags": ["character", "vocaloid"],
            "created": "2025-01-20",
            "source": "custom_trained",
            "license": "personal_use_only"
        }
    ]
}
```

## 6.9 Summary of this chapter

What you learned in this chapter:

**LoRA basics**
- How LoRA works and its benefits
- Type and parameter size
- Differences from full fine tuning

**Using LoRA**
- Basic workflow
- Combination of multiple LoRAs
- Adjustment of strength parameters

**LoRA creation**
- Dataset preparation
- Using Lora-Training-in-Comfy
- Learning with Kohya ss-scripts

**MS-S1 Max optimization**
- 128GB memory utilization
- Parallel learning strategy
- Efficient workflow

**Practice and Management**
- Maintain character consistency
- Style mixing
- Library management

In the next chapter, we will learn more about AMDGPU and ROCm optimization.

---

**Reference resources**
- LoRA paper: https://arxiv.org/abs/2106.09685
- Kohya ss-scripts: https://github.com/kohya-ss/sd-scripts
- CivitAI: https://civitai.com/
- Lora-Training-in-Comfy: https://github.com/LarryJane491/Lora-Training-in-Comfy

