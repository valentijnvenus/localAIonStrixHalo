# Chapter 3: SDXL Basics and Prompt Engineering

## 3.1 Understanding SDXL Architecture

### 3.1.1 SDXL vs SD 1.5

**Architectural evolution**
```
SD 1.5:
- U-Net parameters: 860M
- Text encoder: CLIP ViT-L/14
- Resolution: 512x512 native
- Total parameters: about 1B

SDXL:
- U-Net parameters: 2.6B
- Text encoder: CLIP ViT-L + OpenCLIP ViT-G
- Resolution: 1024x1024 native
- Total parameters: about 6.6B
- Improvements: text understanding, composition, detailed expression
```

### 3.1.2 SDXL two-stage model

**Base Model (required)**
```
role:
- Main image generation
- Native generation at 1024x1024
- Main interpretation of prompts

Parameter: 2.6B
Training data: Pre-training with high quality images
Usage: Basis of all generation
```

**Refiner Model (optional)**
```
role:
- Detail enhancement of Base generated images
- Improved high frequency detail
- Final finishing

Parameter: 2.3B
Training data: fine-tune with high-quality images
Usage timing: From 80-85% completion of Base
```

**Decision on using Refiner on MS-S1 Max**
```python
# Refiner recommended case
Recommended use:
✅ Focus on final output quality
✅ Detailed texture expression required
✅ Professional use
✅ You have plenty of time (+40-50% time)

Unnecessary case:
❌ Prototyping
❌ Mass production
❌ Emphasis on real-time performance
❌ Memory constraints exist

# Generation time comparison (MS-S1 Max, 1024x1024)
Base only: 10-12 seconds
Base + Refiner: 16-20 seconds
```

## 3.2 Prompt Engineering Fundamentals

### 3.2.1 SDXL Prompt Characteristics

**Differences from SD 1.5**
```
SD 1.5:
- Keyword focused
- Focus on comma separated tags
- Strict order dependence
- Negative prompt required

SDXL:
- Improved natural language understanding
- Recognizes text format as well
- Contextual understanding possible
- Minimal negative prompts possible
- Higher semantic understanding
```

**Strengths of SDXL Prompts**
```python
# SD 1.5 style (works but not optimal)
prompt_sd15 = "cat, sitting, window, sunlight, detailed fur, 4k"

# SDXL style (recommended)
prompt_sdxl = "A fluffy orange cat sitting on a windowsill, bathed in warm sunlight. The cat's fur is detailed and realistic, with individual strands visible. High quality, 4k resolution."

# Both work, but SDXL format reflects the intent more accurately
```

### 3.2.2 Prompt Structure Best Practices

**Recommended prompt structure**
```
1. Subject
- Be clear about what to generate

2. Detailed Imagery
- Color, texture, material

3. Environment
- Scene, foreground/background

4. Mood/Atmosphere
- lighting, emotion, style

5. Technical specification
- resolution, quality, artist
```

**Example: Landscape image**
```
# Basic prompt
"mountain landscape"
→ Generated but ambiguous

# Structured prompts (recommended)
"A majestic snow-capped mountain range during golden hour.
The peaks are illuminated by warm sunset light, creating
long shadows across alpine meadows in the foreground.
Dramatic clouds gather around the summits. Crystal clear
lake reflects the scene. Professional landscape photography,
high detail, 4k quality, cinematic composition."

Components:
1. Subject: mountain range
2. Details: snow-capped, golden hour, warm sunset light
3. Environment: alpine meadows, crystal clear lake
4. Mood: dramatic clouds, majestic
5. Technology: professional photography, 4k, cinematic
```

**Illustrative example: portrait**
```
"A portrait of a young woman with long flowing auburn hair,
piercing green eyes, and a gentle smile. Soft natural lighting
from a window creates a warm glow on her face. She's wearing
a cream-colored sweater. Shallow depth of field blurs the
cozy interior background. Professional portrait photography,
85mm lens, f/1.8, soft focus, high quality."

Components:
1. Subject: young woman
2. Details: auburn hair, green eyes, cream sweater
3. Environment: cozy interior, window light
4. Mood: gentle smile, warm glow
5. Technology: 85mm, f/1.8, soft focus
```

### 3.2.3 Keyword weight

**Wait syntax**
```python
# Weight specification in ComfyUI
Basic: (keyword) # 1.1x
Emphasis: ((keyword)) # 1.21x (1.1^2)
Furthermore: (((keyword))) # 1.331x (1.1^3)

# Specify numerical value (more accurate)
(keyword:1.2) # 1.2 times
(keyword:1.5) # 1.5 times
(keyword:0.8) # 0.8x (weaken)

# Notes on SDXL
⚠️ SDXL is weight sensitive
⚠️ 1.4 or higher is usually not required
⚠️ If you overdo it, it will collapse.
```

**Example of proper weight usage**
```
# Proper use
"A (detailed:1.2) portrait of a woman with (flowing hair:1.1),
natural lighting, high quality"

# Excessive use (not recommended)
"A (((detailed:1.5))) portrait of a (((woman:1.8))) with
(((flowing hair:2.0))), (((natural lighting:1.7)))"
→ Risk of artifact generation
```

**MS-S1 Max recommended weight strategy**
```yaml
Standard emphasis: 1.1 - 1.2
Medium emphasis: 1.2 - 1.3
Strong emphasis: 1.3 - 1.4
Max: 1.5 (use with caution)

Weak: 0.7 - 0.9
Significantly weaken: 0.5 - 0.7
```

## 3.3 Negative prompting strategy

### 3.3.1 Negative Prompts in SDXL

**SD 1.5 vs SDXL**
```
SD 1.5:
- Large amount of negative keywords required
- "ugly, bad, deformed, extra fingers, ..." (50+ words)
- Without negatives, quality decreases

SDXL:
- Minimal and effective
- Only specific elements you want to avoid
- Excessive negativity has the opposite effect
```

**SDXL Optimized Negative Prompt**
```python
# minimal (recommended)
negative_minimal = "low quality, blurry"

# Standard (general use)
negative_standard = "low quality, blurry, distorted, watermark"

# Details (specific use)
negative_detailed = """
low quality, blurry, out of focus,
distorted, watermark, text,
oversaturated, underexposed
"""

# Excessive (not recommended)
negative_excessive = """
ugly, bad, deformed, extra fingers,
mutated hands, poorly drawn, bad anatomy,
wrong anatomy, extra limbs, missing limbs,
floating limbs, disconnected limbs, mutation,
mutated, ugly, disgusting, blurry, amputation,
JPEG artifacts, signature, watermark, username,
sketch, cartoon, drawing, anime, text, cropped,
out of frame, worst quality, low quality,
jpeg artifacts, ugly, duplicate, morbid,
mutilated, extra digits, fewer digits, ...
"""
→ Not necessary with SDXL, rather a negative effect
```

**Negative prompts by use**
```
Realistic photos:
"cartoon, illustration, 3d render, painting"

Illustration generation:
"photograph, realistic, photorealistic"

Portrait:
"multiple people, crowd, extra arms, extra legs"

Landscape:
"People, buildings (if you want to avoid them), vehicles"

Anime style:
"realistic, photograph, 3d"
```

### 3.3.2 MS-S1 Max recommended settings

**Settings by quality level**
```yaml
# Rapid prototyping
positive: "simple subject description"
negative: "low quality"
steps: 20
cfg: 7.0

# Standard quality
positive: "detailed subject with environment and mood"
negative: "low quality, blurry, distorted"
steps: 25
cfg: 7.5

# Top quality
positive: "comprehensive description with all elements"
negative: "low quality, blurry, distorted, watermark, oversaturated"
steps: 30-35
cfg: 7.0-8.0
use_refiner: true
```

## 3.4 Resolution and aspect ratio

### 3.4.1 SDXL Native Resolution

**Importance of total number of pixels**
```
SDXL training resolution: 1024x1024 = 1,048,576 pixels

Recommended resolution (same total number of pixels):
✅ 1024x1024 (1:1)   - 1,048,576
✅ 1152x896  (9:7)   - 1,032,192
✅ 896x1152  (7:9)   - 1,032,192
✅ 1216x832  (3:2)   - 1,011,712
✅ 832x1216  (2:3)   - 1,011,712
✅ 1344x768  (16:9)  - 1,032,192
✅ 768x1344  (9:16)  - 1,032,192
✅ 1536x640  (21:9)  - 983,040

⚠️ What to avoid:
❌ 512x512 - blurry
❌ 2048x2048 - Out of memory, artifacts
```

### 3.4.2 Recommended resolution by MS-S1 Max memory

**By VRAM allocation**
```python
# 16GB VRAM allocation
resolutions_16gb = {
"standard": (1024, 1024), # 10-12 seconds
"portrait": (832, 1216), # 10-12 seconds
"landscape": (1216, 832), # 10-12 seconds
"wide": (1344, 768), # 11-13 seconds
"max_safe": (1152, 896), # 10-12 seconds
}

# 20GB VRAM allocation
resolutions_20gb = {
"standard": (1024, 1024), # 10-12 seconds
"high_res": (1536, 1536), # 26-30 seconds
"ultrawide": (1728, 768), # 18-22 seconds
"batch_2": (1024, 1024, 2), # 18-22 seconds
}

# 24GB VRAM allocation (MS-S1 Max maximum)
resolutions_24gb = {
"standard": (1024, 1024), # 10-12 seconds
"high_res": (1536, 1536), # 26-30 seconds
"ultra_res": (2048, 2048), # 55-65 seconds (note)
"batch_4": (1024, 1024, 4), # 35-40 seconds
"batch_2_hires": (1536, 1536, 2), # 50-60 seconds
}
```

## 3.5 Sampler and Scheduler Details

### 3.5.1 SDXL Recommended Sampler

**Sampler performance comparison (MS-S1 Max)**
```
┌──────────────────┬────────┬────────┬──────────┬─────────┐
│ Sampler          │ Speed  │ Quality│ Steps    │ Use Case│
├──────────────────┼────────┼────────┼──────────┼─────────┤
│ Euler a │ ★★★★★ │ ★★★☆☆ │ 20-40 │ Experiment │
│ Euler │ ★★★★★ │ ★★★☆☆ │ 25-50 │ High speed │
│ DPM++ 2M Karras │ ★★★★☆ │ ★★★★★ │ 20-30 │ Recommended │
│ DPM++ SDE Karras │ ★★★☆☆ │ ★★★★★ │ 25-35 │ High quality │
│ DPM++ 2M SDE │ ★★★☆☆ │ ★★★★☆ │ 20-30 │ Balance │
│ DDIM │ ★★★★☆ │ ★★★★☆ │ 30-50 │ Reproducibility │
│ UniPC │ ★★★★☆ │ ★★★☆☆ │ 15-25 │ Super high speed │
└──────────────────┴────────┴────────┴──────────┴─────────┘
```

**Select sampler by application**
```yaml
# Prototyping (emphasis on speed)
sampler: euler_a
steps: 20
time: ~8 seconds @ 1024x1024

# General use (balance)
sampler: dpmpp_2m_karras
steps: 25
time: ~11 seconds @ 1024x1024

# High quality output
sampler: dpmpp_sde_karras
steps: 30
time: ~15 seconds @ 1024x1024

# Animation (reproducibility)
sampler: ddim
steps: 40
time: ~20 seconds @ 1024x1024
```

### 3.5.2 CFG (Classifier Free Guidance) optimization

**Impact of CFG value**
```python
cfg_effects = {
    1.0: {
"adherence": "none",
"creativity": "maximum",
"result": "Ignore prompt, random",
"use": "experimental"
    },
    3.0: {
"adherence": "low",
"creativity": "high",
"result": "free interpretation, diversity",
"use": "artistic expression"
    },
    5.0: {
"adherence": "medium",
"creativity": "medium",
"result": "Good balance",
"use": "exploratory generation"
    },
    7.0: {
"adherence": "high",
"creativity": "medium",
"result": "Exactly reflects prompt",
"use": "Standard Recommended"
    },
    10.0: {
"adherence": "very high",
"creativity": "low",
"result": "Follow detailed instructions",
"use": "Specific request"
    },
    15.0: {
"adherence": "excessive",
"creativity": "almost none",
"result": "color saturation, artifacts",
"use": "Generally not recommended"
    }
}
```

**CFG recommended values ​​by application**
```yaml
Realistic photos:
  cfg: 7.0-8.0
reason: Accurate depiction is important

Illustration:
  cfg: 6.0-7.5
reason: balance with artistic freedom

Anime style:
  cfg: 6.5-8.0
reason: Emphasis on style consistency

Concept art:
  cfg: 5.0-7.0
reason: Emphasis on creativity

Technical drawings:
  cfg: 9.0-11.0
reason: Accuracy is top priority

abstract art:
  cfg: 3.0-6.0
reason: free expression
```

## 3.6 Practical prompt examples

### 3.6.1 Landscape photography

**Basic**
```
Prompt:
A serene mountain lake at dawn, perfectly still water
reflecting snow-capped peaks. Mist rising from the
surface, soft pink and orange sunrise colors. Pine
trees frame the foreground. Professional landscape
photography, high detail.

Negative:
low quality, people, buildings

Settings:
- Resolution: 1344x768 (16:9)
- Steps: 25
- CFG: 7.5
- Sampler: dpmpp_2m_karras
```

**Altitude**
```
Prompt:
Dramatic alpine landscape during the golden hour,
viewed from an elevated vantage point. Jagged peaks
pierce through layers of clouds, creating a sea of
mist below. Warm sunset light bathes the mountain
faces in golden and amber tones, while shadows define
deep valleys. In the foreground, weathered rocks and
hardy alpine flowers add depth. A winding hiking trail
disappears into the distance. Professional outdoor
photography with wide-angle lens, f/11 for maximum
depth of field, graduated ND filter for balanced
exposure, tack sharp details, National Geographic
quality.

Negative:
low quality, blurry, people, man-made structures,
watermark, oversaturated

Settings:
- Resolution: 1216x832 (3:2)
- Steps: 30
- CFG: 7.0
- Sampler: dpmpp_sde_karras
- Refiner: Yes (0.85 denoise start)
```

### 3.6.2 Portrait

**Basic**
```
Prompt:
Portrait of a young woman with wavy brown hair and
blue eyes, natural smile. Soft window light from the
left creates a gentle glow. Wearing a simple white
shirt. Blurred background. Professional headshot,
85mm lens.

Negative:
multiple people, cartoon, low quality

Settings:
- Resolution: 832x1216 (2:3 portrait)
- Steps: 25
- CFG: 7.5
- Sampler: dpmpp_2m_karras
```

**Altitude**
```
Prompt:
Cinematic portrait of an elderly craftsman in his
workshop, captured in dramatic Rembrandt lighting.
Deep wrinkles and weathered hands tell stories of
decades of skilled work. A single window provides
directional light from the left, creating strong
chiaroscuro effect with half the face illuminated.
Warm amber tones from workshop lamps add depth.
Background shows blurred tools and wooden surfaces.
He's wearing a worn leather apron, hands holding a
handcrafted item. Eyes reflect wisdom and pride. Shot
with 50mm f/1.4 lens at f/2 for shallow depth of
field, professional color grading, film grain texture,
award-winning photography.

Negative:
low quality, blurry, multiple people, cartoon,
oversaturated, modern clothing

Settings:
- Resolution: 832x1216 (2:3 portrait)
- Steps: 32
- CFG: 7.5
- Sampler: dpmpp_sde_karras
- Refiner: Yes (0.80 denoise start)
```

### 3.6.3 Architecture

**Basic**
```
Prompt:
Modern minimalist house with large glass windows,
clean white walls, and flat roof. Set in a green
landscape with trees. Blue sky. Architectural
photography, clear details.

Negative:
low quality, blurry, people, cars

Settings:
- Resolution: 1216x832 (3:2)
- Steps: 25
- CFG: 8.0
- Sampler: dpmpp_2m_karras
```

**Altitude**
```
Prompt:
Stunning contemporary architectural masterpiece
featuring cantilevered volumes and floor-to-ceiling
glass curtain walls. The structure seamlessly blends
geometric concrete forms with natural wood cladding.
Set on a dramatic hillside overlooking a valley,
captured during blue hour when interior lights create
warm glows against the deepening sky. A reflective
infinity pool in the foreground mirrors both building
and sky. Carefully designed landscape with native
plants frames the composition. Shot with tilt-shift
lens to correct perspective, long exposure for smooth
water, professional architectural photography,
published in Architectural Digest, hyperdetailed,
perfect symmetry where intended.

Negative:
low quality, distorted perspective, people,
vehicles, watermark, oversaturated

Settings:
- Resolution: 1344x768 (16:9)
- Steps: 35
- CFG: 8.5
- Sampler: dpmpp_sde_karras
- Refiner: Yes (0.85 denoise start)
```

### 3.6.4 Character Design

**Anime style**
```
Prompt:
Anime-style character portrait of a female mage with
long silver hair and purple eyes, wearing an ornate
blue and gold robe with intricate magical patterns.
She's holding a glowing staff with a crystal at the
top. Confident expression. Fantasy setting with soft
magical particles floating around. High quality anime
artwork, detailed shading, vibrant colors.

Negative:
photograph, realistic, 3d, low quality, blurry

Settings:
- Resolution: 832x1216 (2:3)
- Steps: 28
- CFG: 7.0
- Sampler: dpmpp_2m_karras
```

**Fantasy Realism**
```
Prompt:
Highly detailed character concept art of a battle-worn
female warrior in dark fantasy setting. She wears
weathered leather and chainmail armor with visible
scratches and battle damage. Long dark hair tied back
practically, intense green eyes show determination.
Holding a notched longsword, stance ready for combat.
Background shows a misty battlefield at dawn. Painted
in the style of high-end game concept art with
dramatic lighting, rich textures, and careful
attention to material properties. Photorealistic
rendering with painterly touches, trending on ArtStation.

Negative:
low quality, cartoon, anime, oversexualized,
impractical armor, blurry

Settings:
- Resolution: 832x1216 (2:3)
- Steps: 32
- CFG: 7.5
- Sampler: dpmpp_sde_karras
- Refiner: Yes (0.80 denoise start)
```

## 3.7 Iterative prompt improvement

### 3.7.1 Phased approach

**Step 1: Basic prompt**
```
"a cat"
→ generated is too common
```

**Step 2: Add details**
```
"a fluffy orange tabby cat sitting on a windowsill"
→ More specific, but lacks mood
```

**Step 3: Environment and Mood**
```
"a fluffy orange tabby cat sitting on a windowsill,
warm sunlight streaming through the window, cozy
indoor atmosphere"
→ Good, further quality can be specified
```

**Step 4: Technical specifications**
```
"a fluffy orange tabby cat sitting on a windowsill,
warm sunlight streaming through the window creating
soft shadows, cozy indoor atmosphere with blurred
background. Professional pet photography, shallow
depth of field, 85mm lens, high detail, 4k quality"
→ Optimization completed
```

### 3.7.2 A/B testing strategy

**Parameter change test**
```python
# Baseline
baseline = {
"prompt": "basic prompt",
    "negative": "low quality",
    "steps": 25,
    "cfg": 7.5,
    "sampler": "dpmpp_2m_karras",
    "seed": 42
}

# Test 1: CFG change
test_cfg = baseline.copy()
test_cfg["cfg"] = 6.5
# compare results

# Test 2: Change Steps
test_steps = baseline.copy()
test_steps["steps"] = 30
# compare results

# Test 3: Sampler change
test_sampler = baseline.copy()
test_sampler["sampler"] = "dpmpp_sde_karras"
# compare results

# Identify the best combination
```

## 3.8 MS-S1 Max optimization workflow

### 3.8.1 Batch generation strategy

**Exploration phase**
```yaml
Purpose: Generate multiple variations
setting:
  resolution: 1024x1024
  batch_size: 4
  steps: 20
  sampler: euler_a
  cfg: 7.0
time_per_batch: 35-40 seconds
purpose: Quickly explore ideas with 4 different seeds
```

**Refining phase**
```yaml
Purpose: Improve the quality of 1-2 selected photos
setting:
resolution: 1536x1536 or 1024x1024
  batch_size: 1
  steps: 30
  sampler: dpmpp_sde_karras
  cfg: 7.5
  refiner: true
time_per_image: 26-35 seconds (without Refiner) or 40-50 seconds (with Refiner)
```

### 3.8.2 Memory efficient workflow

**128GB memory utilization**
```
Workflow design:
1. Load multiple models simultaneously
   - SDXL Base: 8GB
   - SDXL Refiner: 8GB
   - VAE: 0.5GB
- ControlNet (if required): 2-4GB
- Multiple LoRA: 1-2GB

2. Cache with remaining memory
- Generated image history: 10-20GB
- Workflow buffer: 5-10GB
- System free space: 80GB or more

3. Can be executed simultaneously
- Prepare next prompt during generation
- Upscale in the background
- Multiple workflows in parallel
```

## 3.9 Troubleshooting

### 3.9.1 General issues

**Problem: Generated image is different from expected**
```
Cause 1: Ambiguous prompt
Solution: Add more specific description

Cause 2: Incorrect CFG
Solution: Adjust in the range 7.0-8.0

Cause 3: Insufficient Steps
Solution: increase to 25-30

Cause 4: Bad Seed
Solution: Try multiple batch generation
```

**Problem: Artifact occurs**
```
Cause 1: Incorrect resolution
Solution: Use 1024x1024 based resolution

Cause 2: CFG is too high
Solution: lower to below 7.5

Cause 3: Excessive use of weights
Solution: Limit to 1.4 or below

Cause 4: Too many negative prompts
Solution: Minimize
```

## 3.10 Summary of this chapter

What you learned in this chapter:

**SDXL Architecture**
- Two-stage model of Base + Refiner
- Evolution points from SD 1.5
- Optimal use with MS-S1 Max

**Prompt Engineering**
- The importance of structured prompts
- SDXL natural language understanding
- Proper use of keyword weights
- Minimize negative prompts

**Technical parameters**
- Optimized resolution and aspect ratio
- Sampler and CFG selection
- Settings for MS-S1 Max

**Practical approach**
- Examples of usage-specific prompts
- Iterative improvement method
- Batch generation strategy

In the next chapter, we will learn more about creating workflows in ComfyUI.

---

**Reference resources**
- SDXL paper: https://arxiv.org/abs/2307.01952
- Stable Diffusion Art Guide: https://stable-diffusion-art.com/
- ComfyUI workflow examples: https://github.com/comfyanonymous/ComfyUI_examples

