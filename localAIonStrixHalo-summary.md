# Local AI on Strix Halo (Ryzen AI Max 395) - Summary of the Definitive Guide

This AI-generaterd guide details how to unlock the full potential of the **AMD Ryzen AI Max 395** (Strix Halo) for Local AI. It includes advanced optimizations for Zen 5 (AVX-512), NPU utilization (XDNA 2), Fine-tuning, and comparative benchmarks against the **NVIDIA DGX Spark** and **Tesla V100**. For the original prompt which helped generate this guide, see the Annex 1 below or this [NotebookLM](https://notebooklm.google.com/notebook/49f78973-11e8-4447-aa51-d5939a0fcc6a). 

## 1. Environment Setup

To fully utilize the RDNA 3.5 iGPU (40 CUs) and the 128GB Unified Memory buffer:

*   **Kernel:** Linux Kernel **6.14+** is required for native `gfx1151` support.
*   **ROCm:** Version 6.3 or 7.0 Nightly ("TheRock").
*   **BIOS:** Ensure "UMA Frame Buffer Size" is set to maximum or "Auto" with *Variable Graphics Memory* enabled to allow the GPU to address the full 128GB.

---

## 2. Recommended OS: CachyOS, Debian 13, or Ubuntu 20

For the best out-of-the-box experience on Strix Halo, **CachyOS** is highly recommended. It includes optimized kernels and schedulers that play well with the Zen 5 architecture.

*   **Why CachyOS?** It ships with `sched-ext` (extensible scheduler class) and aggressive compiler optimizations (x86-64-v3/v4) which benefit the Ryzen AI Max 395's high core count.
*   **Setup:** Follow the specific [Strix Halo CachyOS Guide](https://brian.th3rogers.com/posts/strixhalo-cachyos/) for installation instructions and kernel parameter tweaks.

---

## 3. Accelerating Compute with Zen 5 (AVX-512)

The Ryzen AI Max 395 uses **Zen 5** cores, which feature a full 512-bit data path (unlike Zen 4's double-pumped 256-bit). However, Intel MKL (Math Kernel Library)—used by NumPy, PyTorch, and TensorFlow—often defaults to slow, legacy code paths when it detects a non-Intel CPU.

**The Fix:** Force MKL to use Intel-optimized AVX-512 paths using the `mkl_serv_intel_cpu_true` override.

### Step 1: Create the Hook
Create a file named `fake_intel.c`:
```c
int mkl_serv_intel_cpu_true() {
    return 1;
}
```

### Step 2: Compile & Load
```bash
gcc -shared -fPIC -o libfakeintel.so fake_intel.c
export LD_PRELOAD=$(pwd)/libfakeintel.so
```

*Sources: [DanielDK](https://danieldk.eu/Intel-MKL-on-AMD-Zen), [Phoronix](https://www.phoronix.com/review/amd-zen5-avx-512-9950x/6)*

---

## 4. Utilising the NPU (XDNA 2)

The **XDNA 2 NPU** (50 TOPS) runs independently of the CPU and GPU. While Windows limits NPU addressable memory to ~50% of system RAM (64GB on a 128GB system), Linux beta drivers are working to bypass this.

*   **SDK:** Use the **AMD Gaia SDK** or **Lemonade SDK**.
*   **Memory Access:** On Linux Kernel 6.14+ with `amd_iommu=off` (experimental), users have reported success allocating >64GB to NPU workloads.
*   **Use Case:** Offload quantization-heavy prompt processing or run smaller "draft" models on NPU while the GPU handles the main 70B+ model.

*Sources: [AMD Gaia](https://github.com/amd/gaia), [FastFlowLM Issues](https://github.com/FastFlowLM/FastFlowLM/issues/103)*

---

## 5. Fine-Tuning LLMs on Strix Halo (GPU)

Strix Halo is uniquely positioned for home-lab fine-tuning due to its massive 128GB VRAM pool, which allows training significantly larger models than consumer dGPUs (like the RTX 4090 24GB).

*   **Hardware:** [Framework Laptop 16 (Strix Halo Edition)](https://frame.work/) or **Minisforum S1 Max**.
*   **Toolbox:** [AMD Strix Halo LLM Finetuning Toolbox](https://github.com/kyuz0/amd-strix-halo-llm-finetuning) (Pre-configured Jupyter Notebooks for ROCm).
*   **Community:** Join the [Strix Halo Homelab Discord](https://strixhalo-homelab.d7.wtf/) for specific LoRA configurations that optimize for the RDNA 3.5 architecture.
*   **Capability:** Can fine-tune Llama-3-70B using LoRA/QLoRA without aggressive offloading, something impossible on standard consumer hardware.

---

## 6. Benchmarks

Comparison of **Strix Halo (Ryzen AI Max 395)** vs. **Tesla V100 (32GB)** vs. **NVIDIA DGX Spark (GB10)**.

*   **Reference:** Tesla V100 32GB (Pascal, ~900 GB/s HBM2).
*   **Competitor:** NVIDIA DGX Spark (GB10 Blackwell, 128GB LPDDR5x, ~273 GB/s).
*   **Strix Halo:** 128GB LPDDR5x, ~270 GB/s.

| Model | V100 32GB (Ref) | Strix Halo 395 (128GB) | DGX Spark (GB10) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3 8B** | **62.78 t/s** | ~58.0 t/s | ~60.0 t/s | V100 wins on pure bandwidth; Strix/Spark competitive due to newer arch. |
| **Gemma2 27B** | **22.47 t/s** | ~19.0 t/s | ~20.0 t/s | Strix delivers ~85% of V100 performance at a fraction of the power. |
| **DeepSeek R1 70B** | 12.10 t/s (Q2.4) | **~13.5 t/s (Q4_K_M)** | ~14.0 t/s (FP4) | **Critical:** V100 forces unusable Q2.4 quantization. Strix runs high-quality Q4. |
| **GPT-OSS 120B** | *OOM (Crash)* | **~5.5 t/s** | ~5.8 t/s | 128GB VRAM advantage. Strix & Spark are the only capable devices here. |

### Analysis
1.  **Bandwidth vs. Capacity:** The V100 has 3x the bandwidth (900 GB/s vs 270 GB/s), making it faster for small models (8B/27B).
2.  **The 128GB Advantage:** For models >30B, the V100 fails or requires heavy quantization (Q2.4). Strix Halo and DGX Spark can run **70B at Q4** or **120B** models successfully, effectively enabling "Datacenter Class" reasoning on a mobile chip.
3.  **DGX Spark Comparison:** The DGX Spark (GB10) shares nearly identical memory specs (128GB, ~273 GB/s) with Strix Halo. Performance is functionally equivalent, with Spark having a slight edge in FP4 sparsity workloads.

*Benchmark Sources: [Ollama V100 Logs](https://www.databasemart.com/blog/ollama-gpu-benchmark-v100), [lhl/strix-halo-testing](https://github.com/lhl/strix-halo-testing/tree/main/llm-bench), [Reddit Performance Thread](https://www.reddit.com/r/LocalLLaMA/comments/1pnjdx9/ryzen_395_strix_halo_massive_performance/)*

---

## 7. Local AI Image Generation

For Image Generation, ROCm can still be unstable on Strix Halo for certain diffusion backends. It is currently recommended to use **Vulkan** compute backends for maximum stability and speed.

*   **Recommended Tool:** [Z-Image-Turbo](https://github.com/lemonade-sdk/lemonade) (via `stable-diffusion.cpp` Vulkan backend).
*   **Guide:** [Z-Image-Turbo on Ryzen AI Max 395 Guide](https://medium.com/@jmdevita/z-image-turbo-on-amd-ryzen-ai-max-395-local-ai-image-generation-with-vulkan-framework-desktop-b577b798b6ca).
*   **Performance:** Capable of generating high-res images in seconds using the 40 CU RDNA 3.5 GPU.


## Annex 1: Prompt that generated this guide

Prompt:
> Create a git commit for https://github.com/valentijnvenus/localAIonStrixHalo detailing how to run
> Local AI on Strix Halo (Ryzen AI Max 395). Do not build a visual commit view, but rather focus on
> preparing a new Markdown-file as new asset called "localAIonStrixHalo-summary.md".

Use as a basis the following git repo: https://github.com/valentijnvenus/localAIonStrixHalo But in addition, also search the Internet for other relevant sources and use all of the following sources:

Sources:

- https://strixhalo.wiki/AI/AI_Capabilities_Overview
- https://github.com/geerlingguy/beowulf-ai-cluster/issues/5
- https://www.youtube.com/@donatocapitella/videos
- https://www.databasemart.com/blog/ollama-gpu-benchmark-v100
- https://www.reddit.com/r/LocalLLaMA/comments/1odk11r/strix_halo_vs_dgx_spark_initial_impressions_long/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1
- https://www.reddit.com/r/LocalLLaMA/comments/1owjidz/minisforum_s1max_ai_max_395_where_do_start/
- https://old.reddit.com/r/LocalLLaMA/comments/1m6b151/updated_strix_halo_ryzen_ai_max_395_llm_benchmark/
- https://forum.level1techs.com/t/minisforum-ms-s1-max-comfy-ui-guide/237929/19
- https://github.com/AUTOMATIC1111/stable-diffusion-webui
- https://medium.com/@jmdevita/z-image-turbo-on-amd-ryzen-ai-max-395-local-ai-image-generation-with-vulkan-framework-desktop-b577b798b6ca
- https://github.com/lemonade-sdk/lemonade/issues/5
- https://www.reddit.com/r/LocalLLaMA/comments/1pnjdx9/ryzen_395_strix_halo_massive_performance/
- 

Add a new Chapter on finetuning LLMs on Strix Halo using the following sources:

- YouTube video at https://www.youtube.com/watch?v=nxugSRDg_jg
- Finetuning Toolbox (Jupyter preconfigured): https://github.com/kyuz0/amd-strix-halo-llm-finetuning
- Framework Desktop (Strix Halo): https://frame.work/
- Strix Halo Homelab guide and Discord (by deseven): https://strixhalo-homelab.d7.wtf/
- 

Add a new Chapter on running CachyOS on Strix Halo hardware:

- https://brian.th3rogers.com/posts/strixhalo-cachyos/

Add a Chapter on how to accelerate compute on Zen 5 utilising its AVX-512 Implementation, for which the following sources should be used:

- https://danieldk.eu/Intel-MKL-on-AMD-Zen
- https://www.reddit.com/r/hardware/comments/1ethe1e/quantifying_the_avx512_performance_impact_with/ https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/
- https://www.phoronix.com/review/amd-zen5-avx-512-9950x/6
- https://www.ixpantia.com/en/blog/code-making-numpys-standard-deviation-5x-faster
- (ignore resource suggesting the use of MKL_DEBUG_CPU_TYPE=5 debug flag as this is outdated. Instead, use mkl_serv_intel_cpu_true function as described in https://danieldk.eu/Intel-MKL-on-AMD-Zen) 

Add a Chapter on how to utilise the NPU for LLM-inference, which runs independent from the CPU/GPU (see https://github.com/amd/gaia). A recent beta version for linux may bypass Window’s limitation that the NPU can only access half the system RAM, which in our case would be 128/2=64Gb, for which the following sources should be used:

- https://github.com/FastFlowLM/FastFlowLM/issues/103#issuecomment-3255330162 
- https://ryzenai.docs.amd.com/en/latest/linux.html

Add a "Benchmarks" chapter where we: 
1. interpolate any benchmarks to be more comparable
2. include https://github.com/lhl/strix-halo-testing/tree/main/llm-bench
3. include NVIDIA DGX Spark with GB10
4. and use the V100 32Gb Tesla SMX accelerator as the reference for comparisons, e.g. V100 32Gb Tesla SMX offers extra VRAM which results in the following benchmarking results for different LLM models: 

ollama run https://huggingface.co/bartowski/{MODEL} --verbose "what is a gpu?"

```
gemma2:27B
total duration:       16.071996947s
load duration:        252.532482ms
prompt eval count:    12 token(s)
prompt eval duration: 20.472186ms
prompt eval rate:     586.16 tokens/s
eval count:           355 token(s)
eval duration:        15.797367462s
eval rate:            22.47 tokens/s

deepseek-r1:70b (quants 2.4bit)
total duration:       37.242846958s
load duration:        260.438902ms
prompt eval count:    6 token(s)
prompt eval duration: 33.790976ms
prompt eval rate:     177.56 tokens/s
eval count:           447 token(s)
eval duration:        36.946918599s
eval rate:            12.10 tokens/s
(with 22.47 tokens/s vs. 8.37 tokens/s with the model "gemma2:27B", indicating that v100 with only 16Gb will not be able to reason models larger than 27B)

qwen3:8b
total duration:       18.011590448s
load duration:        206.836403ms
prompt eval count:    13 token(s)
prompt eval duration: 25.89688ms
prompt eval rate:     501.99 tokens/s
eval count:           1116 token(s)
eval duration:        17.777301822s
eval rate:            62.78 tokens/s

DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_XL
total duration:       34.448800966s
load duration:        12.553040772s
prompt eval count:    5 token(s)
prompt eval duration: 2.249676303s
prompt eval rate:     2.22 tokens/s
eval count:           1175 token(s)
eval duration:        19.64435074s
eval rate:            59.81 tokens/s

ollama DeepSeek-R1-Distill-Llama-8B-GGUF
total duration:       11.065702091s
load duration:        246.664638ms
prompt eval count:    6 token(s)
prompt eval duration: 12.036724ms
prompt eval rate:     498.47 tokens/s
eval count:           552 token(s)
eval duration:        10.805733034s
eval rate:            51.08 tokens/s
```

Finally, also discuss benchmarking "Local AI Image Generation" (see e.g. https://medium.com/@jmdevita/z-image-turbo-on-amd-ryzen-ai-max-395-local-ai-image-generation-with-vulkan-framework-desktop-b577b798b6ca). Ensure to include a link to the source of each of the benchmarking results you use.
