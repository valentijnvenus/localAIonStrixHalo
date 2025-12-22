# Using AI locally on Strix Halo hardware

## 📚 Complete Guide Series

This is a comprehensive guidebook for fully utilizing the local AI environment on the Minisforum MS-S1 Max equipped with the AMD Ryzen AI Max+ 395 (128GB memory). Comprised of 5 parts and 45 chapters, , starting with LMStudio [Complete Edition], this is the most comprehensive local AI guide written for using AI locally on Strix Halo hardware (and later hardware revisions).

---

## 🎯 Features of this book

### ✅ Fully optimized for AMD Ryzen AI Max+ 395

- Strategic use of 128GB of unified, large memory
- Optimal Settings for RDNA 3.5 GPU (Radeon 8060S)
- Complete guide to building the ROCm 6.2 environment
- Four performance modes (Performance/Balance/Quiet/Rack)

  <img width="1024" height="559" alt="unified-mem" src="https://github.com/user-attachments/assets/a79848d0-1f92-4070-81c8-c54e74a5c813" />

### ✅ Practical and comprehensive

- 45 chapters, approximately 1,500 pages
- 200+ code examples and scripts
- Specific optimal settings for each application
- Complete troubleshooting

### ✅ For beginners to advanced players

- Step-by-step learning from basics to applications
- Immediately actionable examples
- How to deploy to a production environment
- Introducing community resources

---

## 📖 5 parts in total

### [🔷Part 1: Complete Guide to LM Studio](part1_lmstudio/) (9 Chapters)

**Building an intuitive LLM execution environment using LM Studio**

#### Chapter 1: [Introduction - LM Studio and the World of Local AI](part1_lmstudio/chapter01_introduction.md)

- The benefits of local AI
- Features of LM Studio
- MS-S1 Max performance explanation

#### Chapter 2: [Hardware Specifications and System Requirements](part1_lmstudio/chapter02_hardware_specs.md)

- AMD Ryzen AI Max+ 395 Detailed Specifications
- Memory Bandwidth and Performance
- Memory requirements by model

#### Chapter 3: [Installing and Initializing LM Studio](part1_lmstudio/chapter03_installation.md)

- Windows/Linux environment construction
- ROCm setup
- Checking GPU recognition

#### Chapter 4: [The Complete Guide to AMD GPU Configuration](part1_lmstudio/chapter04_amd_gpu_settings.md)

- GPU Offload Optimization
- Flash Attention Settings
- Thermal Management and Thermal Throttling

#### Chapter 5: [Downloading and Managing Models](part1_lmstudio/chapter05_model_management.md)

- Recommended Model Catalog
- Quantization Level Selection
- Model Consolidation Strategy

#### Chapter 6: [A complete explanation of inference settings](part1_lmstudio/chapter06_inference_settings.md)

- Temperature, Top P, Top K
- Repeat Penalty
- Optimal settings for each application

#### Chapter 7: [Optimization Settings for MS-S1 Max](part1_lmstudio/chapter07_optimization.md)

- 128GB memory utilization strategy
- Recommended settings for each performance mode
- Workflow-specific configuration

#### Chapter 8: [Practical Use](part1_lmstudio/chapter08_practical_usage.md)

- Utilizing the chat interface
- API Server Mode
- VS Code Integration

#### Chapter 9: [Advanced Features and Customization](part1_lmstudio/chapter09_advanced_features.md)

- RAG (Search Extension Generation) implementation
- Prompt Engineering
- Security and Privacy

---

### [🔶Part 2: The Complete Guide to Ollama](part2_ollama/) (9 Chapters)

**Building a flexible CLI/API-based LLM execution environment**

#### Chapter 1: [Introduction - What is Ollama?](part2_ollama/chapter01_introduction.md)

- Ollama's Features and Philosophy
- Differences from LM Studio
- MS-S1 Max usage scenario

#### Chapter 2: [Installation and Setup](part2_ollama/chapter02_installation.md)

- Windows/Linux installation
- ROCm environment settings
- Initial settings and operation check

#### Chapter 3: [Basic Usage](part2_ollama/chapter03_basic_usage.md)

- Complete CLI Command Guide
- Running and Managing Models
- Prompt Template

#### Chapter 4: [Customizing the Modelfile](part2_ollama/chapter04_modelfile.md)

- Modelfile Syntax
- Custom Model Creation
- Parameter adjustment

#### Chapter 5: [Optimization for MS-S1 Max](part2_ollama/chapter05_optimization.md)

- Memory management optimization
- Leveraging parallel execution
- Performance Tuning

#### Chapter 6: [API Integration and Development](part2_ollama/chapter06_api_development.md)

- REST API Utilization
- Python/Node.js integration
- Practical Application Development

#### Chapter 7: [Creating and Sharing Models](part2_ollama/chapter07_model_creation.md)

- Finetune Model Integration
- Model Export/Import
- Private registry construction

#### Chapter 8: [Practical Use Cases](part2_ollama/chapter08_practical_usage.md)

- CLI Automation and Scripting
- System Integration Example
- Multi-model environment construction

#### Chapter 9: [Advanced Techniques](part2_ollama/chapter09_advanced_techniques.md)

- Distributed Execution
- Custom Backend
- troubleshooting

---

### [🔷Part 3: Complete Guide to Text-Generating Web UI](part3_textgen_webui/) (9 Chapters)

**Advanced LLM execution environment using oobabooga's text-generation-webui**

#### Chapter 1: [Introduction - What is Text Generation WebUI?](part3_textgen_webui/chapter01_introduction.md)

- WebUI Features and Functions
- Understanding the ecosystem
- Advantages of the MS-S1 Max

#### Chapter 2: [Installation and Setup](part3_textgen_webui/chapter02_installation.md)

- Environment setup (Windows/Linux)
- ROCm optimization
- Dependency resolution

#### Chapter 3: [ExLlamaV2 and Loader Configuration](part3_textgen_webui/chapter03_loaders.md)

- ExLlamaV2 optimization
- Comparison of various loaders
- Memory-efficient loading

#### Chapter 4: [Interface and Modes](part3_textgen_webui/chapter04_interfaces.md)

- Chat, Default, Notebook modes
- Creating a Custom UI
- API Integration

#### Chapter 5: [Parameters and Generation Settings](part3_textgen_webui/chapter05_parameters.md)

- Detailed parameter explanation
- Preset Creation
- Optimal settings for each application

#### Chapter 6: [Characters and Personas](part3_textgen_webui/chapter06_characters.md)

- Character Definition
- Persona Customization
- Roleplay Settings

#### Chapter 7: [Extensions and Plugins](part3_textgen_webui/chapter07_extensions.md)

- Major Enhancements
- Creating a Custom Extension
- API Extensions

#### Chapter 8: [Practical Use](part3_textgen_webui/chapter08_practical_usage.md)

- Complex Dialogue Systems
- Fine Tuning
- Dataset creation

#### Chapter 9: [Advanced Techniques and Troubleshooting](part3_textgen_webui/chapter09_advanced_techniques.md)

- Performance Optimization
- Memory Management
- Common problems and solutions

---

### [🔶Part 4: The Complete Guide to ComfyUI and Stable Diffusion](part4_comfyui/) (9 Chapters)

**Building and optimizing a local image generation environment**

#### Chapter 1: [Introduction - ComfyUI and Stable Diffusion](part4_comfyui/chapter01_introduction.md)

- Features of ComfyUI
- Stable Diffusion Basics
- Image generation with MS-S1 Max

#### Chapter 2: [Installation and Setup](part4_comfyui/chapter02_installation.md)

- Install ComfyUI
- AMD GPU Settings (ROCm)
- Download the model

#### Chapter 3: [Basic Workflow](part4_comfyui/chapter03_basic_workflow.md)

- Understanding Nodes
- Simple workflow creation
- Prompt Engineering

#### Chapter 4: [SDXL Optimization](part4_comfyui/chapter04_sdxl.md)

- Running the SDXL model
- Using Refiner
- High-resolution generation

#### Chapter 5: [ControlNet and Pause Control](part4_comfyui/chapter05_controlnet.md)

- ControlNet introduced
- Various control methods
- Practical examples

#### Chapter 6: [LoRA and Custom Models](part4_comfyui/chapter06_lora.md)

- Using LoRA
- Custom Model Integration
- Style Control

#### Chapter 7: [Optimization for MS-S1 Max](part4_comfyui/chapter07_optimization.md)

- Memory Management
- Batch generation optimization
- AMD GPU Optimal Settings

#### Chapter 8: [Practical Workflow](part4_comfyui/chapter08_practical_workflows.md)

- Complex Workflow Example
- Animation Generation
- Batch Processing

#### Chapter 9: [Advanced Techniques](part4_comfyui/chapter09_advanced_techniques.md)

- Custom Node Creation
- API Integration
- troubleshooting

---

### [Part 5: Local AI Application Development](part5_app_development/) (9 chapters)

**Practical development and operation of local AI applications**

#### Chapter 1: [Introduction - The world of local AI development](part5_app_development/chapter01_introduction.md)

- Application Architecture
- Technology Stack Selection
- MS-S1 Max Utilization Strategy

#### Chapter 2: [Building a development environment](part5_app_development/chapter02_dev_environment.md)

- Python environment setup
- Framework Selection
- Integrated Development Environment

#### Chapter 3: [Building a RAG System](part5_app_development/chapter03_rag_system.md)

- RAG Architecture
- Vector Database
- Implementation example

#### Chapter 4: [Chatbot Development](part5_app_development/chapter04_chatbot.md)

- Chatbot Architecture
- Conversation Management
- UI Design

#### Chapter 5: [API Design and Integration](part5_app_development/chapter05_api_integration.md)

- RESTful API design
- Integration of multiple AI backends
- Authentication and Security

#### Chapter 6: [Multimodal Applications](part5_app_development/chapter06_multimodal.md)

- Text + Image Processing
- Voice Recognition Integration
- Integrated Applications

#### Chapter 7: [Performance and Scalability](part5_app_development/chapter07_performance.md)

- Caching Strategies
- load balancing
- MS-S1 Max optimization

#### Chapter 8: [Deployment and Operations](part5_app_development/chapter08_deployment.md)

- Containerization (Docker)
- monitoring
- Log Management

#### Chapter 9: [Hands-on Projects](part5_app_development/chapter09_real_projects.md)

- Complete implementation example
- Best Practices
- Future outlook

---

## 🚀 Quick Start

### Suggested Reading Order

**For beginners:**

```
Part 1 → Part 2 → Basic part of Part 3
```

**Intermediate:**

```
Part 1 (review) → Part 2 and Part 3 in parallel → Part 4
```

**For advanced users:**

```
Start with the part you are interested in → Integrate in Part 5
```

### System Requirements

**Recommended environment:**

- **CPU** : AMD Ryzen AI Max+ 395 (16 cores/32 threads)
- **GPU** : Radeon 8060S (RDNA 3.5, integrated)
- **Memory** : 128GB LPDDR5X-8000
- **Storage** : 2TB+ NVMe SSD
- **System** : Minisforum MS-S1 Max
- **OS** : Windows 11 Pro or Ubuntu 24.04 LTS
- **ROCm** (Linux): 6.2 or later

**Minimum requirements:**

- CPU: AVX2 compatible processor
- Memory: 32GB or more
- Storage: 500GB or more
- GPU: AMD Radeon RX 5700 or higher (recommended)

---

## 📊 Comparison of features of each part

item | Part 1<br> LM Studio | Part 2<br> Ollama | Part 3<br> WebUI | Part 4<br> ComfyUI | Part 5 Development
--- | --- | --- | --- | --- | ---
**Difficulty** | ⭐ Beginner | ⭐⭐ Intermediate | ⭐⭐⭐ Intermediate to advanced | ⭐⭐ Intermediate | ⭐⭐⭐⭐ Advanced
**GUI** | ✅ Intuitive | ❌ CLI | ✅ Web | ✅ Node-based | 📱 Proprietary Development
**Customizability** | Medium | high | Very high | Very high | the best
**Purpose** | General Chat | CLI automation | Advanced Interactions | Image generation | App Development
**API** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | 🔧 Create
**Recommended Model** | LLM | LLM | LLM | Stable Diffusion | all

---

## 💡 Highlights of each section

### Part 1: LM Studio

```yaml
Ideal for:
- Introduction to LLM
- Daily chat
- Easy API integration

Recommended Model:
- Qwen2.5 7B (daily use)
- Llama 3.1 70B (high quality)
- DeepSeek-Coder (coding)

Expected speed (MS-S1 Max):
- 7B: 35-45 tokens/s
- 32B: 8-12 tokens/s
- 70B: 3-5 tokens/s
```

### Part 2: Ollama

```yaml
Ideal for:
- CLI automation
- Script Integration
- Multiple model management

Recommended configuration:
- Parallel execution environment
- Custom Modelfile
- Private Registry

Features:
- Lightweight and fast startup
- Simple API
- Superior model management
```

### Part 3: Text Generation WebUI

```yaml
Ideal for:
- Advanced parameter adjustment
- Character dialogue
- Fine Tuning

Strengths:
- The most extensive settings
- Extensions ecosystem
- Community Support

Recommended:
- Experimental research
- Deep customization
- Complex dialogue system
```

### Part 4: ComfyUI + Stable Diffusion

```yaml
Ideal for:
- Local image generation (see https://medium.com/@jmdevita/z-image-turbo-on-amd-ryzen-ai-max-395-local-ai-image-generation-with-vulkan-framework-desktop-b577b798b6ca)
- Workflow automation
- Creative work

MS-S1 Max performance:
- SDXL: approx. 8-12 seconds/image
- SD 1.5: approx. 3-5 seconds/image
- Batch generation: efficient

Examples of use:
- Illustration generation
- Concept Art
- Design work
```

### Part 5: Application Development

```yaml
What you will learn:
- RAG system construction
- Chatbot development
- Multimodal apps
- Production Operations

Technology stack:
- Python, FastAPI
- LangChain, LlamaIndex
- Docker, Kubernetes
- Monitoring tools

Deliverables:
- Practical Applications
- Deployable Systems
- Maintenance and operation know-how
```

---

## 📈 Recommended Learning Path

### Path 1: Specialized in chat and text generation

```
Part 1 → Part 2 → Part 3 → Part 5 (RAG/Chatbot)
```

### Path 2: Creative Specialization

```
Part 1 (Basics) → Part 4 → Part 5 (Multimodal)
```

### Path 3: Full-Stack Developer

```
Part 1 → Part 2 → Part 3 → Part 4 → Part 5 (Complete Conquest)
```

### Path 4: Researcher/Experimenter

```
Part 3 (Advanced Configuration) → Part 2 (CLI Automation) → Part 5 (Custom Development)
```

---

## 🛠 Recommended Tool Set (MS-S1 Max)

### Daily use configuration (memory usage: approx. 30GB)

```yaml
LM Studio:
- Qwen2.5 7B Q4_K_M (4.8GB)
- Llama 3.2 3B Q4_K_M (2GB)

Ollama:
- Mistral 7B (background service)
- Gemma 2 9B (for experimentation)

ComfyUI:
- SDXL Base (6.9GB)
- Several lightweight LoRAs

Remaining memory: 98GB (usable for browser, IDE, etc.)
```

### Professional configuration (memory usage: approx. 80GB)

```yaml
LM Studio:
- Qwen2.5 32B Q5_K_M (24GB)
- Llama 3.1 70B Q4_K_M (42GB)

Ollama:
- Multiple special-purpose models

Text Generation WebUI:
- Experimental Model and LoRA

ComfyUI:
- SDXL + Refiner
- Multiple ControlNet

Remaining memory: 48GB
```

### Maximum utilization configuration (memory usage: approx. 110GB)

```yaml
Run all tools simultaneously:
- 70B LLM loaded
- Multiple mid-size models
- ComfyUI is now operational
- Full development environment
- Multiple Docker containers

Remaining memory: 18GB (system reserved)
```

---

## 📚 Supplementary Materials

### Official Resources

**LM Studio:**

- Official website: https://lmstudio.ai/
- Discord: https://discord.gg/lmstudio

**Ollama:**

- Official website: https://ollama.ai/
- GitHub: https://github.com/ollama/ollama

**Text Generation WebUI:**

- GitHub: https://github.com/oobabooga/text-generation-webui

**ComfyUI:**

- GitHub: https://github.com/comfyanonymous/ComfyUI

**AMD ROCm:**

- Official documentation: https://rocm.docs.amd.com/

### community

- Reddit r/LocalLLaMA
- Reddit r/StableDiffusion
- Hugging Face Community
- GitHub Discussions

---

## 🎓 Target Audience

### Recommended for:

✅People **who value privacy**

- I don't want to send data outside
- You need to handle trade secrets
- Seeking complete control

✅Those **who want to keep costs down**

- Monthly cloud AI fees are a burden
- I want an unlimited environment
- Zero running costs after initial investment

✅ **People who enjoy technical exploration**

- I want to gain a deeper understanding of how AI works.
- I want to enjoy customization
- I want to try the latest technology

✅People **who engage in creative activities**

- Utilizing AI in creative activities
- Build your own workflow
- Commercial use also in sight

✅Developers and engineers

- AI Application Development
- Integrated system construction
- Production environment operation

---

## ⚖️ License and Notices

### About this book

This document is for informational purposes only. Actual performance may vary depending on the environment, model, and configuration.

### License for the model you are using

Each AI model has its own license, please be sure to check before using it for commercial purposes.

**Licenses for major models:**

- **Llama 3** : Llama 3 Community License (commercial use available)
- **Qwen** : Apache 2.0 (commercial use available)
- **Mistral** : Apache 2.0 (commercial use available)
- **Stable Diffusion** : CreativeML Open RAIL-M (conditional commercial use allowed)

---

## 📊 Statistics

```
Total number of pages: Approximately 1,500 pages
Total number of characters: Approximately 750,000
Total number of chapters: 45 (9 chapters per part x 5 parts)
Code example: 200+
Setting table: 100+
Screenshot: Coming soon
Chart: In preparation
```

---

## 🔄 Update history

- **v1.0.0** (2025-10-30): First release of 5 parts and 45 chapters
    - Part 1: Complete Guide to LM Studio
    - Part 2: The Complete Guide to Ollama
    - Part 3: The Complete Guide to Text-Generating WebUI
    - Part 4: A Complete Guide to ComfyUI and Stable Diffusion
    - Part 5: Local AI application development

---

## 👥 Author/Producer

- **Written by** : Claude (Anthropic)
- **Technical cooperation** : Claude Code
- **Moderated by** : Community Feedback
- **Target hardware** : AMD Ryzen AI Max+ 395 / Minisforum MS-S1 Max

---

## 🙏 Acknowledgments

We would like to thank the following projects and communities for their contributions to this book:

- LM Studio Development Team
- Ollama Development Team
- oobabooga (Text Generation WebUI)
- ComfyUI Development Team
- AMD ROCm Team
- Hugging Face Community
- r/LocalLLaMA Community

---

## 📞 Feedback/Questions

We welcome your feedback, questions, and suggestions regarding this book.

---

**© 2025 - All Rights Reserved**

**Use this book to explore the infinite possibilities of local AI!**

🚀 **Let's Build Amazing AI Applications Locally!** 🚀
