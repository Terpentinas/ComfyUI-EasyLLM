← [Documentation Home](README.md)

---

# 🚀 Features

### ⚡ Tier 1 — Why People Install

| Feature | Description |
|---------|-------------|
| | **Local GGUF Inference** | Run GGUF models directly inside ComfyUI via `llama-cpp-python` — 30–50+ tok/sec with Q4_K_M |
| | **LLM Image Generation Agent** | Chat with the LLM — it decides when to generate or edit images, writes structured prompts (with negative prompts), and auto-queues your KSampler pipeline. Includes loadable workflows for Flux Klein 4B and other SD models. Works via Trigger Router + Image Capture nodes |
| | **Built-in Model Browser** | Search, filter, and manage local GGUF models with automatic vision detection, family/architecture identification, and mmproj matching |
| | **Vision / Multimodal Support** | Use vision-language models (LLaVA, Qwen-VL, Gemma-3-Vision) with automatic mmproj detection — upload images directly from chat, no wires needed |
| | **Streaming Chat** | Interactive chat popup with real-time token streaming, conversation history, and markdown rendering |
| | **Prompt Enhancement Workflows** | Enhancer mode transforms simple concepts into detailed image prompts — wire directly to CLIP Text Encode → KSampler |
| | **One-Click Installers** | Platform launchers (.bat / .sh / .command) with interactive menus — double-click to install with auto GPU detection, no terminal commands needed |

### 📚 Tier 2 — Why People Keep Using It

| Feature | Description |
|---------|-------------|
| | **System Prompt Manager** | 11 built-in prompt templates plus custom creation, editing, import, export, and sharing — a complete prompt engineering toolkit with drag-to-reorder and search |
| | **Smart History Database** | Persistent chat & enhancer history with atomic writes (no corruption on crash), auto-cleanup by age and size limits, per-node storage. Export to Markdown or Plain Text |
| | **Model Persistence** | Keep models loaded on GPU between generations for instant chat — no reload delays |
| | **Text Utility Nodes** | 7 lightweight text processing nodes — Joiner, Replacer, Extractor, Limiter, Whitespace Cleaner, Duplicate Remover, Text Input |
| | **Socket-Chainable Inputs** | Both `text` and `system_prompt` accept forceInput sockets — chain multiple LLM nodes, feed prompts from other nodes, build dynamic multi-node pipelines |
| | **ComfyUI Manager Ready** | `install.py` is auto-discovered by ComfyUI Manager — GGUF backend installs automatically when adding via Manager (no manual steps) |

### 🧠 Tier 3 — Power-User Features

| Feature | Description |
|---------|-------------|
| | **CLIP Backend (Optional)** | Use Qwen-based text encoders already loaded by Anima, Z-Image, Flux Klein — zero additional downloads |
| | **Auto Chat-Template Detection** | 3-tier pipeline detects the correct chat template from GGUF metadata, architecture, or filename heuristics |
| | **Auto-Tuned Sampling** | Model size detection sets optimal temperature, top-K, top-P, and repetition penalty automatically |
| | **GPU Acceleration** | TF32 matmul, torch.compile, GPU-vectorized sampling — 1.2–1.5× speedup on Ampere+ GPUs |
| | **Memory Management** | Three VRAM modes (unload / keep_loaded / aggressive_free) for any GPU budget |
| | **Upgrade-Safe Configuration** | `config_user.py` persists your custom settings across reinstalls and upgrades — `REATTACH_IMAGES`, `HISTORY_DB_MAX_AGE_DAYS`, `HISTORY_DB_MAX_SIZE_MB`, and more |

---

← [Back to Documentation](README.md)
