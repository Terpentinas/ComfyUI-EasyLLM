# EasyLLM Documentation

Welcome to the EasyLLM for ComfyUI documentation. This index organizes all pages by category — use the links below to find what you need.

---

## 🚀 Features & Setup

| Page | Description |
|------|-------------|
| [Features](features.md) | Complete feature list with all 3 tiers — why people install, keep using, and power-user features |
| [Installation](installation.md) | Full install guide: basic setup, C++ acceleration, all GPU variants, launchers, manual install |
| [Compatible Models](models.md) | GGUF and CLIP model recommendations, quantization guide, model families |
| [Speed Optimization](speed-optimization.md) | Unlock 30–50+ tok/sec — VRAM modes, CPU offload, C++ engine guide |

---

## 🔌 Node Reference

| Page | Description |
|------|-------------|
| [EasyLLM GGUF](nodes/easyllm-gguf.md) | High-speed GGUF chat node: all 6 sections of parameters, socket inputs, outputs |
| [EasyLLM CLIP](nodes/easyllm-clip.md) | Interactive conversation (CLIP backend): all 5 sections of parameters, outputs |
| [Trigger Router](nodes/trigger-router.md) | Decompose `trigger_prompt` JSON into individual sockets for multi-turn workflows |
| [Image Capture](nodes/image-capture.md) | Persist generated images to chat history database, keyed by session UUID |
| [EasyLLM Text](nodes/easyllm-text.md) | Display text output on canvas with an Open Chat button |

---

## 📖 User Guides

| Page | Description |
|------|-------------|
| [System Prompts](system-prompts.md) | System Prompt Manager: 11 built-in templates, custom creation, import/export |
| [Workflow Examples](workflows.md) | Prompt enhancement, chat, GGUF chat, text processing pipeline, multi-turn generation |
| [Chat UI](chat-ui.md) | Popup chat, canvas widgets, chat modes, markdown rendering, export |
| [Text Utilities](text-utilities.md) | 7 lightweight text processing nodes — no dependencies |
| [Vision / Multimodal](vision.md) | Image upload, mmproj auto-detection, vision-language models |

---

## 🧠 Technical Reference

| Page | Description |
|------|-------------|
| [Auto-Tuned Sampling](auto-tuning.md) | How automatic parameter tuning works based on model size |
| [Chat Templates](chat-templates.md) | 3-tier auto-detection pipeline for GGUF models |
| [Technical Details](technical.md) | Internals: generation chain, GGUF backend, GPU acceleration, VRAM, architecture tree, limitations |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions: garbled output, slow generation, OOM, model browser |
| [Configuration](configuration.md) | `config_user.py` override system for persistent custom settings |

---

← [Back to Main README](../README.md)
