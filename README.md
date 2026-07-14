# EasyLLM for ComfyUI

Chat, prompt enhancement, and high-speed GGUF inference — all inside ComfyUI.
**No external services. No Ollama. No LM Studio. No API keys.**

> Already have models for LM Studio? Point EasyLLM at those folders and use the same GGUF files directly inside ComfyUI — no background servers required.
>
> **Note for Ollama users:** Ollama stores models in a proprietary format. Download the original `.gguf` file from Hugging Face instead.

---

## 📸 Screenshots

### Main Node & Chat Popup

![Main Node on Canvas](media/main%20-node.png)

![Chat Popup Interface](media/chat-pop-up.png)

### Model Browser & Prompt Library

![GGUF Model Browser](media/model-browser-pop-up.png)

![System Prompt Library](media/Prompt-Library.png)

### 🎥 Demo Video

<video src="media/simple-chat.mp4" poster="media/chat-pop-up.png" controls width="100%"></video>

> Streaming chat demo — click play to watch.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| ⚡ **Local GGUF Inference** | 30–50+ tok/sec with llama.cpp C++ engine — same as Ollama/LM Studio |
| 🔍 **Built-in Model Browser** | Search, filter, and manage GGUF models with auto vision detection |
| 👁️ **Vision / Multimodal** | Upload images directly from chat — LLaVA, Qwen-VL, Gemma-3-Vision |
| 💬 **Streaming Chat** | Real-time token streaming with markdown rendering and conversation history |
| 🔧 **Prompt Enhancement** | Transform simple concepts into detailed image prompts |
| 📚 **System Prompt Manager** | 11 built-in templates + custom creation, import, export |
| 📝 **Text Utility Nodes** | 7 lightweight text processing nodes — no dependencies |

📖 [See all features →](docs/features.md)

---

## 🚀 Quick Start

### ⚡ GGUF Backend (Recommended — 30–50+ tok/sec)

Install llama-cpp-python once:
```bash
python custom_nodes\ComfyUI-EasyLLM\install.py
```

Minimal workflow:
```
[EasyLLM GGUF] ──> [EasyLLM Text]
     │ model_path: "path/to/model.gguf"
     │ text: "explain how diffusion models work"
```

### 🤖 CLIP Backend (No extra install)

```
[Load CLIP (Anima)] ──> [EasyLLM] ──> [CLIP Text Encode] ──> [KSampler]
```

Uses the Qwen model already loaded inside your CLIP text encoder.

---

## 📦 Installation

1. Clone into `ComfyUI/custom_nodes/`
2. Restart ComfyUI
3. (Optional) Run `python install.py` for GGUF acceleration

📖 [Full installation guide →](docs/installation.md)

---

## 📖 Documentation

| Topic | Description |
|-------|-------------|
| [Features](docs/features.md) | Complete feature list with all 3 tiers |
| [Installation](docs/installation.md) | All GPU variants, launchers, manual install |
| [Node Reference](docs/nodes/) | Full parameter tables for every node |
| [Workflow Examples](docs/workflows.md) | Prompt enhancement, chat, multi-turn generation |
| [Chat UI](docs/chat-ui.md) | Popup chat, canvas widgets, modes, export |
| [System Prompts](docs/system-prompts.md) | Built-in templates, custom creation, import/export |
| [Text Utilities](docs/text-utilities.md) | 7 text processing nodes |
| [Compatible Models](docs/models.md) | GGUF and CLIP model recommendations |
| [Speed Optimization](docs/speed-optimization.md) | Unlock 30–50+ tok/sec |
| [Auto-Tuned Sampling](docs/auto-tuning.md) | How automatic parameter tuning works |
| [Chat Templates](docs/chat-templates.md) | 3-tier auto-detection pipeline |
| [Vision / Multimodal](docs/vision.md) | Image upload, mmproj, vision-language models |
| [Technical Details](docs/technical.md) | Internals, architecture, GPU acceleration |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Configuration](docs/configuration.md) | config_user.py override system |
