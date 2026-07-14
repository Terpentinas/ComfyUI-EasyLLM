← [Documentation Home](README.md)

---

# ⚡ Speed Optimization Guide

Getting 3-5 tokens/sec? Here's how to unlock 30-50+ tokens/sec.

### Why It's Slow (The CPU Offload Trap)

Looking at your terminal log:
```
CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu
```

ComfyUI aggressively offloads the CLIP text encoder to **CPU** to save VRAM for image models. This means every generation requires:
1. Move model from CPU → GPU (slow PCIe transfer)
2. Generate tokens
3. Move model back from GPU → CPU
4. Clear CUDA cache

This constant swapping is what limits you to **~3.68 tokens/sec**.

### Speed Comparison

| Backend | Model | Quant | Tokens/sec | VRAM Use | How To |
|---------|-------|-------|:----------:|:--------:|--------|
| **PyTorch (unload)** | 8B | Q8_0 | **~3-4** | Swaps | Default behavior |
| **PyTorch (keep_loaded)** | 8B | Q8_0 | **~8-12** | ~8.5GB | Set `vram_mode=keep_loaded` |
| **PyTorch (aggressive_free)** | 8B | Q8_0 | **~3-4** | Minimal | Set `vram_mode=aggressive_free` for max VRAM clearance |
| **C++ llama.cpp** | 8B | Q4_K_M | **~30-50** | ~4.8GB | Use `EasyLLM GGUF` node |
| **C++ llama.cpp** | 8B | Q8_0 | **~20-30** | ~8.5GB | Use `EasyLLM GGUF` node |

### PyTorch Backend: VRAM Mode Controls

The **EasyLLM** node provides three `vram_mode` settings to control GPU memory:

| Mode | Behavior | Best For |
|------|----------|----------|
| `unload` (default) | Frees VRAM after generation. Safe for image workflows. | One-shot generation in image pipelines |
| `keep_loaded` | Keeps model on GPU between generations. Eliminates 5-15s reload delay. | Popup chat sessions, sequential generations |
| `aggressive_free` | Unloads ALL non-essential models before and after generation. Max VRAM clearance. | Low-VRAM GPUs, running alongside large image models |

**`use_mlock`** (checkbox) locks model memory to prevent Windows/Linux from swapping it to disk. Recommended when using `vram_mode=keep_loaded`.

> **Expected improvement**: 3-4 tok/s → **8-12 tok/s** for an 8B model in chat mode with `keep_loaded`.

### C++ Engine (10-15× Faster — One Command Install)

For **Ollama/LM Studio level speed** (30-50+ tokens/sec), use the **EasyLLM GGUF** node. This loads a GGUF model file directly via `llama-cpp-python` — the **same C++ engine** that powers Ollama and LM Studio. The setup is a single command — see the [Installation guide](installation.md).

#### 🛠 Recommended Settings

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_gpu_layers` | `-1` | All layers on GPU (max speed) |
| `use_mlock` | `True` | Lock memory, prevent swapping |
| `vram_mode` | `keep_loaded` | Model stays on GPU for instant chat |
| Quantization | **Q4_K_M** | ~4.8GB VRAM for 8B (best speed/quality) |

> See [Compatible Models](models.md) for the GGUF quantization guide and model recommendations.

---

← [Back to Documentation](README.md)
