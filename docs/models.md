← [Documentation Home](README.md)

---

# Compatible Models

### GGUF Backend (EasyLLM GGUF) — Recommended

Any GGUF model compatible with llama.cpp. Recommended:
- [Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
- [Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF)
- [Search HuggingFace for GGUF models](https://huggingface.co/models?search=gguf)

### CLIP Backend (EasyLLM) — Optional

| Model | Text Encoder | Size | Notes |
|-------|-------------|------|-------|
| **Anima** | Qwen3-0.6B | 0.6B | Smallest, fastest. Auto-tuned for stable output. |
| **Z-Image** | Qwen3-4B | 4B | Good balance of speed and quality |
| **Flux Klein** | Qwen3-4B / Qwen3-8B | 4-8B | Best quality, larger models |
| **Qwen-Image** | Qwen2.5-7B-VL | 7B | Vision-language capable |

Models **without** generation support (will show a clear error):
- Standard SD1.5 CLIP (ViT-L/14)
- Standard SDXL CLIP (ViT-bigG)
- Flux T5-only (no Qwen)
- Standard T5 text encoders

---

### 📦 GGUF Quantization Guide

GGUF is a file format for quantized LLMs. Smaller quantization = less VRAM:

| Quantization | Size (8B model) | VRAM | Quality | Speed |
|:-----------:|:--------------:|:----:|:-------:|:-----:|
| **Q4_K_M** | ~4.8 GB | ✅ Fits most GPUs | ✅ Excellent | ⚡ 30-50 tok/s |
| **Q5_K_M** | ~5.5 GB | ⚠ Tight fit | ✅ Great | ⚡~25-40 tok/s |
| **Q8_0** | ~8.5 GB | ❌ 12GB+ only | 🏆 Best | ⚡~20-30 tok/s |

**Why Q4_K_M?** A 4-bit quantized model uses ~4.8GB VRAM (vs ~8.5GB for Q8_0). The smaller size fits entirely in GPU memory alongside other models, and the quality difference between Q4_K_M and Q8_0 is virtually invisible for text generation.

**Download recommended models:**
- [Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) — use the `q4_k_m.gguf` file
- [Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF)
- [Search HuggingFace for GGUF models](https://huggingface.co/models?search=gguf)

### 🔄 Background Model Pre-loading

When you select a model in the GGUF Model Browser and click **Apply**, the model starts loading in the background. By the time you queue the workflow, the model is already loaded — eliminating the load-time delay from your first generation. A WebSocket event (`easyllm_model_ready`) notifies the popup when loading completes.

---

← [Back to Documentation](README.md)
