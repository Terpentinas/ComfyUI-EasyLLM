← [Documentation Home](README.md)

---

# Troubleshooting

### Gibberish output (repeating characters, garbled Unicode symbols like ⚙ or ⚐)

**This is now handled automatically** — the node:

1. **Detects model size** and tunes parameters accordingly: 0.6B models get `temperature=0.3`, `top_k=20`, `repetition_penalty=1.3` to prevent random token generation.
2. **Filters garbled Unicode** from the output — symbols in the U+2300–27BF range (Miscellaneous Technical, Box Drawing, Geometric Shapes, Dingbats, etc.) are stripped.
3. **Uses an optimized system prompt** that instructs the model to describe visual scenes concisely without explanations or code.

If you still see issues on a 0.6B model, try:
- **Select `temperature=0.3`** explicitly in the dropdown (already the auto default, but confirms override)
- **Lower `max_length`** to 64–128 — shorter generations are less likely to drift
- **Use more specific prompts** — the 0.6B model benefits from clear, direct input

### Garbled output on 4B+ models

Try **selecting a lower temperature** from the dropdown (e.g., `0.3` or `0.5`) to reduce randomness. The auto-tuning for 4B models uses `temperature=0.6`, which is fine for most cases but can be adjusted down for more focused responses.

### Very slow generation (3+ minutes for 256 tokens)

Generation running on CPU instead of GPU. The node now **automatically** moves the model to CUDA with bfloat16 before generation, and restores it afterward. Expected performance:

| Model | Tokens/sec (GPU) |
|-------|------------------|
| Anima (Qwen3-0.6B) | ~100+ |
| Z-Image (Qwen3-4B) | ~30–50 |
| Flux Klein (Qwen3-4B/8B) | ~15–40 |

If generation is still slow, check that ComfyUI's `--gpu-only` flag is not interfering, or that you have enough free VRAM.

### Out of memory (OOM) after generation

The node calls `torch.cuda.empty_cache()` after generation. If you still get OOM errors, reduce `max_length` or use a smaller model (Anima 0.6B instead of 4B/8B). For the GGUF backend, try `vram_mode=unload` or reduce `n_gpu_layers`.

### GGUF Model Browser

![GGUF Model Browser](../media/model-browser-pop-up.png)

If the GGUF Model Browser doesn't find your models, ensure:
1. Your `.gguf` files are in a directory ComfyUI searches (e.g., `ComfyUI/models/LLM/`, `ComfyUI/models/GGUF/`)
2. The model index has been refreshed via the **Refresh** button in the browser
3. The directory hasn't been accidentally excluded via the browser's exclusion controls

### GGUF Model Not Found / Path Issues

The node searches multiple locations when resolving model paths:
- Absolute paths (e.g., `C:/models/qwen.gguf`)
- ComfyUI's registered model folders (`models/`, `text_encoders/`, `llm/`, `gguf/`, `clip/`)
- Custom directories added via the Model Browser

If a model file can't be found, use the **Browse** button in the popup to locate it manually.

### Console Log Levels

The node uses three log prefixes in your ComfyUI terminal. Here's what they mean:

| Prefix | What it shows | Always visible? |
|--------|---------------|-----------------|
| `[LLM Chat GGUF]` | Model load/unload, version detection, VRAM status | ✅ Yes — core lifecycle |
| `[DIAG]` | Generation timing (TTFT, tok/s, batch eval times) | ✅ Yes — performance telemetry |
| `[LLM Chat Debug]` | Deep profiling: module imports, phase timings, box charts | ❌ Only if enabled |

**Enabling deep debug profiling:**
- Set environment variable `LLM_CHAT_DEBUG=1`, or
- Create an empty file named `.debug` in the EasyLLM custom_node directory

When reporting an issue on GitHub, paste your full console log — the `[DIAG]` lines help identify performance bottlenecks quickly.

---

← [Back to Documentation](README.md)
