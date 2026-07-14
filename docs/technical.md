← [Documentation Home](README.md)

---

# Technical Details

### How generation works under the hood (CLIP backend)

The chain of calls:

```
comfy.sd.CLIP
  └── .cond_stage_model  (SD1ClipModel subclass, e.g., AnimaTEModel)
        └── .generate(token_dict, ...)     ← SD1ClipModel.generate()
              └── .qwen3_4b.generate(...)  ← SDClipModel.generate()
                    └── .transformer.generate(embeds, ...)  ← BaseGenerate.generate()
                          └── autoregressive token loop with KV cache
```

All of this infrastructure already exists in ComfyUI at:
- [`SD1ClipModel.generate()`](comfy/sd1_clip.py:743)
- [`SDClipModel.generate()`](comfy/sd1_clip.py:311)
- [`BaseGenerate.generate()`](comfy/text_encoders/llama.py:863)

### GGUF Backend

The `EasyLLM GGUF` node loads a `.gguf` model file directly via `llama-cpp-python` — the **same C++ engine** that powers Ollama and LM Studio. The entire generation loop runs in native C++ CUDA kernels, completely bypassing Python/PyTorch overhead.

### Model size detection

The node detects which model is loaded by reading the `clip_name` attribute on the inner cond_stage model (e.g., `"qwen3_06b"`, `"qwen3_4b"`). The name is matched against patterns to determine the model's parameter count:

- `"06b"` or `"0.6"` → **0.6B** (small) — strictest sampling
- `"4b"` or `"2b"` → **4B** (medium) — balanced sampling
- `"8b"` → **8B** (large) — relaxed sampling
- Anything else → **8B+ defaults** (safe fallback)

### Chat template

The CLIP backend uses the standard Qwen chat format:

```
<|im_start|>system
You are an expert prompt engineer...
<|im_end|>
<|im_start|>user
a cool cyberpunk cat
<|im_end|>
<|im_start|>assistant
```

This is handled internally — you just type your message. The GGUF backend uses the [auto-detected](chat-templates.md) chat template.

### GPU acceleration

The node automatically moves the text encoder model to your GPU with the correct dtype (bfloat16) before generation begins. ComfyUI normally offloads text encoders to CPU to save VRAM — this node temporarily moves it back to CUDA, runs the generation loop, then restores the model to its original location so ComfyUI's memory management is not disrupted.

### GPU-vectorized optimizations

The CLIP backend applies several GPU-level optimizations automatically:
- **GPU-vectorized `sample_token`** — no Python loops for repetition_penalty
- **TF32 matmul precision** — 1.2–1.5× speedup on Ampere+ GPUs
- **`torch.inference_mode()`** — faster than `no_grad`
- **`torch.compile()`** of transformer forward pass
- **Selective quantized forward path** — Q8_0 native, NVFP4 full-precision

### VRAM usage

- **Temporary GPU usage**: The model is moved to GPU only during generation, then restored
- **Cache cleared after generation**: `torch.cuda.empty_cache()` is called to prevent VRAM fragmentation for downstream KSampler
- **Generation is sequential**: ComfyUI executes nodes one at a time, so the KSampler only runs after generation completes
- **Small models are fast**: Qwen3-0.6B generates ~100+ tokens/sec on a modern GPU

---

## Limitations

- **CLIP backend model quality**: The 0.6B Qwen in Anima is small and can hallucinate. The 4B/8B models in Z-Image and Flux Klein are significantly better. Auto-tuning helps, but larger models produce more coherent output.

---

## Technical Architecture

```
easyllm/
├── __init__.py              # Module init, NODE_CLASS_MAPPINGS export
├── _version_config.py       # Centralized version config (install.py + backend)
├── chat_node.py             # Core node classes: EasyLLM, EasyLLMText, EasyLLMGGUF
├── text_utils.py            # Text Utility Nodes (7 nodes)
├── streaming.py             # WebSocket push, API routes (/easyllm/*), generation
├── prompt_manager.py        # System prompt templates API (/easyllm/prompts/*)
├── utils.py                 # Utilities, GGUF model index, text cleaning, chat templates
├── trigger_router_node.py   # 🎛️ Trigger Router — decomposes trigger_prompt JSON
├── image_capture_node.py    # 🖼️ Image Capture — persists generated images to DB
├── generation_state.py      # Shared generation state (popup tracking, abort, progress)
├── cuda_optimizations.py    # GPU-vectorized sample, TF32, torch.compile
├── memory_manager.py        # GPU memory management (prepare, offload, restore)
├── llama_cpp_backend.py     # llama.cpp Python wrapper (LlamaCppModel)
├── debug_profiler.py        # Import-time performance profiling
├── profiler.py              # Per-generation profiling (TTFT, tokens/sec)
├── install.py               # llama-cpp-python auto-installer
├── system_prompts.json      # Built-in system prompt templates (11 templates)
├── requirements.txt         # Dependency documentation
├── launchers/               # Platform-specific install launchers
│   ├── README.md
│   ├── run_launcher.bat     # Windows double-click
│   ├── run_launcher.sh      # Linux terminal
│   └── run_launcher.command # macOS double-click
└── js/
    ├── llm_chat.js          # Main extension entry point
    ├── api.js               # API layer (prompts, model browser)
    ├── buttons.js           # Canvas button widgets
    ├── constants.js         # NODE_NAMES, shared constants
    ├── db_manager.js        # IndexedDB wrapper for local storage
    ├── editor.js            # Prompt management dialog
    ├── history_store.js     # Chat history storage
    ├── popup.js             # Shared popup utilities
    ├── popup_bubble.js      # Chat bubbles, export, rendering
    ├── popup_chat.js        # Chat popup modal
    ├── popup_model_browser.js  # GGUF model browser
    ├── popup_settings.js    # Settings popup dialog logic
    ├── popup_utils.js       # Additional popup helpers
    ├── prompt_select.js     # 📚 Prompt Select — node canvas prompt library
    ├── text_input.js        # Canvas text input handling
    ├── ui_utils.js          # UI utilities
    ├── websocket_bridge.js  # WebSocket streaming
    └── llm_chat.css         # Stylesheet
```

---

← [Back to Documentation](README.md)
