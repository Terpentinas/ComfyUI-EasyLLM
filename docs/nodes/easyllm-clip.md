← [Documentation Home](../README.md)

---

# 🤖 `EasyLLM` — Interactive conversation (CLIP)

Talk to the LLM model loaded inside your CLIP text encoder. **No additional model downloads required.** This backend is optional — use it if you already have a compatible Qwen-based model loaded for image generation, and want to reuse it for text generation.

#### Section 1: Setup

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `clip` | `CLIP` | — | The CLIP object from Load CLIP / DualCLIPLoader |
| `mode` | `COMBO` | `chat` | `chat`: Interactive conversation with popup + history. `enhancer`: Direct prompt-to-output, wire to CLIP Text Encode. |

#### Section 2: Persona & Input

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `STRING` | `""` | Your message to the LLM |
| `prompt_template` | `COMBO` | First template | Select a system prompt template, or `Custom` to write your own |
| `system_prompt_text` | `STRING` | `""` | Custom system prompt (used when template is `Custom`) |
| `system_prompt` | `STRING` (forceInput) | `""` | Connect from another node. Overrides template and custom text. |

#### Section 3: Socket Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `text_input` | `STRING` (forceInput) | `""` | Chain input: accepts text from another node. Overrides the text widget when connected. |

#### Section 4: Generation Parameters

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `COMBO` | `auto` | `auto` = model-optimized; or select `0.0` (greedy), `0.3`, `0.5`, `0.7`, `0.9` to override |
| `max_length` | `INT` | `768` | Maximum tokens to generate (16–4096) |
| `seed` | `INT` | `0` | Random seed. `0` = auto-randomize each generation |

#### Section 5: Hardware (Advanced)

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `vram_mode` | `COMBO` | `unload` | `unload`: Free VRAM after gen (safe for image workflows). `keep_loaded`: Keep model on GPU for fast sequential chat. `aggressive_free`: Unload ALL non-essential models before & after gen (max VRAM clearance). |
| `use_mlock` | `BOOLEAN` | `False` | Lock model memory to prevent OS swapping |

> **Other sampling parameters (top_k, top_p, repetition_penalty) are automatically tuned** based on the detected model size — see [Auto-Tuned Sampling](../auto-tuning.md).

| Output | Type | Description |
|--------|------|-------------|
| `text` | `STRING` | The generated response (cleaned — think tags and artifacts removed) |
| `raw_text` | `STRING` | The raw generated text as decoded from the model (preserves think tags, channel tags) |
| `clip` | `CLIP` | Original CLIP object, passed through |

| New Output (v2) | Type | Description |
|-----------------|------|-------------|
| `image_output` | `IMAGE` | Passthrough of attached image — enables image-to-image workflows via the chat popup |

---

← [Back to Documentation](../README.md)
