← [Documentation Home](../README.md)

---

# ⚡ `EasyLLM GGUF` — High-speed GGUF chat

Directly loads a `.gguf` model via llama.cpp C++ engine. **30–50+ tok/sec** with Q4_K_M. This is the flagship backend.

#### Section 1: Setup

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | `STRING` | — | Path to `.gguf` file (e.g., `Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf`) |
| `mode` | `COMBO` | `chat` | `chat`: Interactive conversation with popup + history. `enhancer`: Direct prompt-to-output. |

#### Section 2: Persona & Input

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `STRING` | `""` | Your message to the LLM |
| `prompt_template` | `COMBO` | First template | Select a system prompt template, or `Custom` to write your own |
| `system_prompt_text` | `STRING` | `""` | Custom system prompt (used when template is `Custom`) |

#### Section 3: Generation Parameters

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `FLOAT` | `0.7` | Sampling temperature. `0.0` = greedy/deterministic. |
| `max_length` | `INT` | `768` | Maximum tokens to generate (16–16384) |
| `seed` | `INT` | `0` | Random seed. `0` = auto-randomize |

#### Section 4: Socket Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `text_input` | `STRING` (forceInput) | `""` | Chain input: accepts text from another node. Overrides the text widget when connected. |
| `system_prompt` | `STRING` (forceInput) | `""` | Connect from another node. Overrides template and custom text. |

#### Section 5: Hardware (Advanced)

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `chat_template` | `COMBO` | `auto` | `auto` = 3-tier auto-detection (recommended). Or select a specific template. |
| `n_ctx` | `INT` | `4096` | Context window size (512–32768) |
| `n_gpu_layers` | `INT` | `-1` | `-1` = all layers on GPU (max speed). `0` = CPU. `24+` = balanced. |
| `use_mlock` | `BOOLEAN` | `True` | Lock model memory to prevent OS swapping |
| `vram_mode` | `COMBO` | `unload` | `unload`: Free VRAM after gen. `keep_loaded`: Model stays on GPU for fast chat. `aggressive_free`: Max VRAM clearance. |
| `top_k` | `INT` | `50` | Top-K sampling. Higher = more diverse. |
| `top_p` | `FLOAT` | `0.9` | Nucleus sampling threshold |
| `repetition_penalty` | `FLOAT` | `1.1` | Repetition penalty. `1.0` = none, `>1.0` discourages repeats |

#### Section 6: Vision / Multimodal (Optional)

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `IMAGE` (forceInput) | — | Connect from Load Image node for vision-language models (LLaVA, BakLLaVA, Qwen-VL, etc.) |
| `mmproj_path` | `STRING` | `""` | Path to multimodal projection `.gguf` file. Auto-detected if placed in same folder as model. |
| `image_filename` | `STRING` | `""` | Internal: stores uploaded image filename for no-wire chat mode. Managed automatically by the chat popup. |

| Output | Type | Description |
|--------|------|-------------|
| `text` | `STRING` | The generated response (cleaned — think tags and artifacts removed) |
| `raw_text` | `STRING` | The raw generated text as decoded from the model (preserves think tags, channel tags) |

---

← [Back to Documentation](../README.md)
