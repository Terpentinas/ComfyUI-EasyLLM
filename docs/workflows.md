← [Documentation Home](README.md)

---

## Workflow Examples

### Prompt Enhancement (Enhancer Mode)

Set the `EasyLLM` or `EasyLLM GGUF` node's mode to `enhancer` for direct prompt-to-output workflow:

```
[Load CLIP (Anima)] ──> [EasyLLM (mode=enhancer)] ──> [CLIP Text Encode] ──> [KSampler]
                            │                                               │
                            │  "a mystical forest"                         │
                            │  ─────────────────>                          │
                            │  "A mystical forest at twilight, ancient     │
                            │   oaks with glowing runes, fireflies... "    │
```

### Chat (inline response display)

The **EasyLLM** node shows the generated response **directly inside the node** — no separate "Show Text" node needed.

```
[Load CLIP (Anima)] ──> [EasyLLM]
                            │
                            │  "What styles work well with this model?"
                            │
                            │  Response appears inside the node
```

The response text is displayed in a read-only text area with distinct styling. You can still wire the `text` output to a "Show Text" node if needed.

### Chat + EasyLLM Text (alternative)

```
[Load CLIP (Anima)] ──> [EasyLLM] ──> [EasyLLM Text]
                            │
                            │  "What styles work well with this model?"
```

### GGUF Chat

```
[EasyLLM GGUF] ──> [EasyLLM Text]
     │
     │ model_path: "Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf"
     │ text: "Explain how diffusion models work"
```

### Text Processing Pipeline

Use the Text Utility nodes to clean, transform, and combine LLM output before it reaches the KSampler:

```
[EasyLLM GGUF] ──> [🧽 Whitespace Cleaner] ──> [🗃️ Duplicate Remover] ──> [CLIP Text Encode]
```

### 🎛️ Multi-Turn Generation with Auto-Queue & Group Router

The **Trigger Router**, **Group Router**, and **Image Capture** nodes enable a powerful multi-turn chat-to-image workflow:

```
EasyLLM ──trigger_prompt──► Trigger Router ──prompt──► CLIP Text Encode (x2)
                                      │               └──negative_prompt──► CLIP Text Encode (neg)
                                      └──session_uuid──► Image Capture ◄── VAE Decode
```

**How it works:**
1. The LLM outputs structured JSON via `trigger_prompt` with an `action` field (`generate_image`, `edit_image`, `just_chat`)
2. The **Trigger Router** decomposes the JSON into individual sockets (prompt, negative_prompt, session_uuid)
3. The frontend **auto-queues** the full pipeline to the target Image Capture node when image generation is requested
4. **Image Capture** saves generated images with `session_uuid` for chat history reconciliation

**Loadable example workflows** are in [`example_workflows/`](../example_workflows/):
- [`Enhancer.json`](../example_workflows/Enhancer.json) — Single-shot prompt enhancement
- [`InteractiveChat + Generate.json`](../example_workflows/InteractiveChat + Generate.json) — Chat + single `[GENERATE]` group
- [`InteractiveChat + Generate + Edit.json`](../example_workflows/InteractiveChat + Generate + Edit.json) — Chat + `[GENERATE]` + `[EDIT]` groups

---

← [Back to Documentation](README.md)
