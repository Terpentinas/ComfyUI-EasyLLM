← [Documentation Home](../README.md)

---

# 🎛️ `LLM_TriggerRouter` — Decompose trigger_prompt JSON

Decomposes the structured `trigger_prompt` JSON from EasyLLM / EasyLLM GGUF into individual sockets. Required for all multi-turn generation workflows.

| Input | Type | Description |
|-------|------|-------------|
| `trigger_prompt` | `STRING` (forceInput) | Connect to LLM node's `trigger_prompt` output. Expects JSON with fields: `action`, `prompt`, `negative_prompt`, `session_uuid` |

| Output | Type | Description |
|--------|------|-------------|
| `prompt` | `STRING` | The image generation prompt (text to feed to CLIP Text Encode) |
| `negative_prompt` | `STRING` | Negative prompt for CLIP Text Encode |
| `session_uuid` | `STRING` | Unique ID for this generation turn — wire to Image Capture for image reconciliation |

---

← [Back to Documentation](../README.md)
