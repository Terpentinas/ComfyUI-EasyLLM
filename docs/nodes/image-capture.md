← [Documentation Home](../README.md)

---

# 🖼️ `EasyLLM_ImageCapture` — Persist generated images to chat history

Saves generated images to the chat history database, keyed by `session_uuid`. Required for multi-turn generation workflows so generated images appear in the chat popup.

| Input | Type | Description |
|-------|------|-------------|
| `images` | `IMAGE` | Connect VAE Decode output |
| `session_uuid` | `STRING` (forceInput) | Connect Trigger Router's `session_uuid` output |

| Output | Type | Description |
|--------|------|-------------|
| `images` | `IMAGE` | Passthrough of input images for downstream nodes (Preview Image, Save Image) |

---

← [Back to Documentation](../README.md)
