← [Documentation Home](README.md)

---

# Chat Template Auto-Detection (GGUF)

The **EasyLLM GGUF** node automatically detects the correct chat template for your model using a **3-tier pipeline**:

1. **Tier 1 — GGUF Metadata**: Reads `tokenizer.chat_template` from the GGUF file's header. Most modern models include this in their metadata.

2. **Tier 2 — Architecture Mapping**: Falls back to `general.architecture` field. Maps architectures like `qwen2`, `llama`, `gemma2`, `phi3`, etc. to known chat templates.

3. **Tier 3 — Filename Heuristics**: As a last resort, analyzes the model filename for known prefixes (e.g., `phi-3.5-vision` → `phi3v`, `llama-3` → `llama`).

If all three tiers fail, the template safely defaults to `"llama"` (the most widely compatible format).

The detected template is shown in the node's model info panel. You can also manually select a template from the `chat_template` dropdown to override auto-detection.

---

← [Back to Documentation](README.md)
