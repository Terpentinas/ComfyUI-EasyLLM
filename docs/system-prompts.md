← [Documentation Home](README.md)

---

# 📚 System Prompt Manager

The node includes a built-in **system prompt template manager** with **11 pre-built templates** stored in [`system_prompts.json`](../system_prompts.json). This is a complete prompt engineering toolkit — not just a text box.

| Template | Use Case |
|----------|----------|
| **Simple Minimal Enhance** | Add 2-3 visual details to a concept |
| **Natural Language Enhancer** | Transform concepts into natural language descriptions |
| **Danbooru Tag Generator** | Output comma-separated Danbooru-style tags |
| **Mixed Tag + NL** | Combine quality tags with natural language |
| **Tag to Natural Language** | Convert tags into fluent prose |
| **Image Describer** | Brief 1-2 sentence image description |
| **Detail Modifier** | Change one specific visual detail |
| **Art Style Lite** | Identify art medium, technique, and era |
| **Short Budget** | Maximum 50-word description |
| **Prompt Compressor** | Condense prompts to essential information |
| **Negative Prompt Generator** | Generate negative prompts |

### Why it's more than a text box

Most ComfyUI LLM nodes give you a single `system_prompt` text widget. EasyLLM gives you a complete management system:

| Capability | What it means |
|------------|---------------|
| **📝 Edit built-in templates** | Adjust the 11 default prompts to your style |
| **➕ Create custom templates** | Save your own system prompts with names |
| **📥 Import from JSON** | Load prompt libraries shared by the community |
| **📤 Export to JSON** | Share your prompt collections — perfect for Civitai workflows |
| **🔄 Reorder templates** | Drag to arrange by frequency of use |
| **🔍 Search** | Find templates by name instantly |
| **🔗 Share across installations** | Export templates from one ComfyUI, import into another |

### How to access

1. Double-click any node to open the **popup chat**
2. Click **⚙ Manage Prompts...** in the popup footer
3. Or click **⚙ Manage Prompts...** on the node canvas

The management dialog connects to the backend API at `/easyllm/prompts/*` and updates all nodes in real-time when you save changes.

![Prompt Management Dialog](../media/Prompt-Library.png)

---

← [Back to Documentation](README.md)
