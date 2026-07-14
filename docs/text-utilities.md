← [Documentation Home](README.md)

---

# 📝 Text Utility Nodes

These lightweight text processing nodes have **no dependencies** on the LLM engine (no llama-cpp, no CLIP). They're found under the **`EasyLLM/text`** category.

| Node | Description | Key Features |
|------|-------------|--------------|
| 📝 **EasyLLM Text Input** | Standalone multi-line text input box. ComfyUI lacks a native text entry node — this fills that gap. | Simple passthrough, dynamic prompts disabled |
| 🔗 **EasyLLM Text Joiner** | Merge two text strings with a configurable delimiter. | Empty-input safe; delimiters: `, `, space, newline, ` - ` |
| 🧹 **EasyLLM Text Replacer** | General-purpose find-and-replace on any string output. | Case-sensitive toggle; empty find = safe passthrough |
| ✂️ **EasyLLM Text Extractor** | Keep text before or after a delimiter word/phrase. | Nth-occurrence selection; case-insensitive mode |
| 📏 **EasyLLM Text Limiter** | Truncate text to max characters, words, or lines. | Choose which side to cut; optional ellipsis |
| 🧽 **EasyLLM Whitespace Cleaner** | Strip leading/trailing whitespace, collapse blank lines. | Trim modes: both, leading, trailing, all_lines |
| 🗃️ **EasyLLM Text Duplicate Remover** | Remove duplicate tags, words, or lines. | First-occurrence wins; case-insensitive by default; configurable separator |

---

← [Back to Documentation](README.md)
