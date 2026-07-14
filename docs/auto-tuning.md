← [Documentation Home](README.md)

---

# Auto-Tuned Sampling

The `EasyLLM` node **automatically detects your model's size** and sets optimal sampling parameters. No more fiddling with `top_k`, `top_p`, or `repetition_penalty`.

| Model | Temperature | top_k | top_p | repetition_penalty |
|-------|:-----------:|:-----:|:-----:|:------------------:|
| **Anima (Qwen3-0.6B)** | 0.3 | 20 | 0.90 | 1.3 |
| **Z-Image / Flux Klein 4B** | 0.6 | 40 | 0.90 | 1.2 |
| **Flux Klein 8B / Unknown** | 0.7 | 50 | 0.95 | 1.1 |

### Why auto-tuning matters

- **Small models (0.6B)** need strict sampling — low temperature, tight top-K, higher repetition penalty. Without this, they easily generate random token IDs that decode as garbled Unicode symbols (⚙, ⚐, �).
- **Medium models (4B)** can handle moderate creativity with good coherence.
- **Large models (8B+)** benefit from more relaxed settings that let their larger capacity shine.

### The temperature dropdown

The `temperature` control is the **only** sampling parameter exposed in the CLIP backend UI:

| Option | Effect |
|--------|--------|
| `auto` | Uses the model-optimized temperature from the table above |
| `0.0` | Greedy — fully deterministic, same output for same seed |
| `0.3` | Low — more focused, deterministic output |
| `0.5` | Medium-low — good balance for most prompts |
| `0.7` | Medium — creative but may risk garbled output on 0.6B |
| `0.9` | High — very creative, best for 4B+ models |

When you select a specific value (e.g., `0.5`), it **overrides only the temperature**. The other parameters (`top_k`, `top_p`, `repetition_penalty`) remain auto-tuned for your model.

The **GGUF backend** exposes all sampling parameters (`top_k`, `top_p`, `repetition_penalty`) directly for manual tuning.

---

← [Back to Documentation](README.md)
