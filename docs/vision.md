← [Documentation Home](README.md)

---

# Vision / Multimodal Support (GGUF)

The **EasyLLM GGUF** node supports **vision-language models** that can process images alongside text. These models require a **mmproj** (multimodal projection) file — a small GGUF file that maps image embeddings to the text embedding space.

### How it works

1. Connect an image from a **Load Image** node to the `image` socket
2. Set `mmproj_path` to the companion mmproj `.gguf` file (e.g., `llava-v1.5-7b-mmproj.gguf`)
3. The node sends both the image and your text message to the model
4. The model "sees" the image and responds accordingly

### Auto-Detection

Place a `*mmproj*.gguf` file in the **same directory** as your main model. The node will:
- Match it by GGUF architecture (preferred — reads `general.architecture` from both files)
- Fall back to filename similarity scoring
- Cache the result in the model index for near-instant subsequent lookups

In the **GGUF Model Browser**, vision-capable models are marked with a 🖼️ **mmproj** badge.

### No-wire chat mode

Upload an image directly from the **popup chat** — no Load Image node needed. The image is uploaded to ComfyUI's input directory and passed to the model automatically.

**Requirements:**
- llama-cpp-python v0.3.23 or newer (auto-checked)
- A vision-capable GGUF model (LLaVA, BakLLaVA, Qwen-VL, Gemma-3-Vision, etc.)
- The companion mmproj file

### Image persistence

Images are sent to the model **once** — with the message they're attached to. On subsequent turns, only text from previous messages is preserved. If you need the model to re-examine an image (e.g., follow-up visual questions), re-upload the image with your new question. Make sure you're using a vision-capable model with an mmproj file.

---

← [Back to Documentation](README.md)
