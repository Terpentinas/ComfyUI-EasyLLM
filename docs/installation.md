← [Documentation Home](README.md)

---

# Installation

### Basic (No external dependencies)
1. Clone or copy this directory into `ComfyUI/custom_nodes/`
2. Restart ComfyUI
3. Find the nodes under the **"EasyLLM"** category in the node menu
4. Connect them to any compatible `Load CLIP` node

No pip install needed. No extra requirements.

### Optional: C++ acceleration (30-50+ tok/sec)

#### 🚀 Automatic Installation (Recommended)

Run the install helper script once:

```bash
python custom_nodes\ComfyUI-EasyLLM\install.py
```

The script will:
1. 🔍 **Auto-detect** your ComfyUI Python installation (including Windows Portable's embedded Python)
2. 🎯 **Detect your GPU** — runs `nvidia-smi` to find your CUDA version, or detects ROCm/CPU
3. ⚙️ **Install** the matching pre-built wheel from the [llama-cpp-python wheel index](https://abetlen.github.io/llama-cpp-python/whl/cu124) — **no C++ compiler needed**
4. ✅ **Verify** the installation works

> **ComfyUI Manager users:** The `install.py` script is auto-discovered and executed by ComfyUI Manager during node installation. No action needed.

#### 🪟🐧🍎 Launcher Scripts

For a simple interactive menu (no terminal commands needed), use the platform-specific launchers:

| Platform | File | How to run |
|----------|------|------------|
| **Windows** | [`launchers/run_launcher.bat`](../launchers/run_launcher.bat) | Double-click in File Explorer |
| **Linux** | [`launchers/run_launcher.sh`](../launchers/run_launcher.sh) | `./run_launcher.sh` in terminal |
| **macOS** | [`launchers/run_launcher.command`](../launchers/run_launcher.command) | Double-click in Finder (after `chmod +x`) |

See [`launchers/README.md`](../launchers/README.md) for full instructions.

#### 🔧 Manual Installation

If the automatic script doesn't work, pick one command for your GPU:

| Your GPU | Install Command |
|----------|----------------|
| **NVIDIA CUDA 12.x** (RTX 30/40/50 series) | `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124` |
| **NVIDIA CUDA 11.x** (older GPUs) | `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118` |
| **CPU only** (any system) | `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` |
| **AMD ROCm 5.x** | `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/rocm5` |
| **Any GPU** (Vulkan) | `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan` |

> **ComfyUI Windows Portable users:** Replace `pip` with `python_embeded\python.exe -m pip` in the commands above.

After installing, restart ComfyUI and use the **"EasyLLM GGUF"** node with a `.gguf` model file.

---

← [Back to Documentation](README.md)
