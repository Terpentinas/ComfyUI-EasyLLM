# Changelog

All notable changes to EasyLLM for ComfyUI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — First Public Release

### Added
- **⚡ EasyLLM GGUF node** — High-speed local GGUF inference via llama.cpp C++ engine (30–50+ tok/sec, Q4_K_M)
- **🤖 EasyLLM node (CLIP backend)** — Use Qwen-based text encoders already loaded by Anima, Z-Image, Flux Klein
- **📄 EasyLLM Text node** — Display LLM output on the canvas with passthrough
- **🎛️ EasyLLM Trigger Router** — Decompose structured trigger_prompt JSON into individual sockets
- **🖼️ EasyLLM Image Capture node** — Persist generated images to chat history for multi-turn workflows
- **👁️ Vision / Multimodal support** — LLaVA, Qwen-VL, Gemma-3-Vision with automatic mmproj detection
- **🔍 GGUF Model Browser** — Search, filter, and manage local GGUF models from a popup dialog
- **📚 System Prompt Manager** — 11 built-in templates plus custom creation, editing, import, and export
- **💬 Streaming Chat Popup** — Real-time token streaming, markdown rendering, image upload, conversation history
- **📝 7 Text Utility Nodes** — Joiner, Replacer, Extractor, Limiter, Whitespace Cleaner, Duplicate Remover, Text Input
- **📚 EasyLLM Prompt Select node** — Prompt library browser on the canvas
- **💾 Persistent History Database** — Chat and enhancer history persisted to disk via JSON, survives restarts
- **⚙️ config_user.py override system** — User settings survive updates without modifying main config
- **🧠 Auto Chat-Template Detection** — 3-tier pipeline (GGUF metadata → architecture → filename heuristics)
- **🎯 Auto-Tuned Sampling** — Model size detection sets optimal temperature, top-K, top-P, repetition_penalty
- **⚡ GPU Acceleration** — TF32 matmul, torch.compile, GPU-vectorized sampling (1.2–1.5× speedup on Ampere+)
- **💾 VRAM Modes** — unload / keep_loaded / aggressive_free for any GPU budget
- **🔄 Background Model Pre-loading** — Models load asynchronously, ready before first generation
- **🚀 Automatic Installer** — install.py auto-detects CUDA version and installs correct llama-cpp-python wheel
- **📦 Platform Launchers** — Windows (.bat), Linux (.sh), macOS (.command) installer scripts
