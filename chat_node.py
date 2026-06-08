"""
EasyLLM Nodes - Interactive chat and prompt enhancement using the loaded CLIP model.

Provides the main nodes:
1. EasyLLM - Interactive conversation / prompt enhancement with the LLM loaded in CLIP
2. EasyLLMText - 🤖 Display generated text on the node surface (EasyLLM Text Display)
3. EasyLLMGGUF - High-speed GGUF inference via llama.cpp C++ engine

All chat/enhance nodes pass through the CLIP object so it can be used
with CLIP Text Encode downstream.
"""

import functools
import json
import logging
import time

from .utils import (
    auto_seed,
    supports_generation,
    resolve_text,
    resolve_system_prompt,
    format_prompt_by_template,
    clean_generated_text,
    detect_chat_template_from_metadata,
    detect_template_from_architecture,
    heuristic_template_from_filename,
    _read_gguf_architecture_only,
    CHAT_TEMPLATES,
    _UNIVERSAL_ROLE_STOP_TOKENS,
)
from .streaming import execute_chat_generation
from .prompt_manager import get_prompt_names, get_prompt_by_name

# Cache: unique_id -> (cleaned_text, raw_text); preserves think tags in raw_text
_cache = {}

# GGUF cache: avoids unique_id collisions with CLIP _cache
_cache_gguf: dict = {}

# Shared error message for models without generation support
NO_GENERATION_ERROR = (
    "ERROR: The loaded CLIP model does not support text generation.\n\n"
    "This node works with models that use an LLM-based text encoder:\n"
    "- Anima (Qwen3-0.6B)\n"
    "- Z-Image (Qwen3-4B)\n"
    "- Flux Klein (Qwen3-4B / Qwen3-8B)\n"
    "- Qwen-Image (Qwen2.5-7B-VL)\n\n"
    "Standard SDXL/SD1.5 CLIP models do NOT support generation."
)

NO_MODEL_PATH_ERROR = (
    "ERROR: No model path provided.\n\n"
    "Set model_path to a .gguf file, e.g., "
    "Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf"
)

LLAMA_NOT_INSTALLED_ERROR = (
    "ERROR: llama-cpp-python is not installed.\n\n"
    "Install it with:\n"
    "  pip install llama-cpp-python --extra-index-url "
    "https://abetlen.github.io/llama-cpp-python/whl/cu124\n\n"
    "Then restart ComfyUI."
)

class EasyLLM:
    """
    Interactive chat with the LLM model loaded inside a CLIP object.

    Accepts a CLIP connection (from Load CLIP / DualCLIPLoader) and a user message.
    Uses the underlying LLM's generate() capability to produce a text response.
    Sampling parameters (temperature, top_k, top_p, repetition_penalty) are
    automatically tuned based on the detected model size for optimal quality.

    System prompt can come from one of four sources (priority order):
    1. A connected LLMSystemPromptSelector node (system_prompt forceInput)
    2. A selected template from the prompt_template dropdown
    3. Custom text typed in the system_prompt_text widget
    4. The hardcoded default from utils.py (when all are empty)

    Text input can come from two sources (priority order):
    1. A connected node's STRING output (text_input forceInput socket)
    2. The canvas text widget (typed by user or set by popup)

    Passes through the CLIP object for downstream use.
    """

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        # Load prompt names from JSON
        prompt_names = get_prompt_names()
        return {
            "required": {
            },
            "optional": {
                # Section 1: Setup — What you need to get started
                "clip": ("CLIP", {"tooltip": "The CLIP model containing an LLM text encoder (Anima, Z-Image, Flux Klein, etc.)"}),
                "mode": (["chat", "enhancer"], {
                    "default": "chat",
                    "tooltip": "chat: Interactive conversation with popup + history. enhancer: Direct prompt-to-output, wire to CLIP Text Encode.",
                }),

                # Section 2: Persona & Input — Configure behavior and enter your message
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Your message to the LLM. Managed via the LLM Lab popup.",
                }),
                "prompt_template": (prompt_names, {
                    "default": prompt_names[0] if prompt_names else "Custom",
                    "tooltip": "Select a system prompt template, or Custom to write your own below.",
                }),
                "system_prompt_text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Type a custom system prompt. Used when prompt_template is 'Custom'.",
                }),
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Connect from LLMSystemPromptSelector. Overrides dropdown and custom text.",
                }),

                # Section 3: Socket Inputs — External text pipe connections
                "text_input": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Chain input: accepts text from another node. Overrides the text widget when connected.",
                }),

                # Section 4: Tuning — Generation parameters
                "temperature": (["auto", "0.0", "0.3", "0.5", "0.7", "0.9"], {
                    "default": "auto",
                    "tooltip": "Sampling temperature. Auto = model-optimized (recommended). 0.0 = greedy/deterministic, no sampling.",
                }),
                "max_length": ("INT", {
                    "default": 768,
                    "min": 16,
                    "max": 4096,
                    "step": 16,
                    "tooltip": "Maximum tokens to generate",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "step": 1,
                    "tooltip": "Random seed. 0 = auto-randomize each generation.",
                }),

                # Section 5: Hardware (Less Used) — Advanced system settings
                "vram_mode": (["unload", "keep_loaded", "aggressive_free"], {
                    "default": "unload",
                    "tooltip": "unload: Free image models VRAM before LLM gen, free LLM after (safe for image workflows). "
                               "keep_loaded: Free image models VRAM before LLM gen, keep LLM on GPU for fast sequential chat. "
                               "aggressive_free: Unload ALL non-essential models before & after LLM gen (max VRAM clearance).",
                }),
                "use_mlock": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Lock model memory to prevent OS swapping. Stabilizes shared RAM performance. Recommended for keep_loaded mode.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "CLIP",)
    RETURN_NAMES = ("text", "raw_text", "clip",)
    OUTPUT_TOOLTIPS = (
        "The generated response text (cleaned, with artifacts removed)",
        "The raw generated text (uncleaned, as decoded from the model)",
        "The original CLIP model, passed through for downstream use",
    )
    FUNCTION = "chat"
    CATEGORY = "EasyLLM"
    DESCRIPTION = "Chat with the LLM model loaded inside a CLIP text encoder. Sampling params auto-tune for model size. Supports system prompt templates."

    def _resolve_system_prompt(self, system_prompt, prompt_template, system_prompt_text):
        """Resolve system prompt via utils.resolve_system_prompt() (4-source priority)."""
        from .utils import resolve_system_prompt as _resolve
        return _resolve(system_prompt, prompt_template, system_prompt_text)

    def _resolve_text(self, text_input, text):
        """Resolve input text via utils.resolve_text() (2-source priority)."""
        from .utils import resolve_text as _resolve
        return _resolve(text_input, text)

    def chat(self, clip=None, text="", text_input="", mode="chat", system_prompt="",
             prompt_template="Custom", system_prompt_text="", max_length=768,
             temperature="auto", seed=0, vram_mode="unload", use_mlock=False,
             unique_id=None):
        """
        Execute a chat interaction with the loaded CLIP model.

        Auto-mode: When effective text is empty (Queue Prompt without popup or chain),
        returns cached output from `_cache[unique_id]` without GPU generation.

        Returns:
            tuple: (cleaned_text, raw_text, clip)
        """
        def _ui_result(text_val, raw_val, clip, system_prompt_val=""):
            """Wrap text + CLIP in ui dict and result tuple; logs think-tag diagnostics."""
            if text_val != raw_val and ("<think>" in raw_val or "<|channel>" in raw_val):
                logging.debug(
                    f"[LLM Chat] _ui_result: text={len(text_val)} chars (no think), "
                    f"raw={len(raw_val)} chars (with think tags)"
                )
            elif text_val == raw_val and ("<think>" in text_val or "<|channel>" in text_val):
                logging.warning(
                    f"[LLM Chat] _ui_result: BOTH outputs contain think tags — "
                    f"raw_text not properly separated!"
                )
            return {
                "ui": {
                    "text": [text_val],
                    "raw_text": [raw_val],
                    "input_text": [effective_text],
                    "system_prompt": [system_prompt_val],
                },
                "result": (text_val, raw_val, clip),
            }

        # — RESOLVE EFFECTIVE TEXT —
        # Must run before mode/cache checks so chain input is considered when widget is empty.
        effective_text = self._resolve_text(text_input, text)

        # — MODE-SPECIFIC BEHAVIOR —
        if mode == "enhancer":
            if not effective_text or not effective_text.strip():
                return _ui_result("", "", clip)
        else:
            # Chat mode: return cached output when text is empty (auto-mode)
            if not effective_text or not effective_text.strip():
                cached = _cache.get(unique_id, None)
                if cached:
                    cleaned, raw = cached
                    return _ui_result(cleaned, raw, clip)
                return _ui_result("", "", clip)

        # — CLIP CONNECTION CHECK —
        if clip is None:
            return _ui_result(
                "ERROR: No CLIP model connected.\n\n"
                "This node requires a CLIP model that supports text generation:\n"
                "- Anima (Qwen3-0.6B)\n"
                "- Z-Image (Qwen3-4B)\n"
                "- Flux Klein (Qwen3-4B / Qwen3-8B)\n"
                "- Qwen-Image (Qwen2.5-7B-VL)\n\n"
                "Connect a CLIP model to this node to enable text generation.",
                "ERROR: No CLIP model connected.",
                clip
            )

        if not supports_generation(clip):
            return _ui_result(NO_GENERATION_ERROR, NO_GENERATION_ERROR, clip)

        seed = auto_seed(seed)

        # Retrieve chat history from server-side store (populated by frontend
        # before queue). Returns None on first turn — format_prompt() handles this.
        parsed_history = None
        try:
            from .streaming import get_chat_history
            parsed_history = get_chat_history(unique_id)
        except Exception as e:
            logging.warning(f"[LLM Chat] Failed to retrieve history for node {unique_id}: {e}")

        try:
            effective_system_prompt = self._resolve_system_prompt(
                system_prompt, prompt_template, system_prompt_text
            )

            generated_text, raw_text = execute_chat_generation(
                clip=clip,
                text=effective_text,
                mode=mode,
                effective_system_prompt=effective_system_prompt,
                max_length=max_length,
                temperature=temperature,
                seed=seed,
                vram_mode=vram_mode,
                use_mlock=use_mlock,
                unique_id=unique_id,
                chat_history=parsed_history,
            )

            # Cache for auto-mode replay (preserves think tags in raw_text)
            _cache[unique_id] = (generated_text, raw_text)

            return _ui_result(generated_text, raw_text, clip, effective_system_prompt)

        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = f"Error during generation: {str(e)}"
            return _ui_result(err_msg, err_msg, clip, effective_system_prompt)

class EasyLLMText:
    """
    Display generated text on the node surface with real-time popup interaction.

    Shows text in a scrollable display area on the canvas node.
    Updates in real-time via JavaScript CustomEvents when connected
    to an EasyLLM node's text output (no queue wait).

    Features:
    - Text display on node surface
    - Real-time updates via CustomEvent with popup chat
    - "Open Chat" button to launch popup for further interaction
    - Text passthrough output for downstream nodes
    """

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "forceInput": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = (
        "Passes through the input text for downstream nodes.",
    )
    FUNCTION = "show"
    CATEGORY = "EasyLLM"
    DESCRIPTION = (
        "Displays text on the node surface with real-time popup interaction. "
        "Connect to EasyLLM's text output and click 'Open Chat' to continue "
        "the conversation in the popup."
    )

    def show(self, text=""):
        """Pass through the input text to downstream nodes."""
        return {"ui": {"text": [text]}, "result": (text,)}

class EasyLLMGGUF:
    """
    High-speed LLM chat using llama.cpp C++ inference engine.

    Loads a GGUF model file directly via llama-cpp-python, bypassing
    ComfyUI's Python/PyTorch generation loop entirely. The generation
    runs in native C++ CUDA kernels (same engine as Ollama/LM Studio),
    achieving 30-50+ tokens/sec for an 8B model.

    Requires: pip install llama-cpp-python
    Recommended: Q4_K_M quantization (~4.8GB VRAM for 8B model)

    Features:
    - C++ inference engine: 10-15x faster than PyTorch path
    - n_gpu_layers: Control GPU offloading (-1 = all layers on GPU)
    - use_mlock: Lock memory to prevent OS swapping
    - vram_mode: Keep model on GPU for instant chat responses
    - Model caching: Same model_path reuses loaded model across calls
    - Independent from ComfyUI's CLIP pipeline (no CLIP input needed)
    """

    OUTPUT_NODE = True

    # Module-level model cache: model_path -> LlamaCppModel
    _model_cache = {}
    _gguf_call_count = 0  # DIAG: track consecutive Queue Prompt runs

    @classmethod
    def INPUT_TYPES(cls):
        prompt_names = get_prompt_names()
        from .utils import CHAT_TEMPLATES
        chat_template_names = ["auto"] + list(CHAT_TEMPLATES.keys())
        return {
            "required": {
                # Section 1: Setup — What you need to get started
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a .gguf model file. Use Q4_K_M quantization for best speed.",
                }),
                "mode": (["chat", "enhancer"], {
                    "default": "chat",
                    "tooltip": "chat: Interactive conversation with popup + history. enhancer: Direct prompt-to-output.",
                }),

                # Section 2: Persona & Input — Configure behavior and enter your message
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Your message to the LLM. Managed via the LLM Lab popup.",
                }),
                "prompt_template": (prompt_names, {
                    "default": prompt_names[0] if prompt_names else "Custom",
                    "tooltip": "Select a system prompt template, or Custom to write your own below.",
                }),
                "system_prompt_text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Type a custom system prompt. Used when prompt_template is 'Custom'.",
                }),
            },
            "optional": {
                # Section 4: Tuning — Generation parameters
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature. 0 = greedy/deterministic.",
                }),
                "max_length": ("INT", {
                    "default": 768,
                    "min": 16,
                    "max": 16384,
                    "step": 16,
                    "tooltip": "Maximum tokens to generate",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "step": 1,
                    "tooltip": "Random seed. 0 = auto-randomize.",
                }),

                # — Socket inputs (kept here for clean node socket layout) —
                "text_input": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Chain input: accepts text from another node. Overrides the text widget when connected.",
                }),
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Connect from LLMSystemPromptSelector. Overrides dropdown and custom text.",
                }),

                # Section 5: Hardware (Less Used) — Advanced system settings
                "chat_template": (chat_template_names, {
                    "default": "auto",
                    "tooltip": "Auto-detect (Recommended) or select the chat template matching your model. Auto reads GGUF metadata, falls back to architecture, then filename heuristics.",
                }),
                "n_ctx": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 32768,
                    "step": 512,
                    "tooltip": "Context window size (max tokens the model can remember).",
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "-1 = all layers on GPU (max speed, more VRAM). 0 = CPU. 24+ = balanced.",
                }),
                "use_mlock": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Lock model memory to prevent OS swapping. Stabilizes shared RAM.",
                }),
                "vram_mode": (["keep_loaded", "unload", "aggressive_free"], {
                    "default": "unload",
                    "tooltip": "unload: Free VRAM after generation (safer for image workflows). "
                               "keep_loaded: Model stays on GPU for fast sequential chat. "
                               "aggressive_free: Unload ALL non-essential models before & after LLM gen (max VRAM clearance).",
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Top-K sampling. Higher = more diverse output.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold.",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Repetition penalty. 1.0 = no penalty. >1.0 discourages repetition, <1.0 encourages repetition. Default 1.1 provides mild repetition prevention.",
                }),

                # Section 6: Vision / Multimodal (Optional) — Image input support
                "image": ("IMAGE", {
                    "forceInput": True,
                    "tooltip": "Connect an image from Load Image node for vision-language models (LLaVA, BakLLaVA, Qwen-VL, etc.)",
                }),
                "mmproj_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to multimodal projection .gguf file (e.g., llava-v1.5-7b-mmproj.gguf). Required when an image is connected.",
                }),
                "image_filename": ("STRING", {
                    "default": "",
                    "tooltip": "Internal: stores uploaded image filename for chat mode (no-wire path). Managed automatically by the chat popup.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("text", "raw_text",)
    OUTPUT_TOOLTIPS = (
        "The generated response text (cleaned, with think tags and artifacts removed)",
        "The raw generated text (uncleaned, as decoded from the model)",
    )
    FUNCTION = "generate"
    CATEGORY = "EasyLLM"
    DESCRIPTION = (
        "High-speed LLM chat using llama.cpp C++ engine. "
        "Loads .gguf files directly. Requires: pip install llama-cpp-python. "
        "10-15x faster than PyTorch path. 30-50+ tok/s on GPU."
    )

    def _get_cached_model(self, model_path, mmproj_path, n_gpu_layers, use_mlock, n_ctx):
        """Get or create a cached LlamaCppModel instance.

        Cached by composite key (model_path, mmproj_path), allowing separate
        slots for text-only and multimodal paths. Purges duplicate entries.
        """
        from .llama_cpp_backend import LlamaCppModel

        cache_key = (model_path, mmproj_path or "")

        # — Check preload cache first —
        # Move pre-loaded model (via Apply button) into regular cache.
        from .streaming import _make_preload_key, _preload_results, _preload_lock
        preload_key = _make_preload_key(
            model_path, mmproj_path or "", n_gpu_layers, use_mlock, n_ctx
        )
        with _preload_lock:
            preloaded = _preload_results.pop(preload_key, None)
        if preloaded is not None:
            logging.info(
                f"[LLM Chat GGUF] Using pre-loaded model for {model_path} "
                f"(mmproj={mmproj_path or 'none'})"
            )
            # Purge stale cache entries with same model_path but different mmproj
            for key in list(self._model_cache.keys()):
                if isinstance(key, tuple) and key[0] == model_path and key != cache_key:
                    self._model_cache[key].unload()
                    del self._model_cache[key]
            for key in list(self._model_cache.keys()):
                other = self._model_cache[key]
                other.unload()
                del self._model_cache[key]
            self._model_cache[cache_key] = preloaded
            return preloaded

        # Check cached model with matching params
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            # Reload if params changed
            if (cached.n_gpu_layers != n_gpu_layers or
                    cached.use_mlock != use_mlock or
                    cached.n_ctx != n_ctx):
                logging.info(
                    f"[LLM Chat GGUF] Model params changed — reloading "
                    f"{model_path} (mmproj={mmproj_path or 'none'})"
                )
                cached.unload()
                del self._model_cache[cache_key]
            else:
                return cached

        # Purge duplicate model_path entries (only one copy per model due to VRAM)
        for key in list(self._model_cache.keys()):
            if isinstance(key, tuple) and key[0] == model_path and key != cache_key:
                logging.info(
                    f"[LLM Chat GGUF] Purging stale cache entry {key} "
                    f"for model {model_path}"
                )
                self._model_cache[key].unload()
                del self._model_cache[key]

        # Unload all previously cached models before loading new one.
        # Prevents VRAM accumulation when switching models with keep_loaded.
        for key in list(self._model_cache.keys()):
            cached = self._model_cache[key]
            cached.unload()
            del self._model_cache[key]
            logging.info(
                f"[LLM Chat GGUF] Unloaded previous cached model {key} "
                f"before loading new model {model_path}"
            )

        # Load new model
        model = LlamaCppModel(
            model_path=model_path,
            mmproj=mmproj_path,
            n_gpu_layers=n_gpu_layers,
            use_mlock=use_mlock,
            n_ctx=n_ctx,
        )
        model.load()
        self._model_cache[cache_key] = model
        return model

    # — GGUF Architecture Detection (pre-load metadata reader) —

    @staticmethod
    def _get_gguf_architecture(filepath: str) -> str | None:
        """Read `general.architecture` from a GGUF file's binary header.

        Uses ``_read_gguf_architecture_only`` — a minimal binary scanner
        that reads only the target key, skipping tensor data (~1-5ms).

        Returns:
            Lowercase architecture string (``"qwen2vl"``, ``"llama"``, etc.)
            or ``None`` if unreadable.
        """
        import os

        if not filepath or not os.path.isfile(filepath):
            return None
        try:
            return _read_gguf_architecture_only(filepath)
        except Exception as e:
            logging.debug(
                f"[LLM Chat GGUF] Failed to read architecture from "
                f"{os.path.basename(filepath)}: {e}"
            )
            return None

    # — LRU cache for auto-detect results (keyed by resolved model_path) —
    _mmproj_cache: dict[str, str] = {}
    _MMPROJ_CACHE_MAX = 32

    @staticmethod
    def _auto_detect_mmproj(model_path: str) -> str:
        """Auto-detect a companion mmproj file for the given model path.

        Only mmproj files in the **same directory** as the model are considered.
        Two-tier resolution: 0) GGUF metadata architecture matching, 1) filename
        similarity scoring (shared prefix >= 4 chars + substring containment).
        Results LRU-cached keyed by resolved model path.

        Returns:
            Absolute path to the detected mmproj file, or ``""`` if none found.
        """
        import os
        from .utils import resolve_gguf_path

        resolved = resolve_gguf_path(model_path)
        if not resolved:
            return ""

        # — Check LRU cache —
        cached = EasyLLMGGUF._mmproj_cache.get(resolved)
        if cached is not None:
            # Promote to most-recently-used by re-inserting
            del EasyLLMGGUF._mmproj_cache[resolved]
            EasyLLMGGUF._mmproj_cache[resolved] = cached
            return cached

        model_dir = os.path.dirname(resolved)
        model_basename = os.path.basename(resolved).lower()
        model_stem = os.path.splitext(model_basename)[0]

        # — TIER 0: GGUF metadata architecture matching —
        # Match `general.architecture` between model and mmproj files.
        # Falls through to Tier 1 on any failure.
        try:
            model_arch = EasyLLMGGUF._get_gguf_architecture(resolved)
            if model_arch:
                logging.info(
                    f"[LLM Chat GGUF] Model architecture detected: "
                    f"'{model_arch}' for {os.path.basename(resolved)}"
                )
                # Scan model's own directory only
                if model_dir and os.path.isdir(model_dir):
                    try:
                        for f in os.listdir(model_dir):
                            if "mmproj" in f.lower() and f.endswith(".gguf"):
                                mmproj_path = os.path.join(model_dir, f)
                                mmproj_arch = EasyLLMGGUF._get_gguf_architecture(mmproj_path)
                                if mmproj_arch and mmproj_arch == model_arch:
                                    logging.info(
                                        f"[LLM Chat GGUF] Auto-detected mmproj via "
                                        f"architecture match ('{model_arch}'): {mmproj_path}"
                                    )
                                    if len(EasyLLMGGUF._mmproj_cache) >= EasyLLMGGUF._MMPROJ_CACHE_MAX:
                                        oldest = next(iter(EasyLLMGGUF._mmproj_cache))
                                        del EasyLLMGGUF._mmproj_cache[oldest]
                                    EasyLLMGGUF._mmproj_cache[resolved] = mmproj_path
                                    return mmproj_path
                    except Exception:
                        pass
                logging.info(
                    f"[LLM Chat GGUF] No mmproj with matching architecture "
                    f"'{model_arch}' found — falling back to filename heuristics"
                )
        except Exception:
            logging.debug(
                f"[LLM Chat GGUF] Tier 0 architecture matching failed for "
                f"{os.path.basename(resolved)} — falling back to filename heuristics",
                exc_info=True
            )
        # Minimum prefix/substring thresholds to prevent false positives
        _MIN_MMPROJ_PREFIX = 4
        _MIN_MMPROJ_SUBSTRING = 4

        def _mmproj_score(mmproj_stem: str) -> int:
            """Score mmproj stem vs model stem by shared prefix + substring containment."""
            m_stem = mmproj_stem.replace("mmproj-", "").replace("mmproj_", "")
            if not m_stem:
                return 0

            # 1) Shared prefix length
            prefix_score = 0
            for i in range(min(len(model_stem), len(m_stem)), 0, -1):
                if model_stem.startswith(m_stem[:i]) or m_stem.startswith(model_stem[:i]):
                    prefix_score = i
                    break

            # Zero out prefix scores below minimum threshold
            if prefix_score < _MIN_MMPROJ_PREFIX:
                prefix_score = 0

            # 2) Substring containment bonus
            substring_bonus = 0
            if m_stem and (m_stem in model_stem or model_stem in m_stem):
                match_len = min(len(model_stem), len(m_stem))
                if match_len >= _MIN_MMPROJ_SUBSTRING:
                    substring_bonus = match_len

            return prefix_score + substring_bonus

        def _find_mmproj_in_dir(directory: str) -> str | None:
            """Scan a single directory for mmproj files, best-match first."""
            if not directory or not os.path.isdir(directory):
                return None
            try:
                candidates = []
                for f in os.listdir(directory):
                    if "mmproj" in f.lower() and f.endswith(".gguf"):
                        f_stem = os.path.splitext(f.lower())[0]
                        score = _mmproj_score(f_stem)
                        candidates.append((score, os.path.join(directory, f)))
                if candidates:
                    candidates.sort(key=lambda x: -x[0])  # highest score first
                    best_score, best_path = candidates[0]
                    if best_score > 0:
                        return best_path
            except Exception:
                pass
            return None

        result = ""

        # Tier 1: Scan model's own directory
        local_match = _find_mmproj_in_dir(model_dir)
        if local_match:
            logging.info(
                f"[LLM Chat GGUF] Auto-detected mmproj in model dir: {local_match}"
            )
            result = local_match

        # — Store in LRU cache —
        if len(EasyLLMGGUF._mmproj_cache) >= EasyLLMGGUF._MMPROJ_CACHE_MAX:
            oldest = next(iter(EasyLLMGGUF._mmproj_cache))
            del EasyLLMGGUF._mmproj_cache[oldest]
        EasyLLMGGUF._mmproj_cache[resolved] = result

        return result

    def _ui_result(self, text_val, raw_val, model_info=None, b64_image=None,
                   input_text="", system_prompt_val=""):
        """Wrap text outputs in ui dict + result tuple for ComfyUI OUTPUT_NODE."""
        ui_payload = {
            "text": [text_val],
            "raw_text": [raw_val],
        }
        if input_text:
            ui_payload["input_text"] = [input_text]
        if system_prompt_val:
            ui_payload["system_prompt"] = [system_prompt_val]
        if model_info:
            ui_payload["model_info"] = [model_info]
        if b64_image:
            ui_payload["image"] = [b64_image]
        return {
            "ui": ui_payload,
            "result": (text_val, raw_val),
        }

    def generate(self, model_path="", text="", max_length=768,
                 temperature=0.7, seed=0, system_prompt="",
                 prompt_template="Custom", system_prompt_text="",
                 mode="chat", text_input="", chat_template="qwen",
                 n_gpu_layers=-1, use_mlock=True, n_ctx=4096,
                 vram_mode="keep_loaded", top_k=50, top_p=0.9,
                 repetition_penalty=1.0, unique_id=None,
                 image=None, mmproj_path="", image_filename=""):
        """Generate text using the C++ llama.cpp inference engine.

        Loads GGUF model (cached if previously loaded), formats prompt
        via selected chat template, runs CUDA inference.

        Supports:
        - 4-source system prompt resolution
        - Chat mode (popup + history) and enhancer mode (direct output)
        - Chain text input via text_input forceInput socket
        - Multi-turn chat with history retrieval
        - Auto-mode replay via _cache_gguf
        - Streaming with progress events (when popup is open)
        - Vision/multimodal: image input + mmproj_path for vision models

        Returns:
            tuple: (cleaned_text, raw_text) or UI dict for enhancer mode
        """
        # — RESOLVE EFFECTIVE TEXT —
        effective_text = resolve_text(text_input, text)

        # — MODE-SPECIFIC BEHAVIOR —
        if mode == "enhancer":
            if not effective_text or not effective_text.strip():
                return self._ui_result("", "", input_text=effective_text)
        else:
            # Chat mode: return cached output when text is empty (auto-mode)
            if not effective_text or not effective_text.strip():
                cached = _cache_gguf.get(unique_id, None)
                if cached:
                    cleaned, raw = cached
                    return self._ui_result(cleaned, raw, input_text=effective_text)
                return self._ui_result("", "", input_text=effective_text)

        # — VALIDATE MODEL PATH —
        if not model_path or not model_path.strip():
            return self._ui_result(
                NO_MODEL_PATH_ERROR, NO_MODEL_PATH_ERROR,
                input_text=effective_text,
            )

        seed = auto_seed(seed)

        # — DIAG: Track consecutive Queue Prompt runs —
        type(self)._gguf_call_count += 1
        run_num = self._gguf_call_count
        logging.info(
            f"[DIAG] GGUF generate call #{run_num} "
            f"(vram_mode={vram_mode}, model_cache_size={len(self._model_cache)})"
        )

        try:
            # — IMPORT BACKEND (graceful fallback if not installed) —
            try:
                from .llama_cpp_backend import LlamaCppModel, LLAMA_CPP_AVAILABLE
                if not LLAMA_CPP_AVAILABLE:
                    return self._ui_result(
                        LLAMA_NOT_INSTALLED_ERROR, LLAMA_NOT_INSTALLED_ERROR,
                        input_text=effective_text,
                    )
            except ImportError:
                return self._ui_result(
                    "ERROR: llama_cpp_backend.py not found.\n\n"
                    "Ensure the file exists in the EasyLLM custom_node directory.",
                    "ERROR: llama_cpp_backend.py not found.",
                    input_text=effective_text,
                )

            # — GET OR CREATE CACHED MODEL —
            # Text-only path uses empty mmproj_path; re-acquired with mmproj if image present.
            model = self._get_cached_model(
                model_path, "", n_gpu_layers, use_mlock, n_ctx
            )

            # — RESOLVE SYSTEM PROMPT (4-source resolution) —
            effective_system_prompt = resolve_system_prompt(
                system_prompt, prompt_template, system_prompt_text
            )

            # — RETRIEVE CHAT HISTORY (needed for text and multimodal paths) —
            parsed_history = None
            try:
                from .streaming import get_chat_history
                parsed_history = get_chat_history(unique_id)
            except Exception as e:
                logging.warning(
                    f"[LLM Chat GGUF] Failed to retrieve history for node {unique_id}: {e}"
                )

            # — IMAGE INPUT HANDLING (vision-language / multimodal) —
            b64_image = None
            detected_template = None
            has_image = image is not None
            uploaded_file = image_filename.strip() if not has_image else None

            if has_image:
                # — AUTO-DETECT mmproj FILE (lazy fallback) —
                if not mmproj_path or not mmproj_path.strip():
                    mmproj_path = self._auto_detect_mmproj(model_path)

                # — VALIDATE mmproj_path —
                if not mmproj_path or not mmproj_path.strip():
                    return self._ui_result(
                        "ERROR: Image input requires a multimodal projection "
                        "file mmproj_path.\n\n"
                        "Set mmproj_path to the path of your .gguf mmproj "
                        "file, e.g.,\n"
                        "  llava-v1.5-7b-mmproj.gguf\n\n"
                        "The mmproj file is a small GGUF file that maps image "
                        "embeddings to the language model's text embedding "
                        "space. It's usually named like your main model with "
                        "-mmproj in the filename.\n\n"
                        "💡 TIP: Place a *mmproj*.gguf file in the same "
                        "directory as your main model and it will be "
                        "auto-detected.",
                        "ERROR: mmproj_path required for image input.",
                        input_text=effective_text,
                    )

                # — RE-ACQUIRE MODEL WITH mmproj PARAMETER —
                model = self._get_cached_model(
                    model_path, mmproj_path, n_gpu_layers, use_mlock, n_ctx
                )

                # — CHAT TEMPLATE DETECTION FOR VISION PATH —
                # Apply same 3-tier detection as the text-only path to derive
                # proper stop tokens for create_chat_completion(), preventing
                # template token bleeding (USER:/ASSISTANT: generated as text).
                _vision_chat_template = chat_template
                _vision_stop_tokens = None
                if _vision_chat_template == "auto":
                    _detected = None
                    try:
                        _detected = detect_chat_template_from_metadata(
                            model.metadata
                        )
                    except Exception:
                        pass
                    if not _detected:
                        try:
                            _detected = detect_template_from_architecture(
                                model.metadata
                            )
                        except Exception:
                            pass
                    if not _detected:
                        try:
                            _detected = heuristic_template_from_filename(
                                model_path
                            )
                        except Exception:
                            pass
                    if not _detected:
                        _detected = "llama"
                        logging.info(
                            f"[LLM Chat GGUF] Vision path: all detection "
                            f"tiers failed — falling back to '{_detected}'"
                        )
                    _vision_chat_template = _detected
                    detected_template = _detected

                _vision_template_config = CHAT_TEMPLATES.get(
                    _vision_chat_template
                )
                if _vision_template_config:
                    _vision_stop_tokens = list(_vision_template_config["stop"])
                    _vision_stop_tokens.extend(_UNIVERSAL_ROLE_STOP_TOKENS)
                    logging.info(
                        f"[LLM Chat GGUF] Vision path: using template "
                        f"'{_vision_chat_template}', stop tokens: "
                        f"{_vision_stop_tokens}"
                    )

                # — CONVERT IMAGE TENSOR TO BASE64 PNG —
                try:
                    from .utils import tensor_to_base64_png
                    b64_image = tensor_to_base64_png(image)
                except Exception as e:
                    return self._ui_result(
                        f"ERROR: Failed to convert image to base64: {e}\n\n"
                        "Ensure the image input is a valid ComfyUI IMAGE "
                        "tensor (from a Load Image node).",
                        f"ERROR: Image conversion failed: {e}",
                        input_text=effective_text,
                    )

                # — BUILD OPENAI-COMPATIBLE MESSAGES ARRAY —
                # create_chat_completion() handles chat templates internally.
                messages = []

                if effective_system_prompt:
                    messages.append({
                        "role": "system",
                        "content": effective_system_prompt
                    })

                # Text-only history; image turns are not persisted.
                if parsed_history:
                    for entry in parsed_history:
                        role = entry.get("role", "user")
                        msg = entry.get("message", "")
                        messages.append({"role": role, "content": msg})

                user_content = [
                    {"type": "image_url",
                     "image_url": {"url": b64_image}},
                    {"type": "text", "text": effective_text}
                ]
                messages.append({"role": "user", "content": user_content})

                # — DIAGNOSTIC: Log messages array structure —
                _has_image_content = any(
                    isinstance(m.get("content"), list) and any(
                        c.get("type") == "image_url" for c in m["content"]
                    )
                    for m in messages
                )
                logging.info(
                    f"[DIAG] Multimodal messages: len={len(messages)}, "
                    f"roles={[m['role'] for m in messages]}, "
                    f"b64_image_len={len(b64_image) if b64_image else 0}, "
                    f"has_image_url={_has_image_content}"
                )

                logging.info(
                    f"[LLM Chat GGUF] Generating {max_length} tokens "
                    f"(mode={mode}, temp={temperature}, top_k={top_k}, "
                    f"top_p={top_p}, multimodal=True)"
                )

                # — CHECK IF POPUP IS OPEN (streaming path) —
                is_streaming = mode == "chat"
                if is_streaming:
                    try:
                        from .streaming import is_popup_mode
                        from .generation_state import get_state
                        _state = get_state()
                        is_streaming = is_popup_mode(unique_id)
                        logging.info(
                            f"[DIAG] GGUF chat stream check: "
                            f"unique_id={unique_id!r}, "
                            f"mode={mode!r}, "
                            f"popup_active_nodes={_state.popup_active_nodes}, "
                            f"is_streaming={is_streaming}"
                        )
                    except Exception as e:
                        logging.warning(
                            f"[DIAG] GGUF chat is_popup_mode exception for "
                            f"unique_id={unique_id!r}: {e}"
                        )

                if is_streaming:
                    # — STREAMING MULTIMODAL PATH (popup open) —
                    from .streaming import execute_gguf_chat_generation
                    raw_text = execute_gguf_chat_generation(
                        model=model,
                        messages=messages,
                        mode=mode,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        repetition_penalty=repetition_penalty,
                        stop=_vision_stop_tokens,
                        unique_id=unique_id,
                    )
                else:
                    # — BLOCKING MULTIMODAL PATH (no popup, or enhancer) —
                    raw_text = model.generate_chat(
                        messages=messages,
                        max_tokens=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        repetition_penalty=repetition_penalty,
                        stop=_vision_stop_tokens,
                    )

                multimodal_mode = True

            elif uploaded_file:
                # — No-wire chat mode: image uploaded from popup —
                # — AUTO-DETECT mmproj FILE (lazy fallback) —
                if not mmproj_path or not mmproj_path.strip():
                    mmproj_path = self._auto_detect_mmproj(model_path)

                # — VALIDATE mmproj_path —
                if not mmproj_path or not mmproj_path.strip():
                    return self._ui_result(
                        "ERROR: Image input requires a multimodal projection "
                        "file mmproj_path.\n\n"
                        "Set mmproj_path to the path of your .gguf mmproj "
                        "file, e.g.,\n"
                        "  llava-v1.5-7b-mmproj.gguf\n\n"
                        "The mmproj file is a small GGUF file that maps image "
                        "embeddings to the language model's text embedding "
                        "space.\n\n"
                        "💡 TIP: Place a *mmproj*.gguf file in the same "
                        "directory as your main model and it will be "
                        "auto-detected.",
                        "ERROR: mmproj_path required for image input.",
                        input_text=effective_text,
                    )

                # — RE-ACQUIRE MODEL WITH mmproj PARAMETER —
                model = self._get_cached_model(
                    model_path, mmproj_path, n_gpu_layers, use_mlock, n_ctx
                )

                # — CHAT TEMPLATE DETECTION FOR NO-WIRE VISION PATH —
                _vision_chat_template = chat_template
                _vision_stop_tokens = None
                if _vision_chat_template == "auto":
                    _detected = None
                    try:
                        _detected = detect_chat_template_from_metadata(
                            model.metadata
                        )
                    except Exception:
                        pass
                    if not _detected:
                        try:
                            _detected = detect_template_from_architecture(
                                model.metadata
                            )
                        except Exception:
                            pass
                    if not _detected:
                        try:
                            _detected = heuristic_template_from_filename(
                                model_path
                            )
                        except Exception:
                            pass
                    if not _detected:
                        _detected = "llama"
                        logging.info(
                            f"[LLM Chat GGUF] No-wire vision path: all "
                            f"detection tiers failed — falling back to "
                            f"'{_detected}'"
                        )
                    _vision_chat_template = _detected
                    detected_template = _detected

                _vision_template_config = CHAT_TEMPLATES.get(
                    _vision_chat_template
                )
                if _vision_template_config:
                    _vision_stop_tokens = list(_vision_template_config["stop"])
                    _vision_stop_tokens.extend(_UNIVERSAL_ROLE_STOP_TOKENS)
                    logging.info(
                        f"[LLM Chat GGUF] No-wire vision path: using "
                        f"template '{_vision_chat_template}', stop tokens: "
                        f"{_vision_stop_tokens}"
                    )

                # — CONVERT FILE TO BASE64 —
                try:
                    from .utils import load_image_to_base64
                    b64_image = load_image_to_base64(uploaded_file, max_size=1024)
                except FileNotFoundError as e:
                    return self._ui_result(
                        f"ERROR: Uploaded image not found: {e}\n\n"
                        "The image file may have been deleted from ComfyUI's "
                        "input/ directory. Please re-attach the image.",
                        f"ERROR: Uploaded image not found: {e}",
                        input_text=effective_text,
                    )
                except Exception as e:
                    return self._ui_result(
                        f"ERROR: Failed to load uploaded image: {e}",
                        f"ERROR: Image load failed: {e}",
                        input_text=effective_text,
                    )

                # — BUILD OPENAI-COMPATIBLE MESSAGES ARRAY —
                messages = []

                if effective_system_prompt:
                    messages.append({
                        "role": "system",
                        "content": effective_system_prompt
                    })

                if parsed_history:
                    for entry in parsed_history:
                        role = entry.get("role", "user")
                        msg = entry.get("message", "")
                        messages.append({"role": role, "content": msg})

                user_content = [
                    {"type": "image_url",
                     "image_url": {"url": b64_image}},
                    {"type": "text", "text": effective_text}
                ]
                messages.append({"role": "user", "content": user_content})

                # — DIAGNOSTIC: Log messages array structure —
                _has_image_content = any(
                    isinstance(m.get("content"), list) and any(
                        c.get("type") == "image_url" for c in m["content"]
                    )
                    for m in messages
                )
                logging.info(
                    f"[DIAG] No-wire image upload messages: len={len(messages)}, "
                    f"roles={[m['role'] for m in messages]}, "
                    f"b64_image_len={len(b64_image) if b64_image else 0}, "
                    f"has_image_url={_has_image_content}, "
                    f"uploaded_file={uploaded_file}"
                )

                logging.info(
                    f"[LLM Chat GGUF] Generating {max_length} tokens "
                    f"(mode={mode}, temp={temperature}, top_k={top_k}, "
                    f"top_p={top_p}, multimodal=uploaded)"
                )

                # — CHECK IF POPUP IS OPEN (streaming path) —
                is_streaming = mode == "chat"
                if is_streaming:
                    try:
                        from .streaming import is_popup_mode
                        from .generation_state import get_state
                        _state = get_state()
                        is_streaming = is_popup_mode(unique_id)
                    except Exception as e:
                        logging.warning(
                            f"[DIAG] No-wire is_popup_mode exception for "
                            f"unique_id={unique_id!r}: {e}"
                        )

                if is_streaming:
                    # — STREAMING MULTIMODAL PATH (popup open) —
                    from .streaming import execute_gguf_chat_generation
                    raw_text = execute_gguf_chat_generation(
                        model=model,
                        messages=messages,
                        mode=mode,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        repetition_penalty=repetition_penalty,
                        stop=_vision_stop_tokens,
                        unique_id=unique_id,
                    )
                else:
                    # — BLOCKING MULTIMODAL PATH (no popup, or enhancer) —
                    raw_text = model.generate_chat(
                        messages=messages,
                        max_tokens=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        repetition_penalty=repetition_penalty,
                        stop=_vision_stop_tokens,
                    )

                multimodal_mode = True

            else:
                # — TEXT-ONLY PATH —
                multimodal_mode = False

                # — AUTO-DETECT CHAT TEMPLATE (3-TIER PIPELINE) —
                # Runs when user has "auto" selected (default).
                detected_template = None
                if chat_template == "auto":
                    _t0 = time.perf_counter()

                    # Tier 1: Read tokenizer.chat_template from GGUF metadata
                    try:
                        detected_template = detect_chat_template_from_metadata(
                            model.metadata
                        )
                        if detected_template:
                            logging.info(
                                f"[LLM Chat GGUF] [Tier 1] Auto-detected "
                                f"chat template '{detected_template}' "
                                f"from GGUF metadata"
                            )
                    except Exception as e:
                        logging.debug(
                            f"[LLM Chat GGUF] [Tier 1] Metadata detection "
                            f"failed: {e}"
                        )

                    # Tier 2: Fall back to general.architecture mapping
                    if not detected_template:
                        try:
                            detected_template = detect_template_from_architecture(
                                model.metadata
                            )
                            if detected_template:
                                logging.info(
                                    f"[LLM Chat GGUF] [Tier 2] Auto-detected "
                                    f"chat template '{detected_template}' "
                                    f"from model architecture"
                                )
                        except Exception as e:
                            logging.debug(
                                f"[LLM Chat GGUF] [Tier 2] Architecture "
                                f"detection failed: {e}"
                            )

                    # Tier 3: Last resort — filename heuristics
                    if not detected_template:
                        try:
                            detected_template = heuristic_template_from_filename(
                                model_path
                            )
                            if detected_template:
                                logging.info(
                                    f"[LLM Chat GGUF] [Tier 3] Auto-detected "
                                    f"chat template '{detected_template}' "
                                    f"from filename heuristics"
                                )
                        except Exception as e:
                            logging.debug(
                                f"[LLM Chat GGUF] [Tier 3] Filename heuristic "
                                f"failed: {e}"
                            )

                    # Final fallback: use "llama" (safer than "qwen" as most
                    # modern models use Llama-style tokenization)
                    if not detected_template:
                        detected_template = "llama"
                        logging.info(
                            f"[LLM Chat GGUF] All auto-detection tiers "
                            f"failed — falling back to '{detected_template}'"
                        )

                    chat_template = detected_template
                    _t1 = time.perf_counter()
                    if (_t1 - _t0) * 1000 > 5:
                        logging.info(
                            f"[DIAG] Template detection (3-tier) took "
                            f"{(_t1 - _t0)*1000:.1f}ms"
                        )

                # — GET STOP TOKENS FROM SELECTED CHAT TEMPLATE —
                template_config = CHAT_TEMPLATES.get(chat_template)
                if template_config is None:
                    logging.warning(
                        f"[LLM Chat GGUF] Unknown chat template "
                        f"'{chat_template}' — falling back to 'qwen'"
                    )
                    template_config = CHAT_TEMPLATES["qwen"]
                stop_tokens = template_config["stop"]

                # — FORMAT PROMPT USING SELECTED CHAT TEMPLATE —
                formatted_prompt = format_prompt_by_template(
                    user_text=effective_text,
                    system_prompt=effective_system_prompt,
                    template_name=chat_template,
                    history=parsed_history,
                )

                logging.info(
                    f"[LLM Chat GGUF] Generating {max_length} tokens "
                    f"(mode={mode}, temp={temperature}, top_k={top_k}, "
                    f"top_p={top_p}, template={chat_template})"
                )

                # — CHECK IF POPUP IS OPEN (streaming path) —
                is_streaming = mode == "chat"
                if is_streaming:
                    try:
                        from .streaming import is_popup_mode
                        from .generation_state import get_state
                        _state = get_state()
                        is_streaming = is_popup_mode(unique_id)
                        logging.info(
                            f"[DIAG] GGUF stream check: "
                            f"unique_id={unique_id!r}, "
                            f"mode={mode!r}, "
                            f"popup_active_nodes={_state.popup_active_nodes}, "
                            f"is_streaming={is_streaming}"
                        )
                    except Exception as e:
                        logging.warning(
                            f"[DIAG] GGUF is_popup_mode exception for "
                            f"unique_id={unique_id!r}: {e}"
                        )

                if is_streaming:
                    # — STREAMING PATH (popup open) —
                    from .streaming import execute_gguf_generation
                    raw_text = execute_gguf_generation(
                        model=model,
                        formatted_prompt=formatted_prompt,
                        mode=mode,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        stop=stop_tokens,
                        repetition_penalty=repetition_penalty,
                        unique_id=unique_id,
                    )
                else:
                    # — BLOCKING PATH (no popup, or enhancer) —
                    raw_text = model.generate(
                        prompt=formatted_prompt,
                        max_tokens=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        stop=stop_tokens,
                        repetition_penalty=repetition_penalty,
                    )

            # — CLEAN GENERATED TEXT —
            cleaned_text = clean_generated_text(raw_text)

            # — COLLECT MODEL INFO FROM GGUF METADATA —
            model_info = {}
            try:
                model_meta = getattr(model, "metadata", {})
                if model_meta:
                    model_info = {
                        "architecture": model_meta.get("general.architecture", ""),
                        "name": model_meta.get("general.name", ""),
                        "description": model_meta.get("general.description", ""),
                        "context_length": str(model_meta.get(
                            "llama.context_length",
                            model_meta.get("llama.max_position_embeddings", ""),
                        )),
                    }
            except Exception:
                pass

            if detected_template:
                model_info["detected_template"] = detected_template

            # Fallback: read architecture via GGUF header reader if not in loaded metadata
            if not model_info.get("architecture"):
                try:
                    arch = EasyLLMGGUF._get_gguf_architecture(model_path)
                    if arch:
                        model_info["architecture"] = arch
                except Exception:
                    pass

            # — UPDATE MODEL INDEX CACHE WITH TRUE ARCHITECTURE —
            # The cache may have guessed from filename prefix; update now
            # that true architecture is known from loaded model metadata.
            true_arch = model_info.get("architecture")
            if true_arch:
                try:
                    from .utils import _update_cached_architecture
                    _update_cached_architecture(
                        model_path,
                        true_arch,
                        context_length=model_info.get("context_length", ""),
                        description=model_info.get("description", ""),
                        model_name=model_info.get("name", ""),
                    )
                except Exception:
                    pass  # non-critical

            # — CACHE FOR AUTO-MODE REPLAY —
            if mode == "chat":
                _cache_gguf[unique_id] = (cleaned_text, raw_text)

            ui_payload = {
                "text": [cleaned_text],
                "raw_text": [raw_text],
            }
            if model_info.get("architecture") or model_info.get("name"):
                ui_payload["model_info"] = [model_info]
            if b64_image:
                ui_payload["image"] = [b64_image]

            # — ENHANCER MODE RETURN —
            if mode == "enhancer":
                ui_payload["input_text"] = [effective_text]
                ui_payload["system_prompt"] = [effective_system_prompt]
                ui_payload["enhancer"] = [json.dumps({
                    "input": effective_text,
                    "output": cleaned_text,
                    "system_prompt": effective_system_prompt,
                })]
                return {
                    "ui": ui_payload,
                    "result": (cleaned_text, raw_text),
                }

            # — CHAT MODE RETURN —
            return {
                "ui": ui_payload,
                "result": (cleaned_text, raw_text),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            err_str = str(e)

            # — CLASSIFY ERROR FOR USER-FRIENDLY MESSAGES —
            err_lower = err_str.lower()
            if any(oom in err_lower for oom in [
                "cuda out of memory", "cudamalloc", "outofmemory",
                "cublas_status_not_supported", "cublas_status_alloc_failed",
            ]):
                err_msg = (
                    "🚨 CUDA OUT OF MEMORY\n\n"
                    "The GGUF model requires more GPU memory than available.\n\n"
                    "Try:\n"
                    "  • Set n_gpu_layers lower (e.g., 24 instead of -1)\n"
                    "  • Use a smaller model or lower quantization (Q4_K_M, Q3_K_M)\n"
                    "  • Close other GPU-intensive apps\n"
                    "  • Check ComfyUI's VRAM usage in the console\n"
                    "  • Set vram_mode to 'unload' (not 'keep_loaded')\n"
                )
            elif any(invalid in err_lower for invalid in [
                "not a valid gguf", "magic number", "invalid file",
                "unexpected file", "corrupted", "not a gguf",
            ]):
                err_msg = (
                    "❌ INVALID OR CORRUPT GGUF FILE\n\n"
                    f"'{model_path}' is not a valid GGUF model file.\n\n"
                    "Try:\n"
                    "  • Download a new copy of the file\n"
                    "  • Verify the file hash matches the source\n"
                    "  • Use a Q4_K_M quantized model from a trusted source\n"
                    "  • Check the file extension is .gguf\n"
                )
            elif any(dl in err_lower for dl in [
                "connection refused", "name or service not known",
                "temporary failure", "connection reset",
                "404", "403", "timeout", "cannot connect",
            ]):
                err_msg = (
                    "🌐 DOWNLOAD / NETWORK ERROR\n\n"
                    "Failed to download or access the model file.\n\n"
                    "Try:\n"
                    "  • Check your internet connection\n"
                    "  • Verify the model path / URL is correct\n"
                    "  • Download the file manually first via HuggingFace\n"
                    "  • Some providers require authentication (tokens)\n"
                )
            elif "model not loaded" in err_lower:
                err_msg = (
                    "⚠️ MODEL NOT LOADED\n\n"
                    "The GGUF model failed to load.\n\n"
                    "Try:\n"
                    "  • Check the console for detailed error logs\n"
                    "  • Verify the model file is compatible with your "
                    "llama-cpp-python version\n"
                    "  • Try a different quantization format (Q4_K_M recommended)\n"
                )
            elif "architecture_mismatch" in err_lower or (
                "tensor shape mismatch" in err_lower
                and "mmproj" in err_lower
            ):
                err_msg = (
                    "🖼️ MMPROJ / MODEL ARCHITECTURE MISMATCH\n\n"
                    "The multimodal projection (mmproj) file is not "
                    "compatible with this model.\n"
                    f"  {err_str}\n\n"
                    "Try:\n"
                    "  • Ensure the mmproj file matches your model's "
                    "architecture (e.g., use the mmproj published by the\n"
                    "    same author as the model)\n"
                    "  • For LLaVA models, use llava-v1.5-7b-mmproj.gguf "
                    "with LLaVA 7B models\n"
                    "  • For Qwen-VL models, use the matching qwen-vl-mmproj.gguf\n"
                    "  • Check the ComfyUI console for the full error details\n"
                )
            elif "multimodal projection file not found" in err_lower:
                err_msg = (
                    f"🖼️ MMPROJ FILE NOT FOUND\n\n"
                    f"{err_str}\n\n"
                    f"💡 How to fix:\n"
                    f"  • Set mmproj_path to the full absolute path of your "
                    f".gguf mmproj file\n"
                    f"  • Or just use the filename (e.g., "
                    f"llava-v1.5-7b-mmproj.gguf)\n"
                    f"    and the node will search registered model folders\n"
                    f"  • The mmproj file is usually named like your main "
                    f"model with -mmproj in the filename"
                )
            elif "gguf model file not found" in err_lower:
                err_msg = (
                    f"❌ GGUF MODEL FILE NOT FOUND\n\n"
                    f"{err_str}\n\n"
                    f"💡 How to fix:\n"
                    f"  • Set model_path to the full absolute path of your .gguf file\n"
                    f"  • Or just use the filename (e.g., Qwen3-8B-Q8_0.gguf)\n"
                    f"    and the node will search registered model folders\n"
                    f"  • Use the Browse button in the popup to locate the file"
                )
            elif (
                "0xc000001d" in err_str
                or "-1073741795" in err_str
                or "STATUS_ILLEGAL_INSTRUCTION" in err_str
            ):
                err_msg = (
                    "🚨 CPU INSTRUCTION SET INCOMPATIBILITY\n\n"
                    "The installed llama-cpp-python wheel was compiled with CPU "
                    "instructions not supported by your processor.\n"
                    f"  Error: {err_str}\n\n"
                    "This is common when a pre-built wheel is compiled for newer "
                    "CPUs (AVX2, AVX-VNNI, AVX512) but your processor is older.\n\n"
                    "An automatic downgrade to the safe baseline (v0.3.23) was attempted.\n"
                    "Check the ComfyUI console to see if the auto-fix succeeded.\n\n"
                    "Manual solutions (try in order):\n"
                    "  1️⃣ Reinstall with CPU-only backend (most compatible):\n"
                    "       python install.py --backend cpu\n\n"
                    "  2️⃣ Or compile from source with conservative flags:\n"
                    "       set CMAKE_ARGS=-DGGML_AVX2=OFF -DGGML_FMA=OFF\n"
                    "       pip install llama-cpp-python --force-reinstall\n\n"
                    "  3️⃣ Or use a different backend (Vulkan):\n"
                    "       python install.py --backend vulkan\n\n"
                    "💡 For help, check the ComfyUI console for the full traceback."
                )
            elif "TemplateSyntaxError" in err_str or "unknown tag" in err_str or \
                 ("template" in err_lower and "generation" in err_lower):
                err_msg = (
                    "⚠️ CHAT TEMPLATE INCOMPATIBILITY\n\n"
                    "This model's GGUF metadata contains a custom Jinja2 tag "
                    "('{% generation %}') that is not supported by the installed "
                    "version of llama-cpp-python. "
                    "This is common with Dolphin-based models.\n\n"
                    "Automatic workaround applied:\n"
                    "  • The built-in chat template was suppressed\n"
                    "  • Text-only chat uses format_prompt_by_template() instead\n"
                    "  • Text-only mode works correctly\n\n"
                    "To fully resolve:\n"
                    "  • Upgrade llama-cpp-python:\n"
                    "    pip install --upgrade llama-cpp-python\n"
                    "  • Or switch to a non-Dolphin model variant\n"
                    "  • Or continue using text-only mode (works fine)"
                )
            else:
                err_msg = (
                    f"❌ GENERATION ERROR\n\n"
                    f"Unexpected error during GGUF inference:\n"
                    f"  {err_str}\n\n"
                    f"Check the ComfyUI console for the full traceback."
                )

            return self._ui_result(
                err_msg, err_msg,
                input_text=effective_text,
            )

        finally:
            if vram_mode in ("unload", "aggressive_free"):
                from .debug_profiler import profiler
                profiler.begin("model_unload")
                # Iterate all keys to match composite (model_path, mmproj_path) tuples
                for key in list(self._model_cache.keys()):
                    if isinstance(key, tuple) and key[0] == model_path:
                        cached = self._model_cache[key]
                        cached.unload()
                        del self._model_cache[key]
                        logging.info(
                            f"[LLM Chat GGUF] Model unloaded (vram_mode={vram_mode}, "
                            f"mmproj={key[1] or 'none'})"
                        )
                profiler.end("model_unload")

            # Aggressive VRAM cleanup after unload
            if vram_mode == "aggressive_free":
                try:
                    import comfy.model_management
                    comfy.model_management.soft_empty_cache(force=True)
                    logging.debug(
                        "[LLM Chat GGUF] Aggressive VRAM cleanup done — "
                        "soft_empty_cache(force=True)"
                    )
                except Exception:
                    pass

# Node registration mappings
NODE_CLASS_MAPPINGS = {
    "EasyLLM": EasyLLM,
    "EasyLLMText": EasyLLMText,
    "EasyLLMGGUF": EasyLLMGGUF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyLLM": "🤖 EasyLLM",
    "EasyLLMText": "🤖 EasyLLM Text Display",
    "EasyLLMGGUF": "⚡ EasyLLM GGUF",
}
