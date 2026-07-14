"""
EasyLLM for ComfyUI
===================

Chat and prompt enhancement using the LLM model already loaded inside your CLIP text encoder.

No external dependencies, no separate model server needed.
Uses the existing BaseGenerate infrastructure built into ComfyUI's Qwen-based text encoders.

Models supported:
- Anima (Qwen3-0.6B)
- Z-Image (Qwen3-4B)
- Flux Klein (Qwen3-4B / Qwen3-8B)
- Qwen-Image (Qwen2.5-7B-VL)

GPU-Vectorized Optimizations:
- GPU-vectorized sample_token (no Python loops for repetition_penalty)
- TF32 matmul precision (1.2-1.5x on Ampere+ GPUs)
- torch.inference_mode() (faster than no_grad)
- torch.compile() of transformer forward pass
- Selective quantized forward path (Q8_0 native, NVFP4 full-precision)
"""

import logging
import time

from .debug_profiler import profiler

profiler.begin("module_import_total")

_t0 = time.perf_counter()
from .chat_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
profiler.record_module_import("chat_node", time.perf_counter() - _t0)

_t0 = time.perf_counter()
from .prompt_manager import setup_routes
profiler.record_module_import("prompt_manager", time.perf_counter() - _t0)

_t0 = time.perf_counter()
from .streaming import setup_streaming_routes
profiler.record_module_import("streaming", time.perf_counter() - _t0)

# GPU-vectorized sample_token patch (replaces BaseGenerate.sample_token).
_t0 = time.perf_counter()
from .cuda_optimizations import apply_vectorized_sample_patch
apply_vectorized_sample_patch()
profiler.record_module_import("cuda_optimizations", time.perf_counter() - _t0)

# Register API routes for the system prompt management dialog
setup_routes()

# Register WebSocket streaming routes for popup chat UI.
setup_streaming_routes()

# Initialize the on-disk JSON history database (idempotent).
# Creates easyllm_db/ directory tree and loads/rebuilds the index.
_t0 = time.perf_counter()
from .history_db import initialize as init_history_db
init_history_db()
profiler.record_module_import("history_db", time.perf_counter() - _t0)

# Register text utility nodes (📝 EasyLLM Text Input, 🔗 EasyLLM Text Joiner, 🧹 EasyLLM Text Cleaner)
_t0 = time.perf_counter()
from .text_utils import NODE_CLASS_MAPPINGS as TEXT_UTILS_MAPPINGS
from .text_utils import NODE_DISPLAY_NAME_MAPPINGS as TEXT_UTILS_DISPLAY
NODE_CLASS_MAPPINGS.update(TEXT_UTILS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(TEXT_UTILS_DISPLAY)
profiler.record_module_import("text_utils", time.perf_counter() - _t0)

# Register prompt select node (📚 EasyLLM Prompt Select)
_t0 = time.perf_counter()
from .prompt_select_node import NODE_CLASS_MAPPINGS as PROMPT_SELECT_MAPPINGS
from .prompt_select_node import NODE_DISPLAY_NAME_MAPPINGS as PROMPT_SELECT_DISPLAY
NODE_CLASS_MAPPINGS.update(PROMPT_SELECT_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROMPT_SELECT_DISPLAY)
profiler.record_module_import("prompt_select_node", time.perf_counter() - _t0)

# Register trigger router node (🎛️ EasyLLM Trigger Router)
_t0 = time.perf_counter()
from .trigger_router_node import NODE_CLASS_MAPPINGS as TRIGGER_ROUTER_MAPPINGS
from .trigger_router_node import NODE_DISPLAY_NAME_MAPPINGS as TRIGGER_ROUTER_DISPLAY
NODE_CLASS_MAPPINGS.update(TRIGGER_ROUTER_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(TRIGGER_ROUTER_DISPLAY)
profiler.record_module_import("trigger_router_node", time.perf_counter() - _t0)

# Register EasyLLM Image Capture node (🖼️ EasyLLM Image Capture)
_t0 = time.perf_counter()
from .image_capture_node import NODE_CLASS_MAPPINGS as IMAGE_CAPTURE_MAPPINGS
from .image_capture_node import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_CAPTURE_DISPLAY
NODE_CLASS_MAPPINGS.update(IMAGE_CAPTURE_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_CAPTURE_DISPLAY)
profiler.record_module_import("image_capture_node", time.perf_counter() - _t0)

profiler.end("module_import_total")

# Web directory for frontend JavaScript extensions
WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
