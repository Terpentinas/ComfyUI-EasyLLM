"""
Memory management utilities.

Provides VRAM optimization and OOM-safe model loading for GPU-constrained environments.
"""

import gc
import logging
import os
import types
from typing import Optional

import torch
import comfy.ops as comfy_ops


# ── Shared VRAM/RAM Optimization Utilities ──────────────────────────


def defragment_vram():
    """Maximum VRAM cleanup before model loading.

    Calls gc.collect(), torch.cuda.empty_cache(), and torch.cuda.synchronize()
    to free contiguous VRAM. Logs VRAM state before/after.
    """
    if not torch.cuda.is_available():
        return
    allocated_before = torch.cuda.memory_allocated() / (1024**3)
    reserved_before = torch.cuda.memory_reserved() / (1024**3)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    allocated_after = torch.cuda.memory_allocated() / (1024**3)
    reserved_after = torch.cuda.memory_reserved() / (1024**3)
    logging.info(
        f"[LLM Chat] VRAM defrag: "
        f"{allocated_before:.1f}→{allocated_after:.1f} GiB allocated, "
        f"{reserved_before:.1f}→{reserved_after:.1f} GiB reserved"
    )


def query_vram_state() -> dict:
    """Query GPU VRAM state, returning a structured dict.

    Centralizes torch.cuda.memory_* calls to avoid duplication across consumers.

    Returns:
        dict with keys:
            - total_bytes: Total GPU memory in bytes (0 if CUDA unavailable)
            - allocated_bytes: Currently allocated memory in bytes
            - reserved_bytes: Reserved (cached allocator) memory in bytes
            - free_bytes: Total minus allocated in bytes
            - pressure_ratio: allocated_bytes / total_bytes (1.0 if unavailable)
    """
    state = {
        "total_bytes": 0,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "free_bytes": 0,
        "pressure_ratio": 1.0,
    }
    if not torch.cuda.is_available():
        return state
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        state["total_bytes"] = total
        state["allocated_bytes"] = allocated
        state["reserved_bytes"] = reserved
        state["free_bytes"] = total - allocated
        state["pressure_ratio"] = allocated / total if total > 0 else 1.0
    except Exception:
        pass
    return state


# ── VRAM Estimation for free_memory() ──────────────────────────────


def estimate_vram_needed(
    *,
    transformer: Optional[torch.nn.Module] = None,
    n_ctx: int = 2048,
    model_path: Optional[str] = None,
    n_gpu_layers: int = -1,
    use_layer_offloading: bool = False,
) -> int:
    """Estimate VRAM needed for comfy.model_management.free_memory().

    Three strategies depending on available inputs:

    **PyTorch path** (``transformer`` provided):
        Sums ``p.numel() * p.element_size()`` for exact weight bytes.
        Adds estimated KV cache from model config plus 15% safety margin.

    **GGUF path** (``model_path`` provided):
        Uses ``os.path.getsize()`` as proxy. Multiplies by 1.3 to account
        for KV cache, engine overhead, and buffers. Scales by ``n_gpu_layers``.

    **Layer offloading** (``use_layer_offloading=True``):
        Returns fixed ~1.5 GiB estimate (embeddings + 1 layer + KV cache).

    Args:
        transformer: PyTorch transformer model.
        n_ctx: Context window size for KV cache estimation.
        model_path: Path to a .gguf file.
        n_gpu_layers: GPU layer count (-1=all, 0=CPU, >0=partial).
        use_layer_offloading: Layer-by-layer offloading active.

    Returns:
        int: Estimated VRAM in bytes. Clamped 1-24 GiB.
    """
    # ── Layer offloading path ─────────────────────────────────────────
    # Only embeddings (~100 MB), 1 transformer layer, and KV cache are GPU-resident
    if use_layer_offloading:
        return int(1.5 * 1024**3)

    # ── GGUF path ─────────────────────────────────────────────────────
    if model_path is not None:
        try:
            file_size = os.path.getsize(model_path)
        except (OSError, FileNotFoundError) as e:
            logging.warning(
                f"[LLM Chat] Cannot determine size of {model_path}: {e} — "
                "using conservative default estimate"
            )
            return int(4 * 1024**3)  # fallback: 4 GiB
        # Layer ratio: -1 = all layers, 0 = CPU (minimum), >0 = partial
        if n_gpu_layers < 0:
            layer_ratio = 1.0  # all layers on GPU
        elif n_gpu_layers == 0:
            layer_ratio = 0.05  # CPU only, just buffers
        else:
            # Assume ~32 layers for most models; scale proportionally
            layer_ratio = max(n_gpu_layers / 32.0, 0.05)
        estimate = int(file_size * 1.3 * layer_ratio)
        min_vram = int(1.5 * 1024**3)  # GGUF has C++ engine overhead

    # ── PyTorch path ──────────────────────────────────────────────────
    elif transformer is not None:
        # Exact weight bytes in current dtype — works for fp16, bf16, fp32,
        # and quantized tensors (each reports its compressed storage size)
        total_weight_bytes = sum(
            p.numel() * p.element_size()
            for p in transformer.parameters()
        )

        # Estimate KV cache from model config if available
        kv_bytes = 0
        try:
            model_obj = getattr(transformer, 'model', transformer)
            config = getattr(model_obj, 'config', None)
            if config:
                n_layers = (
                    getattr(config, 'num_hidden_layers', None)
                    or getattr(config, 'num_layers', 28)
                )
                hidden_size = getattr(config, 'hidden_size', 2048)
                n_heads = getattr(config, 'num_attention_heads', 32)
                n_kv = getattr(config, 'num_key_value_heads', n_heads)
                head_dim = hidden_size // n_heads
                # 2 (k+v) * n_layers * n_ctx * n_kv * head_dim * 2 bytes
                kv_bytes = 2 * n_layers * n_ctx * n_kv * head_dim * 2
        except Exception:
            pass

        if kv_bytes == 0:
            kv_bytes = int(512 * 1024 * 1024)  # fallback: 512 MB

        estimate = int((total_weight_bytes + kv_bytes) * 1.15)
        min_vram = int(1 * 1024**3)  # 1 GiB minimum

    else:
        # No information available — use conservative default
        return int(4 * 1024**3)

    # ── Clamp ─────────────────────────────────────────────────────────
    max_vram = int(24 * 1024**3)  # 24 GiB maximum
    return max(min_vram, min(estimate, max_vram))


# ── OOM-Safe Model Loading ──────────────────────────────────────────


def move_model_to_device_safe(
    transformer,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> bool:
    """Move model to GPU with OOM-safe retry logic.

    On CUDA OOM, defragments cached allocator blocks and retries once.

    Args:
        transformer: The transformer model to move
        target_device: Target GPU device
        target_dtype: Target data type (bfloat16 or float32)

    Returns:
        bool: True if model placed on GPU, False if fell back to CPU
    """
    if not torch.cuda.is_available() or target_device.type == "cpu":
        transformer.to(device=target_device, dtype=target_dtype)
        return False

    for attempt in range(2):  # Try twice; second attempt after defrag
        try:
            transformer.to(device=target_device, dtype=target_dtype)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(target_device) / (1024**3)
            total = torch.cuda.get_device_properties(target_device).total_memory / (1024**3)
            logging.info(
                f"[LLM Chat] Model moved to GPU (attempt {attempt + 1}): "
                f"{allocated:.1f} GiB allocated / {total:.1f} GiB total — "
                f"shared memory {(allocated / total) * 100:.0f}% utilized"
            )
            return True

        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda out of memory" in error_str:
                if attempt == 0:
                    # Defragment cached allocator blocks and retry once
                    logging.warning(
                        f"[LLM Chat] CUDA OOM on first attempt: {e} — "
                        "defragmenting and retrying..."
                    )
                    defragment_vram()
                    torch.cuda.empty_cache()
                else:
                    logging.error(
                        f"[LLM Chat] CUDA OOM after defragment retry: {e} — "
                        "model too large for VRAM + shared memory. "
                        "Consider use_layer_offloading=True to keep only "
                        "one transformer layer in VRAM at a time."
                    )
            else:
                # Non-OOM error — re-raise
                raise

    # Both attempts failed — fall back to CPU
    logging.error(
        f"[LLM Chat] Failed to fit model on GPU. "
        "Falling back to CPU generation — this will be significantly slower."
    )
    transformer.to(device=torch.device("cpu"), dtype=target_dtype)
    return False


# ── Layer Offloading Engine ─────────────────────────────────────────
# Keeps only the currently-executing transformer layer in VRAM by hooking
# into each layer's forward pass. All other layers reside in CPU RAM.
# Uses non-blocking H2D/D2H transfers on a dedicated CUDA stream to
# overlap data movement with computation.


class LayerOffloadEngine:
    """Manages layer-by-layer VRAM offloading for large models.

    Hooks into each transformer layer via register_forward_pre_hook and
    register_forward_hook. Before a layer runs, the pre-hook ensures its
    parameters are on GPU (waiting for any pending async H2D transfer).
    After the layer completes, the hook offloads it back to CPU and
    begins preloading the next layer. Uses a dedicated CUDA stream for
    transfers, allowing overlap with the next layer's compute.

    Enables models larger than physical VRAM by using system RAM
    as overflow. Peak VRAM usage during generation is approximately:
        1 layer + KV cache + embeddings + logits

    Usage:
        engine = LayerOffloadEngine(model, device=torch.device("cuda"))
        engine.install()
        try:
            output = model.forward(embeds, ...)
        finally:
            engine.remove()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.offload_device = offload_device
        self.transfer_stream = torch.cuda.Stream()
        self.compute_done_event = torch.cuda.Event()
        self._hooks: list = []
        self._layer_idx: list = [0]  # mutable for closure access
        self.layers: list = []

        # Discover transformer layers — supports both flat and nested models
        if hasattr(model, "layers"):
            self.layers = list(model.layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = list(model.model.layers)

        self.num_layers = len(self.layers)
        self._installed = False

        if self.num_layers == 0:
            logging.warning(
                "[LLM Chat] LayerOffloadEngine: no layers found on model "
                f"type={type(model).__name__} — offloading has no effect"
            )

    # ── Hooks ────────────────────────────────────────────────────────────

    def _pre_hook_fn(self, module, args):
        """Before layer forward: ensure layer parameters are on GPU.

        Waits for the transfer stream to complete any pending H2D transfer
        for this layer. By the time the wait completes, the layer's
        parameters are resident on GPU and ready for the forward pass.
        """
        if not self._installed:
            return None
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        return None

    def _hook_fn(self, module, args, output):
        """After layer forward: offload this layer, preload the next.

        1. Records compute-done event on the current stream
        2. On the transfer stream (async):
           a. Waits for compute-done event
           b. Offloads this layer's params back to CPU (non-blocking D2H)
           c. Preloads next layer's params to GPU (non-blocking H2D)

        Overlaps D2H/H2D transfers with Python-level work, keeping
        only one layer in VRAM at any time.
        """
        if not self._installed or self.num_layers == 0:
            return output

        idx = self._layer_idx[0]

        # Record compute completion on the current stream
        self.compute_done_event.record(torch.cuda.current_stream())

        if idx < self.num_layers:
            with torch.cuda.stream(self.transfer_stream):
                # Wait for compute to finish before modifying parameters
                self.transfer_stream.wait_event(self.compute_done_event)
                # Offload current layer back to CPU (non-blocking D2H)
                module.to(self.offload_device, non_blocking=True)
                # Preload next layer to GPU (non-blocking H2D)
                if idx + 1 < self.num_layers:
                    self.layers[idx + 1].to(self.device, non_blocking=True)

        self._layer_idx[0] = idx + 1
        return output

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def install(self):
        """Register hooks on all layers and move them to CPU.

        All layers start on CPU. The first layer is immediately preloaded
        to GPU on the transfer stream. Hooks handle the rest during
        model.forward().
        """
        self.remove()  # Clean up any previous installation

        if self.num_layers == 0:
            self._installed = True  # Mark as installed (no-op)
            return

        for i, layer in enumerate(self.layers):
            # Move each layer to CPU initially
            layer.to(self.offload_device, non_blocking=False)
            # Register hooks
            h1 = layer.register_forward_pre_hook(self._pre_hook_fn)
            h2 = layer.register_forward_hook(self._hook_fn)
            self._hooks.extend([h1, h2])

        # Preload first layer to GPU on the transfer stream
        with torch.cuda.stream(self.transfer_stream):
            self.layers[0].to(self.device, non_blocking=True)

        torch.cuda.synchronize()
        self._installed = True

        logging.info(
            f"[LLM Chat] LayerOffloadEngine installed: "
            f"{self.num_layers} layers, device={self.device}, "
            f"offload={self.offload_device}"
        )

    def remove(self):
        """Remove hooks and restore all layers to GPU.

        Should be called after generation completes to return the model
        to its normal (GPU-resident) state.
        """
        self._installed = False
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_idx[0] = 0

        if self.num_layers > 0:
            for layer in self.layers:
                layer.to(self.device, non_blocking=True)
            torch.cuda.synchronize()

        logging.info(
            "[LLM Chat] LayerOffloadEngine removed — all layers restored to GPU"
        )

    # ── Context manager support ──────────────────────────────────────────

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False


# ── Post-Generation Offload ─────────────────────────────────────────


def _conditionally_offload_model(transformer, original_device, inner_sd_clip, vram_mode):
    """After generation, either keep model on GPU or offload based on vram_mode.

    Args:
        transformer: The model transformer (or None).
        original_device: The device to restore the model to (or None).
        inner_sd_clip: The inner SD clip (or None).
        vram_mode: "keep_loaded" to preserve GPU state, anything else to offload.
    """
    if vram_mode == "keep_loaded":
        logging.debug(
            "[LLM Chat] vram_mode=keep_loaded — preserving GPU state "
            "for next generation"
        )
    else:
        if transformer is not None and original_device is not None:
            transformer.to(device=original_device)
        if inner_sd_clip is not None:
            inner_sd_clip.execution_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.debug(f"[LLM Chat] Model offloaded from GPU (vram_mode={vram_mode})")

        # Force PyTorch cache empty after offload
        if vram_mode == "aggressive_free":
            try:
                import comfy.model_management
                comfy.model_management.soft_empty_cache(force=True)
                logging.debug(
                    "[LLM Chat] Aggressive VRAM cleanup done — "
                    "soft_empty_cache(force=True)"
                )
            except Exception:
                pass


# ── Weight Tying for Quantized Models ────────────────────────────────


def _patch_missing_lm_head(transformer):
    """Apply weight tying + logits patching for quantized models missing lm_head.

    Weight ties lm_head.weight to embed_tokens.weight to prevent AttributeError,
    then monkey-patches logits() to bypass the lm_head layer entirely.

    Args:
        transformer: The model transformer with .model.lm_head and .get_input_embeddings().
    """
    if not hasattr(transformer, "model") or not hasattr(transformer.model, "lm_head"):
        return

    if transformer.model.lm_head.weight is not None:
        return

    tied_weight = transformer.get_input_embeddings().weight
    transformer.model.lm_head.weight = tied_weight
    logging.info(
        "[LLM Chat] Missing lm_head.weight detected — applied weight tying"
    )

    lm_head_has_quant = (
        hasattr(transformer.model.lm_head, 'layout_type')
        and transformer.model.lm_head.layout_type is not None
    )
    config_has_lm_head = getattr(transformer.model.config, 'lm_head', False)

    if not config_has_lm_head or lm_head_has_quant:
        return

    def _patched_logits(self, x):
        """Bypass lm_head, use embed_tokens.weight directly."""
        input_tensor = x[:, -1:]
        module = self.model.embed_tokens
        offload_stream = None
        if getattr(module, 'comfy_cast_weights', False):
            weight, _, offload_stream = comfy_ops.cast_bias_weight(
                module, input_tensor, offloadable=True
            )
        else:
            weight = module.weight.to(input_tensor)
        result = torch.nn.functional.linear(input_tensor, weight, None)
        comfy_ops.uncast_bias_weight(module, weight, None, offload_stream)
        return result

    transformer.logits = types.MethodType(_patched_logits, transformer)
    logging.info(
        "[LLM Chat] lm_head lacks quantization metadata — "
        "patched logits() to use embed_tokens.weight path"
    )


# ── Shared Model Setup ─────────────────────────────────────────────


def prepare_model_for_generation(clip, use_layer_offloading=False, vram_mode="unload"):
    """Extract transformer from CLIP, move to GPU, configure quantization.

    Performs shared model setup for generation:
      1. Extracts inner_sd_clip and transformer from the CLIP object
      2. Determines target device (GPU/CPU) and dtype (bfloat16/float32)
      3. Moves model to GPU via OOM-safe retry or layer-offloading
      4. Configures quantized forward path based on format
      5. Patches missing lm_head via weight tying
      6. Logs Phase 3 optimizations banner

    Args:
        clip: A comfy.sd.CLIP object containing the transformer.
        use_layer_offloading: If True, moves only embed_tokens/norm to GPU;
              layers stay on CPU via LayerOffloadEngine.

    Returns:
        Tuple of (inner_sd_clip, transformer, original_device).
        All None if no transformer was found.
    """
    import comfy.model_management

    cond_stage = clip.cond_stage_model  # SD1ClipModel

    inner_sd_clip = None
    transformer = None
    original_device = None

    if hasattr(cond_stage, "clip_name"):
        clip_name = cond_stage.clip_name
        inner_sd_clip = getattr(cond_stage, clip_name, None)
        if inner_sd_clip is None:
            inner_sd_clip = getattr(cond_stage, f"clip_{clip_name}", None)

    if inner_sd_clip is not None and hasattr(inner_sd_clip, "transformer"):
        transformer = inner_sd_clip.transformer
        embed_weight = transformer.get_input_embeddings().weight
        original_device = embed_weight.device

        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            target_dtype = (
                torch.bfloat16
                if comfy.model_management.should_use_bf16(target_device)
                else torch.float32
            )
        else:
            target_dtype = torch.float32

        # Conditional model loading based on layer offloading mode.
        # When use_layer_offloading is True, only essential components
        # (embed_tokens, norm) are moved to GPU; LayerOffloadEngine
        # handles per-layer H2D transfers for the active layer only.
        #
        # When use_layer_offloading is False, normal model loading with
        # OOM-safe retry is used.
        if use_layer_offloading and torch.cuda.is_available():
            # Model stays on CPU — only move non-layer essentials to GPU
            m = transformer
            if hasattr(m, 'embed_tokens') and hasattr(m.embed_tokens, 'weight'):
                m.embed_tokens.to(device=target_device, dtype=target_dtype)
            if hasattr(m, 'norm') and hasattr(m.norm, 'weight'):
                m.norm.to(device=target_device, dtype=target_dtype)
            logging.info(
                "[LLM Chat] Layer offloading active — moved essential "
                "components (embed_tokens, norm) to GPU, keeping layers "
                "on CPU for on-demand transfer"
            )
        else:
            # ── keep_loaded fast path: skip free_memory + reload if model already on GPU ──
            # When vram_mode is "keep_loaded" and the transformer is already resident
            # on the target device, there's no need to free VRAM and re-load.
            # This eliminates the unnecessary VRAM dip between consecutive generations.
            _needs_reload = not (
                vram_mode == "keep_loaded"
                and transformer is not None
                and getattr(transformer.get_input_embeddings().weight, 'device', None) == target_device
            )

            if _needs_reload:
                # ── Ask ComfyUI to free VRAM before loading LLM ──
                # Other models may occupy VRAM; free_memory() unloads them
                # before moving the LLM transformer to GPU.
                try:
                    memory_needed = estimate_vram_needed(
                        transformer=transformer,
                        n_ctx=2048,
                        use_layer_offloading=use_layer_offloading,
                    )
                    comfy.model_management.free_memory(
                        memory_required=memory_needed,
                        device=target_device,
                        keep_loaded=[],  # Don't keep anything — free all non-essential
                    )
                    # Build informative log including weight bytes when available
                    try:
                        total_params = sum(p.numel() for p in transformer.parameters())
                        total_gb = sum(
                            p.numel() * p.element_size() for p in transformer.parameters()
                        ) / (1024**3)
                        log_detail = (
                            f"(~{total_params/1e9:.1f}B params, "
                            f"{total_gb:.1f} GiB weights)"
                        )
                    except Exception:
                        log_detail = ""
                    logging.info(
                        f"[LLM Chat] Requested ComfyUI to free "
                        f"{memory_needed / (1024**3):.1f} GiB VRAM on {target_device} "
                        f"{log_detail}"
                    )
                except Exception as e:
                    logging.warning(
                        f"[LLM Chat] Failed to request VRAM freeing: {e} — "
                        "proceeding without explicit free_memory()"
                    )

                move_model_to_device_safe(
                    transformer,
                    target_device=target_device,
                    target_dtype=target_dtype,
                )
            else:
                logging.debug(
                    f"[LLM Chat] keep_loaded: transformer already on {target_device} — "
                    "skipping free_memory() and model reload, preserving GPU state"
                )
        inner_sd_clip.execution_device = target_device

        # ── Quantized forward path optimization ──
        # Selectively enable/disable quantized matmul based on format.
        # NVFP4 (4-bit): use full_precision for quality.
        # Q8_0 / others: use native quantized path for speed.
        quant_format = "unknown"
        has_quantized_layers = False
        for module in transformer.modules():
            if hasattr(module, '_full_precision_mm'):
                has_quantized_layers = True
                layout = getattr(module, 'layout_type', None)
                if layout is not None:
                    layout_str = str(layout).lower()
                    quant_format = layout_str
                    if 'nvfp4' in layout_str or 'fp4' in layout_str:
                        if not module._full_precision_mm:
                            module._full_precision_mm = True
                        logging.info(
                            f"[LLM Chat] NVFP4 quantized layer detected "
                            f"({layout_str}) — forcing full_precision_mm for quality"
                        )
                    else:
                        if module._full_precision_mm:
                            module._full_precision_mm = False
                        logging.info(
                            f"[LLM Chat] {layout_str} quantized layer detected — "
                            f"using native quantized matmul path for speed"
                        )
                break

        if has_quantized_layers and quant_format == "unknown":
            for module in transformer.modules():
                if hasattr(module, '_full_precision_mm') and not module._full_precision_mm:
                    module._full_precision_mm = True
            logging.info("[LLM Chat] Quantized layers found (unknown format) — defaulting to full_precision_mm=True")

        # ── Weight tying + logits patching for quantized models ──
        _patch_missing_lm_head(transformer)

        logging.info(
            "[LLM Chat] Phase 3 optimizations active: TF32 + inference_mode "
            "+ vectorized sampling + GPU-native generation engine"
        )

    return inner_sd_clip, transformer, original_device
