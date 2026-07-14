"""
Live streaming support.

Token-by-token streaming of generated LLM output into the popup UI
via ComfyUI's WebSocket (PromptServer.instance.send_sync).

Data Flow
---------
generate_text() -> generate_text_gpu() -> _gpu_generate_inner() -> model.forward()
    -> sample_token() (= _vectorized_sample_token) fires _fire_streaming_callback()
    -> _streaming_callback decodes token & sends via PromptServer.instance.send_sync()
    -> frontend WebSocket receives "llm_lab_token"
    Falls back to cond_stage.generate() if GPU-native path unavailable.

Module-level State
------------------
_popup_active_nodes (set)    — nodes with open popups
_abort_flags (dict)          — node_id -> bool abort signals

Callback Registry
-----------------
Registered via cuda_optimizations.set_streaming_callback() before generation,
cleared via clear_streaming_callback() in the finally block.

Progress Events
---------------
Auto-enabled via cuda_optimizations.enable_progress() during streaming
so the frontend progress bar tracks GPU-native generation progress.

API Routes
----------
- POST /easyllm/popup_active/{node_id}   — signal popup open
- POST /easyllm/popup_inactive/{node_id} — signal popup closed
- POST /easyllm/abort_stream/{node_id}   — set abort flag
- POST /easyllm/upload_image/{node_id}   — store uploaded image filename for a node
- POST /easyllm/clear_uploaded_image/{node_id} — clear stored uploaded image filename
"""

import asyncio
import logging
import os

import torch

from .generation_state import get_state

_state = get_state()

# Server-side store for chat history context.
# Keyed by node_id (unique_id), populated by frontend via POST before queue.
# The chat() method retrieves via pop() during execution.
_history_store: dict[str, list] = {}

# Server-side store for uploaded image filenames (no-wire chat mode).
# Keyed by node_id, populated by frontend via POST after user uploads
# to ComfyUI's /upload/image endpoint. Retrieved by generate() during execution.
_uploaded_images: dict[str, str] = {}

# ── Known GGUF model prefix → architecture mappings ──────────────────────
# Maps filename prefixes (observed in common GGUF naming conventions) to
# their ``general.architecture`` value. Used by the model browser to show
# architecture pills and improve mmproj detection when GGUF metadata is
# unavailable (e.g. ``gguf`` library not installed).

_KNOWN_ARCH_PREFIXES: dict[str, str] = {
    # Vision-language models (need mmproj)
    "qwen2-vl":           "qwen2vl",
    "qwen2.5-vl":         "qwen2vl",
    "llava":              "llava",
    "minicpm":            "minicpmv",
    "cogvlm":             "cogvlm",
    "phi-3-vision":       "phi3v",
    "phi-3.5-vision":     "phi3v",
    "internvl":           "internvl2",
    "glm-4v":             "glm4v",
    "yi-vl":              "yivl",
    "deepseek-vl":        "deepseekvl",
    "florence":           "florence",
    "paligemma":          "paligemma",
    "molmo":              "molmo",
    # Text-only models (no mmproj needed, but still useful to label)
    "llama-3":            "llama",
    "llama-2":            "llama",
    "llama-3.2":          "llama",
    "gemma-2":            "gemma2",
    "gemma-3":            "gemma3",
    "phi-3":              "phi3",
    "phi-3.5":            "phi3",
    "phi-4":              "phi4",
    "mistral":            "mistral",
    "mixtral":            "mixtral",
    "qwen-2.5":           "qwen2",
    "qwen2.5":            "qwen2",
    "qwen-2":             "qwen2",
    "deepseek-v3":        "deepseek3",
    "deepseek-r1":        "deepseek3",
    "command-r":          "command-r",
    "dbrx":               "dbrx",
    "starcoder2":         "starcoder2",
    "falcon":             "falcon",
    "tinyllama":          "llama",
}

# Pre-compute sorted prefixes at module level to avoid re-sorting on every call.
_SORTED_ARCH_PREFIXES: list[str] = sorted(
    _KNOWN_ARCH_PREFIXES.keys(), key=len, reverse=True
)

# ── API Routes ──────────────────────────────────────────────────────


def _get_architecture_from_prefix(model_name: str) -> str | None:
    """Guess architecture from a model filename prefix.

    Searches ``_KNOWN_ARCH_PREFIXES`` in longest-first order so that
    e.g. ``"phi-3.5-vision"`` matches before ``"phi-3"``.

    Args:
        model_name: The model filename (e.g. ``"phi-3.5-vision-instruct.Q4_K_M.gguf"``).

    Returns:
        Architecture string (e.g. ``"phi3v"``) or ``None`` if unknown.
    """
    name_lower = model_name.lower()
    sorted_prefixes = _SORTED_ARCH_PREFIXES
    for prefix in sorted_prefixes:
        if name_lower.startswith(prefix) or prefix in name_lower:
            return _KNOWN_ARCH_PREFIXES[prefix]
    return None


def setup_streaming_routes():
    """Register WebSocket push routes for popup state tracking, abort,
    and image upload storage.

    Safe to call multiple times (idempotent — skips if already registered).
    Must be called from __init__.py after ComfyUI server is initialized.
    """
    if getattr(setup_streaming_routes, '_registered', False):
        return

    try:
        from server import PromptServer
        from aiohttp import web

        @PromptServer.instance.routes.post("/easyllm/popup_active/{node_id}")
        async def set_popup_active(request):
            """Called by frontend popup.js before app.queuePrompt()."""
            node_id = request.match_info["node_id"]
            _state.set_popup_active(node_id)
            return web.json_response({"success": True})

        @PromptServer.instance.routes.post("/easyllm/popup_inactive/{node_id}")
        async def clear_popup_active(request):
            """Called by frontend popup.js after popup close or generation done."""
            node_id = request.match_info["node_id"]
            _state.clear_popup_active(node_id)
            return web.json_response({"success": True})

        @PromptServer.instance.routes.post("/easyllm/abort_stream/{node_id}")
        async def abort_stream(request):
            """Set abort flag for mid-stream cancellation.

            The streaming callback checks this flag per token. When set,
            tokens are silently dropped and generation continues (background
            streaming). To fully kill execution, use ComfyUI's native Cancel.
            """
            node_id = request.match_info["node_id"]
            _state.set_abort(node_id)
            return web.json_response({"success": True})

        @PromptServer.instance.routes.post("/easyllm/store_history/{node_id}")
        async def store_chat_history(request):
            """Store serialized chat history for retrieval during execution.

            Called by frontend before app.queuePrompt(). The chat() method
            retrieves the history via _history_store.pop(unique_id, None)
            during execution.
            """
            node_id = request.match_info["node_id"]
            try:
                body = await request.json()
                history = body.get("history", [])
                if isinstance(history, list):
                    # Write to ephemeral store for LLM context during execution
                    _history_store[node_id] = history
                    return web.json_response({"success": True})
                else:
                    return web.json_response({"success": False, "error": "history must be a list"})
            except Exception as e:
                logging.warning(f"[LLM Chat] Failed to store history for node {node_id}: {e}")
                return web.json_response({"success": False, "error": str(e)})

        @PromptServer.instance.routes.post("/easyllm/register_gguf_dir")
        async def register_gguf_dir_api(request):
            """Register a custom directory for GGUF model search.

            Called by the Model Browser popup when the user adds a directory.
            The directory is persisted and searched on all future lookups.
            """
            try:
                body = await request.json()
                directory = body.get("directory", "").strip()
                if not directory:
                    return web.json_response(
                        {"success": False, "error": "No directory provided"},
                        status=400
                    )
                import os
                if not os.path.isdir(directory):
                    return web.json_response(
                        {"success": False, "error": f"Directory does not exist: {directory}"}
                    )
                from .utils import register_browsed_gguf_dir
                register_browsed_gguf_dir(directory)
                return web.json_response(
                    {"success": True, "directory": os.path.abspath(directory)}
                )
            except Exception as e:
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/unregister_gguf_dir")
        async def unregister_gguf_dir_api(request):
            """Remove a custom directory from GGUF model search.

            Called by the Model Browser popup when the user clicks Remove.
            """
            try:
                body = await request.json()
                directory = body.get("directory", "").strip()
                if not directory:
                    return web.json_response(
                        {"success": False, "error": "No directory provided"},
                        status=400
                    )
                from .utils import unregister_browsed_gguf_dir
                removed = unregister_browsed_gguf_dir(directory)
                return web.json_response({"success": removed})
            except Exception as e:
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.get("/easyllm/list_browsed_dirs")
        async def list_browsed_dirs_api(request):
            """Return the current list of browsed/search directories.

            Used by the Model Browser popup to populate the directory list
            on open. Collapses nested children as a read-time safety net
            to ensure no polluted subdirectories are returned.
            """
            try:
                from .utils import _browsed_gguf_dirs, _collapse_nested_dirs
                return web.json_response({
                    "directories": _collapse_nested_dirs(sorted(_browsed_gguf_dirs))
                })
            except Exception as e:
                return web.json_response(
                    {"directories": [], "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/exclude_gguf_dir")
        async def exclude_gguf_dir_api(request):
            """Exclude a directory from GGUF model search.

            Called by the Model Browser popup when the user toggles a directory
            off. The directory is persisted in the exclusion set and will be
            skipped on all future searches.
            """
            try:
                body = await request.json()
                directory = body.get("directory", "").strip()
                if not directory:
                    return web.json_response(
                        {"success": False, "error": "No directory provided"},
                        status=400
                    )
                from .utils import exclude_gguf_search_dir
                result = exclude_gguf_search_dir(directory)
                return web.json_response({"success": result})
            except Exception as e:
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/unexclude_gguf_dir")
        async def unexclude_gguf_dir_api(request):
            """Remove a directory from the exclusion set.

            Called by the Model Browser popup when the user toggles a directory
            back on.
            """
            try:
                body = await request.json()
                directory = body.get("directory", "").strip()
                if not directory:
                    return web.json_response(
                        {"success": False, "error": "No directory provided"},
                        status=400
                    )
                from .utils import unexclude_gguf_search_dir
                result = unexclude_gguf_search_dir(directory)
                return web.json_response({"success": result})
            except Exception as e:
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.get("/easyllm/list_excluded_dirs")
        async def list_excluded_dirs_api(request):
            """Return the current list of excluded search directories."""
            try:
                from .utils import _excluded_gguf_dirs
                return web.json_response({
                    "directories": sorted(_excluded_gguf_dirs)
                })
            except Exception as e:
                return web.json_response(
                    {"directories": [], "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/clear_all_gguf_state")
        async def clear_all_gguf_state_api(request):
            """Clear all persisted GGUF state (browsed dirs, excluded dirs).

            Called by the Model Browser popup when the user clicks "Reset All"
            or "Clear All Custom Directories". Does NOT clear recently used
            pairs — those are frontend localStorage only.
            """
            try:
                from .utils import (
                    _browsed_gguf_dirs, _excluded_gguf_dirs,
                    _save_browsed_dirs, _save_excluded_dirs
                )
                _browsed_gguf_dirs.clear()
                _excluded_gguf_dirs.clear()
                _save_browsed_dirs(_browsed_gguf_dirs)
                _save_excluded_dirs(_excluded_gguf_dirs)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.get("/easyllm/list_all_search_dirs")
        async def list_all_search_dirs_api(request):
            """Return ComfyUI default search directories with their exclusion status.

            Only returns ComfyUI-managed folders (models, text_encoders, llm,
            gguf, clip) — the generic defaults that exist on any ComfyUI install.
            Nested children are collapsed when a parent is already present.

            User-added custom directories are returned separately in the
            ``custom`` field. The frontend renders them in the Custom Directories
            section instead.
            """
            try:
                from .utils import _excluded_gguf_dirs, _browsed_gguf_dirs, _collapse_nested_dirs
                import os
                dirs: list[str] = []
                # Build the ComfyUI folder_paths list (no external/browsed dirs)
                try:
                    import folder_paths
                    if hasattr(folder_paths, "models_dir") and folder_paths.models_dir:
                        dirs.append(os.path.abspath(folder_paths.models_dir))
                    GGUF_RELEVANT_FOLDERS = {
                        "text_encoders", "llm", "LLM", "gguf", "GGUF", "clip",
                    }
                    if hasattr(folder_paths, "folder_names_and_paths"):
                        for folder_name, (paths_list, _) in folder_paths.folder_names_and_paths.items():
                            if folder_name not in GGUF_RELEVANT_FOLDERS:
                                continue
                            for base_dir in paths_list:
                                abs_dir = os.path.abspath(base_dir)
                                if abs_dir not in dirs:
                                    dirs.append(abs_dir)
                except ImportError:
                    pass
                except Exception:
                    pass
                # Deduplicate
                seen: set[str] = set()
                all_dirs: list[str] = []
                for d in dirs:
                    if d not in seen:
                        seen.add(d)
                        all_dirs.append(d)
                # Collapse nested children (parents cover them via os.walk)
                all_dirs = _collapse_nested_dirs(all_dirs)
                return web.json_response({
                    "directories": all_dirs,
                    "excluded": sorted(_excluded_gguf_dirs),
                    "custom": sorted(_browsed_gguf_dirs),
                })
            except Exception as e:
                return web.json_response(
                    {"directories": [], "excluded": [], "custom": [], "error": str(e)},
                    status=500
                )

        @PromptServer.instance.routes.post("/easyllm/validate_gguf_path")
        async def validate_gguf_path_api(request):
            """Validate a full file path exists and return its canonical absolute path.

            On success, registers the parent directory so future filename-only
            lookups can find models in that directory. Skips registration if the
            directory is in the exclusion set.
            """
            try:
                body = await request.json()
                path = body.get("path", "")
                if not path:
                    return web.json_response(
                        {"valid": False, "error": "No path provided"}, status=400
                    )
                import os
                if os.path.isfile(path):
                    resolved = os.path.abspath(path)

                    from .utils import (register_browsed_gguf_dir,
                                        _excluded_gguf_dirs)
                    parent = os.path.dirname(resolved)
                    if parent not in _excluded_gguf_dirs:
                        register_browsed_gguf_dir(parent)

                    return web.json_response({"valid": True, "resolved": resolved})
                else:
                    return web.json_response({"valid": False, "error": "File not found"})
            except Exception as e:
                return web.json_response({"valid": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.post("/easyllm/auto_detect_mmproj")
        async def auto_detect_mmproj_api(request):
            """Auto-detect mmproj file, serving from model index cache if possible.

            Called by the model browser popup after the user selects a model.
            First checks the model index cache (fast, zero GGUF reads). Falls
            back to EasyLLMGGUF._auto_detect_mmproj() only on cache miss.

            On cache miss + successful detection, writes the result back to the
            model index cache so subsequent lookups hit the fast path.
            """
            try:
                body = await request.json()
                model_path = body.get("model_path", "")
                if not model_path:
                    return web.json_response(
                        {"mmproj_path": "", "error": "No model_path provided"},
                        status=400
                    )

                # ── Fast path: check model index cache first ───────────
                from .utils import _load_model_index, _save_model_index
                index = _load_model_index()
                if index is not None:
                    for entry in index.get("files", []):
                        if entry.get("path") == model_path:
                            mmproj_path = entry.get("mmproj_path")
                            if mmproj_path:
                                return web.json_response({
                                    "mmproj_path": mmproj_path,
                                    "from_cache": True,
                                })
                            break  # entry found but no mmproj_path → fall through

                # ── Slow path: delegate to GGUF header reads ──────────
                from .chat_node import EasyLLMGGUF
                mmproj = EasyLLMGGUF._auto_detect_mmproj(model_path)

                # ── Write back to model index cache so fast path works next time ──
                if mmproj and index is not None:
                    for entry in index.get("files", []):
                        if entry.get("path") == model_path:
                            entry["has_mmproj"] = True
                            entry["mmproj_path"] = mmproj
                            _save_model_index(index)
                            break

                return web.json_response({
                    "mmproj_path": mmproj or "",
                    "from_cache": False,
                })
            except Exception as e:
                return web.json_response(
                    {"mmproj_path": "", "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/list_gguf_files")
        async def list_gguf_files_api(request):
            """List all .gguf files from all registered search directories.

            Uses a persistent JSON cache to avoid re-reading GGUF binary
            headers on every popup open.  On cache hit, returns in <10ms.
            On cache miss or staleness, rebuilds the index and persists it.

            Query params:
                stale_ok (str): If ``"true"``, return stale cached data with
                    a ``cache_stale`` flag instead of forcing a full rebuild.
                    Used by directory toggle operations to avoid UI freeze.
            """
            try:
                from .utils import (
                    _load_model_index,
                    _build_model_index, _save_model_index,
                    _compute_search_dirs_hash,
                )

                # Check for stale_ok query parameter
                stale_ok = request.query.get("stale_ok", "").lower() == "true"

                # 1. Try loading existing cache
                from .debug_profiler import profiler
                profiler.begin_sub("cache_build_total", "load_index")
                index = _load_model_index()
                profiler.end_sub("cache_build_total", "load_index")

                profiler.begin_sub("cache_build_total", "hash")
                current_hash = _compute_search_dirs_hash()
                profiler.end_sub("cache_build_total", "hash")

                # 2. Determine if cache is still valid
                # NOTE: _is_model_index_stale() intentionally omitted.
                # The search_dirs_hash detects directory changes; per-file mtime checks
                # add overhead without meaningful benefit. File replacement is handled
                # by the explicit Refresh button.
                cache_valid = (
                    index is not None
                    and index.get("version") == 1
                    and index.get("search_dirs_hash") == current_hash
                )

                if cache_valid:
                    # Fast path — return cached data immediately
                    files = index["files"]
                    return web.json_response({
                        "files": files,
                        "total": len(files),
                        "cached_at": index.get("generated_at", ""),
                        "error": None,
                    })

                # ── Stale-ok fast path: return stale data + flag ──────
                if stale_ok and index is not None:
                    files = index.get("files", [])
                    return web.json_response({
                        "files": files,
                        "total": len(files),
                        "cached_at": index.get("generated_at", ""),
                        "cache_stale": True,
                        "error": None,
                    })

                # 3. Cache miss or stale — rebuild (slow path, but cached afterwards)
                # Run rebuild in a thread executor to avoid blocking the asyncio event loop.
                loop = asyncio.get_event_loop()
                profiler.begin("cache_build_total")
                new_index = await loop.run_in_executor(None, _build_model_index)
                profiler.begin_sub("cache_build_total", "save")
                await loop.run_in_executor(None, _save_model_index, new_index)
                profiler.end_sub("cache_build_total", "save")
                profiler.set_model_count(len(new_index["files"]))
                profiler.end("cache_build_total")
                profiler.print_report()

                return web.json_response({
                    "files": new_index["files"],
                    "total": len(new_index["files"]),
                    "cached_at": new_index["generated_at"],
                    "error": None,
                })

            except Exception:
                logging.exception(
                    "[LLM Chat] Failed to list GGUF files"
                )
                return web.json_response(
                    {"files": [], "total": 0, "error": "Internal error listing GGUF files"},
                    status=500,
                )

        # ── Model Index Refresh (force re-scan) ──────────────────────────

        @PromptServer.instance.routes.post("/easyllm/refresh_model_index")
        async def refresh_model_index_api(request):
            """Force a full re-scan and cache rebuild.

            Runs the scan in a thread executor to avoid blocking the event
            loop.  Returns immediately with the total file count.
            """
            import asyncio

            def _rebuild():
                from .utils import _build_model_index, _save_model_index
                new_index = _build_model_index()
                _save_model_index(new_index)
                return len(new_index["files"])

            try:
                loop = asyncio.get_event_loop()
                count = await loop.run_in_executor(None, _rebuild)
                return web.json_response({"success": True, "total": count})
            except Exception as e:
                logging.error(
                    f"[LLM Chat] Failed to refresh model index: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        # ── Model Cache Management ────────────────────────────────────────

        @PromptServer.instance.routes.post("/easyllm/unload_model_cache")
        async def unload_model_cache(request):
            """Unload all cached models from EasyLLMGGUF._model_cache.

            Called by the frontend Apply button (when vram_mode=keep_loaded)
            to immediately free VRAM before a new model is selected.
            No body required. Returns immediately.
            """
            from .chat_node import EasyLLMGGUF
            try:
                with EasyLLMGGUF._model_cache_lock:
                    for key in list(EasyLLMGGUF._model_cache.keys()):
                        try:
                            EasyLLMGGUF._model_cache[key].unload()
                        except Exception:
                            pass
                        del EasyLLMGGUF._model_cache[key]
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                return web.json_response({"status": "ok"})
            except Exception as e:
                logging.warning(
                    f"[LLM Chat] unload_model_cache error: {e}"
                )
                return web.json_response({"status": "ok"})  # best-effort

        # ── Image Upload Routes (no-wire chat mode) ─────────────────

        @PromptServer.instance.routes.post("/easyllm/upload_image/{node_id}")
        async def upload_chat_image(request):
            """Receive uploaded image filename from chat popup, store for node."""
            node_id = request.match_info["node_id"]
            try:
                body = await request.json()
                filename = body.get("filename", "")
                if not filename:
                    return web.json_response(
                        {"success": False, "error": "No filename provided"},
                        status=400,
                    )
                _uploaded_images[node_id] = filename
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/clear_uploaded_image/{node_id}")
        async def clear_uploaded_image(request):
            """Clear stored uploaded image filename for a node."""
            node_id = request.match_info["node_id"]
            _uploaded_images.pop(node_id, None)
            return web.json_response({"success": True})

        # ── JSON Database Routes (Phase 1) ──────────────────────────

        @PromptServer.instance.routes.get("/easyllm/db/history/{node_id}")
        async def db_get_history(request):
            """Load chat or enhancer history for a node from the on-disk database.

            Supports query parameters:
            - ``?type=enhancer`` to select enhancer history (default: chat)
            - ``?limit=N`` to return only the last N entries (pagination)
            - ``?offset=N`` to skip N entries from the end (used with limit)

            Called by the frontend popup on open to load persisted history.
            """
            node_id = request.match_info["node_id"]
            hist_type = request.query.get("type", "chat")
            limit_str = request.query.get("limit", "0")
            offset_str = request.query.get("offset", "0")
            try:
                limit = int(limit_str) if limit_str.isdigit() else 0
                offset = int(offset_str) if offset_str.isdigit() else 0
            except (ValueError, AttributeError):
                limit = 0
                offset = 0

            try:
                from .history_db import get_history, is_available
                if not is_available():
                    return web.json_response({"entries": [], "available": False})
                entries = get_history(
                    node_id, hist_type, offset=offset, limit=limit,
                )

                # Also return total count for pagination UI
                total_count = len(get_history(
                    node_id, hist_type,
                ) or [])

                return web.json_response({
                    "entries": entries or [],
                    "available": True,
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                })
            except Exception as e:
                logging.warning(
                    f"[LLM Chat DB] GET history failed for node {node_id}: {e}"
                )
                return web.json_response(
                    {"entries": [], "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/db/history/{node_id}")
        async def db_set_history(request):
            """Save chat or enhancer history for a node to the on-disk database.

            Full replacement — all previous entries for this node are discarded.
            Also updates the old ephemeral store for backward compatibility with
            get_chat_history() during execution (Phase 2 bidirectional sync).
            Called by the frontend popup on close, or before queue.
            """
            node_id = request.match_info["node_id"]
            try:
                body = await request.json()
                entries = body.get("entries", [])
                hist_type = body.get("type", "chat")
                metadata = body.get("metadata", None)

                from .history_db import set_history, is_available
                if not is_available():
                    # Fallback: write to old ephemeral store too
                    if isinstance(entries, list):
                        _history_store[node_id] = entries
                    return web.json_response({"success": True, "available": False})

                ok = set_history(node_id, entries, hist_type, metadata)

                # ── NEW: bidirectional sync — update ephemeral store too ──
                if ok and isinstance(entries, list) and hist_type == "chat":
                    _history_store[node_id] = entries

                if ok:
                    return web.json_response({"success": True})
                else:
                    return web.json_response(
                        {"success": False, "error": "Failed to save history (empty entries?)"},
                        status=400,
                    )
            except Exception as e:
                logging.warning(
                    f"[LLM Chat DB] POST history failed for node {node_id}: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        # ── JSON Database Routes — Incremental Append (Phase 6) ──────────

        @PromptServer.instance.routes.post("/easyllm/db/history/{node_id}/append")
        async def db_append_history(request):
            """Append a single entry to history (incremental, not full replace).

            Body::

                {
                    "entry": { "role": "user", "message": "...", ... },
                    "type": "chat" | "enhancer"
                }

            The server appends the entry to the existing session on disk.
            Returns the saved entry so the frontend can sync its display cache.

            Called by ``handlePopupSend()`` and ``onExecuted()`` to persist
            entries one at a time instead of replacing the entire history.
            """
            node_id = request.match_info["node_id"]
            try:
                body = await request.json()
                entry = body.get("entry")
                hist_type = body.get("type", "chat")

                if not entry or not isinstance(entry, dict):
                    return web.json_response(
                        {"success": False, "error": "Missing or invalid entry"},
                        status=400,
                    )

                from .history_db import append_history, is_available
                if not is_available():
                    return web.json_response(
                        {"success": False, "available": False},
                        status=503,
                    )

                ok = append_history(node_id, [entry], hist_type)
                if ok:
                    return web.json_response({
                        "success": True,
                        "entry": entry,
                    })
                else:
                    return web.json_response(
                        {"success": False, "error": "Append failed"},
                        status=500,
                    )
            except Exception as e:
                logging.warning(
                    f"[LLM Chat DB] POST append failed for node {node_id}: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/db/history/{node_id}/append-images")
        async def db_append_images(request):
            """Attach generated images to the history entry matching ``session_uuid``.

            Body::

                {
                    "session_uuid": "...",
                    "images": [{ "type": "generated", "filename": "...", "data": null }, ...],
                    "type": "chat" | "enhancer"
                }

            The server finds the entry with the matching ``_sessionUuid`` field,
            appends the image objects to its ``images`` array, and returns the
            updated entry. The frontend then syncs its display cache.

            Called by ImageCapture's ``onExecuted()`` to persist generated
            images without requiring a full history rewrite.
            """
            node_id = request.match_info["node_id"]
            try:
                body = await request.json()
                session_uuid = body.get("session_uuid")
                images = body.get("images", [])
                hist_type = body.get("type", "chat")

                if not session_uuid or not images:
                    return web.json_response(
                        {"success": False, "error": "Missing session_uuid or images"},
                        status=400,
                    )

                from .history_db import append_images_to_entry, is_available
                if not is_available():
                    return web.json_response(
                        {"success": False, "available": False},
                        status=503,
                    )

                updated_entry = append_images_to_entry(
                    node_id, session_uuid, images, hist_type
                )
                if updated_entry is not None:
                    return web.json_response({
                        "success": True,
                        "entry": updated_entry,
                    })
                else:
                    return web.json_response(
                        {"success": False, "error": "No matching entry found"},
                        status=404,
                    )
            except Exception as e:
                logging.warning(
                    f"[LLM Chat DB] POST append-images failed for node {node_id}: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        # ── Incremental Update / Delete Entry (Cleanup Phase) ─────────────

        @PromptServer.instance.routes.put("/easyllm/db/history/{node_id}/entry")
        async def db_update_entry(request):
            """Update a single history entry by index.

            Body::

                {
                    "index": 0,
                    "entry": { "role": "assistant", "message": "...", ... },
                    "type": "chat" | "enhancer"
                }

            Calls ``history_db.update_entry()`` to replace only the specified
            entry rather than rewriting the entire session.  Used by the
            ``saveEdit()`` and enhancer-edit frontend paths.

            Returns ``{"success": true}`` on success, or an error status.
            """
            node_id = request.match_info["node_id"]
            try:
                body = await request.json()
                entry_index = body.get("index")
                updated_entry = body.get("entry")
                hist_type = body.get("type", "chat")

                if entry_index is None or updated_entry is None:
                    return web.json_response(
                        {"success": False, "error": "Missing index or entry"},
                        status=400,
                    )
                if not isinstance(entry_index, int) or entry_index < 0:
                    return web.json_response(
                        {"success": False, "error": "Invalid index"},
                        status=400,
                    )

                from .history_db import update_entry, is_available
                if not is_available():
                    return web.json_response(
                        {"success": False, "available": False},
                        status=503,
                    )

                ok = update_entry(node_id, entry_index, updated_entry, hist_type)
                if ok:
                    return web.json_response({"success": True})
                else:
                    return web.json_response(
                        {"success": False, "error": "Update failed (index out of range?)"},
                        status=404,
                    )
            except Exception as e:
                logging.warning(
                    f"[LLM Chat DB] PUT entry failed for node {node_id}: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.delete("/easyllm/db/history/{node_id}/entry")
        async def db_delete_entry(request):
            """Delete a single history entry by index.

            Query params::

                index (int): The zero-based index of the entry to delete.
                type (str): "chat" (default) or "enhancer".

            Calls ``history_db.delete_entry()`` to remove only the specified
            entry rather than rewriting the entire session.  Used by the
            ``deleteMessage()`` frontend path.

            Returns ``{"success": true}`` on success, or an error status.
            """
            node_id = request.match_info["node_id"]
            try:
                index_str = request.query.get("index")
                hist_type = request.query.get("type", "chat")

                if index_str is None or not index_str.isdigit():
                    return web.json_response(
                        {"success": False, "error": "Missing or invalid index query param"},
                        status=400,
                    )

                entry_index = int(index_str)

                from .history_db import delete_entry, is_available
                if not is_available():
                    return web.json_response(
                        {"success": False, "available": False},
                        status=503,
                    )

                ok = delete_entry(node_id, entry_index, hist_type)
                if ok:
                    return web.json_response({"success": True})
                else:
                    return web.json_response(
                        {"success": False, "error": "Delete failed (index out of range?)"},
                        status=404,
                    )
            except Exception as e:
                logging.warning(
                    f"[LLM Chat DB] DELETE entry failed for node {node_id}: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.delete("/easyllm/db/history/{node_id}")
        async def db_clear_history(request):
            """Delete all history for a node from the on-disk database.

            Supports query parameter ``?type=chat`` (default) or ``?type=enhancer``
            to selectively clear only one history type.

            Called by the "Clear" button in the chat popup.
            """
            node_id = request.match_info["node_id"]
            hist_type = request.query.get("type", "chat")  # Phase 3: support ?type=enhancer
            try:
                from .history_db import clear_history, is_available
                if not is_available():
                    # Fallback: clear old ephemeral store
                    _history_store.pop(node_id, None)
                    _uploaded_images.pop(node_id, None)
                    return web.json_response({"success": True, "available": False})

                ok = clear_history(node_id, hist_type)  # Pass hist_type
                return web.json_response({"success": ok})
            except Exception as e:
                logging.warning(
                    f"[LLM Chat DB] DELETE history failed for node {node_id}: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        # ── Settings API ─────────────────────────────────────────────────

        @PromptServer.instance.routes.get("/easyllm/db/settings")
        async def db_get_settings(request):
            """Return current database runtime settings.

            Returns auto_cleanup_enabled, interval, max_size, max_age,
            immediate_write, and db_path.
            """
            try:
                from .history_db import get_settings, is_available
                if not is_available():
                    return web.json_response({"available": False})
                settings = get_settings()
                return web.json_response({"available": True, **settings})
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Get settings failed: {e}")
                return web.json_response(
                    {"available": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/db/settings")
        async def db_update_settings(request):
            """Update runtime database settings.

            Accepts JSON with any subset of keys:
            ``auto_cleanup_enabled``, ``auto_cleanup_interval_sec``,
            ``max_size_mb``, ``max_age_days``, ``immediate_write``.

            Changes are persisted to ``settings.json`` and survive restarts.
            """
            try:
                from .history_db import update_settings, is_available
                if not is_available():
                    return web.json_response({"available": False})
                body = await request.json()
                settings = update_settings(body)
                return web.json_response({"status": "ok", **settings})
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Update settings failed: {e}")
                return web.json_response(
                    {"status": "error", "error": str(e)}, status=500
                )

        # ── Destroy API ─────────────────────────────────────────────────

        @PromptServer.instance.routes.post("/easyllm/db/destroy")
        async def db_destroy(request):
            """Destroy and recreate the database.

            WARNING: Irreversible. All history and images will be lost.
            Requires ``{"confirm": true}`` in the request body to proceed.
            """
            try:
                body = await request.json()
                confirm = body.get("confirm", False)
                if not confirm:
                    return web.json_response(
                        {"status": "error", "error": "Confirmation required"},
                        status=400,
                    )

                from .history_db import destroy_database, is_available
                if not is_available():
                    return web.json_response({"available": False})

                result = destroy_database()
                return web.json_response(result)
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Destroy failed: {e}")
                return web.json_response(
                    {"status": "error", "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/db/image")
        async def db_save_image(request):
            """Save an image to the database and return its UUID filename.

            Accepts JSON with ``base64`` (data URI string) and optional
            ``type`` field (``"input"`` or ``"generated"``).
            Returns ``{"filename": "<uuid>_<type>.png"}``.
            """
            try:
                body = await request.json()
                b64_data = body.get("base64", "")
                image_type = body.get("type", "input")

                from .history_db import save_image_from_base64, is_available
                if not is_available():
                    return web.json_response(
                        {"filename": None, "available": False}
                    )

                filename = save_image_from_base64(b64_data, image_type)
                return web.json_response({"filename": filename})
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Save image failed: {e}")
                return web.json_response(
                    {"filename": None, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.get("/easyllm/db/image/{filename}")
        async def db_get_image(request):
            """Serve an image file from the database.

            Returns raw PNG bytes with the correct content type header.
            """
            filename = request.match_info["filename"]
            try:
                from .history_db import get_image_data, is_available
                if not is_available():
                    return web.Response(status=404, text="Database not available")

                data = get_image_data(filename)
                if data is None:
                    return web.Response(status=404, text="Image not found")

                return web.Response(
                    body=data,
                    content_type="image/png",
                    headers={
                        "Cache-Control": "public, max-age=86400",
                    },
                )
            except Exception as e:
                logging.warning(
                    f"[LLM Chat DB] GET image failed for {filename}: {e}"
                )
                return web.Response(status=500, text=str(e))

        @PromptServer.instance.routes.post("/easyllm/db/cleanup")
        async def db_cleanup(request):
            """Run database cleanup: remove orphaned and expired data.

            Idempotent — safe to call repeatedly.
            Returns stats dict with removal counts.
            """
            try:
                from .history_db import cleanup, is_available
                if not is_available():
                    return web.json_response({"status": "unavailable"})

                stats = cleanup()
                return web.json_response({"status": "ok", "stats": stats})
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Cleanup failed: {e}")
                return web.json_response(
                    {"status": "error", "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/db/scan-orphans")
        async def db_scan_orphans(request):
            """Scan for orphaned images without deleting anything.

            Returns count and total bytes of images not referenced by any session.
            """
            try:
                from .history_db import scan_orphaned_images, is_available
                if not is_available():
                    return web.json_response({"available": False})

                result = scan_orphaned_images()
                return web.json_response({"status": "ok", "orphans": result})
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Scan orphans failed: {e}")
                return web.json_response(
                    {"status": "error", "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.get("/easyllm/db/stats")
        async def db_stats(request):
            """Return database statistics.

            Returns total_nodes, total_sessions, total_images, disk_size, etc.
            """
            try:
                from .history_db import get_stats, is_available
                if not is_available():
                    return web.json_response({"available": False})

                stats = get_stats()
                return web.json_response({"available": True, **stats})
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Stats failed: {e}")
                return web.json_response(
                    {"available": False, "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.get("/easyllm/db/search")
        async def db_search(request):
            """Search across all stored conversations for matching entries.

            Query params:
                q (str): The search term.
                max (int, optional): Maximum results (default 50).

            Returns list of matching entries with node_id, session_id,
            hist_type, role, message/input/output, and timestamp.
            """
            try:
                query = request.query.get("q", "").strip()
                if not query:
                    return web.json_response({"results": [], "error": "Missing 'q' parameter"}, status=400)

                max_results = int(request.query.get("max", "50"))

                from .history_db import search_history, is_available
                if not is_available():
                    return web.json_response({"results": [], "available": False})

                results = search_history(query, max_results=max_results)
                return web.json_response({"results": results, "available": True})
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Search failed: {e}")
                return web.json_response(
                    {"results": [], "error": str(e)}, status=500
                )

        @PromptServer.instance.routes.get("/easyllm/db/export")
        async def db_export(request):
            """Export all conversations as a downloadable file.

            Query params:
                format (str): ``"md"`` (default) or ``"json"``.

            Returns the formatted export as plain text with appropriate
            Content-Type and Content-Disposition headers.
            """
            try:
                export_format = request.query.get("format", "md").strip().lower()
                if export_format not in ("md", "json"):
                    return web.json_response(
                        {"error": "Format must be 'md' or 'json'"}, status=400
                    )

                from .history_db import export_all_history, is_available
                if not is_available():
                    return web.json_response({"error": "Database not available"}, status=503)

                content = export_all_history(format=export_format)
                content_type = "text/markdown" if export_format == "md" else "application/json"
                filename = f"easyllm_export.{export_format}"

                return web.Response(
                    text=content,
                    content_type=content_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
            except Exception as e:
                logging.warning(f"[LLM Chat DB] Export failed: {e}")
                return web.json_response(
                    {"error": str(e)}, status=500
                )

        # ── Phase 5: Session listing API ─────────────────────────────────
        @PromptServer.instance.routes.get("/easyllm/db/sessions")
        async def db_list_sessions(request):
            """List all stored sessions with metadata.

            Returns a list of sessions with session_id, node_id, hist_type,
            entry_count, created_at, updated_at, model_name, and preview text.
            """
            try:
                from .history_db import list_sessions, is_available
                if not is_available():
                    return web.json_response({"sessions": [], "available": False})

                sessions = list_sessions()
                return web.json_response({"sessions": sessions, "available": True})
            except Exception as e:
                logging.warning(f"[LLM Chat DB] List sessions failed: {e}")
                return web.json_response(
                    {"sessions": [], "error": str(e)}, status=500
                )

        # ── Enhancer Export API ──────────────────────────────────────────
        @PromptServer.instance.routes.post("/easyllm/export/enhancer")
        async def export_enhancer(request):
            """Export selected enhancer entries to a server directory.

            Request body (JSON):
                entries (list): Enhancer history entries.
                options (dict): Export configuration — see
                    ``history_db.export_enhancer_entries()``.

            Returns JSON with success, file_count, output_path, images_written.
            """
            try:
                data = await request.json()
                entries = data.get("entries", [])
                options = data.get("options", {})

                if not entries:
                    return web.json_response(
                        {"success": False, "error": "No entries to export"},
                        status=400,
                    )

                output_dir = options.get("output_dir", "").strip()
                if not output_dir:
                    return web.json_response(
                        {"success": False, "error": "No output directory specified"},
                        status=400,
                    )

                # Security: normalize path and prevent traversal
                output_dir = os.path.abspath(output_dir)
                # Ensure no traversal beyond root
                if ".." in output_dir.split(os.sep):
                    return web.json_response(
                        {"success": False, "error": "Invalid path (directory traversal detected)"},
                        status=400,
                    )

                from .history_db import export_enhancer_entries

                result = export_enhancer_entries(entries, options)
                status = 200 if result.get("success") else 400
                return web.json_response(result, status=status)

            except Exception as e:
                logging.warning(
                    f"[LLM Chat] Enhancer export failed: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        # ── V2: Enhancer Export (Redesigned) ──
        @PromptServer.instance.routes.post("/easyllm/export/enhancer_v2")
        async def export_enhancer_v2(request):
            """Export enhancer entries using the redesigned v2 schema.

            Request body (JSON):
                entries (list): Enhancer history entries.
                options (dict): Export configuration — see
                    ``history_db.export_enhancer_entries_v2()``.

            Returns JSON with success, file_count, output_path, images_written.
            """
            try:
                data = await request.json()
                entries = data.get("entries", [])
                options = data.get("options", {})

                if not entries:
                    return web.json_response(
                        {"success": False, "error": "No entries to export"},
                        status=400,
                    )

                output_dir = options.get("output_dir", "").strip()
                if not output_dir:
                    return web.json_response(
                        {"success": False, "error": "No output directory specified"},
                        status=400,
                    )

                # Security: normalize path and prevent traversal
                output_dir = os.path.abspath(output_dir)
                if ".." in output_dir.split(os.sep):
                    return web.json_response(
                        {"success": False, "error": "Invalid path (directory traversal detected)"},
                        status=400,
                    )

                from .history_db import export_enhancer_entries_v2

                result = export_enhancer_entries_v2(entries, options)
                status = 200 if result.get("success") else 400
                return web.json_response(result, status=status)

            except Exception as e:
                logging.warning(
                    f"[LLM Chat] Enhancer v2 export failed: {e}"
                )
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        # ── Scan files for auto-increment counter ──
        @PromptServer.instance.routes.get("/easyllm/export/scan_files")
        async def export_scan_files(request):
            """Scan a directory for existing numbered files to determine
            the next auto-increment counter.

            Query params:
                dir (str): Directory to scan.
                base_name (str): Base filename (without counter or extension).
                ext (str): File extension including dot, e.g. ``.txt``.

            Returns JSON with:
                max_counter (int): Highest existing counter (0 if none).
                files (list): Matching filenames found.
            """
            try:
                dir_path = request.query.get("dir", "").strip()
                base_name = request.query.get("base_name", "").strip()
                ext = request.query.get("ext", ".txt").strip()

                if not dir_path:
                    return web.json_response(
                        {"max_counter": 0, "files": []}
                    )

                dir_path = os.path.abspath(dir_path)
                if ".." in dir_path.split(os.sep):
                    return web.json_response(
                        {"error": "Invalid path"}, status=400
                    )
                if not os.path.isdir(dir_path):
                    return web.json_response(
                        {"max_counter": 0, "files": []}
                    )

                import re
                safe_base = re.escape(base_name) if base_name else ""
                pattern = re.compile(
                    rf"^{safe_base}(\d+){re.escape(ext)}$"
                )

                max_counter = 0
                matching_files = []
                try:
                    for fname in os.listdir(dir_path):
                        m = pattern.match(fname)
                        if m:
                            matching_files.append(fname)
                            num = int(m.group(1))
                            if num > max_counter:
                                max_counter = num
                except OSError:
                    pass

                return web.json_response({
                    "max_counter": max_counter,
                    "files": matching_files,
                })

            except Exception as e:
                return web.json_response(
                    {"error": str(e)}, status=500
                )

        @PromptServer.instance.routes.get("/easyllm/export/list_dir")
        async def export_list_dir(request):
            """List directory contents for the export folder browser.

            Query params:
                path (str): Directory to list (default: "" — returns
                    common base directories).

            Returns JSON with path, parent, contents (list of {name, type}).
            """
            try:
                dir_path = request.query.get("path", "").strip()

                if not dir_path:
                    # Return common base directories
                    import folder_paths
                    common = []
                    try:
                        from .history_db import get_db_path
                        common.append({"name": get_db_path(), "type": "dir"})
                    except Exception:
                        pass
                    try:
                        common.append({"name": folder_paths.get_output_directory(), "type": "dir"})
                    except Exception:
                        pass
                    try:
                        common.append({"name": folder_paths.base_path, "type": "dir"})
                    except Exception:
                        pass
                    # Dedup
                    seen = set()
                    unique = []
                    for c in common:
                        if c["name"] not in seen:
                            seen.add(c["name"])
                            unique.append(c)
                    return web.json_response({
                        "path": "",
                        "parent": None,
                        "contents": unique,
                    })

                # List the requested directory
                dir_path = os.path.abspath(dir_path)
                if ".." in dir_path.split(os.sep):
                    return web.json_response(
                        {"error": "Invalid path"}, status=400
                    )
                if not os.path.isdir(dir_path):
                    return web.json_response(
                        {"error": "Directory not found"}, status=404
                    )
                if not os.access(dir_path, os.R_OK):
                    return web.json_response(
                        {"error": "Directory not readable"}, status=403
                    )

                parent = os.path.dirname(dir_path)
                contents = []
                try:
                    for name in sorted(os.listdir(dir_path)):
                        full = os.path.join(dir_path, name)
                        try:
                            if os.path.isdir(full):
                                contents.append({"name": name, "type": "dir"})
                        except OSError:
                            pass
                except OSError:
                    pass

                return web.json_response({
                    "path": dir_path,
                    "parent": parent if parent != dir_path else None,
                    "contents": contents,
                })

            except Exception as e:
                logging.warning(
                    f"[LLM Chat] List dir failed: {e}"
                )
                return web.json_response(
                    {"error": str(e)}, status=500
                )

        @PromptServer.instance.routes.post("/easyllm/export/validate_path")
        async def export_validate_path(request):
            """Validate that a path exists and is writable.

            Request body (JSON):
                path (str): Directory path to validate.

            Returns JSON with valid (bool) and optional error.
            """
            try:
                data = await request.json()
                dir_path = data.get("path", "").strip()

                if not dir_path:
                    return web.json_response(
                        {"valid": False, "error": "No path specified"}
                    )

                dir_path = os.path.abspath(dir_path)
                if ".." in dir_path.split(os.sep):
                    return web.json_response(
                        {"valid": False, "error": "Invalid path"}
                    )

                # Does it exist?
                if not os.path.isdir(dir_path):
                    # Can we create it?
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                    except OSError as e:
                        return web.json_response(
                            {"valid": False, "error": f"Cannot create directory: {e}"}
                        )

                if not os.access(dir_path, os.W_OK):
                    return web.json_response(
                        {"valid": False, "error": "Directory is not writable"}
                    )

                return web.json_response({"valid": True})

            except Exception as e:
                return web.json_response(
                    {"valid": False, "error": str(e)}
                )

        setup_streaming_routes._registered = True
        logging.info("[LLM Chat] Streaming + Database API routes registered")

    except Exception as e:
        logging.error(
            f"[LLM Chat] Failed to register streaming routes: {e} — "
            "streaming will be unavailable until ComfyUI restarts"
        )


def get_chat_history(unique_id: str) -> list | None:
    """Retrieve and remove stored chat history for a node.

    Used by chat_node.py's chat() method during execution.

    Phase 2: Falls back to the on-disk database if the ephemeral
    store is empty (crash recovery after restart).
    """
    # ── OLD: primary path — ephemeral store (frontend POSTs before queue) ──
    history = _history_store.pop(unique_id, None)
    if history is not None:
        return history

    # ── NEW: fallback path — on-disk database (crash recovery) ──
    try:
        from .history_db import get_history, is_available
        if is_available():
            entries = get_history(unique_id, "chat")
            if entries is not None:
                logging.info(
                    f"[LLM Chat DB] History recovered from disk for node {unique_id}"
                )
                return entries
    except Exception as e:
        logging.warning(
            f"[LLM Chat DB] Failed to read history for node {unique_id}: {e}"
        )

    return None


# ── Helpers ─────────────────────────────────────────────────────────


def is_popup_mode(unique_id) -> bool:
    """Check if the given node has an active popup.

    Used by EasyLLM.chat() to decide whether to use the streaming path
    (popup open) or the blocking path (Queue Prompt without popup).
    """
    return _state.is_popup_mode(unique_id)


def is_aborted(unique_id) -> bool:
    """Check if generation was aborted for this node.

    Atomically checks and consumes the abort flag to prevent the same
    signal from affecting subsequent generations.

    Returns:
        True if an abort was requested, False otherwise.
    """
    return _state.is_aborted(unique_id)


# ── Streaming Generation ────────────────────────────────────────────




def generate_text(clip, token_dict, max_length=256,
                  temperature=0.7, top_k=50, top_p=0.9,
                  seed=42, do_sample=True, repetition_penalty=1.2,
                  vram_mode="unload", use_mlock=False,
                  use_layer_offloading=False,
                  streaming_callback=None,
                  node_id=None) -> list:
    """Generate text tokens with optional per-token streaming via callback.

    When `streaming_callback` is None, behaves as standard blocking generation.
    When provided, registers the callback for per-token notification,
    wraps it with profiler tracking, enables progress WebSocket events,
    sends a "done" WebSocket signal on completion, and cleans up in the
    finally block.

    The callback is registered via cuda_optimizations.set_streaming_callback()
    and cleared in the finally block. Fires from the GPU-native generation
    loop (or cond_stage.generate() fallback) each time a token is sampled.

    When vram_mode="keep_loaded", the model stays on GPU after generation
    for faster sequential calls.

    Args:
        clip: A comfy.sd.CLIP object
        token_dict: Token dict from build_token_dict()
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-K sampling
        top_p: Top-P / nucleus sampling
        seed: Random seed
        do_sample: If False, use greedy decoding
        repetition_penalty: Penalty for repeated tokens (>1.0 discourages repeats)
        vram_mode: "unload" (free VRAM after gen) or "keep_loaded" (stay on GPU)
        use_mlock: Lock model memory to prevent OS swapping
        use_layer_offloading: Enable layer-by-layer VRAM offloading for large models
        streaming_callback: Optional callable(token_id) for per-token streaming.
                            When provided, also enables progress events and sends
                            a "done" WebSocket signal on completion.
        node_id: Required when streaming_callback is provided. Used for abort
                 flag cleanup and "done" signal routing.

    Returns:
        list: Generated token IDs. Returns empty list if stream was aborted.
    """
    import torch

    # ── Create profiler for this generation run ──
    from .profiler import GenerationProfiler
    model_name = ""
    try:
        if hasattr(clip, "cond_stage_model") and hasattr(clip.cond_stage_model, "clip_name"):
            model_name = str(clip.cond_stage_model.clip_name)
    except Exception:
        pass
    profiler = GenerationProfiler(model_name=model_name, max_tokens=max_length)

    cond_stage = clip.cond_stage_model  # SD1ClipModel

    # ── Begin profiling: Model Setup phase ──
    profiler.begin_phase("model_setup")

    # ── Shared model setup: GPU acceleration, quantization, weight tying ──
    from .memory_manager import prepare_model_for_generation
    inner_sd_clip, transformer, original_device = prepare_model_for_generation(
        clip,
        use_layer_offloading=use_layer_offloading,
        vram_mode=vram_mode,
    )

    profiler.end_phase("model_setup")

    # ── Streaming setup ──
    from .cuda_optimizations import (
        optimized_generation_context,
        generate_text_gpu,
    )
    from .memory_manager import _conditionally_offload_model

    if streaming_callback is not None:
        # Wrap callback to inject profiler tracking (mark_first_token, record_token).
        def _profiled_callback(token_id):
            profiler.mark_first_token()
            profiler.record_token()
            streaming_callback(token_id)

        _state.set_streaming_callback(_profiled_callback)
        _state.enable_progress(node_id)

    try:
        profiler.begin_phase("generation_loop")

        with optimized_generation_context(
            model=transformer if transformer is not None else None,
            compile_model=(
                torch.cuda.is_available()
                and transformer is not None
                and hasattr(transformer, "model")
            ),
        ):
            # GPU-native generation engine.
            # Keeps token history on GPU, no CPU-GPU sync per token, no Python
            # loop overhead. Falls back to cond_stage.generate() automatically
            # if the GPU-native path is unavailable.
            generated_tokens = generate_text_gpu(
                cond_stage,
                token_dict,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                presence_penalty=0.0,
                profiler=profiler,
                use_layer_offloading=use_layer_offloading,
            )

        profiler.end_phase("generation_loop")
        profiler.print_summary_table(logger=logging)
        return generated_tokens

    except StopIteration:
        # StopIteration raised by streaming callback on abort. Tokens already sent via callback.
        profiler.end_phase("generation_loop")
        profiler.print_summary_table(logger=logging)
        logging.info(f"[LLM Chat] Stream aborted for node {node_id}")
        return []

    finally:
        if streaming_callback is not None:
            _state.clear_streaming_callback()

        # Disable progress (paired with unconditional enable_progress above).
        _state.disable_progress()

        # Clean up abort flag.
        if node_id is not None:
            _state.cleanup_abort(node_id)

        # Send "done" signal to frontend with timing data (streaming path only).
        if streaming_callback is not None:
            from server import PromptServer
            try:
                # Build timing data from profiler
                timing = {}
                if profiler is not None:
                    gen_loop_s = profiler.get_total_generation_time()
                    timing["duration_ms"] = round(gen_loop_s * 1000, 1)
                    timing["ttft_ms"] = round(profiler.get_ttft() * 1000, 1)
                    timing["token_count"] = profiler._token_count if hasattr(profiler, '_token_count') else 0
                    timing["tokens_per_second"] = round(profiler.get_overall_tps(), 1)

                PromptServer.instance.send_sync("llm_lab_token", {
                    "node_id": node_id,
                    "token": "",
                    "done": True,
                    "timing": timing,
                })
            except Exception:
                pass

        # Conditionally offload based on vram_mode.
        _conditionally_offload_model(transformer, original_device, inner_sd_clip, vram_mode)


# ── Generation Orchestration ──


def execute_chat_generation(clip, text, mode, effective_system_prompt, max_length=256,
                            temperature="auto", seed=0, vram_mode="unload",
                            use_mlock=False, unique_id=None,
                            chat_history=None):
    """Full generation orchestration for EasyLLM node.

    Handles all steps from prompt formatting through text cleaning:
      1. Format prompt (chat or enhancer template)
      2. Tokenize (bypassing outer template wrapping)
      3. Get optimal sampling params for this model
      4. Generate tokens (streaming if popup open, blocking otherwise)
      5. Decode token IDs to text
      6. Clean generated text

    Args:
        clip: The CLIP object containing a generation-capable LLM.
        text: The user's message text.
        mode: 'chat' or 'enhancer'.
        effective_system_prompt: The resolved system prompt string.
        max_length: Maximum tokens to generate.
        temperature: 'auto' or a string float like '0.5'.
        seed: Random seed (0 = auto-randomized upstream).
        vram_mode: 'unload' or 'keep_loaded'.
        use_mlock: Lock model memory to prevent OS swapping.
        unique_id: The node's unique ID for popup routing.

    Returns:
        tuple: (cleaned_text, raw_text)
    """
    from .utils import (
        format_prompt,
        format_prompt_with_history,
        build_token_dict,
        get_optimal_sampling_params,
        decode_token_ids,
        clean_generated_text,
    )
    from .streaming import generate_text

    # Step 1: Format prompt — history-aware if available, else standard.
    if chat_history:
        formatted_text = format_prompt_with_history(
            text, chat_history, effective_system_prompt
        )
    else:
        formatted_text = format_prompt(text, effective_system_prompt)

    # Step 2: Tokenize (bypassing outer template wrapping)
    token_dict = build_token_dict(clip, formatted_text)

    # Step 3: Get optimal sampling params for this model
    params = get_optimal_sampling_params(clip, temperature)
    do_sample = params["temperature"] > 0.0
    temp = params["temperature"] if do_sample else 0.0

    # ── Enable progress events for all executions ──
    # Powers the canvas green bar (enhancer) and popup progress bar (chat).
    _state.enable_progress(unique_id)

    # Step 4: Generate text tokens — streaming path (popup) or blocking path
    if is_popup_mode(unique_id):
        # ── Tokenizer for per-token decoding in streaming callback ──
        from .utils import _get_tokenizer
        inner_tokenizer = _get_tokenizer(clip)
        hf_tokenizer = getattr(inner_tokenizer, 'tokenizer', inner_tokenizer)

        def _streaming_callback(token_id):
            # Check abort flag — raise StopIteration to abort generation
            if is_aborted(unique_id):
                raise StopIteration(f"Stream aborted for node {unique_id}")

            # Decode single token to text
            try:
                token_text = (
                    hf_tokenizer.decode([token_id], skip_special_tokens=False)
                    if hf_tokenizer is not None else ""
                )
            except Exception:
                token_text = ""

            # Strip replacement characters from per-token decode
            # (single-token decode can produce U+FFFD for partial multi-byte sequences)
            if token_text:
                token_text = token_text.replace("\ufffd", "")

            if token_text:
                from server import PromptServer
                PromptServer.instance.send_sync("llm_lab_token", {
                    "node_id": unique_id,
                    "token": token_text,
                    "done": False,
                })

        token_ids = generate_text(
            clip, token_dict,
            max_length=max_length,
            temperature=temp,
            top_k=params["top_k"],
            top_p=params["top_p"],
            seed=seed,
            do_sample=do_sample,
            repetition_penalty=params["repetition_penalty"],
            vram_mode=vram_mode,
            use_mlock=use_mlock,
            streaming_callback=_streaming_callback,
            node_id=unique_id,
        )
    else:
        token_ids = generate_text(
            clip, token_dict,
            max_length=max_length,
            temperature=temp,
            top_k=params["top_k"],
            top_p=params["top_p"],
            seed=seed,
            do_sample=do_sample,
            repetition_penalty=params["repetition_penalty"],
            vram_mode=vram_mode,
            use_mlock=use_mlock,
        )

    # Step 5: Decode token IDs to text
    raw_text = decode_token_ids(clip, token_ids)

    # Step 6: Clean up any artifacts
    cleaned_text = clean_generated_text(raw_text)

    return cleaned_text, raw_text


# ── GGUF Generation Orchestration ──────────────────────────────────────


def execute_gguf_generation(
    model,
    formatted_prompt: str,
    mode: str = "chat",
    max_length: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int = 0,
    stop: list[str] | None = None,
    repetition_penalty: float = 1.0,
    unique_id: str | None = None,
) -> str:
    """Full generation orchestration for GGUF node (streaming/popup path).

    Handles streaming with progress events, timing data collection,
    per-token WebSocket emission, and abort support.

    Args:
        model: A loaded LlamaCppModel instance (caller handles caching).
        formatted_prompt: The fully formatted prompt string (system + history + user).
        mode: 'chat' or 'enhancer'.
        max_length: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-K sampling.
        top_p: Top-P / nucleus sampling.
        seed: Random seed (0 = auto-randomized upstream).
        stop: List of stop tokens for this chat template.
        repetition_penalty: Repetition penalty (1.0 = none).
        unique_id: The node's unique ID for popup/progress routing.

    Returns:
        str: The raw generated text (caller applies clean_generated_text()).
    """
    import time

    from .generation_state import get_state
    _state_gguf = get_state()

    # Enable progress events for the frontend progress bar
    _state_gguf.enable_progress(unique_id)

    start_time = time.perf_counter()
    ttft = None
    token_count = 0
    accumulated_text = []
    _diag_log_interval = 50  # log every N tokens

    logging.info(
        f"[DIAG] execute_gguf_generation START for node={unique_id}, "
        f"prompt_len={len(formatted_prompt)}, max_tokens={max_length}"
    )

    try:
        # Stream tokens via the C++ engine's streaming generator
        for token_text in model.generate_stream(
            prompt=formatted_prompt,
            max_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=stop or [],
            repetition_penalty=repetition_penalty,
        ):
            # Check abort flag (mid-stream cancellation)
            if is_aborted(unique_id):
                logging.info(
                    f"[LLM Chat GGUF] Stream aborted for node {unique_id}"
                )
                break

            # Record TTFT (Time To First Token)
            if ttft is None:
                ttft = time.perf_counter() - start_time
                logging.info(
                    f"[DIAG] TTFT for node {unique_id}: {ttft*1000:.1f}ms"
                )

            token_count += 1

            # Diagnostic: log first few tokens and periodic stats
            if token_count <= 5 or token_count % _diag_log_interval == 0:
                elapsed = time.perf_counter() - start_time
                logging.info(
                    f"[DIAG] Stream token #{token_count} for node {unique_id}: "
                    f"len={len(token_text)}, "
                    f"repr={repr(token_text[:80])}, "
                    f"elapsed={elapsed*1000:.1f}ms"
                )

            # Emit progress event for canvas green bar + popup progress bar
            if _state_gguf.progress_enabled:
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync("progress", {
                        "value": token_count,
                        "max": max_length,
                        "node": unique_id,
                    })
                except Exception:
                    pass

            # Emit token for streaming display in the popup
            try:
                from server import PromptServer
                PromptServer.instance.send_sync("llm_lab_token", {
                    "node_id": unique_id,
                    "token": token_text,
                    "done": False,
                })
            except Exception:
                pass

            # Accumulate tokens for the final text result
            accumulated_text.append(token_text)

        # Log final token count
        duration = time.perf_counter() - start_time
        tps = token_count / duration if duration > 0 else 0.0
        logging.info(
            f"[DIAG] execute_gguf_generation DONE for node={unique_id}: "
            f"total_tokens={token_count}, duration={duration*1000:.1f}ms, "
            f"tps={tps:.1f}"
        )

        # Compute timing data
        duration = time.perf_counter() - start_time
        timing = {}
        if token_count > 0:
            timing = {
                "duration_ms": round(duration * 1000, 1),
                "ttft_ms": round(ttft * 1000, 1) if ttft is not None else 0,
                "token_count": token_count,
                "tokens_per_second": round(
                    token_count / duration, 1
                ) if duration > 0 else 0,
            }

        # Emit final 100% progress event
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("progress", {
                "value": max_length,
                "max": max_length,
                "node": unique_id,
            })
        except Exception:
            pass

        # Emit "done" signal with timing data
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("llm_lab_token", {
                "node_id": unique_id,
                "token": "",
                "done": True,
                "timing": timing,
            })
        except Exception:
            pass

    except Exception as e:
        logging.error(
            f"[LLM Chat GGUF] Error during streaming generation "
            f"for node {unique_id}: {e}"
        )
        # Return partial accumulation on error
        import traceback
        traceback.print_exc()

    finally:
        _state_gguf.disable_progress()
        _state_gguf.cleanup_abort(unique_id)

    # Join accumulated tokens into the final raw text
    raw_text = "".join(accumulated_text)
    return raw_text


def execute_gguf_chat_generation(
    model,
    messages: list[dict],
    mode: str = "chat",
    max_length: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int = 0,
    repetition_penalty: float = 1.0,
    stop: list[str] | None = None,
    unique_id: str | None = None,
    response_format: dict | None = None,
) -> str:
    """Full generation orchestration for GGUF multimodal (vision) streaming path.

    Uses create_chat_completion(stream=True) via model.generate_chat_stream(),
    which takes an OpenAI-compatible messages array instead of a formatted prompt.
    Accepts optional stop tokens to halt generation when template tokens appear.

    Args:
        model: A loaded LlamaCppModel instance with vision support (mmproj loaded).
        messages: OpenAI-compatible messages array, e.g.:
                  [{"role": "system", "content": "..."},
                   {"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                       {"type": "text", "text": "What's in this image?"}
                   ]}]
        mode: 'chat' or 'enhancer'.
        max_length: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-K sampling.
        top_p: Top-P / nucleus sampling.
        seed: Random seed (0 = auto-randomized upstream).
        repetition_penalty: Repetition penalty (1.0 = none).
        stop: Optional list of stop strings. Forwarded to
              model.generate_chat_stream() to halt generation when
              these sequences appear. Use CHAT_TEMPLATES["stop"] values.
        unique_id: The node's unique ID for popup/progress routing.

    Returns:
        str: The raw generated text (caller applies clean_generated_text()).

    Raises:
        RuntimeError: If the model does not support vision (no mmproj loaded).
    """
    import time

    from .generation_state import get_state
    _state_gguf = get_state()

    # Validate vision support
    if not getattr(model, 'supports_vision', False):
        raise RuntimeError(
            "Model does not support vision/multimodal input. "
            "An mmproj (multimodal projection) file must be provided "
            "and loaded for image inputs."
        )

    # Enable progress events for the frontend progress bar
    _state_gguf.enable_progress(unique_id)

    start_time = time.perf_counter()
    ttft = None
    token_count = 0
    accumulated_text = []
    _diag_log_interval = 50  # log every N tokens

    _has_image = any(
        isinstance(m.get("content"), list) and any(
            c.get("type") == "image_url" for c in m["content"]
        )
        for m in messages
    )
    logging.info(
        f"[DIAG] execute_gguf_chat_generation START for node={unique_id}, "
        f"messages_len={len(messages)}, max_tokens={max_length}, "
        f"has_image_url={_has_image}, "
        f"roles={[m['role'] for m in messages]}"
    )

    try:
        # Stream tokens via the C++ engine's chat completion streaming
        for token_text in model.generate_chat_stream(
            messages=messages,
            max_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            repetition_penalty=repetition_penalty,
            stop=stop,
            response_format=response_format,
        ):
            # Check abort flag (mid-stream cancellation)
            if is_aborted(unique_id):
                logging.info(
                    f"[LLM Chat GGUF] Chat stream aborted for node {unique_id}"
                )
                break

            # Record TTFT (Time To First Token)
            if ttft is None:
                ttft = time.perf_counter() - start_time
                logging.info(
                    f"[DIAG] Chat TTFT for node {unique_id}: {ttft*1000:.1f}ms"
                )

            token_count += 1

            # Diagnostic: log first few tokens and periodic stats
            if token_count <= 5 or token_count % _diag_log_interval == 0:
                elapsed = time.perf_counter() - start_time
                logging.info(
                    f"[DIAG] Chat stream token #{token_count} for node {unique_id}: "
                    f"len={len(token_text)}, "
                    f"repr={repr(token_text[:80])}, "
                    f"elapsed={elapsed*1000:.1f}ms"
                )

            # Emit progress event for canvas green bar + popup progress bar
            if _state_gguf.progress_enabled:
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync("progress", {
                        "value": token_count,
                        "max": max_length,
                        "node": unique_id,
                    })
                except Exception:
                    pass

            # Emit token for streaming display in the popup
            try:
                from server import PromptServer
                PromptServer.instance.send_sync("llm_lab_token", {
                    "node_id": unique_id,
                    "token": token_text,
                    "done": False,
                })
            except Exception:
                pass

            # Accumulate tokens for the final text result
            accumulated_text.append(token_text)

        # Log final token count
        duration = time.perf_counter() - start_time
        tps = token_count / duration if duration > 0 else 0.0
        logging.info(
            f"[DIAG] execute_gguf_chat_generation DONE for node={unique_id}: "
            f"total_tokens={token_count}, duration={duration*1000:.1f}ms, "
            f"tps={tps:.1f}"
        )

        # Compute timing data
        duration = time.perf_counter() - start_time
        timing = {}
        if token_count > 0:
            timing = {
                "duration_ms": round(duration * 1000, 1),
                "ttft_ms": round(ttft * 1000, 1) if ttft is not None else 0,
                "token_count": token_count,
                "tokens_per_second": round(
                    token_count / duration, 1
                ) if duration > 0 else 0,
            }

        # Emit final 100% progress event
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("progress", {
                "value": max_length,
                "max": max_length,
                "node": unique_id,
            })
        except Exception:
            pass

        # Emit "done" signal with timing data
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("llm_lab_token", {
                "node_id": unique_id,
                "token": "",
                "done": True,
                "timing": timing,
            })
        except Exception:
            pass

    except Exception as e:
        logging.error(
            f"[LLM Chat GGUF] Error during chat streaming generation "
            f"for node {unique_id}: {e}"
        )
        # Return partial accumulation on error.
        import traceback
        traceback.print_exc()

    finally:
        _state_gguf.disable_progress()
        _state_gguf.cleanup_abort(unique_id)

    # Join accumulated tokens into the final raw text
    raw_text = "".join(accumulated_text)
    return raw_text
