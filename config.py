"""
User-configurable settings for LLM Chat node behaviour.

All settings in this file can be overridden by creating a ``config_user.py``
in the same directory with the same variable names. The user file is imported
after this one and takes precedence.

Import in any module::

    from .config import REATTACH_IMAGES, USE_HISTORY_DATABASE
"""

# ── Image Persistence Across Chat Turns ───────────────────────────────────
# When ``True``, images sent in previous chat turns are re-attached to
# subsequent messages so the vision model can re-examine them across the
# conversation (e.g. follow-up questions about the same image).
#
# When ``False``, only text from previous turns is preserved (the original
# behaviour). Images are sent once and discarded after the turn they
# appeared in.
#
# **Disable this on low-VRAM GPUs** (6 GB or less) to keep the context
# window from growing unnecessarily when images are persisted across turns.
REATTACH_IMAGES: bool = False

# ── Iterative Image Refinement ─────────────────────────────────────────────
# Default value for the per-node ``iterative_refinement`` boolean widget
# (accessible via the Settings popup in the LLM Chat popup).
#
# When ``True``, the generated/edited output image replaces the uploaded
# input image in the pipeline cache after each ``edit_image`` turn.
# Subsequent "refine" or "edit" requests without re-uploading will use
# the latest generated result as the source instead of the original.
#
# When ``False`` (default), all edits use the original uploaded image
# as the source — the current (pre-iterative) behavior.
#
# Note: This only affects the pipeline (img2img source for Flux Klein).
# The vision context sent to the LLM is never modified.
#
# The runtime value can be toggled per-node via the Settings popup
# (checkbox labelled "Iterative Refine").  This config value acts as
# the default for nodes where the widget hasn't been explicitly changed.
ITERATIVE_REFINEMENT: bool = False


# ── Trigger Prompt Output ──────────────────────────────────────────
# When ``True``, the node attempts to parse structured JSON from model
# output and exposes it as a ``trigger_prompt`` STRING output socket.
#
# When ``False`` (default), the trigger_prompt socket still exists but
# always contains an empty string. Existing workflows are unaffected.
#
# Enabling this adds a small regex/JSON parse step to every generation.
# The extra output socket is harmless when left unwired.
ENABLE_TRIGGER_PROMPT: bool = True


# ── JSON Database (History Persistence) ───────────────────────────────────
# When ``True``, chat and enhancer history is persisted to disk in
# ComfyUI's user data directory.  Survives restarts without requiring
# the user to save the workflow.
#
# When ``False``, uses the old in-memory + workflow JSON approach
# (existing behaviour).
USE_HISTORY_DATABASE: bool = True

# Directory for the history database.
# Default: resolves to ``ComfyUI/user/default/easyllm_db/``
# Override to any writable path to customise storage location.
HISTORY_DB_PATH: str = ""  # empty = use default

# Maximum disk usage for the history database in MB.
# 0 = unlimited.
HISTORY_DB_MAX_SIZE_MB: int = 500

# Automatically delete history entries older than this many days.
# 0 = never auto-delete.
HISTORY_DB_MAX_AGE_DAYS: int = 0

# Flush history to disk immediately on every change.
# When ``True`` (default), every append/set/clear writes to disk straight
# away — crash-safe but more I/O.
# When ``False``, writes are batched and flushed opportunistically
# (less I/O, but up to 1-2 seconds of data could be lost on crash).
HISTORY_DB_IMMEDIATE_WRITE: bool = True


# ── User Override ─────────────────────────────────────────────────────────
# Import config_user.py if it exists — values there override the defaults
# above. This lets users keep custom settings across updates.
import os as _os
import logging as _logging

_config_dir = _os.path.dirname(_os.path.abspath(__file__))
_user_cfg = _os.path.join(_config_dir, "config_user.py")
if _os.path.isfile(_user_cfg):
    try:
        import importlib.util as _util
        _spec = _util.spec_from_file_location("config_user", _user_cfg)
        if _spec and _spec.loader:
            _mod = _util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            for _key in dir(_mod):
                if not _key.startswith("_"):
                    globals()[_key] = getattr(_mod, _key)
            _logging.info(
                f"[LLM Chat] Loaded user config overrides from config_user.py"
            )
    except Exception as _exc:
        _logging.warning(
            f"[LLM Chat] Failed to load config_user.py: {_exc}"
        )
