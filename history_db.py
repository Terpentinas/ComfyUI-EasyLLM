"""
EasyLLM — Node-ID-Based History Storage

Stores chat history, enhancer output logs, and images as simple JSON files
keyed by node_id. No index.json, no session UUIDs, no chunked directories.

Directory layout::

    <base_path>/easyllm_db/
    ├── chat/
    │   ├── <node_id>.json        # Single file per node
    │   └── ...
    ├── enhancer/
    │   ├── <node_id>.json        # Single file per node
    │   └── ...
    ├── images/
    │   ├── <uuid>_input.png
    │   ├── <uuid>_generated.png
    │   └── ...
    └── settings.json

Thread safety:
    All public write operations use a module-level ``threading.Lock`` to prevent
    concurrent access from multiple API routes or execution threads.

Atomic writes:
    JSON files are written to a ``.tmp`` path first, then atomically renamed
    to the final path. This prevents partial/corrupt files on power loss or crash.
"""

import base64
import json
import logging
import os
import re
import threading
import uuid
import io
import time
import shutil
from datetime import datetime
from typing import Any

from .config import (
    USE_HISTORY_DATABASE,
    HISTORY_DB_PATH,
    HISTORY_DB_MAX_SIZE_MB,
    HISTORY_DB_MAX_AGE_DAYS,
    HISTORY_DB_IMMEDIATE_WRITE,
)

# ── Runtime overrides (set via update_settings / settings.json) ────────
_runtime_max_size_mb: int | None = None
_runtime_max_age_days: int | None = None
_runtime_immediate_write: bool | None = None


def _effective_max_size_mb() -> int:
    return _runtime_max_size_mb if _runtime_max_size_mb is not None else HISTORY_DB_MAX_SIZE_MB


def _effective_max_age_days() -> int:
    return _runtime_max_age_days if _runtime_max_age_days is not None else HISTORY_DB_MAX_AGE_DAYS


def _effective_immediate_write() -> bool:
    return _runtime_immediate_write if _runtime_immediate_write is not None else HISTORY_DB_IMMEDIATE_WRITE


# ── Logging ────────────────────────────────────────────────────────────
_logger = logging.getLogger("LLM Chat DB")

# ── Module-level state ─────────────────────────────────────────────────
_db_lock = threading.Lock()
_db_initialized = False
_db_base_path: str = ""  # Absolute path to easyllm_db/ directory

# ── Image filename pattern for GC ──────────────────────────────────────
_DB_IMAGE_PATTERN = re.compile(r"^[0-9a-f]{32}_(input|generated)\.png$")

# ── Settings persistence ───────────────────────────────────────────────
_SETTINGS_FILE = "settings.json"


def _settings_filepath() -> str:
    return os.path.join(_db_base_path, _SETTINGS_FILE)


def _load_settings_file() -> dict:
    """Load persisted settings from settings.json. Returns empty dict if absent."""
    fpath = _settings_filepath()
    if not os.path.isfile(fpath):
        return {}
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _logger.warning(f"[LLM Chat DB] Failed to load settings: {e}")
        return {}


def _save_settings_file(settings: dict) -> None:
    """Atomically write settings to settings.json."""
    fpath = _settings_filepath()
    tmp = fpath + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        os.replace(tmp, fpath)
    except OSError as e:
        _logger.warning(f"[LLM Chat DB] Failed to save settings: {e}")


# ── Path Resolution ────────────────────────────────────────────────────


def _resolve_base_path() -> str:
    """Resolve the database root directory.

    Priority:
    1. ``HISTORY_DB_PATH`` from config (if non-empty).
    2. ``<ComfyUI user directory>/easyllm_db/``.
    """
    if HISTORY_DB_PATH:
        base = HISTORY_DB_PATH
    else:
        try:
            import folder_paths
            base = os.path.join(folder_paths.get_output_directory(), "..", "easyllm_db")
        except ImportError:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "easyllm_db")
    return os.path.abspath(base)


def _ensure_dirs(path: str) -> None:
    """Create the standard subdirectory structure under *path*."""
    for subdir in ("chat", "enhancer", "images"):
        dirpath = os.path.join(path, subdir)
        try:
            os.makedirs(dirpath, exist_ok=True)
        except OSError as e:
            _logger.warning(f"[LLM Chat DB] Cannot create {dirpath}: {e}")


# ── Initialization ─────────────────────────────────────────────────────


def initialize() -> str:
    """Initialize the database engine.

    Creates the directory structure and deletes the old ``index.json``
    if present (legacy format cleanup).

    Returns:
        The absolute path to the database directory.
    """
    global _db_initialized, _db_base_path

    if not USE_HISTORY_DATABASE:
        _logger.info("[LLM Chat DB] Database is disabled via USE_HISTORY_DATABASE=False")
        return ""

    _db_base_path = os.path.join(_resolve_base_path(), "easyllm_db")
    _ensure_dirs(_db_base_path)

    # ── Delete legacy index.json if present (old format cleanup) ──
    legacy_index = os.path.join(_db_base_path, "index.json")
    if os.path.isfile(legacy_index):
        try:
            os.remove(legacy_index)
            _logger.info("[LLM Chat DB] Removed legacy index.json")
        except OSError as e:
            _logger.warning(f"[LLM Chat DB] Could not remove legacy index.json: {e}")

    # ── Delete any legacy session directories (UUID-named dirs) ──
    # These are session_id directories that don't match node_id pattern.
    # We only want files matching <node_id>.json in chat/ and enhancer/.
    for subdir in ("chat", "enhancer"):
        dirpath = os.path.join(_db_base_path, subdir)
        if not os.path.isdir(dirpath):
            continue
        try:
            for name in os.listdir(dirpath):
                item_path = os.path.join(dirpath, name)
                # Remove directories (legacy chunked session dirs)
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                        _logger.info(f"[LLM Chat DB] Removed legacy session directory: {name}")
                    except OSError as e:
                        _logger.warning(f"[LLM Chat DB] Could not remove legacy dir {name}: {e}")
                # Remove legacy monolithic .json files (they use session UUIDs)
                # We keep files that match typical node_id patterns (no hyphens if they're
                # ComfyUI node IDs like "123", not UUIDs with dashes)
                elif name.endswith(".json") and not name.startswith("."):
                    # Remove UUID-format files (with hyphens), keep simple node_id files
                    base_name = name[:-5]
                    if "-" in base_name or len(base_name) > 36:
                        try:
                            os.remove(item_path)
                            _logger.info(f"[LLM Chat DB] Removed legacy session file: {name}")
                        except OSError as e:
                            _logger.warning(f"[LLM Chat DB] Could not remove legacy file {name}: {e}")
        except OSError:
            pass

    # ── Apply persisted settings ──
    _apply_persisted_settings()

    # ── Start auto-cleanup if enabled ──
    _start_auto_cleanup_if_enabled()

    _db_initialized = True
    _logger.info(f"[LLM Chat DB] Initialized at {_db_base_path}")
    return _db_base_path


def get_db_path() -> str:
    return _db_base_path


def is_available() -> bool:
    return _db_initialized and USE_HISTORY_DATABASE


# ── Node File Helpers ──────────────────────────────────────────────────


def _node_filepath(node_id: str, hist_type: str) -> str:
    """Return the path to a node's history file.

    Args:
        node_id: The ComfyUI node ID.
        hist_type: ``"chat"`` or ``"enhancer"``.

    Returns:
        Absolute path to ``<base>/<hist_type>/<node_id>.json``.
    """
    return os.path.join(_db_base_path, hist_type, f"{node_id}.json")


def _read_node_file(node_id: str, hist_type: str) -> dict | None:
    """Read a node's history file and return the parsed dict.

    Returns ``None`` if the file doesn't exist or is corrupt.
    """
    return _read_json_safe(_node_filepath(node_id, hist_type))


def _write_node_file(node_id: str, hist_type: str, data: dict) -> None:
    """Write a node's history file atomically.

    Automatically updates ``updated_at`` and ``entry_count``.
    """
    data["node_id"] = node_id
    data["hist_type"] = hist_type
    data["updated_at"] = int(time.time() * 1000)
    data["entry_count"] = len(data.get("entries", []))
    _atomic_write_json(_node_filepath(node_id, hist_type), data)


# ── Atomic JSON I/O ────────────────────────────────────────────────────


def _atomic_write_json(filepath: str, data: Any) -> None:
    """Write JSON data to a file atomically.

    Writes to a ``.tmp`` path first, then renames atomically via
    ``os.replace()``. This prevents partial/corrupt files on power loss.
    """
    tmp = filepath + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, filepath)
    except OSError as e:
        _logger.error(f"[LLM Chat DB] Atomic write failed: {filepath} — {e}")


def _read_json_safe(filepath: str) -> dict | None:
    """Read a JSON file and return the parsed dict.

    Returns ``None`` if the file doesn't exist, is empty, or is corrupt.
    """
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _logger.warning(f"[LLM Chat DB] Failed to read {filepath}: {e}")
        return None


def _is_db_filename(name: str) -> bool:
    """Check if a filename matches the DB image pattern (32 hex chars + _ + type + .png)."""
    return bool(_DB_IMAGE_PATTERN.match(name))


# ═══════════════════════════════════════════════════════════════════════
# Core CRUD Operations
# ═══════════════════════════════════════════════════════════════════════


def get_history(
    node_id: str,
    hist_type: str = "chat",
    offset: int = 0,
    limit: int = 0,
    session_id: str | None = None,  # Kept for backward compat, ignored
) -> list[dict] | None:
    """Load history entries for a node from the on-disk database.

    Args:
        node_id: The ComfyUI node ID.
        hist_type: ``"chat"`` or ``"enhancer"``.
        offset: Number of entries to skip from the end (for pagination).
        limit: Max entries to return (0 = all).
        session_id: **Deprecated.** Ignored in the new node-ID-based storage.

    Returns:
        List of entry dicts, or ``None`` if the file doesn't exist.
    """
    if not is_available():
        return None

    data = _read_node_file(node_id, hist_type)
    if data is None:
        return None

    entries = data.get("entries", [])
    if not entries:
        return []

    if offset > 0 or limit > 0:
        # Pagination: offset from end, limit count
        start = max(0, len(entries) - offset - limit) if limit > 0 else 0
        end = len(entries) - offset if offset > 0 else len(entries)
        return entries[start:end]

    return entries


def set_history(
    node_id: str,
    entries: list[dict],
    hist_type: str = "chat",
    metadata: dict | None = None,
) -> bool:
    """Replace all history entries for a node.

    Args:
        node_id: The ComfyUI node ID.
        entries: List of entry dicts.
        hist_type: ``"chat"`` or ``"enhancer"``.
        metadata: Optional metadata dict (stored in the file's ``meta`` field).

    Returns:
        ``True`` on success.
    """
    if not is_available():
        return False

    if not isinstance(entries, list):
        return False

    with _db_lock:
        data = {
            "node_id": node_id,
            "hist_type": hist_type,
            "entries": entries,
            "created_at": int(time.time() * 1000),
            "updated_at": int(time.time() * 1000),
            "entry_count": len(entries),
            "image_count": _count_images_in_entries(entries),
            "meta": metadata or {},
        }
        _write_node_file(node_id, hist_type, data)

    return True


def append_history(
    node_id: str,
    entries: list[dict],
    hist_type: str = "chat",
) -> bool:
    """Append entries to an existing history file, or create a new one.

    Args:
        node_id: The ComfyUI node ID.
        entries: List of entry dicts to append.
        hist_type: ``"chat"`` or ``"enhancer"``.

    Returns:
        ``True`` on success.
    """
    if not is_available():
        return False

    if not entries:
        return False

    with _db_lock:
        data = _read_node_file(node_id, hist_type)
        if data is None:
            # Create new file
            data = {
                "node_id": node_id,
                "hist_type": hist_type,
                "entries": list(entries),
                "created_at": int(time.time() * 1000),
                "updated_at": int(time.time() * 1000),
                "entry_count": len(entries),
                "image_count": _count_images_in_entries(entries),
                "meta": {},
            }
        else:
            existing = data.get("entries", [])
            existing.extend(entries)
            data["entries"] = existing
            # Preserve original created_at
            if "created_at" not in data:
                data["created_at"] = int(time.time() * 1000)
            data["entry_count"] = len(existing)
            data["image_count"] = _count_images_in_entries(existing)

        _write_node_file(node_id, hist_type, data)

    return True


def append_images_to_entry(
    node_id: str,
    session_uuid: str,
    images: list[dict],
    hist_type: str = "chat",
) -> dict | None:
    """Find a history entry by ``_sessionUuid`` and append images to it.

    Args:
        node_id: The ComfyUI node ID.
        session_uuid: The ``_sessionUuid`` value to match.
        images: List of image dicts with ``{"filename": ..., "type": ...}``.
        hist_type: ``"chat"`` or ``"enhancer"``.

    Returns:
        The updated entry dict if found and updated, or ``None`` on failure.
    """
    if not is_available():
        return None

    if not session_uuid or not images:
        return None

    with _db_lock:
        data = _read_node_file(node_id, hist_type)
        if data is None:
            return None

        entries = data.get("entries", [])
        matched_entry = None
        for entry in entries:
            if isinstance(entry, dict) and entry.get("_sessionUuid") == session_uuid:
                existing_images = entry.get("images", [])
                if not isinstance(existing_images, list):
                    existing_images = []
                existing_images.extend(images)
                entry["images"] = existing_images
                matched_entry = entry
                break

        if matched_entry is None:
            return None

        _write_node_file(node_id, hist_type, data)

    return matched_entry


def update_entry(
    node_id: str,
    index: int,
    updates: dict,
    hist_type: str = "chat",
) -> bool:
    """Update a single history entry by its index within the node file.

    Args:
        node_id: The ComfyUI node ID.
        index: The zero-based index of the entry to update.
        updates: Dict of fields to update (merged into the entry).
        hist_type: ``"chat"`` or ``"enhancer"``.

    Returns:
        ``True`` if the entry was found and updated.
    """
    if not is_available():
        return False

    with _db_lock:
        data = _read_node_file(node_id, hist_type)
        if data is None:
            return False

        entries = data.get("entries", [])
        if index < 0 or index >= len(entries):
            return False

        entry = entries[index]
        if not isinstance(entry, dict):
            return False

        entry.update(updates)
        entries[index] = entry
        data["entries"] = entries
        _write_node_file(node_id, hist_type, data)

    return True


def delete_entry(
    node_id: str,
    index: int,
    hist_type: str = "chat",
) -> bool:
    """Delete a single history entry by its index within the node file.

    Args:
        node_id: The ComfyUI node ID.
        index: The zero-based index of the entry to delete.
        hist_type: ``"chat"`` or ``"enhancer"``.

    Returns:
        ``True`` if the entry was found and deleted.
    """
    if not is_available():
        return False

    with _db_lock:
        data = _read_node_file(node_id, hist_type)
        if data is None:
            return False

        entries = data.get("entries", [])
        if index < 0 or index >= len(entries):
            return False

        del entries[index]
        data["entries"] = entries
        _write_node_file(node_id, hist_type, data)

    return True


def clear_history(node_id: str, hist_type: str = "chat") -> bool:
    """Remove all history for a node.

    Deletes the node's JSON file and optionally removes referenced images.

    Args:
        node_id: The ComfyUI node ID.
        hist_type: ``"chat"`` or ``"enhancer"``.

    Returns:
        ``True`` if the file was deleted.
    """
    if not is_available():
        return False

    with _db_lock:
        filepath = _node_filepath(node_id, hist_type)
        if not os.path.isfile(filepath):
            return False

        try:
            os.remove(filepath)
            _logger.info(f"[LLM Chat DB] Cleared history for node {node_id} ({hist_type})")
            return True
        except OSError as e:
            _logger.warning(f"[LLM Chat DB] Failed to clear history for {node_id}: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════
# Image Storage
# ═══════════════════════════════════════════════════════════════════════


def save_image(image_data: bytes, image_type: str = "input") -> str | None:
    """Save an image to the database and return its UUID filename.

    Args:
        image_data: Raw PNG bytes.
        image_type: ``"input"`` (user-uploaded) or ``"generated"``.

    Returns:
        The UUID filename (e.g. ``"abc..._input.png"``) or ``None`` on failure.
    """
    if not is_available():
        return None

    filename = f"{uuid.uuid4().hex}_{image_type}.png"
    img_dir = os.path.join(_db_base_path, "images")
    filepath = os.path.join(img_dir, filename)

    with _db_lock:
        try:
            with open(filepath, "wb") as f:
                f.write(image_data)
            return filename
        except OSError as e:
            _logger.error(f"[LLM Chat DB] Failed to save image: {e}")
            return None


def save_image_from_base64(b64_data: str, image_type: str = "input") -> str | None:
    """Decode a base64 data URI and save it as a PNG file.

    Args:
        b64_data: Base64-encoded image data (with or without ``data:`` prefix).
        image_type: ``"input"`` or ``"generated"``.

    Returns:
        The UUID filename or ``None`` on failure.
    """
    try:
        if b64_data.startswith("data:"):
            b64_part = b64_data.split(",", 1)[-1]
        else:
            b64_part = b64_data
        image_bytes = base64.b64decode(b64_part)
        return save_image(image_bytes, image_type)
    except Exception as e:
        _logger.error(f"[LLM Chat DB] Failed to save image from base64: {e}")
        return None


def save_image_force(image_data: bytes, image_type: str = "input") -> str | None:
    """Save an image to disk without requiring USE_HISTORY_DATABASE.

    Used by image capture node which needs file storage even when the
    chat history database is disabled.
    """
    if not _db_base_path:
        return None

    filename = f"{uuid.uuid4().hex}_{image_type}.png"
    img_dir = os.path.join(_db_base_path, "images")
    try:
        os.makedirs(img_dir, exist_ok=True)
    except OSError:
        return None

    filepath = os.path.join(img_dir, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(image_data)
        return filename
    except OSError as e:
        _logger.error(f"[LLM Chat DB] save_image_force failed: {e}")
        return None


def get_image_path(filename: str) -> str | None:
    """Return the absolute path to an image file.

    Args:
        filename: The UUID filename (e.g. ``"abc..._input.png"``).

    Returns:
        Absolute path, or ``None`` if the file doesn't exist.
    """
    if not _db_base_path:
        return None
    filepath = os.path.join(_db_base_path, "images", filename)
    return filepath if os.path.isfile(filepath) else None


def get_image_data(filename: str) -> bytes | None:
    """Read an image file and return its raw bytes.

    Args:
        filename: The UUID filename.

    Returns:
        Raw PNG bytes, or ``None`` if the file doesn't exist.
    """
    filepath = get_image_path(filename)
    if filepath is None:
        return None
    try:
        with open(filepath, "rb") as f:
            return f.read()
    except OSError:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Session Listing (for Database Manager)
# ═══════════════════════════════════════════════════════════════════════


def list_sessions() -> list[dict]:
    """List all stored node files with metadata.

    Scans the ``chat/`` and ``enhancer/`` directories for ``.json`` files
    and returns their metadata. This replaces the old session-based listing.

    Returns:
        List of dicts with keys: ``node_id``, ``hist_type``, ``entry_count``,
        ``created_at``, ``updated_at``, ``model_name``, ``preview``.
    """
    if not is_available():
        return []

    sessions: list[dict] = []

    with _db_lock:
        for subdir, hist_type in (("chat", "chat"), ("enhancer", "enhancer")):
            dirpath = os.path.join(_db_base_path, subdir)
            if not os.path.isdir(dirpath):
                continue
            try:
                for name in sorted(os.listdir(dirpath)):
                    if not name.endswith(".json") or name.startswith("."):
                        continue
                    node_id = name[:-5]
                    filepath = os.path.join(dirpath, name)
                    data = _read_json_safe(filepath)
                    if data is None:
                        continue

                    entries = data.get("entries", [])
                    preview = _get_preview(entries)

                    sessions.append({
                        "node_id": node_id,
                        "session_id": node_id,  # Keep for backward compat with frontend
                        "hist_type": hist_type,
                        "entry_count": len(entries),
                        "created_at": data.get("created_at", 0),
                        "updated_at": data.get("updated_at", 0),
                        "model_name": data.get("meta", {}).get("model_name", ""),
                        "preview": preview,
                        "image_count": data.get("image_count", 0),
                        "size_bytes": os.path.getsize(filepath),
                    })
            except OSError:
                pass

    return sessions


def _get_preview(entries: list) -> str:
    """Build a short preview text from the first few entries."""
    preview_parts = []
    for entry in entries[:3]:
        if isinstance(entry, dict):
            text = (
                entry.get("message")
                or entry.get("input")
                or entry.get("output")
                or ""
            )
            if text:
                # Take first 120 chars of text
                preview_parts.append(text[:120])
    return " | ".join(preview_parts) if preview_parts else ""


def _count_images_in_entries(entries: list) -> int:
    """Count entries that contain image data (``images`` or ``image`` field)."""
    return sum(1 for e in entries if isinstance(e, dict) and (e.get("image") or e.get("images")))


# ═══════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════


def cleanup() -> dict:
    """Remove orphaned and expired data.

    In the new node-ID-based storage, cleanup handles:
    1. **Expired node files**: Remove files older than ``HISTORY_DB_MAX_AGE_DAYS``.
    2. **Size limit**: If ``HISTORY_DB_MAX_SIZE_MB > 0``, evict oldest files
       until under the limit.
    3. **Orphaned images**: Remove image files not referenced by any node file.

    Returns:
        dict with cleanup stats.
    """
    if not _db_initialized:
        return {"error": "Database not initialized"}

    stats: dict = {
        "removed_expired_entries": 0,
        "removed_session_files": 0,
        "removed_images": 0,
        "bytes_freed": 0,
    }

    with _db_lock:
        now_ts = int(time.time() * 1000)

        # ── Pass 1: Expired node files ──
        effective_age = _effective_max_age_days()
        if effective_age > 0:
            max_age_ms = effective_age * 86400 * 1000
            cutoff = now_ts - max_age_ms

            for subdir in ("chat", "enhancer"):
                dirpath = os.path.join(_db_base_path, subdir)
                if not os.path.isdir(dirpath):
                    continue
                try:
                    for name in os.listdir(dirpath):
                        if not name.endswith(".json") or name.startswith("."):
                            continue
                        filepath = os.path.join(dirpath, name)
                        data = _read_json_safe(filepath)
                        if data:
                            updated_at = data.get("updated_at", 0)
                            if updated_at > 0 and updated_at < cutoff:
                                try:
                                    fsize = os.path.getsize(filepath)
                                    os.remove(filepath)
                                    stats["removed_expired_entries"] += 1
                                    stats["bytes_freed"] += fsize
                                    _logger.info(
                                        f"[LLM Chat DB] Removed expired node file: {name}"
                                    )
                                except OSError:
                                    pass
                except OSError:
                    pass

        # ── Pass 2: Size limit enforcement ──
        effective_size = _effective_max_size_mb()
        if effective_size > 0:
            max_bytes = effective_size * 1024 * 1024
            _enforce_size_limit(max_bytes, stats)

        # ── Pass 3: Orphaned images ──
        _cleanup_orphaned_images(stats)

    _logger.info(
        f"[LLM Chat DB] Cleanup complete: "
        f"removed {stats['removed_expired_entries']} expired, "
        f"{stats['removed_session_files']} evicted, "
        f"{stats['removed_images']} orphaned images, "
        f"freed {stats['bytes_freed'] / 1024:.1f} KB"
    )
    return stats


def _cleanup_orphaned_images(stats: dict) -> None:
    """Remove image files not referenced by any node file."""
    referenced: set[str] = set()

    for subdir in ("chat", "enhancer"):
        dirpath = os.path.join(_db_base_path, subdir)
        if not os.path.isdir(dirpath):
            continue
        try:
            for name in os.listdir(dirpath):
                if not name.endswith(".json") or name.startswith("."):
                    continue
                filepath = os.path.join(dirpath, name)
                data = _read_json_safe(filepath)
                if data and isinstance(data.get("entries"), list):
                    for entry in data["entries"]:
                        _collect_image_refs(entry, referenced)
        except OSError:
            pass

    images_dir = os.path.join(_db_base_path, "images")
    if not os.path.isdir(images_dir):
        return

    removed = 0
    freed = 0
    try:
        for fname in os.listdir(images_dir):
            if not _is_db_filename(fname):
                continue
            if fname in referenced:
                continue
            fpath = os.path.join(images_dir, fname)
            try:
                freed += os.path.getsize(fpath)
                os.remove(fpath)
                removed += 1
            except OSError:
                pass
    except OSError:
        pass

    stats["removed_images"] = removed
    stats["bytes_freed"] = stats.get("bytes_freed", 0) + freed
    if removed > 0:
        _logger.info(
            f"[LLM Chat DB] Removed {removed} orphaned images ({freed / 1024:.1f} KB)"
        )


def scan_orphaned_images() -> dict:
    """Count orphaned images without deleting anything.

    Returns:
        dict with ``count`` (int) and ``bytes`` (int) of orphaned images.
    """
    if not _db_initialized:
        return {"count": 0, "bytes": 0, "error": "Database not initialized"}

    referenced: set[str] = set()

    for subdir in ("chat", "enhancer"):
        dirpath = os.path.join(_db_base_path, subdir)
        if not os.path.isdir(dirpath):
            continue
        try:
            for name in os.listdir(dirpath):
                if not name.endswith(".json") or name.startswith("."):
                    continue
                filepath = os.path.join(dirpath, name)
                data = _read_json_safe(filepath)
                if data and isinstance(data.get("entries"), list):
                    for entry in data["entries"]:
                        _collect_image_refs(entry, referenced)
        except OSError:
            pass

    images_dir = os.path.join(_db_base_path, "images")
    if not os.path.isdir(images_dir):
        return {"count": 0, "bytes": 0}

    orphan_count = 0
    orphan_bytes = 0
    try:
        for fname in os.listdir(images_dir):
            if not _is_db_filename(fname):
                continue
            if fname in referenced:
                continue
            fpath = os.path.join(images_dir, fname)
            try:
                orphan_bytes += os.path.getsize(fpath)
                orphan_count += 1
            except OSError:
                pass
    except OSError:
        pass

    return {"count": orphan_count, "bytes": orphan_bytes}


def _collect_image_refs(entry: dict, referenced: set[str]) -> None:
    """Collect referenced image filenames from a history entry."""
    if not isinstance(entry, dict):
        return
    img = entry.get("image", "")
    if isinstance(img, str) and _is_db_filename(img):
        referenced.add(img)
    for img_obj in entry.get("images", []):
        if isinstance(img_obj, dict):
            fn = img_obj.get("filename")
            if fn and _is_db_filename(fn):
                referenced.add(fn)


def _enforce_size_limit(max_bytes: int, stats: dict) -> None:
    """Remove oldest node files until total size is under ``max_bytes``."""
    files: list[tuple[int, str, int]] = []  # (updated_at, filepath, size)
    total_size = 0

    for subdir in ("chat", "enhancer"):
        dirpath = os.path.join(_db_base_path, subdir)
        if not os.path.isdir(dirpath):
            continue
        try:
            for name in os.listdir(dirpath):
                if not name.endswith(".json") or name.startswith("."):
                    continue
                filepath = os.path.join(dirpath, name)
                data = _read_json_safe(filepath)
                if data is None:
                    continue
                updated_at = data.get("updated_at", 0)
                try:
                    fsize = os.path.getsize(filepath)
                except OSError:
                    continue
                files.append((updated_at, filepath, fsize))
                total_size += fsize
        except OSError:
            pass

    if total_size <= max_bytes:
        return

    # Sort by last_updated ascending (oldest first)
    files.sort(key=lambda x: x[0])

    for updated_at, filepath, fsize in files:
        if total_size <= max_bytes:
            break
        try:
            os.remove(filepath)
            total_size -= fsize
            stats["bytes_freed"] += fsize
            stats["removed_session_files"] += 1
            _logger.info(
                f"[LLM Chat DB] Size limit: removed {os.path.basename(filepath)} "
                f"(updated {updated_at}, freed {fsize} bytes)"
            )
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════


def get_stats() -> dict:
    """Return statistics about the current database state.

    Returns:
        dict with keys: ``total_nodes``, ``total_sessions``, ``total_images``,
        ``disk_size_bytes``, ``db_path``.
    """
    if not _db_initialized:
        return {"error": "Database not initialized"}

    total_size = 0
    chat_count = 0
    enhancer_count = 0
    image_count = 0

    for subdir, counter_name in (("chat", "chat"), ("enhancer", "enhancer")):
        dirpath = os.path.join(_db_base_path, subdir)
        if not os.path.isdir(dirpath):
            continue
        try:
            for name in os.listdir(dirpath):
                if not name.endswith(".json") or name.startswith("."):
                    continue
                item_path = os.path.join(dirpath, name)
                if counter_name == "chat":
                    chat_count += 1
                else:
                    enhancer_count += 1
                try:
                    total_size += os.path.getsize(item_path)
                except OSError:
                    pass
        except OSError:
            pass

    # Count images
    images_dir = os.path.join(_db_base_path, "images")
    if os.path.isdir(images_dir):
        try:
            for fname in os.listdir(images_dir):
                fpath = os.path.join(images_dir, fname)
                if os.path.isfile(fpath):
                    image_count += 1
                    total_size += os.path.getsize(fpath)
        except OSError:
            pass

    return {
        "total_nodes": chat_count + enhancer_count,
        "total_sessions": chat_count + enhancer_count,
        "total_chat_sessions": chat_count,
        "total_enhancer_sessions": enhancer_count,
        "total_images": image_count,
        "disk_size_bytes": total_size,
        "disk_size_mb": round(total_size / (1024 * 1024), 2),
        "db_path": _db_base_path,
    }


# ═══════════════════════════════════════════════════════════════════════
# Settings Management
# ═══════════════════════════════════════════════════════════════════════


def get_settings() -> dict:
    """Return current database runtime settings."""
    return {
        "db_path": _db_base_path,
        "auto_cleanup_enabled": _auto_cleanup_timer is not None,
        "auto_cleanup_interval_sec": _AUTO_CLEANUP_INTERVAL,
        "max_size_mb": _effective_max_size_mb(),
        "max_age_days": _effective_max_age_days(),
        "immediate_write": _effective_immediate_write(),
    }


def _apply_persisted_settings() -> None:
    """Load settings.json and apply overrides to module state."""
    global _AUTO_CLEANUP_INTERVAL, _runtime_max_size_mb, _runtime_max_age_days, _runtime_immediate_write

    persisted = _load_settings_file()
    if not persisted:
        return

    if "auto_cleanup_interval_sec" in persisted:
        _AUTO_CLEANUP_INTERVAL = max(60, int(persisted["auto_cleanup_interval_sec"]))

    if "max_size_mb" in persisted:
        _runtime_max_size_mb = int(persisted["max_size_mb"])

    if "max_age_days" in persisted:
        _runtime_max_age_days = int(persisted["max_age_days"])

    if "immediate_write" in persisted:
        _runtime_immediate_write = bool(persisted["immediate_write"])

    _logger.info(
        f"[LLM Chat DB] Applied persisted settings from settings.json: "
        f"interval={_AUTO_CLEANUP_INTERVAL}s, "
        f"max_size={_effective_max_size_mb()}MB, "
        f"max_age={_effective_max_age_days()}d, "
        f"immediate_write={_effective_immediate_write()}"
    )


def update_settings(new_settings: dict) -> dict:
    """Update runtime settings and persist to settings.json.

    Accepts any subset of keys: ``auto_cleanup_enabled``,
    ``auto_cleanup_interval_sec``, ``max_size_mb``, ``max_age_days``,
    ``immediate_write``.

    Returns the full current settings dict after applying changes.
    """
    global _AUTO_CLEANUP_INTERVAL, _runtime_max_size_mb, _runtime_max_age_days, _runtime_immediate_write

    if "auto_cleanup_interval_sec" in new_settings:
        val = int(new_settings["auto_cleanup_interval_sec"])
        if val < 60:
            val = 60
        _AUTO_CLEANUP_INTERVAL = val

    if "auto_cleanup_enabled" in new_settings:
        enabled = bool(new_settings["auto_cleanup_enabled"])
        if enabled and _auto_cleanup_timer is None:
            start_auto_cleanup()
        elif not enabled and _auto_cleanup_timer is not None:
            stop_auto_cleanup()

    # max_size_mb has been removed from the UI — always disable size-based enforcement
    _runtime_max_size_mb = 0

    if "max_age_days" in new_settings:
        _runtime_max_age_days = int(new_settings["max_age_days"])

    if "immediate_write" in new_settings:
        _runtime_immediate_write = bool(new_settings["immediate_write"])

    # Persist to settings.json for next restart
    current = get_settings()
    _save_settings_file(current)

    return current


def destroy_database() -> dict:
    """Delete all database files and directories, then re-initialize.

    WARNING: This is irreversible. All history and images will be lost.
    """
    global _db_initialized, _db_base_path

    with _db_lock:
        # Stop auto-cleanup
        stop_auto_cleanup()

        # Remove all sub-directories
        for subdir in ("chat", "enhancer", "images"):
            dirpath = os.path.join(_db_base_path, subdir)
            if os.path.isdir(dirpath):
                shutil.rmtree(dirpath)

        # Re-create directory structure
        _ensure_dirs(_db_base_path)

        # Preserve runtime settings across database reset
        _save_settings_file(get_settings())

        _logger.warning("[LLM Chat DB] Database destroyed and recreated")

    return {"status": "ok", "db_path": _db_base_path}


# ═══════════════════════════════════════════════════════════════════════
# Auto-Cleanup Timer
# ═══════════════════════════════════════════════════════════════════════

_AUTO_CLEANUP_INTERVAL = 3600  # 1 hour in seconds, mutable at runtime
_auto_cleanup_timer: threading.Timer | None = None


def _auto_cleanup_worker() -> None:
    """Background worker that runs cleanup periodically."""
    global _auto_cleanup_timer

    try:
        if _db_initialized and USE_HISTORY_DATABASE:
            cleanup()
    except Exception as e:
        _logger.warning(f"[LLM Chat DB] Auto-cleanup failed: {e}")

    # Schedule next run
    _auto_cleanup_timer = threading.Timer(_AUTO_CLEANUP_INTERVAL, _auto_cleanup_worker)
    _auto_cleanup_timer.daemon = True
    _auto_cleanup_timer.start()


def _start_auto_cleanup_if_enabled() -> None:
    """Start auto-cleanup timer if settings.json says enabled."""
    persisted = _load_settings_file()
    if persisted.get("auto_cleanup_enabled", False):
        start_auto_cleanup()


def start_auto_cleanup() -> None:
    """Start the periodic auto-cleanup timer.

    Safe to call multiple times (only starts once).
    """
    global _auto_cleanup_timer
    if _auto_cleanup_timer is not None:
        return
    _auto_cleanup_timer = threading.Timer(_AUTO_CLEANUP_INTERVAL, _auto_cleanup_worker)
    _auto_cleanup_timer.daemon = True
    _auto_cleanup_timer.start()
    _logger.info(
        f"[LLM Chat DB] Auto-cleanup started (interval={_AUTO_CLEANUP_INTERVAL}s)"
    )


def stop_auto_cleanup() -> None:
    """Stop the periodic auto-cleanup timer."""
    global _auto_cleanup_timer
    if _auto_cleanup_timer is not None:
        _auto_cleanup_timer.cancel()
        _auto_cleanup_timer = None
        _logger.info("[LLM Chat DB] Auto-cleanup stopped")


# ═══════════════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════════════


def search_history(query: str, max_results: int = 50) -> list[dict]:
    """Search across all stored conversations for matching entries.

    Searches both chat and enhancer history files for entries whose
    ``message``, ``input``, or ``output`` fields contain the query
    string (case-insensitive substring match).

    Args:
        query: The search term to look for.
        max_results: Maximum number of matching entries to return.

    Returns:
        List of dicts with keys: ``node_id``, ``session_id``, ``hist_type``,
        ``role``, ``message``/``input``/``output``, ``timestamp``.
    """
    if not USE_HISTORY_DATABASE or not _db_initialized:
        return []

    query_lower = query.lower()
    results: list[dict] = []

    with _db_lock:
        for subdir, hist_type in (("chat", "chat"), ("enhancer", "enhancer")):
            dirpath = os.path.join(_db_base_path, subdir)
            if not os.path.isdir(dirpath):
                continue
            try:
                for name in sorted(os.listdir(dirpath)):
                    if not name.endswith(".json") or name.startswith("."):
                        continue
                    filepath = os.path.join(dirpath, name)
                    data = _read_json_safe(filepath)
                    if not data or not isinstance(data.get("entries"), list):
                        continue
                    node_id = data.get("node_id", name[:-5])
                    session_id = node_id  # Use node_id as session_id for backward compat
                    for entry in data["entries"]:
                        if _entry_matches_query(
                            entry, query_lower, results,
                            session_id, node_id, hist_type, max_results
                        ):
                            return results
            except OSError:
                pass

    return results


def _entry_matches_query(
    entry: Any,
    query_lower: str,
    results: list[dict],
    session_id: str,
    node_id: str,
    hist_type: str,
    max_results: int,
) -> bool:
    """Check if an entry matches the search query and append to results.

    Returns:
        ``True`` if the results list has reached ``max_results`` (caller
        should stop searching).
    """
    if not isinstance(entry, dict):
        return False

    text_fields = []
    for field in ("message", "input", "output"):
        val = entry.get(field, "")
        if isinstance(val, str) and val:
            text_fields.append(val)

    if not any(query_lower in tf.lower() for tf in text_fields):
        return False

    result: dict = {
        "session_id": session_id,
        "node_id": node_id,
        "hist_type": hist_type,
        "timestamp": entry.get("timestamp", 0),
    }
    result["role"] = entry.get("role", "")
    for field in ("message", "input", "output"):
        if field in entry:
            result[field] = entry[field]
    results.append(result)
    return len(results) >= max_results


# ═══════════════════════════════════════════════════════════════════════
# Bulk Export
# ═══════════════════════════════════════════════════════════════════════


def export_all_history(format: str = "md") -> str:
    """Export all conversations as a single formatted string.

    Args:
        format: ``"md"`` for Markdown, ``"json"`` for raw JSON.

    Returns:
        Formatted string containing all conversations.
    """
    if not USE_HISTORY_DATABASE or not _db_initialized:
        return ""

    if format == "json":
        return _export_all_json()
    return _export_all_markdown()


def _export_all_markdown() -> str:
    """Export all conversations as Markdown."""
    lines = [
        "# EasyLLM Chat History Export",
        "",
        f"**Exported:** {datetime.now().isoformat()}",
        f"**Database:** {_db_base_path}",
        "",
        "---",
        "",
    ]

    for subdir, label in (("chat", "Chat"), ("enhancer", "Enhancer")):
        dirpath = os.path.join(_db_base_path, subdir)
        if not os.path.isdir(dirpath):
            continue
        try:
            for name in sorted(os.listdir(dirpath)):
                if not name.endswith(".json") or name.startswith("."):
                    continue
                filepath = os.path.join(dirpath, name)
                data = _read_json_safe(filepath)
                if not data:
                    continue

                node_id = data.get("node_id", name[:-5])
                entries = data.get("entries", [])
                created = data.get("created_at", 0)
                created_str = (
                    datetime.fromtimestamp(created / 1000).isoformat()
                    if created
                    else "unknown"
                )

                lines.append(f"## {label} Node: {node_id}")
                lines.append(f"- **Date:** {created_str}")
                lines.append(f"- **Entries:** {len(entries)}")
                lines.append("")

                for entry in entries:
                    if isinstance(entry, dict):
                        role = entry.get("role", "unknown")
                        msg = (
                            entry.get("message")
                            or entry.get("input")
                            or entry.get("output")
                            or ""
                        )
                        timestamp = entry.get("timestamp", 0)
                        ts_str = (
                            datetime.fromtimestamp(timestamp / 1000).isoformat()
                            if timestamp
                            else ""
                        )
                        lines.append(f"### [{role.upper()}] ({ts_str})")
                        lines.append("")
                        lines.append(msg)
                        lines.append("")

                lines.append("---")
                lines.append("")
        except OSError:
            pass

    return "\n".join(lines)


def _export_all_json() -> str:
    """Export all conversations as a JSON array."""
    all_data: dict = {"exported_at": int(time.time() * 1000), "sessions": []}
    for subdir in ("chat", "enhancer"):
        dirpath = os.path.join(_db_base_path, subdir)
        if not os.path.isdir(dirpath):
            continue
        try:
            for name in sorted(os.listdir(dirpath)):
                if not name.endswith(".json") or name.startswith("."):
                    continue
                filepath = os.path.join(dirpath, name)
                data = _read_json_safe(filepath)
                if data:
                    all_data["sessions"].append(data)
        except OSError:
            pass
    return json.dumps(all_data, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# Enhancer Export (server-side file writing)
# ═══════════════════════════════════════════════════════════════════════


def export_enhancer_entries(entries: list, options: dict) -> dict:
    """Export enhancer history entries to the server filesystem.

    Called from the ``POST /easyllm/export/enhancer`` API route.
    Writes text files (MD/JSONL/TXT) and image files to *output_dir*.

    Args:
        entries: List of enhancer entry dicts.
        options: Export configuration dict.

    Returns:
        Dict with ``success``, ``file_count``, ``output_path``, ``images_written``.
    """
    output_dir = os.path.abspath(options.get("output_dir", ""))
    if not output_dir:
        return {"success": False, "error": "No output directory specified"}
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            return {"success": False, "error": f"Cannot create directory: {e}"}
    if not os.access(output_dir, os.W_OK):
        return {"success": False, "error": f"Directory is not writable: {output_dir}"}

    include = options.get("include", {})
    file_structure = options.get("file_structure", "single")
    export_format = options.get("format", "md")
    filename_pattern = options.get("filename_pattern", "{node}-{timestamp}")
    node_label = options.get("node_label", "Enhancer")

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in " _-" else "_" for c in node_label).strip()

    def _resolve_template(template: str, index: int = 0, model_name: str = "") -> str:
        s = template.replace("{node}", safe_label)
        s = s.replace("{timestamp}", now_str)
        safe_model = "".join(c if c.isalnum() or c in " _-" else "_" for c in model_name).strip()
        s = s.replace("{model}", safe_model or "unknown")
        s = s.replace("{index}", f"{index:06d}")
        s = s.replace("{", "").replace("}", "")
        return s.strip() or "export"

    ext = {"md": ".md", "jsonl": ".jsonl", "txt": ".txt"}.get(export_format, ".txt")

    def _parse_think_blocks(text: str) -> tuple:
        if not text:
            return "", ""
        thinking = ""
        response = text
        import re as _re
        pattern = _re.compile(r"↩(.*?)↩", _re.DOTALL)
        match = pattern.search(text)
        if match:
            thinking = match.group(1).strip()
            response = pattern.sub("", text).strip()
        return thinking, response

    def _build_entry_content(entry: dict, idx: int) -> str:
        thinking_text, response_text = _parse_think_blocks(entry.get("output", ""))

        if export_format == "jsonl":
            obj = {}
            if include.get("input", True):
                obj["input"] = entry.get("input", "")
            if include.get("output", True):
                obj["output"] = response_text
            if include.get("thinking", True) and thinking_text:
                obj["thinking"] = thinking_text
            if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                obj["system_prompt"] = entry["systemPromptText"]
            if include.get("modelName", False) and entry.get("modelName"):
                obj["model"] = entry["modelName"]
            if include.get("metadata", False):
                obj["timestamp"] = entry.get("timestamp", 0)
            return json.dumps(obj, ensure_ascii=False)

        ts = ""
        if entry.get("timestamp") and include.get("metadata", False):
            d = datetime.fromtimestamp(entry["timestamp"] / 1000)
            ts = d.strftime("%Y-%m-%d %H:%M:%S")

        parts = []
        if export_format == "md":
            entry_title = f"## Entry {idx + 1}"
            if ts:
                entry_title += f" ({ts})"
            parts.append(entry_title)
            parts.append("")
            if include.get("input", True):
                parts.append("**User Prompt:**")
                parts.append(entry.get("input", "") or "*empty*")
                parts.append("")
            if include.get("output", True):
                parts.append("**Answer:**")
                parts.append(response_text or "*empty*")
                parts.append("")
            if include.get("thinking", True) and thinking_text:
                parts.append("**Thinking:**")
                parts.append(thinking_text)
                parts.append("")
            if include.get("modelName", False) and entry.get("modelName"):
                parts.append(f"**Model:** {entry['modelName']}")
                parts.append("")
            if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                parts.append("**System Prompt:**")
                parts.append(entry["systemPromptText"])
                parts.append("")
            parts.append("---")
            parts.append("")
        else:
            entry_title = f"--- Entry {idx + 1}"
            if ts:
                entry_title += f" ({ts})"
            entry_title += " ---"
            parts.append(entry_title)
            if include.get("input", True):
                parts.append(f"USER PROMPT:")
                parts.append(entry.get("input", "") or "*empty*")
                parts.append("")
            if include.get("output", True):
                parts.append(f"ANSWER:")
                parts.append(response_text or "*empty*")
                parts.append("")
            if include.get("thinking", True) and thinking_text:
                parts.append(f"THINKING:")
                parts.append(thinking_text)
                parts.append("")
            if include.get("modelName", False) and entry.get("modelName"):
                parts.append(f"MODEL: {entry['modelName']}")
                parts.append("")
            if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                parts.append(f"SYSTEM PROMPT:")
                parts.append(entry["systemPromptText"])
                parts.append("")
            parts.append("")
        return "\n".join(parts)

    file_count = 0
    images_written = 0

    try:
        if file_structure == "single":
            if export_format == "jsonl":
                lines = []
                for idx, entry in enumerate(entries):
                    lines.append(_build_entry_content(entry, idx))
                content = "\n".join(lines)
            else:
                header = f"# EasyLLM Enhancer Export — {node_label}\n"
                header += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                body = ""
                for idx, entry in enumerate(entries):
                    body += _build_entry_content(entry, idx) + "\n"
                content = header + body

            base_name = _resolve_template(filename_pattern)
            filepath = os.path.join(output_dir, f"{base_name}{ext}")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            file_count += 1

            if include.get("images", True):
                images_dir = os.path.join(output_dir, f"{base_name}_images")
                os.makedirs(images_dir, exist_ok=True)
                for idx, entry in enumerate(entries):
                    entry_images = entry.get("images", [])
                    if not entry_images:
                        continue
                    for img_idx, img in enumerate(entry_images):
                        img_data = img.get("data")
                        img_filename = img.get("filename")
                        if not img_data and not img_filename:
                            continue
                        img_bytes = None
                        img_ext = "png"
                        if img_data:
                            if img_data.startswith("data:"):
                                mime_match = re.match(r"data:image/(\w+);", img_data)
                                if mime_match:
                                    img_ext = mime_match.group(1)
                                b64_part = img_data.split(",", 1)[-1]
                            else:
                                b64_part = img_data
                            import base64 as _b64
                            img_bytes = _b64.b64decode(b64_part)
                        elif img_filename:
                            raw_bytes = get_image_data(img_filename)
                            if raw_bytes:
                                img_bytes = raw_bytes
                                if "." in img_filename:
                                    img_ext = img_filename.rsplit(".", 1)[-1]
                        if img_bytes is None:
                            continue
                        img_type = img.get("type", "unknown")
                        suffix = f"_{img_type}" if img_type != "unknown" else ""
                        if len(entry_images) > 1:
                            img_filename_out = f"{base_name}_{idx+1:06d}{suffix}_{img_idx}.{img_ext}"
                        else:
                            img_filename_out = f"{base_name}_{idx+1:06d}{suffix}.{img_ext}"
                        img_path = os.path.join(images_dir, img_filename_out)
                        try:
                            with open(img_path, "wb") as img_f:
                                img_f.write(img_bytes)
                            images_written += 1
                        except Exception:
                            pass
        else:
            model_name = entries[0].get("modelName", "") if entries else ""
            base_name = _resolve_template(filename_pattern, model_name=model_name)
            for idx, entry in enumerate(entries):
                content = _build_entry_content(entry, idx)
                entry_name = _resolve_template(filename_pattern, idx + 1, entry.get("modelName", ""))
                filepath = os.path.join(output_dir, f"{entry_name}{ext}")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                file_count += 1
                if include.get("images", True):
                    entry_images = entry.get("images", [])
                    if not entry_images:
                        continue
                    for img_idx, img in enumerate(entry_images):
                        img_data = img.get("data")
                        img_filename = img.get("filename")
                        if not img_data and not img_filename:
                            continue
                        img_bytes = None
                        img_ext = "png"
                        if img_data:
                            if img_data.startswith("data:"):
                                mime_match = re.match(r"data:image/(\w+);", img_data)
                                if mime_match:
                                    img_ext = mime_match.group(1)
                                b64_part = img_data.split(",", 1)[-1]
                            else:
                                b64_part = img_data
                            import base64 as _b64
                            img_bytes = _b64.b64decode(b64_part)
                        elif img_filename:
                            raw_bytes = get_image_data(img_filename)
                            if raw_bytes:
                                img_bytes = raw_bytes
                                if "." in img_filename:
                                    img_ext = img_filename.rsplit(".", 1)[-1]
                        if img_bytes is None:
                            continue
                        img_type = img.get("type", "unknown")
                        suffix = f"_{img_type}" if img_type != "unknown" else ""
                        if len(entry_images) > 1:
                            img_filename_out = f"{entry_name}{suffix}_{img_idx}.{img_ext}"
                        else:
                            img_filename_out = f"{entry_name}{suffix}.{img_ext}"
                        img_path = os.path.join(output_dir, img_filename_out)
                        try:
                            with open(img_path, "wb") as img_f:
                                img_f.write(img_bytes)
                            images_written += 1
                        except Exception:
                            pass

        return {
            "success": True,
            "file_count": file_count,
            "output_path": output_dir,
            "images_written": images_written,
        }

    except Exception as e:
        _logger.error(f"[LLM Chat DB] Enhancer export failed: {e}")
        return {"success": False, "error": str(e)}


def export_enhancer_entries_v2(entries: list, options: dict) -> dict:
    """Export enhancer entries using the redesigned v2 options schema.

    Called from the ``POST /easyllm/export/enhancer_v2`` API route.

    Args:
        entries: List of enhancer entry dicts (same as v1).
        options: Dict with keys ``output_dir``, ``structure``, ``include``,
            ``format``, ``separator``, ``base_name``, ``node_label``.

    Returns:
        Dict with ``success``, ``file_count``, ``output_path``, ``images_written``.
    """
    output_dir = os.path.abspath(options.get("output_dir", ""))
    if not output_dir:
        return {"success": False, "error": "No output directory specified"}
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            return {"success": False, "error": f"Cannot create directory: {e}"}
    if not os.access(output_dir, os.W_OK):
        return {"success": False, "error": f"Directory is not writable: {output_dir}"}

    structure = options.get("structure", "pairs")
    include = options.get("include", {})
    export_format = options.get("format", "txt")
    separator = options.get("separator", "\n")
    base_name = options.get("base_name", "").strip()
    node_label = options.get("node_label", "Enhancer")

    safe_base = "".join(c if c.isalnum() else "_" for c in base_name).strip() or "export"
    ext = {"md": ".md", "jsonl": ".jsonl", "txt": ".txt"}.get(export_format, ".txt")

    def _parse_think_blocks(text: str) -> tuple:
        if not text:
            return "", ""
        thinking = ""
        response = text
        import re as _re
        pattern = _re.compile(r"↩(.*?)↩", _re.DOTALL)
        match = pattern.search(text)
        if match:
            thinking = match.group(1).strip()
            response = pattern.sub("", text).strip()
        return thinking, response

    def _resolve_image_bytes(img: dict) -> tuple:
        img_data = img.get("data")
        img_filename = img.get("filename")
        img_bytes = None
        img_ext = "png"
        if img_data:
            if img_data.startswith("data:"):
                mime_match = re.match(r"data:image/(\w+);", img_data)
                if mime_match:
                    img_ext = mime_match.group(1)
                b64_part = img_data.split(",", 1)[-1]
            else:
                b64_part = img_data
            import base64 as _b64
            img_bytes = _b64.b64decode(b64_part)
        elif img_filename:
            raw_bytes = get_image_data(img_filename)
            if raw_bytes:
                img_bytes = raw_bytes
                if "." in img_filename:
                    img_ext = img_filename.rsplit(".", 1)[-1]
        return img_bytes, img_ext

    file_count = 0
    images_written = 0

    try:
        if structure == "pairs":
            import re as _re
            counter_pattern = _re.compile(
                rf"^{_re.escape(safe_base)}(\d+){_re.escape(ext)}$"
            )
            max_counter = 0
            try:
                for fname in os.listdir(output_dir):
                    m = counter_pattern.match(fname)
                    if m:
                        num = int(m.group(1))
                        if num > max_counter:
                            max_counter = num
            except OSError:
                pass
            next_counter = max_counter + 1

            for idx, entry in enumerate(entries):
                thinking_text, response_text = _parse_think_blocks(entry.get("output", ""))

                if export_format == "txt":
                    parts = []
                    if include.get("input", True):
                        parts.append(entry.get("input", ""))
                    if include.get("output", True):
                        parts.append(response_text)
                    if include.get("thinking", True) and thinking_text:
                        parts.append(thinking_text)
                    if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                        parts.append(entry["systemPromptText"])
                    if include.get("modelName", False) and entry.get("modelName"):
                        parts.append(entry["modelName"])
                    if include.get("metadata", False) and entry.get("timestamp"):
                        from datetime import datetime as _dt
                        d = _dt.fromtimestamp(entry["timestamp"] / 1000)
                        parts.append(d.strftime("%Y-%m-%d %H:%M:%S"))
                    content = separator.join(parts)
                else:
                    ts = ""
                    if entry.get("timestamp") and include.get("metadata", False):
                        from datetime import datetime as _dt
                        d = _dt.fromtimestamp(entry["timestamp"] / 1000)
                        ts = d.strftime("%Y-%m-%d %H:%M:%S")

                    parts = []
                    if export_format == "md":
                        entry_title = f"## Entry {idx + 1}"
                        if ts:
                            entry_title += f" ({ts})"
                        parts.append(entry_title)
                        parts.append("")
                        if include.get("input", True):
                            parts.append("**User Prompt:**")
                            parts.append(entry.get("input", "") or "*empty*")
                            parts.append("")
                        if include.get("output", True):
                            parts.append("**Answer:**")
                            parts.append(response_text or "*empty*")
                            parts.append("")
                        if include.get("thinking", True) and thinking_text:
                            parts.append("**Thinking:**")
                            parts.append(thinking_text)
                            parts.append("")
                        if include.get("modelName", False) and entry.get("modelName"):
                            parts.append(f"**Model:** {entry['modelName']}")
                            parts.append("")
                        if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                            parts.append("**System Prompt:**")
                            parts.append(entry["systemPromptText"])
                            parts.append("")
                        parts.append("---")
                        parts.append("")
                    else:
                        import json as _json
                        obj = {}
                        if include.get("input", True):
                            obj["input"] = entry.get("input", "")
                        if include.get("output", True):
                            obj["output"] = response_text
                        if include.get("thinking", True) and thinking_text:
                            obj["thinking"] = thinking_text
                        if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                            obj["system_prompt"] = entry["systemPromptText"]
                        if include.get("modelName", False) and entry.get("modelName"):
                            obj["model"] = entry["modelName"]
                        if include.get("metadata", False):
                            obj["timestamp"] = entry.get("timestamp", 0)
                        content = _json.dumps(obj, ensure_ascii=False)

                entry_counter = next_counter + idx
                text_filename = f"{safe_base}{entry_counter}{ext}"
                text_path = os.path.join(output_dir, text_filename)
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(content)
                file_count += 1

                if include.get("images", True):
                    entry_images = entry.get("images", [])
                    for img_idx, img in enumerate(entry_images):
                        img_bytes, img_ext_resolved = _resolve_image_bytes(img)
                        if img_bytes is None:
                            continue
                        img_type = img.get("type", "unknown")
                        suffix = f"_{img_type}" if img_type != "unknown" else ""
                        if len(entry_images) > 1:
                            img_name = f"{safe_base}{entry_counter}{suffix}_{img_idx}.{img_ext_resolved}"
                        else:
                            img_name = f"{safe_base}{entry_counter}{suffix}.{img_ext_resolved}"
                        img_path = os.path.join(output_dir, img_name)
                        try:
                            with open(img_path, "wb") as img_f:
                                img_f.write(img_bytes)
                            images_written += 1
                        except Exception:
                            pass
        else:
            images_dir = os.path.join(output_dir, f"{safe_base}_images")
            image_refs = []

            if export_format == "jsonl":
                lines = []
                for idx, entry in enumerate(entries):
                    thinking_text, response_text = _parse_think_blocks(entry.get("output", ""))
                    obj = {}
                    if include.get("input", True):
                        obj["input"] = entry.get("input", "")
                    if include.get("output", True):
                        obj["output"] = response_text
                    if include.get("thinking", True) and thinking_text:
                        obj["thinking"] = thinking_text
                    if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                        obj["system_prompt"] = entry["systemPromptText"]
                    if include.get("modelName", False) and entry.get("modelName"):
                        obj["model"] = entry["modelName"]
                    if include.get("metadata", False):
                        obj["timestamp"] = entry.get("timestamp", 0)
                    if include.get("images", True):
                        entry_images = entry.get("images", [])
                        img_refs = []
                        for img_idx, img in enumerate(entry_images):
                            img_bytes, img_ext_resolved = _resolve_image_bytes(img)
                            if img_bytes is None:
                                continue
                            img_name = f"{safe_base}_{idx + 1}.{img_ext_resolved}"
                            img_refs.append(f"{safe_base}_images/{img_name}")
                            image_refs.append((idx, img_name))
                        if img_refs:
                            obj["image_ref"] = img_refs[0] if len(img_refs) == 1 else img_refs
                    lines.append(json.dumps(obj, ensure_ascii=False))
                content = "\n".join(lines)
            else:
                header = f"# EasyLLM Enhancer Export — {node_label}\n"
                header += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                body = ""

                for idx, entry in enumerate(entries):
                    thinking_text, response_text = _parse_think_blocks(entry.get("output", ""))
                    ts = ""
                    if entry.get("timestamp") and include.get("metadata", False):
                        d = datetime.fromtimestamp(entry["timestamp"] / 1000)
                        ts = d.strftime("%Y-%m-%d %H:%M:%S")

                    parts = []
                    if export_format == "md":
                        entry_title = f"## Entry {idx + 1}"
                        if ts:
                            entry_title += f" ({ts})"
                        parts.append(entry_title)
                        parts.append("")
                        if include.get("input", True):
                            parts.append("**User Prompt:**")
                            parts.append(entry.get("input", "") or "*empty*")
                            parts.append("")
                        if include.get("images", True):
                            entry_images = entry.get("images", [])
                            for img_idx, img in enumerate(entry_images):
                                img_bytes, img_ext_resolved = _resolve_image_bytes(img)
                                if img_bytes is None:
                                    continue
                                img_name = f"{safe_base}_{idx + 1}.{img_ext_resolved}"
                                parts.append(f"**Image Ref:** {safe_base}_images/{img_name}")
                                parts.append("")
                                image_refs.append((idx, img_name))
                        if include.get("output", True):
                            parts.append("**Answer:**")
                            parts.append(response_text or "*empty*")
                            parts.append("")
                        if include.get("thinking", True) and thinking_text:
                            parts.append("**Thinking:**")
                            parts.append(thinking_text)
                            parts.append("")
                        if include.get("modelName", False) and entry.get("modelName"):
                            parts.append(f"**Model:** {entry['modelName']}")
                            parts.append("")
                        if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                            parts.append("**System Prompt:**")
                            parts.append(entry["systemPromptText"])
                            parts.append("")
                        parts.append("---")
                        parts.append("")
                    else:
                        entry_title = f"--- Entry {idx + 1}"
                        if ts:
                            entry_title += f" ({ts})"
                        entry_title += " ---"
                        parts.append(entry_title)
                        if include.get("input", True):
                            parts.append(f"USER PROMPT:")
                            parts.append(entry.get("input", "") or "*empty*")
                            parts.append("")
                        if include.get("images", True):
                            entry_images = entry.get("images", [])
                            for img_idx, img in enumerate(entry_images):
                                img_bytes, img_ext_resolved = _resolve_image_bytes(img)
                                if img_bytes is None:
                                    continue
                                img_name = f"{safe_base}_{idx + 1}.{img_ext_resolved}"
                                parts.append(f"IMAGE REF: {safe_base}_images/{img_name}")
                                parts.append("")
                                image_refs.append((idx, img_name))
                        if include.get("output", True):
                            parts.append(f"ANSWER:")
                            parts.append(response_text or "*empty*")
                            parts.append("")
                        if include.get("thinking", True) and thinking_text:
                            parts.append(f"THINKING:")
                            parts.append(thinking_text)
                            parts.append("")
                        if include.get("modelName", False) and entry.get("modelName"):
                            parts.append(f"MODEL: {entry['modelName']}")
                            parts.append("")
                        if include.get("systemPrompt", False) and entry.get("systemPromptText"):
                            parts.append(f"SYSTEM PROMPT:")
                            parts.append(entry["systemPromptText"])
                            parts.append("")
                        parts.append("")
                    body += "\n".join(parts) + "\n"

                content = header + body

            text_path = os.path.join(output_dir, f"{safe_base}{ext}")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(content)
            file_count += 1

            if include.get("images", True) and image_refs:
                os.makedirs(images_dir, exist_ok=True)
                for idx, entry in enumerate(entries):
                    entry_images = entry.get("images", [])
                    for img_idx, img in enumerate(entry_images):
                        img_bytes, img_ext_resolved = _resolve_image_bytes(img)
                        if img_bytes is None:
                            continue
                        matching_refs = [r for r in image_refs if r[0] == idx]
                        if not matching_refs:
                            continue
                        img_name = f"{safe_base}_{idx + 1}.{img_ext_resolved}"
                        img_path = os.path.join(images_dir, img_name)
                        try:
                            with open(img_path, "wb") as img_f:
                                img_f.write(img_bytes)
                            images_written += 1
                        except Exception:
                            pass

        return {
            "success": True,
            "file_count": file_count,
            "output_path": output_dir,
            "images_written": images_written,
        }

    except Exception as e:
        _logger.error(f"[LLM Chat DB] Enhancer v2 export failed: {e}")
        return {"success": False, "error": str(e)}
