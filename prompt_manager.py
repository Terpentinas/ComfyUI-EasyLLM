"""
System Prompt Manager - Load, save, and manage system prompt templates.

Prompts are stored in system_prompts.json in the node directory.
Provides public API functions for prompt CRUD + ComfyUI API routes
for the frontend management dialog.

Data model (v3 — flat prompts, no hardcoded defaults, "System Prompts" locked at bottom):
    {
        "categories": ["Favorites", "describe", "System Prompts"],
        "prompts": [
            {"name": "...", "prompt": "...", "category": "System Prompts"},
            {"name": "...", "prompt": "...", "category": "Favorites"}
        ]
    }
"""

import json
import os
import logging
import shutil
import tempfile

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(NODE_DIR, "system_prompts.json")

LOCKED_CATEGORY = "System Prompts"

# ── Internal helpers ────────────────────────────────────────────────────


def _get_prompts_file_path() -> str:
    """Return the path to the prompts JSON file."""
    return PROMPTS_FILE


def _create_fresh_structure():
    """Build minimal v3 structure with an empty System Prompts category."""
    return {
        "categories": [LOCKED_CATEGORY],
        "prompts": [],
    }


def _deduplicate_names(prompts: list) -> list:
    """
    Append (dup1), (dup2), ... suffixes for prompts sharing the same name
    within the SAME category.

    Operates on a copy; returns the deduplicated list.
    """
    # Group by category, then deduplicate names within each group
    by_category: dict[str, list] = {}
    for p in prompts:
        cat = p.get("category", LOCKED_CATEGORY)
        by_category.setdefault(cat, []).append(dict(p))

    result = []
    for cat, cat_prompts in by_category.items():
        seen = {}
        for p in cat_prompts:
            name = p.get("name", "").strip()
            if not name:
                result.append(p)
                continue
            if name in seen:
                seen[name] += 1
                dup_name = f"{name} (dup{seen[name]})"
                p["name"] = dup_name
            else:
                seen[name] = 0
            result.append(p)
    return result


def _ensure_system_prompts_last(categories: list) -> list:
    """Return categories with LOCKED_CATEGORY forced to the last position."""
    others = [c for c in categories if c != LOCKED_CATEGORY]
    return others + [LOCKED_CATEGORY]


def _migrate_v2_to_v3(data: dict) -> dict:
    """
    Migrate v2 structure (default_prompts + custom_prompts) to v3 flat model.

    - All default_prompts get category = LOCKED_CATEGORY
    - All custom_prompts keep their original categories
    - LOCKED_CATEGORY is placed last in categories list
    - Names are deduplicated
    """
    old_categories = data.get("categories", [])
    old_defaults = data.get("default_prompts", [])
    old_custom = data.get("custom_prompts", [])

    new_prompts = []

    # default_prompts → System Prompts category
    for p in old_defaults:
        new_prompts.append({
            "name": p.get("name", ""),
            "prompt": p.get("prompt", ""),
            "category": LOCKED_CATEGORY,
        })

    # custom_prompts → keep their original categories
    for p in old_custom:
        new_prompts.append({
            "name": p.get("name", ""),
            "prompt": p.get("prompt", ""),
            "category": p.get("category", "Favorites"),
        })

    # Build categories: keep old (minus any dupes), ensure LOCKED_CATEGORY last
    new_categories = []
    for c in old_categories:
        if c != LOCKED_CATEGORY and c not in new_categories:
            new_categories.append(c)
    # Collect categories from custom prompts that weren't in the list
    for p in old_custom:
        cat = p.get("category", "Favorites")
        if cat and cat != LOCKED_CATEGORY and cat not in new_categories:
            new_categories.append(cat)
    # Default "Favorites" if list is empty
    if not new_categories:
        new_categories.append("Favorites")
    # Ensure LOCKED_CATEGORY exists
    new_categories = _ensure_system_prompts_last(new_categories)

    # Deduplicate names
    new_prompts = _deduplicate_names(new_prompts)

    return {
        "categories": new_categories,
        "prompts": new_prompts,
    }


def _migrate_v1_to_v3(data) -> dict:
    """
    Migrate old v1 format (flat array or {prompts: [...]}) directly to v3.
    All existing prompts become "Favorites" category.
    """
    old_prompts = []
    if isinstance(data, dict) and "prompts" in data:
        old_prompts = data["prompts"]
    elif isinstance(data, list):
        old_prompts = data
    else:
        return _create_fresh_structure()

    new_prompts = []
    for p in old_prompts:
        new_prompts.append({
            "name": p.get("name", ""),
            "prompt": p.get("prompt", ""),
            "category": p.get("category", "Favorites"),
        })

    cats = ["Favorites"]
    for p in new_prompts:
        cat = p.get("category", "Favorites")
        if cat and cat not in cats:
            cats.append(cat)
    cats = _ensure_system_prompts_last(cats)

    new_prompts = _deduplicate_names(new_prompts)

    return {
        "categories": cats,
        "prompts": new_prompts,
    }


# ── Public API ───────────────────────────────────────────────────────────


def load_all_prompts() -> dict:
    """
    Load the full prompt structure from the JSON file.

    Auto-migrates v1 (flat) or v2 (default_prompts/custom_prompts) format to v3.
    If the file doesn't exist, creates a minimal structure with an empty
    "System Prompts" category and no hardcoded prompts.

    Returns:
        dict: { "categories": [...], "prompts": [...] }
    """
    filepath = _get_prompts_file_path()

    if not os.path.exists(filepath):
        logging.info(f"[LLM Chat] Creating system_prompts.json with minimal structure")
        struct = _create_fresh_structure()
        _save_structure(struct)
        return struct

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # v3 format — already current
        if isinstance(data, dict) and "prompts" in data and "categories" in data:
            # Ensure LOCKED_CATEGORY exists and is last
            cats = data.get("categories", [])
            if LOCKED_CATEGORY not in cats:
                cats.append(LOCKED_CATEGORY)
            data["categories"] = _ensure_system_prompts_last(cats)
            # Ensure prompts key exists
            if "prompts" not in data:
                data["prompts"] = []
            return data

        # v2 format — has default_prompts / custom_prompts
        if isinstance(data, dict) and ("default_prompts" in data or "custom_prompts" in data):
            logging.info("[LLM Chat] Detected v2 prompts format — migrating to v3")
            struct = _migrate_v2_to_v3(data)
            _save_structure(struct)
            return struct

        # v1 format — flat array or {prompts: [...]}
        logging.info("[LLM Chat] Detected old prompts format — auto-migrating to v3")
        struct = _migrate_v1_to_v3(data)
        _save_structure(struct)
        return struct

    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"[LLM Chat] Failed to load prompts file ({e}), creating fresh structure")
        struct = _create_fresh_structure()
        _save_structure(struct)
        return struct


def _save_structure(struct: dict) -> bool:
    """Write the full v3 structure to disk (atomic write)."""
    filepath = _get_prompts_file_path()
    tmp_path = None
    try:
        # Ensure System Prompts is last and deduplicate before saving
        struct["categories"] = _ensure_system_prompts_last(struct.get("categories", [LOCKED_CATEGORY]))
        struct["prompts"] = _deduplicate_names(struct.get("prompts", []))

        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(filepath),
            prefix=".system_prompts_tmp_",
            suffix=".json",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(struct, f, indent=2, ensure_ascii=False)

        shutil.move(tmp_path, filepath)
        tmp_path = None
        return True

    except (IOError, OSError) as e:
        logging.error(f"[LLM Chat] Failed to save prompts: {e}")
        return False

    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def save_all_prompts(prompts_data) -> bool:
    """
    Save prompts to the JSON file. Accepts:
    - v3 dict: { "categories": [...], "prompts": [...] }
    - v2 dict: { "categories": [...], "default_prompts": [...], "custom_prompts": [...] } — auto-converts
    - flat list (v1 fallback): [{name, prompt}, ...] — auto-converts

    Returns True on success.
    """
    if isinstance(prompts_data, dict):
        # Check if it's v2 format (legacy fallback from old frontend)
        if "default_prompts" in prompts_data or "custom_prompts" in prompts_data:
            struct = _migrate_v2_to_v3(prompts_data)
            return _save_structure(struct)
        # v3 format
        struct = dict(prompts_data)
        if "categories" not in struct:
            struct["categories"] = [LOCKED_CATEGORY]
        if "prompts" not in struct:
            struct["prompts"] = []
        return _save_structure(struct)

    if isinstance(prompts_data, list):
        # v1 flat list — migrate
        struct = _migrate_v1_to_v3({"prompts": prompts_data})
        return _save_structure(struct)

    logging.error(f"[LLM Chat] save_all_prompts: unexpected data type {type(prompts_data)}")
    return False


def get_prompt_names(category: str = LOCKED_CATEGORY) -> list:
    """
    Get prompt names for combo widgets, filtered by category.

    Args:
        category: Category name to filter by (default: LOCKED_CATEGORY "System Prompts").
                  Pass None to return ALL prompt names (for export/admin purposes).

    Returns:
        list: ["Custom", "Name1", "Name2", ...]
    """
    struct = load_all_prompts()
    names = ["Custom"]
    seen = set()

    for p in struct.get("prompts", []):
        if category is not None and p.get("category") != category:
            continue
        name = p.get("name", "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def get_prompt_names_for_category(category: str) -> list:
    """
    Get prompt names only in the specified category.
    Convenience wrapper around get_prompt_names().

    Args:
        category: Category name to filter by

    Returns:
        list: ["Name1", "Name2", ...] (no "Custom" prefix)
    """
    struct = load_all_prompts()
    names = []
    for p in struct.get("prompts", []):
        if p.get("category") == category:
            name = p.get("name", "").strip()
            if name and name not in names:
                names.append(name)
    return names


def get_prompt_by_name(name: str) -> str:
    """
    Get the prompt text for a given template name.
    Searches the flat prompts array across all categories.

    Args:
        name: The prompt name to look up

    Returns:
        str: The prompt text, or empty string if not found / "Custom"
    """
    if not name or name == "Custom":
        return ""

    struct = load_all_prompts()
    for p in struct.get("prompts", []):
        if p.get("name") == name:
            return p.get("prompt", "")
    return ""


# ── Category CRUD (non-API functions, used by endpoints) ────────────────


def get_categories() -> list:
    """Get the ordered list of category names."""
    struct = load_all_prompts()
    return struct.get("categories", [LOCKED_CATEGORY])


def save_categories(categories: list) -> bool:
    """
    Save an updated category list (name changes, reordering, additions).
    Existing prompts with old category names are updated to match.

    - LOCKED_CATEGORY ("System Prompts") is always forced last.
    - LOCKED_CATEGORY cannot be renamed.

    Returns True on success.
    """
    struct = load_all_prompts()
    old_cats = struct.get("categories", [])

    # Ensure LOCKED_CATEGORY is always present and last
    cleaned = _ensure_system_prompts_last(categories)

    # Map old → new names for prompts (if a category was renamed)
    # LOCKED_CATEGORY cannot be renamed, so exclude it from mapping
    name_map = {}
    for i, old_name in enumerate(old_cats):
        if old_name == LOCKED_CATEGORY:
            continue
        if i < len(cleaned):
            new_name = cleaned[i]
            if old_name != new_name:
                name_map[old_name] = new_name

    # Update prompt categories
    for p in struct.get("prompts", []):
        cat = p.get("category", LOCKED_CATEGORY)
        if cat in name_map:
            p["category"] = name_map[cat]

    struct["categories"] = list(cleaned)
    return _save_structure(struct)


def delete_category(category: str, reassign_to: str = "Favorites") -> bool:
    """
    Delete a category. All prompts in that category are reassigned to reassign_to.
    Cannot delete LOCKED_CATEGORY ("System Prompts").
    Returns True on success. Returns False if deleting the last category.
    """
    if category == LOCKED_CATEGORY:
        logging.error(f"[LLM Chat] Cannot delete the {LOCKED_CATEGORY} category")
        return False

    struct = load_all_prompts()
    cats = struct.get("categories", [])

    if category not in cats:
        return True  # already gone

    if len(cats) <= 1:
        logging.error("[LLM Chat] Cannot delete the last category")
        return False

    # Remove category
    cats.remove(category)
    struct["categories"] = _ensure_system_prompts_last(cats)

    # Reassign prompts
    for p in struct.get("prompts", []):
        if p.get("category") == category:
            p["category"] = reassign_to

    return _save_structure(struct)


def export_all_prompts() -> dict:
    """
    Build an export-friendly structure with all prompts + categories.
    Used by /easyllm/prompts/export endpoint.
    """
    struct = load_all_prompts()
    return {
        "categories": list(struct.get("categories", [])),
        "prompts": [dict(p) for p in struct.get("prompts", [])],
    }


# ── ComfyUI API Routes (deferred registration) ──────────────────────────


def setup_routes():
    """
    Register API routes for the frontend management dialog.

    Safe to call multiple times (idempotent — skips if already registered).

    Must be called from __init__.py after ComfyUI server is initialized,
    NOT at module import time (to avoid import-order issues).
    """
    # Idempotency guard — skip if routes have already been registered
    if getattr(setup_routes, "_registered", False):
        return

    try:
        from server import PromptServer
        from aiohttp import web

        @PromptServer.instance.routes.post("/easyllm/prompts/save")
        async def save_prompts_api(request):
            """Save all prompts (v3 structure). Called from management dialog."""
            try:
                data = await request.json()
                success = save_all_prompts(data)
                struct = load_all_prompts() if success else load_all_prompts()
                return web.json_response({
                    "success": success,
                    "categories": struct.get("categories", []),
                    "prompts": struct.get("prompts", []),
                })
            except Exception as e:
                logging.error(f"[LLM Chat] Save prompts API error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.get("/easyllm/prompts/load")
        async def load_prompts_api(request):
            """Load full prompt structure (v3). Called from management dialog."""
            try:
                struct = load_all_prompts()
                return web.json_response({
                    "success": True,
                    "categories": struct.get("categories", []),
                    "prompts": struct.get("prompts", []),
                })
            except Exception as e:
                logging.error(f"[LLM Chat] Load prompts API error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.post("/easyllm/prompts/import")
        async def import_prompts_api(request):
            """
            Import prompts from uploaded JSON data.

            Body: {
                "prompts": [{"name": "...", "prompt": "...", "category": "..."}],
                "strategy": "append" | "replace" | "skip_duplicates"
            }

            If imported prompts have a category, it's preserved.
            If not, they default to "Favorites".
            New categories from imported data are auto-added to the categories list.
            """
            try:
                data = await request.json()
                imported = data.get("prompts", [])
                strategy = data.get("strategy", "append")

                if not isinstance(imported, list):
                    return web.json_response(
                        {"success": False, "error": "Invalid format: 'prompts' must be an array"},
                        status=400,
                    )

                struct = load_all_prompts()
                existing_names = set()
                for p in struct.get("prompts", []):
                    n = p.get("name", "").strip()
                    if n:
                        existing_names.add(n)

                imported_count = 0
                skipped_count = 0

                if strategy == "replace":
                    # Replace all prompts with imported ones
                    struct["prompts"] = [dict(p) for p in imported]
                    imported_count = len(imported)
                    # Rebuild categories from imported data
                    new_cats = ["Favorites"]
                    for p in imported:
                        cat = p.get("category", "Favorites")
                        if cat and cat not in new_cats and cat != LOCKED_CATEGORY:
                            new_cats.append(cat)
                    new_cats = _ensure_system_prompts_last(new_cats)
                    struct["categories"] = new_cats

                elif strategy == "skip_duplicates":
                    for p in imported:
                        name = p.get("name", "").strip()
                        if name and name not in existing_names:
                            entry = {
                                "name": name,
                                "prompt": p.get("prompt", ""),
                                "category": p.get("category", "Favorites"),
                            }
                            struct["prompts"].append(entry)
                            existing_names.add(name)
                            imported_count += 1
                        else:
                            skipped_count += 1
                    _add_missing_categories(struct, imported)

                else:  # append (default)
                    for p in imported:
                        struct["prompts"].append({
                            "name": p.get("name", ""),
                            "prompt": p.get("prompt", ""),
                            "category": p.get("category", "Favorites"),
                        })
                    imported_count = len(imported)
                    _add_missing_categories(struct, imported)

                success = _save_structure(struct)
                if success:
                    struct = load_all_prompts()
                    return web.json_response({
                        "success": True,
                        "categories": struct.get("categories", []),
                        "prompts": struct.get("prompts", []),
                        "imported_count": imported_count,
                        "skipped_count": skipped_count,
                    })
                else:
                    return web.json_response(
                        {"success": False, "error": "Failed to save prompts file"},
                        status=500,
                    )

            except Exception as e:
                logging.error(f"[LLM Chat] Import prompts API error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.post("/easyllm/prompts/categories/save")
        async def save_categories_api(request):
            """Save updated categories list (after add/rename/reorder/delete)."""
            try:
                data = await request.json()
                categories = data.get("categories", [])
                if not categories or not isinstance(categories, list):
                    return web.json_response(
                        {"success": False, "error": "categories must be a non-empty array"},
                        status=400,
                    )
                # save_categories handles LOCKED_CATEGORY protection
                success = save_categories(categories)
                struct = load_all_prompts() if success else load_all_prompts()
                return web.json_response({
                    "success": success,
                    "categories": struct.get("categories", []),
                    "prompts": struct.get("prompts", []),
                })
            except Exception as e:
                logging.error(f"[LLM Chat] Save categories API error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.post("/easyllm/prompts/categories/delete")
        async def delete_category_api(request):
            """Delete a category, reassigning its prompts to another category."""
            try:
                data = await request.json()
                category = data.get("category", "")
                reassign_to = data.get("reassign_to", "Favorites")
                if not category:
                    return web.json_response(
                        {"success": False, "error": "category is required"},
                        status=400,
                    )
                if category == LOCKED_CATEGORY:
                    return web.json_response(
                        {"success": False, "error": f"Cannot delete the {LOCKED_CATEGORY} category"},
                        status=400,
                    )
                success = delete_category(category, reassign_to)
                struct = load_all_prompts() if success else load_all_prompts()
                return web.json_response({
                    "success": success,
                    "categories": struct.get("categories", []),
                    "prompts": struct.get("prompts", []),
                })
            except Exception as e:
                logging.error(f"[LLM Chat] Delete category API error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.get("/easyllm/prompts/export")
        async def export_prompts_api(request):
            """Export all prompts with categories as downloadable JSON."""
            try:
                export_data = export_all_prompts()
                return web.json_response({
                    "success": True,
                    "export": export_data,
                })
            except Exception as e:
                logging.error(f"[LLM Chat] Export prompts API error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        setup_routes._registered = True
        logging.info("[LLM Chat] System prompt API routes registered (v3)")

    except Exception as e:
        logging.error(
            f"[LLM Chat] Failed to register API routes: {e} — "
            "prompt management will be unavailable until ComfyUI restarts"
        )


# ── Helper for import ───────────────────────────────────────────────────


def _add_missing_categories(struct: dict, prompts: list):
    """Add any categories from imported prompts that don't exist yet."""
    cats = struct.get("categories", [LOCKED_CATEGORY])
    changed = False
    for p in prompts:
        cat = p.get("category", "Favorites")
        if cat and cat not in cats and cat != LOCKED_CATEGORY:
            cats.append(cat)
            changed = True
    if changed:
        cats = _ensure_system_prompts_last(cats)
        struct["categories"] = cats
