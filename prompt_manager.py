"""
System Prompt Manager - Load, save, and manage system prompt templates.

Prompts are stored in system_prompts.json in the node directory.
Provides public API functions for prompt CRUD + ComfyUI API routes
for the frontend management dialog.
"""

import json
import os
import logging
import shutil
import tempfile

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(NODE_DIR, "system_prompts.json")

# ── Default prompts (written to file on first load) ──────────────────────

DEFAULT_PROMPTS = [
    {
        "name": "Simple Minimal Enhance",
        "prompt": (
            "Enhance the user's concept by adding 2-3 specific visual details. "
            "Keep the original meaning intact. "
            "Do not rewrite everything, just expand slightly. "
            "Output exactly 2-3 sentences. "
            "No explanations, no preamble."
        ),
    },
    {
        "name": "Natural Language Enhancer",
        "prompt": (
            "Transform the user's simple concept into a clear visual "
            "description using natural language prose.\n\n"
            "Structure (adapt to input length):\n"
            "- Subject: appearance, expression, pose, clothing\n"
            "- Setting: environment, time of day, background\n"
            "- Lighting and Color: direction, quality, palette\n"
            "- Style and Mood: artistic style, emotional tone\n\n"
            "Rules:\n"
            "- Output ONLY the enhanced prompt, no explanations\n"
            "- Use natural language prose, not tags or keywords\n"
            "- Keep to 2-4 sentences\n"
            "- No meta tags like \"masterpiece\", \"8K\"\n"
            "- No emphasis weights like (word:1.2)\n"
            "- Wrap text that must appear in the image "
            "in \"double quotation marks\""
        ),
    },
    {
        "name": "Danbooru Tag Generator",
        "prompt": (
            "Output ONLY comma-separated Danbooru tags. "
            "Use underscores (e.g., blue_hair, red_eyes). "
            "No sentences. No explanations. Tags:"
        ),
    },
    {
        "name": "Mixed Tag + NL",
        "prompt": (
            "Transform the user's concept into a mixed prompt: "
            "start with Danbooru-style tags, end with 1 natural "
            "language sentence.\n\n"
            "Format:\n"
            "[quality tags], [character count], [descriptive tags]. "
            "[Short natural language sentence adding atmosphere or context]\n\n"
            "Example: "
            "masterpiece, best quality, 1girl, brown hair, blue eyes, smile, "
            "dress, outdoors, sunset. "
            "A girl with flowing brown hair smiles warmly in a sunlit field.\n\n"
            "Rules:\n"
            "- Tags first, then a period, then natural language\n"
            "- Use lowercase for tags, spaces between words\n"
            "- The NL sentence adds context, mood, or action "
            "not covered by tags\n"
            "- Output ONLY the mixed prompt, no explanations\n"
            "- Keep total under 3 lines"
        ),
    },
    {
        "name": "Tag to Natural Language",
        "prompt": (
            "Convert the following Danbooru-style tags into a "
            "fluent natural language description.\n\n"
            "Rules:\n"
            "- Write exactly 2 sentences of natural English prose\n"
            "- No tags, no parentheses, no symbols except punctuation\n"
            "- Preserve all visual information: appearance, pose, "
            "expression, clothing, background, lighting\n"
            "- Maintain spatial and compositional relationships\n"
            "- Include colors, lighting, and mood if present in tags\n"
            "- Do NOT add new elements not implied by the tags\n"
            "- Output ONLY the natural language description"
        ),
    },
    {
        "name": "Image Describer",
        "prompt": (
            "Describe the image in 1-2 short sentences. "
            "Focus only on what is visually present: subject, setting, "
            "colors, lighting, composition. "
            "No preamble, no explanations, no extra words."
        ),
    },
    {
        "name": "Detail Modifier",
        "prompt": (
            "Modify one specific visual detail in the user's prompt "
            "as requested. Preserve ALL other elements exactly.\n\n"
            "User will specify what to change (e.g., 'change hair color "
            "to red', 'make it nighttime', 'add a hat'). "
            "Change only the requested detail. "
            "Output only the modified prompt. "
            "Maximum 3 sentences. No explanations."
        ),
    },
    {
        "name": "Art Style Lite",
        "prompt": (
            "Describe the art style in 1 sentence. "
            "Identify: medium (oil painting, digital, watercolor, "
            "sketch, 3D render), brushwork or rendering technique, "
            "color palette tendencies, and artistic era or movement. "
            "No explanations. Output only the style description."
        ),
    },
    {
        "name": "Short Budget",
        "prompt": (
            "Describe the user's concept in a maximum of 50 words. "
            "Output exactly 1-2 short sentences. "
            "Focus only on the most essential visual elements: "
            "subject, setting, key colors. "
            "No extra words. No explanations. "
            "Count your words and stay under 50."
        ),
    },
    {
        "name": "Prompt Compressor",
        "prompt": (
            "Condense the following prompt down to its essential "
            "visual information. Remove redundant descriptions, "
            "filler words, and meta tags.\n\n"
            "Rules:\n"
            "- Output maximum 2 sentences\n"
            "- Keep the same format as input "
            "(tags stay tags, NL stays NL)\n"
            "- Remove all duplicate concepts\n"
            "- Remove meta tags like masterpiece, 8K, best quality\n"
            "- Preserve: subject, key actions, important setting "
            "details, lighting, style\n"
            "- Output ONLY the compressed prompt"
        ),
    },
    {
        "name": "Negative Prompt Generator",
        "prompt": (
            "Generate a concise negative prompt matching "
            "the user's request.\n\n"
            "Focus on:\n"
            "- anatomy errors\n"
            "- extra limbs\n"
            "- duplicate objects\n"
            "- cropped subjects\n"
            "- text artifacts\n"
            "- low quality details\n"
            "- any visual elements the user specifically "
            "wants to avoid\n\n"
            "Output only the negative prompt. No explanations."
        ),
    },
]


# ── Public API ───────────────────────────────────────────────────────────


def _get_prompts_file_path() -> str:
    """Return the path to the prompts JSON file."""
    return PROMPTS_FILE


def load_all_prompts() -> list:
    """
    Load all prompts from the JSON file.

    If the file doesn't exist, create it with default prompts.
    If the file is corrupt, log a warning, overwrite it with defaults, and return defaults.

    Returns:
        list: List of {name, prompt} dicts
    """
    filepath = _get_prompts_file_path()

    if not os.path.exists(filepath):
        logging.info(f"[LLM Chat] Creating system_prompts.json with defaults")
        save_all_prompts(DEFAULT_PROMPTS)
        return list(DEFAULT_PROMPTS)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both {"prompts": [...]} and direct [...] format
        if isinstance(data, dict) and "prompts" in data:
            return data["prompts"]
        if isinstance(data, list):
            return data

        logging.warning(f"[LLM Chat] Unexpected prompts file format, overwriting with defaults")
        save_all_prompts(DEFAULT_PROMPTS)
        return list(DEFAULT_PROMPTS)

    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"[LLM Chat] Failed to load prompts file ({e}), overwriting with defaults")
        save_all_prompts(DEFAULT_PROMPTS)
        return list(DEFAULT_PROMPTS)


def save_all_prompts(prompts: list) -> bool:
    """
    Save all prompts to the JSON file, overwriting any existing content.

    Uses an atomic write pattern (temp file + rename) to prevent file
    corruption from interrupted writes.

    Args:
        prompts: List of {name, prompt} dicts

    Returns:
        bool: True on success, False on failure
    """
    filepath = _get_prompts_file_path()
    tmp_path = None
    try:
        # Write to a temporary file in the same directory first
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(filepath),
            prefix=".system_prompts_tmp_",
            suffix=".json",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"prompts": prompts}, f, indent=2, ensure_ascii=False)

        # Atomic rename — replaces the target atomically on the same filesystem
        shutil.move(tmp_path, filepath)
        tmp_path = None  # prevent cleanup on success
        return True

    except (IOError, OSError) as e:
        logging.error(f"[LLM Chat] Failed to save prompts: {e}")
        return False

    finally:
        # Clean up temp file if the write or rename failed
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def get_prompt_names() -> list:
    """
    Get the list of prompt names for combo widgets.

    Always includes "Custom" as the first option so users can
    always fall back to typing their own prompt text.

    Returns:
        list: ["Custom", "Art Style Descriptor", "Prompt Engineer", ...]
    """
    prompts = load_all_prompts()
    names = ["Custom"]
    seen = set()
    for p in prompts:
        name = p.get("name", "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def get_prompt_by_name(name: str) -> str:
    """
    Get the prompt text for a given template name.

    Args:
        name: The prompt name to look up (e.g. "Art Style Descriptor")

    Returns:
        str: The prompt text, or empty string if not found or name is "Custom"
    """
    if not name or name == "Custom":
        return ""

    prompts = load_all_prompts()
    for p in prompts:
        if p.get("name") == name:
            return p.get("prompt", "")
    return ""


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
            """Save all prompts (overwrite). Called from management dialog."""
            try:
                data = await request.json()
                prompts = data.get("prompts", [])
                success = save_all_prompts(prompts)
                return web.json_response({
                    "success": success,
                    "prompts": prompts if success else load_all_prompts(),
                })
            except Exception as e:
                logging.error(f"[LLM Chat] Save prompts API error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.get("/easyllm/prompts/load")
        async def load_prompts_api(request):
            """Load all prompts. Called from management dialog."""
            try:
                prompts = load_all_prompts()
                return web.json_response({"success": True, "prompts": prompts})
            except Exception as e:
                logging.error(f"[LLM Chat] Load prompts API error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.post("/easyllm/prompts/import")
        async def import_prompts_api(request):
            """
            Import prompts from uploaded JSON data.

            Body: {
                "prompts": [{"name": "...", "prompt": "..."}, ...],
                "strategy": "append" | "replace" | "skip_duplicates"
            }

            Strategies:
            - append: Add imported prompts to existing list (duplicates allowed)
            - replace: Overwrite all existing prompts with imported ones
            - skip_duplicates: Only add prompts whose name doesn't already exist

            Returns: {"success": true, "prompts": [...], "imported_count": N, "skipped_count": M}
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

                current = load_all_prompts()
                existing_names = {p.get("name", "") for p in current}

                imported_count = 0
                skipped_count = 0

                if strategy == "replace":
                    # Replace all existing prompts
                    current = list(imported)
                    imported_count = len(imported)

                elif strategy == "skip_duplicates":
                    # Only add prompts whose name doesn't already exist
                    for p in imported:
                        name = p.get("name", "").strip()
                        if name and name not in existing_names:
                            current.append(p)
                            existing_names.add(name)
                            imported_count += 1
                        else:
                            skipped_count += 1

                else:  # append (default)
                    # Simply append all imported prompts
                    current.extend(imported)
                    imported_count = len(imported)

                success = save_all_prompts(current)
                if success:
                    return web.json_response({
                        "success": True,
                        "prompts": current,
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

        setup_routes._registered = True
        logging.info("[LLM Chat] System prompt API routes registered")

    except Exception as e:
        logging.error(
            f"[LLM Chat] Failed to register API routes: {e} — "
            "prompt management will be unavailable until ComfyUI restarts"
        )
