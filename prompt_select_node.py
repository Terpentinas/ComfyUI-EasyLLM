"""
📚 EasyLLM Prompt Select Node
==============================

Selects a prompt from the Prompt Library (system_prompts.json) and outputs it
with optional text chaining and formatting.

Features:
    - Category filter: "All" or a specific category from the library
    - Prompt modes: "🔀 Random", "🔄 Moving", or a named prompt
    - Format modes: prompt_only, prepend, append, template ({input} substitution)
    - Seed widget for reproducible random selection
    - Text chaining: input_text socket for pipeline composition

No dependencies on the core LLM engine (no llama-cpp, no CLIP, no generation).
"""

import logging
import random

from . import prompt_manager

# ── Module-level state for Moving mode ──────────────────────────────
# Keyed by unique_id, persists across executions within a session.
# Resets on ComfyUI restart.
_prompt_select_counters: dict[str, int] = {}

# ── Constants ────────────────────────────────────────────────────────

RANDOM_LABEL = "🔀 Random"
MOVING_LABEL = "🔄 Moving"
ALL_LABEL = "All"

FORMAT_OPTIONS = ["prompt_only", "prepend", "append", "template"]


# ── Node ─────────────────────────────────────────────────────────────


class LLM_PromptSelect:
    """
    📚 EasyLLM Prompt Select — Pick a prompt from the Prompt Library
    and pipe it into your workflow.

    Inputs:
        text_input (STRING, optional): Text to chain through. Combined with
            the selected prompt based on ``format`` mode.

    Widgets:
        category:     Filter prompts by category ("All" = all categories).
        prompt_name:  "🔀 Random", "🔄 Moving", or a named prompt.
        format:       How to combine with input_text.
        separator:    Text between prompt and input (prepend/append modes).
        seed:         Seed for Random mode (0 = auto-randomize).

    Outputs:
        STRING: The resulting text after prompt selection + formatting.

    Format modes:
        prompt_only  → Output just the selected prompt text.
        prepend      → prompt + separator + input_text.
        append       → input_text + separator + prompt.
        template     → prompt.replace("{input}", input_text).
                       Falls back to append if prompt has no {input}.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define widgets and sockets for the node."""
        # Build category list: "All" + all categories from the library
        try:
            cats = prompt_manager.get_categories()
        except Exception:
            cats = []
        categories = [ALL_LABEL] + cats

        # Build prompt names list: Random + Moving + all prompt names
        names = [RANDOM_LABEL, MOVING_LABEL]
        try:
            struct = prompt_manager.load_all_prompts()
            seen = set()
            for p in struct.get("prompts", []):
                n = p.get("name", "").strip()
                if n and n not in seen:
                    names.append(n)
                    seen.add(n)
        except Exception:
            pass

        return {
            "required": {
                "category": (categories, {
                    "default": ALL_LABEL,
                    "tooltip": "Filter prompts by category. 'All' = search across all categories.",
                }),
                "prompt_name": (names, {
                    "default": RANDOM_LABEL,
                    "tooltip": (
                        "'🔀 Random' = picks one each execution (use seed for reproducibility). "
                        "'🔄 Moving' = cycles through prompts sequentially. "
                        "Named prompts are stable selections."
                    ),
                }),
                "format": (FORMAT_OPTIONS, {
                    "default": "prompt_only",
                    "tooltip": (
                        "'prompt_only' = output just the prompt. "
                        "'prepend' = prompt + separator + input. "
                        "'append' = input + separator + prompt. "
                        "'template' = replace {input} in prompt text."
                    ),
                }),
                "separator": ("STRING", {
                    "multiline": False,
                    "dynamicPrompts": False,
                    "default": "\\n\\n",
                    "tooltip": "Text placed between prompt and input when prepending/appending.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "step": 1,
                    "tooltip": "Seed for Random mode. 0 = auto-randomize each execution.",
                }),
            },
            "optional": {
                "text_input": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Chain input: text from another node. Combined with the selected prompt based on format mode.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = (
        "The resulting text after prompt selection and formatting.",
    )
    FUNCTION = "select_prompt"
    CATEGORY = "EasyLLM/🎛️ Advanced"
    DESCRIPTION = (
        "Select a prompt from the Prompt Library with Random/Moving modes "
        "and text chaining. Supports {input} placeholders for template-style wrapping."
    )

    # ── Public methods ───────────────────────────────────────────────

    def select_prompt(
        self,
        category=ALL_LABEL,
        prompt_name=RANDOM_LABEL,
        format="prompt_only",
        separator="\\n\\n",
        seed=0,
        text_input="",
        unique_id=None,
    ):
        """
        Select and format a prompt from the Prompt Library.

        Args:
            category: Category filter ("All" or a specific category name).
            prompt_name: "🔀 Random", "🔄 Moving", or a named prompt.
            format: Output format mode.
            separator: Text between prompt and input (prepend/append).
            seed: Random seed (0 = auto-randomize).
            text_input: Optional chained input text.
            unique_id: ComfyUI node unique ID (for Moving mode persistence).

        Returns:
            tuple: (result_text,)
        """
        # Normalise separator: STRING widget sends literal "\n" as text
        if separator == "\\n\\n":
            separator = "\n\n"
        elif separator == "\\n":
            separator = "\n"
        elif separator == "\\t":
            separator = "\t"

        # ── RESOLVE PROMPT TEXT ──────────────────────────────────────
        prompt_text = self._resolve_prompt_text(
            category=category,
            prompt_name=prompt_name,
            seed=seed,
            unique_id=unique_id,
        )

        # ── APPLY FORMATTING ─────────────────────────────────────────
        result = self._apply_format(
            prompt_text=prompt_text,
            input_text=text_input,
            format_mode=format,
            separator=separator,
        )

        return (result,)

    # ── Internal helpers ─────────────────────────────────────────────

    def _resolve_prompt_text(
        self,
        category: str,
        prompt_name: str,
        seed: int,
        unique_id: str,
    ) -> str:
        """
        Resolve the selected prompt text based on category, prompt_name, and mode.

        Steps:
            1. Load all prompts from the library.
            2. Filter by category (if not "All").
            3. Apply selection mode (Random / Moving / Named).
            4. Return the prompt text (or empty string on failure).

        Returns:
            str: The resolved prompt text, or "" if not found.
        """
        try:
            struct = prompt_manager.load_all_prompts()
        except Exception as e:
            logging.error(f"[LLM_PromptSelect] Failed to load prompts: {e}")
            return ""

        prompts = struct.get("prompts", [])
        if not prompts:
            logging.warning("[LLM_PromptSelect] No prompts found in library")
            return ""

        # Filter by category
        if category != ALL_LABEL:
            filtered = [p for p in prompts if p.get("category") == category]
        else:
            filtered = list(prompts)

        if not filtered:
            logging.warning(
                f"[LLM_PromptSelect] No prompts found for category '{category}'"
            )
            return ""

        # Resolve by mode
        if prompt_name == RANDOM_LABEL:
            return self._random_select(filtered, seed)
        elif prompt_name == MOVING_LABEL:
            return self._moving_select(filtered, unique_id)
        else:
            # Named prompt — find exact match in filtered list
            for p in filtered:
                if p.get("name") == prompt_name:
                    return p.get("prompt", "")
            # Fallback: search all prompts (in case category filter removed it)
            try:
                return prompt_manager.get_prompt_by_name(prompt_name)
            except Exception:
                logging.warning(
                    f"[LLM_PromptSelect] Prompt '{prompt_name}' not found "
                    f"in category '{category}'"
                )
                return ""

    def _random_select(self, prompts: list, seed: int) -> str:
        """
        Pick a random prompt from the filtered list.

        Args:
            prompts: List of prompt dicts with "prompt" key.
            seed: Random seed. 0 = auto-randomize (time-based).

        Returns:
            str: The selected prompt text, or "" if list is empty.
        """
        if not prompts:
            return ""

        effective_seed = seed if seed != 0 else random.randint(1, 0xFFFFFFFF)
        rng = random.Random(effective_seed)
        chosen = rng.choice(prompts)
        prompt_text = chosen.get("prompt", "")

        logging.debug(
            f"[LLM_PromptSelect] Random select: seed={effective_seed}, "
            f"chosen='{chosen.get('name', '?')}'"
        )

        return prompt_text

    def _moving_select(self, prompts: list, unique_id: str) -> str:
        """
        Cycle through prompts sequentially, returning the next one each execution.

        Persists the current index per unique_id in a module-level dict.
        Resets on ComfyUI restart.

        Args:
            prompts: List of prompt dicts with "prompt" key.
            unique_id: ComfyUI node unique ID for per-instance tracking.

        Returns:
            str: The selected prompt text, or "" if list is empty.
        """
        if not prompts:
            return ""

        idx = _prompt_select_counters.get(unique_id, 0)
        idx = idx % len(prompts)
        _prompt_select_counters[unique_id] = (idx + 1) % len(prompts)

        chosen = prompts[idx]
        prompt_text = chosen.get("prompt", "")

        logging.debug(
            f"[LLM_PromptSelect] Moving select: unique_id={unique_id}, "
            f"index={idx}, name='{chosen.get('name', '?')}'"
        )

        return prompt_text

    def _apply_format(
        self,
        prompt_text: str,
        input_text: str,
        format_mode: str,
        separator: str,
    ) -> str:
        """
        Apply the selected format mode to produce the final output.

        Args:
            prompt_text: The resolved prompt text.
            input_text: Optional chained input text.
            format_mode: One of FORMAT_OPTIONS.
            separator: Delimiter for prepend/append modes.

        Returns:
            str: The formatted output text.
        """
        has_input = bool(input_text and input_text.strip())

        if format_mode == "prompt_only" or not has_input:
            # When no input text, all modes fall back to prompt_only
            return prompt_text

        if format_mode == "prepend":
            return prompt_text + separator + input_text

        if format_mode == "append":
            return input_text + separator + prompt_text

        if format_mode == "template":
            if "{input}" in prompt_text:
                return prompt_text.replace("{input}", input_text)
            else:
                # Graceful fallback: append with separator
                logging.debug(
                    "[LLM_PromptSelect] Template mode but prompt has no "
                    "{input} placeholder — falling back to append"
                )
                return input_text + separator + prompt_text

        # Fallback (shouldn't reach here)
        return prompt_text


# ── Node Registration ─────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LLM_PromptSelect": LLM_PromptSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_PromptSelect": "📚 EasyLLM Prompt Select",
}
