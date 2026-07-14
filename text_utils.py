"""
EasyLLM Text Utility Nodes
===========================

Lightweight text processing nodes for the EasyLLM prompt engineering toolkit.
No dependencies on the core LLM engine (no llama-cpp, no CLIP, no generation).

Nodes:
    1. LLM_TextInput           — 📝 EasyLLM Text Input (multi-line raw text entry)
    2. LLM_TextJoiner          — 🔗 EasyLLM Text Joiner (merge two strings with a delimiter)
    3. LLM_TextReplacer        — 🧹 EasyLLM Text Replacer (find-and-replace / scrubbing)
    4. LLM_TextExtractor       — ✂️ EasyLLM Text Extractor (keep text before/after a delimiter)
    5. LLM_TextLimiter         — 📏 EasyLLM Text Limiter (truncate to max characters/words/lines)
    6. LLM_TextWhitespaceCleaner — 🧽 EasyLLM Whitespace Cleaner (strip whitespace / blank lines)
    7. LLM_TextDuplicateRemover — 🗃️ EasyLLM Text Duplicate Remover (deduplicate tags/words/lines)
"""

import json
import logging
import re

from . import prompt_manager

# ──────────────────────────────────────────────────────────────────────
# Node 1: LLM_TextInput — 📝 EasyLLM Text Input
# ──────────────────────────────────────────────────────────────────────

class LLM_TextInput:
    """
    📝 EasyLLM Text Input — A simple multi-line raw text entry box.

    ComfyUI natively lacks a standalone text input node; users typically
    install third-party custom nodes just to type text.  This node fills
    that gap for the EasyLLM ecosystem.

    Inputs:
        text (STRING, multiline=True): The typed text.

    Outputs:
        STRING: The text as-is.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Build prompt list: "Custom" + all prompt names from library
        names = ["Custom"]
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
                "prompt_selector": (names, {
                    "default": "Custom",
                    "tooltip": (
                        "'Custom' = use the text field below. "
                        "Select a named prompt to output that prompt instead."
                    ),
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Type or paste your text here. Disabled when a named prompt is selected.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("The typed text, passed through as-is.",)
    FUNCTION = "passthrough"
    CATEGORY = "EasyLLM"
    DESCRIPTION = "A simple multi-line text input. Wire the output to any STRING input."

    def passthrough(self, text="", prompt_selector="Custom"):
        """
        Return the input text, or resolve a named prompt from the library.

        Args:
            text: The typed text (used when prompt_selector is "Custom").
            prompt_selector: "Custom" or a named prompt from the library.

        Returns:
            tuple: (resolved_text,)
        """
        if prompt_selector != "Custom" and prompt_selector.strip():
            try:
                resolved = prompt_manager.get_prompt_by_name(prompt_selector)
                if resolved:
                    return (resolved,)
            except Exception as e:
                logging.warning(
                    f"[LLM_TextInput] Failed to resolve prompt "
                    f"'{prompt_selector}': {e}"
                )
        return (text,)


# ──────────────────────────────────────────────────────────────────────
# Node 2: LLM_TextJoiner — 🔗 EasyLLM Text Joiner
# ──────────────────────────────────────────────────────────────────────

class LLM_TextJoiner:
    """
    🔗 EasyLLM Text Joiner — Merge two text strings using a configurable delimiter.

    Useful for appending style triggers, negative prompts, or Lora keywords
    to an LLM output before it reaches the sampler.

    Handles empty inputs gracefully — if either input is empty, the other
    is returned without the delimiter.

    Inputs:
        string_a  (STRING): First text input.
        string_b  (STRING): Second text input.
        delimiter (COMBO):  Delimiter placed between the two strings.
                            Options: [", ", " ", "\\n", " - "]

    Outputs:
        STRING: string_a + delimiter + string_b (when both non-empty).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_a": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "First text input.",
                }),
                "string_b": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Second text input.",
                }),
                "delimiter": ([", ", " ", "\\n", " - "], {
                    "default": ", ",
                    "tooltip": "Delimiter placed between the two strings.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("The joined text.",)
    FUNCTION = "join"
    CATEGORY = "EasyLLM/🔧 Tools"
    DESCRIPTION = (
        "Merge two strings with a clean delimiter. "
        "Handles empty inputs gracefully — no dangling delimiters."
    )

    def join(self, string_a="", string_b="", delimiter=", "):
        """Join two strings with the selected delimiter."""
        # Normalise delimiter: the Combo widget sends literal "\n" as text
        # (two characters backslash + n), but users expect a real newline.
        if delimiter == "\\n":
            delimiter = "\n"

        if string_a and string_b:
            return (string_a + delimiter + string_b,)
        elif string_a:
            return (string_a,)
        elif string_b:
            return (string_b,)
        return ("",)


# ──────────────────────────────────────────────────────────────────────
# Node 3: LLM_TextReplacer — 🧹 EasyLLM Text Replacer
# ──────────────────────────────────────────────────────────────────────

class LLM_TextReplacer:
    """
    🧹 EasyLLM Text Replacer — General-purpose find-and-replace / text scrubbing for any STRING output.

    Since EasyLLM and EasyLLMGGUF already output cleaned ``text`` (think
    tags, DeepSeek thinking blocks, Gemma channel tags, garbled Unicode,
    and repetitive punctuation are all stripped by ``clean_generated_text``),
    this node is **not** specialised for artifact cleaning.  Instead it
    provides flexible, user-defined text transformations on whichever output
    stream the user chooses (``text`` or ``raw_text``).

    Inputs:
        text           (STRING):   Input text to process.
        find_text      (STRING):   Substring to search for (default "" → passthrough).
        replace_with   (STRING):   Replacement text (default "" → removal).
        case_sensitive (BOOLEAN):  Toggle case-sensitive matching (default True).

    Outputs:
        STRING: The transformed text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Input text to process. Connect to 'text' or 'raw_text' from EasyLLM / EasyLLMGGUF.",
                }),
                "find_text": ("STRING", {
                    "multiline": False,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Substring to search for. Empty = passthrough (nothing to replace).",
                }),
                "replace_with": ("STRING", {
                    "multiline": False,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Replacement text. Empty = remove the matched text.",
                }),
                "case_sensitive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable for exact case matching. Disable for case-insensitive replacement.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("The transformed text after find-and-replace.",)
    FUNCTION = "replace"
    CATEGORY = "EasyLLM/🔧 Tools"
    DESCRIPTION = (
        "General-purpose find-and-replace. Works on any text stream "
        "(text or raw_text). Case-sensitive toggle. "
        "Empty find_text = safe passthrough."
    )

    def replace(self, text="", find_text="", replace_with="", case_sensitive=True):
        """Perform find-and-replace on the input text."""
        # Passthrough when nothing to find
        if not find_text:
            return (text,)

        # Passthrough when input is empty
        if not text:
            return (text,)

        if case_sensitive:
            result = text.replace(find_text, replace_with)
        else:
            # Case-insensitive: escape the literal find_text, then compile with IGNORECASE
            try:
                pattern = re.compile(re.escape(find_text), re.IGNORECASE)
                result = pattern.sub(replace_with, text)
            except re.error:
                logging.warning(
                    f"[LLM_TextReplacer] Regex compilation failed for "
                    f"find_text={find_text!r} — falling back to passthrough"
                )
                return (text,)

        return (result,)


# ──────────────────────────────────────────────────────────────────────
# Node 6: LLM_TextWhitespaceCleaner — 🧽 EasyLLM Whitespace Cleaner
# ──────────────────────────────────────────────────────────────────────


class LLM_TextWhitespaceCleaner:
    """
    🧽 EasyLLM Whitespace Cleaner — Strip leading/trailing whitespace
    or clean up blank lines in text.

    Also collapses runs of 3+ consecutive blank lines into a single blank
    line when ``collapse_blank_lines`` is enabled.

    Useful for cleaning raw model output that often has leading newlines,
    trailing spaces, or excessive blank lines before downstream processing.

    Inputs:
        text                 (STRING):  Input text to clean.
        trim_mode            (COMBO):   Where to trim: ["both", "leading", "trailing", "all_lines"]
        collapse_blank_lines (BOOLEAN): Collapse 3+ blank lines into 1 and strip leading/trailing blanks (default True).

    Outputs:
        STRING: The cleaned text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Input text to clean.",
                }),
                "trim_mode": ([
                    "both",
                    "leading",
                    "trailing",
                    "all_lines",
                ], {
                    "default": "both",
                    "tooltip": (
                        "'both' = strip both ends; "
                        "'leading' = only start; "
                        "'trailing' = only end; "
                        "'all_lines' = remove every empty/whitespace-only line."
                    ),
                }),
                "collapse_blank_lines": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Collapse 3+ consecutive blank lines into 1 blank line. "
                        "Also strips any leading/trailing blank lines that remain. "
                        "Has no effect when trim_mode is 'all_lines' "
                        "(which removes all blank lines regardless)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("The cleaned text.",)
    FUNCTION = "clean"
    CATEGORY = "EasyLLM/🔧 Tools"
    DESCRIPTION = (
        "Strip unwanted whitespace from text. "
        "Supports leading/trailing/both trimming and removal of blank lines."
    )

    def clean(self, text="", trim_mode="both", collapse_blank_lines=True):
        """Clean whitespace from the input text according to the chosen mode."""
        if not text:
            return ("",)

        # ── Primary trim based on mode ──
        if trim_mode == "leading":
            result = text.lstrip()
        elif trim_mode == "trailing":
            result = text.rstrip()
        elif trim_mode == "all_lines":
            # Remove every line that is empty or whitespace-only
            result = "\n".join(
                line for line in text.splitlines() if line.strip()
            )
        else:  # "both" (default)
            result = text.strip()

        # ── Secondary: collapse excessive blank lines ──
        if collapse_blank_lines and trim_mode != "all_lines":
            # Collapse runs of 3+ blank lines into a single blank line,
            # and strip any leading/trailing blank lines that remain.
            lines = result.splitlines()
            cleaned = []
            blank_run = 0
            for line in lines:
                if line.strip() == "":
                    blank_run += 1
                    if blank_run <= 2:   # keep at most one blank line (3+ → 1)
                        cleaned.append(line)
                else:
                    blank_run = 0
                    cleaned.append(line)
            # Strip leading/trailing blank lines
            while cleaned and cleaned[0].strip() == "":
                cleaned.pop(0)
            while cleaned and cleaned[-1].strip() == "":
                cleaned.pop()
            result = "\n".join(cleaned)

        return (result,)


# ──────────────────────────────────────────────────────────────────────
# Node 4: LLM_TextExtractor — ✂️ EasyLLM Text Extractor
# ──────────────────────────────────────────────────────────────────────


class LLM_TextExtractor:
    """
    ✂️ EasyLLM Text Extractor — Keep text before or after a delimiter word/phrase.

    Useful for stripping LLM preamble (e.g. "Here are the tags: cat, dog" →
    keep after ":" → " cat, dog") or extracting a specific section from
    structured output.

    Inputs:
        text              (STRING):   Input text to process.
        delimiter         (STRING):   Reference word/phrase. Empty = passthrough.
        mode              (COMBO):    ["keep_before", "keep_after"] — which side to keep.
        include_delimiter (BOOLEAN):  Include the delimiter itself in the output (default False).
        occurrence        (INT):      Which occurrence to anchor on. 1 = first, 2 = second, -1 = last (default 1).
        case_sensitive    (BOOLEAN):  Case-sensitive matching (default True).

    Outputs:
        STRING: The extracted portion, or original text if delimiter not found.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Input text to extract from.",
                }),
                "delimiter": ("STRING", {
                    "multiline": False,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Reference word or phrase. Empty = safe passthrough.",
                }),
                "mode": (["keep_before", "keep_after"], {
                    "default": "keep_after",
                    "tooltip": "'keep_after' = return text AFTER delimiter; 'keep_before' = return text BEFORE delimiter.",
                }),
                "include_delimiter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include the delimiter itself in the output.",
                }),
                "occurrence": ("INT", {
                    "default": 1,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Which occurrence to use. 1 = first, 2 = second, -1 = last.",
                }),
                "case_sensitive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable for exact case matching. Disable for case-insensitive search.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("The extracted text, or original if delimiter not found.",)
    FUNCTION = "extract"
    CATEGORY = "EasyLLM/🔧 Tools"
    DESCRIPTION = (
        "Keep text before or after a delimiter word/phrase. "
        "Supports occurrence selection (first, Nth, last) and case-insensitive matching. "
        "Safe passthrough when delimiter is empty or not found."
    )

    def _find_occurrence(self, text, delimiter, occurrence, case_sensitive):
        """
        Find the position of the Nth occurrence of delimiter in text.

        Args:
            text: The text to search in.
            delimiter: The substring to find.
            occurrence: 1-based index (1 = first, 2 = second, ...). -1 = last.
            case_sensitive: Whether matching is case-sensitive.

        Returns:
            int: Starting position of the occurrence, or -1 if not found.
        """
        if not case_sensitive:
            search_in = text.lower()
            search_for = delimiter.lower()
        else:
            search_in = text
            search_for = delimiter

        if occurrence == -1:
            # Last occurrence
            return search_in.rfind(search_for)

        # Nth occurrence (1-based)
        pos = -1
        for _ in range(occurrence):
            pos = search_in.find(search_for, pos + 1 if pos >= 0 else 0)
            if pos == -1:
                break
        return pos

    def extract(self, text="", delimiter="", mode="keep_after",
                include_delimiter=False, occurrence=1, case_sensitive=True):
        """Extract text before or after a delimiter."""
        # Safe passthrough when nothing to extract
        if not delimiter or not text:
            return (text,)

        # Find the delimiter occurrence
        pos = self._find_occurrence(text, delimiter, occurrence, case_sensitive)
        if pos == -1:
            return (text,)  # passthrough if not found

        if mode == "keep_after":
            start = pos + (0 if include_delimiter else len(delimiter))
            result = text[start:]
        else:  # keep_before
            end = pos + (len(delimiter) if include_delimiter else 0)
            result = text[:end]

        return (result,)


# ──────────────────────────────────────────────────────────────────────
# Node 5: LLM_TextLimiter — 📏 EasyLLM Text Limiter
# ──────────────────────────────────────────────────────────────────────


class LLM_TextLimiter:
    """
    📏 EasyLLM Text Limiter — Truncate text to a maximum number of
    characters, words, or lines.

    Useful for fitting model output into context windows, preview fields,
    or downstream nodes with strict input length limits.

    Inputs:
        text           (STRING):  Input text to truncate.
        mode           (COMBO):   Unit of measurement: ["characters", "words", "lines"]
        max_length     (INT):     Maximum allowed length (1–100000, default 500).
        truncate_side  (COMBO):   Which side to cut from: ["end", "start"]
        add_ellipsis   (BOOLEAN): Append "..." when truncation occurs (default False).

    Outputs:
        STRING: Truncated text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Input text to truncate.",
                }),
                "mode": (["characters", "words", "lines"], {
                    "default": "characters",
                    "tooltip": "Unit: 'characters', 'words', or 'lines'.",
                }),
                "max_length": ("INT", {
                    "default": 500,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Maximum length in the chosen unit.",
                }),
                "truncate_side": (["end", "start"], {
                    "default": "end",
                    "tooltip": "'end' = keep the beginning; 'start' = keep the end.",
                }),
                "add_ellipsis": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Append '...' when truncation occurs (end) or prepend (start).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("The truncated text.",)
    FUNCTION = "limit"
    CATEGORY = "EasyLLM/🔧 Tools"
    DESCRIPTION = (
        "Truncate text to a maximum number of characters, words, or lines. "
        "Choose which side to cut from and optionally add ellipsis."
    )

    def limit(self, text="", mode="characters", max_length=500,
              truncate_side="end", add_ellipsis=False):
        """Truncate text to the specified maximum length."""
        if not text or max_length <= 0:
            return ("",)

        truncated = False

        if mode == "characters":
            if len(text) <= max_length:
                return (text,)
            truncated = True
            if truncate_side == "end":
                result = text[:max_length]
            else:
                result = text[-max_length:]

        elif mode == "words":
            words = text.split()
            if len(words) <= max_length:
                return (text,)
            truncated = True
            if truncate_side == "end":
                result = " ".join(words[:max_length])
            else:
                result = " ".join(words[-max_length:])

        else:  # lines
            lines = text.splitlines()
            if len(lines) <= max_length:
                return (text,)
            truncated = True
            if truncate_side == "end":
                result = "\n".join(lines[:max_length])
            else:
                result = "\n".join(lines[-max_length:])

        if add_ellipsis and truncated:
            if truncate_side == "end":
                result += "..."
            else:
                result = "..." + result

        return (result,)


# ──────────────────────────────────────────────────────────────────────
# Node 7: LLM_TextDuplicateRemover — 🗃️ EasyLLM Text Duplicate Remover
# ──────────────────────────────────────────────────────────────────────


class LLM_TextDuplicateRemover:
    """
    🗃️ EasyLLM Text Duplicate Remover — Remove duplicate tags, words, or lines
    from text while preserving the original order (first occurrence wins).

    Ideal for cleaning Danbooru-style tag lists where the LLM may repeat
    tags, or for deduplicating any comma-separated, whitespace-split, or
    line-delimited text.

    Inputs:
        text              (STRING):  Input text to deduplicate.
        mode              (COMBO):   How to split: ["comma_separated", "words", "lines"]
        case_sensitive    (BOOLEAN): Treat "cat" and "Cat" as duplicates (default False).
        strip_whitespace  (BOOLEAN): Strip whitespace around each item before comparing (default True).
        separator         (STRING):  Used to rejoin items in comma_separated mode (default ", ").

    Outputs:
        STRING: Deduplicated text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Input text containing potential duplicates.",
                }),
                "mode": (["comma_separated", "words", "lines"], {
                    "default": "comma_separated",
                    "tooltip": (
                        "'comma_separated' = split by comma; "
                        "'words' = split by whitespace; "
                        "'lines' = split by newline."
                    ),
                }),
                "case_sensitive": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable so 'cat' and 'Cat' are treated as duplicates (recommended for tags).",
                }),
                "strip_whitespace": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Strip whitespace around each item before comparing.",
                }),
                "separator": ("STRING", {
                    "multiline": False,
                    "dynamicPrompts": False,
                    "default": ", ",
                    "tooltip": "Separator used to rejoin items in 'comma_separated' mode.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("Deduplicated text with first-occurrence order preserved.",)
    FUNCTION = "deduplicate"
    CATEGORY = "EasyLLM/🔧 Tools"
    DESCRIPTION = (
        "Remove duplicate tags, words, or lines. First occurrence wins. "
        "Case-insensitive by default — ideal for Danbooru tag cleanup."
    )

    def deduplicate(self, text="", mode="comma_separated",
                    case_sensitive=False, strip_whitespace=True,
                    separator=", "):
        """Deduplicate items in text while preserving first-occurrence order."""
        if not text:
            return ("",)

        # ── Split into items ──
        if mode == "comma_separated":
            raw_items = text.split(",")
        elif mode == "words":
            raw_items = text.split()
        else:  # lines
            raw_items = text.splitlines()

        # ── Deduplicate (first occurrence wins) ──
        seen = set()
        result_items = []
        for item in raw_items:
            key = item
            if strip_whitespace:
                key = key.strip()
            if not key:  # skip empty items
                continue
            if not case_sensitive:
                key = key.lower()

            if key not in seen:
                seen.add(key)
                result_items.append(item.strip() if strip_whitespace else item)

        # ── Rejoin ──
        if mode == "comma_separated":
            result = separator.join(result_items)
        elif mode == "words":
            result = " ".join(result_items)
        else:  # lines
            result = "\n".join(result_items)

        return (result,)


# ──────────────────────────────────────────────────────────────────────
# Node 8: LLM_TextAutoClean — 🧼 EasyLLM Auto Clean (no settings)
# ──────────────────────────────────────────────────────────────────────


class LLM_TextAutoClean:
    """
    🧼 EasyLLM Auto Clean (no settings) — Zero-config text cleaner for LLM output.

    Applies a cascading pipeline of regex passes to strip model artifacts,
    JSON blocks, stray punctuation, and (optionally) markdown syntax and
    conversational prefixes from generated text.

    Pipeline (least → most destructive):
      1. Model Artifacts       — <think>, Thinking..., <channel>, USER:/ASSISTANT:, garbled Unicode
      2. JSON Extraction       — find {…} blocks, extract string values via json.loads()
      3. Markdown Strip        — headers, bold/italic, lists, code fences, rules (aggressive only)
      4. Conversational Prefix — "Here is:", "Sure!", "Certainly:" etc. (aggressive only)
      5. Stray Punctuation     — leading/trailing unmatched quotes and brackets
      6. Whitespace Normalize  — collapse 3+ blank lines, strip edges

    Inputs:
        text       (STRING): Input text to clean.
        aggressive (BOOLEAN): When True, also strips markdown syntax and
                              conversational prefixes. Default: False.

    Outputs:
        STRING: Cleaned text.
    """

    # ── Conversational prefix patterns (Pass 4) ──
    _PREFIX_PATTERNS = [
        re.compile(r'^(Here(?: is|\'s| are).*?:)\s*'),
        re.compile(r'^(Sure(?:, I can help with that)?[.!:]\s*)'),
        re.compile(r'^(Certainly[!.:]?\s*)'),
        re.compile(r'^(Of course[!.:]?\s*)'),
        re.compile(r'^(I hope this helps[!.:]?\s*)'),
        re.compile(r'^(Hope this helps[!.:]?\s*)'),
        re.compile(r'^(The (?:generated|resulting|final|output) (?:prompt|text|tags)(?: would be| is|:)\s*)'),
        re.compile(r'^(As (?:an AI|a language model)[,.]?\s*)'),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Input text to clean — typically LLM output.",
                }),
                "aggressive": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When OFF (default): safe passes only (model artifacts, "
                        "JSON extraction, stray punctuation, whitespace). "
                        "When ON: also strips markdown syntax and conversational prefixes."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("Cleaned text with artifacts removed.",)
    FUNCTION = "clean"
    CATEGORY = "EasyLLM/🔧 Tools"
    DESCRIPTION = (
        "Zero-config text cleaner. Removes model artifacts, JSON blocks, "
        "stray punctuation. Aggressive mode also strips markdown and "
        "conversational prefixes. Logs what it removes."
    )

    # ──────────────────────────────────────────────────────────────
    # Pass 1: Model Artifacts (zero risk, always runs)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _pass_model_artifacts(text: str) -> str:
        """Strip model-internal artifacts — think tags, channel tags, garbled Unicode, etc."""
        if not text:
            return text

        from .utils import (
            THINK_TAG_PATTERN,
            DEEPSEEK_THINK_PATTERN,
            GEMMA_CHANNEL_PATTERN,
            _CHAT_TEMPLATE_LEAK_PATTERN,
            GARBLED_UNICODE_PATTERN,
        )

        original = text
        text = THINK_TAG_PATTERN.sub("", text)
        text = DEEPSEEK_THINK_PATTERN.sub("", text)
        text = GEMMA_CHANNEL_PATTERN.sub("", text)
        text = _CHAT_TEMPLATE_LEAK_PATTERN.sub("\n", text)
        text = GARBLED_UNICODE_PATTERN.sub("", text)

        if text != original:
            logging.info(
                "[LLM_TextAutoClean] Pass 1 (model_artifacts): "
                "removed artifacts — output length %d (was %d)",
                len(text), len(original)
            )
        return text

    # ──────────────────────────────────────────────────────────────
    # Pass 2: JSON Block Extraction (low risk, always runs)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _collect_strings(data) -> list:
        """Recursively collect all string values from a parsed JSON structure."""
        strings = []
        if isinstance(data, str):
            strings.append(data)
        elif isinstance(data, dict):
            for v in data.values():
                strings.extend(LLM_TextAutoClean._collect_strings(v))
        elif isinstance(data, (list, tuple)):
            for item in data:
                strings.extend(LLM_TextAutoClean._collect_strings(item))
        return strings

    @staticmethod
    def _pass_json_extraction(text: str) -> str:
        """Find JSON {…} blocks, extract string values, replace block with prose."""
        if not text:
            return text

        def _try_extract(match):
            block = match.group(0)
            try:
                data = json.loads(block)
                values = LLM_TextAutoClean._collect_strings(data)
                if values:
                    replacement = ", ".join(values)
                    logging.info(
                        "[LLM_TextAutoClean] Pass 2 (json_extraction): "
                        "extracted %d values from JSON block",
                        len(values)
                    )
                    return replacement
                return block
            except json.JSONDecodeError:
                return block  # not valid JSON — leave unchanged

        result = re.sub(r'\{[^{}]*\}', _try_extract, text)
        return result

    # ──────────────────────────────────────────────────────────────
    # Pass 3: Markdown Strip (medium risk, aggressive only)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _pass_markdown_strip(text: str) -> str:
        """Strip markdown formatting syntax while preserving semantic content."""
        if not text:
            return text

        original = text

        # Headers: ### Title → Title
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Bold/italic: **text** or *text* → text
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)

        # Bullet lists: "- cat\n- dog\n- bird" → "cat, dog, bird"
        # First pass: collect bullet lines and convert to comma-separated
        def _replace_bullet(m):
            items = [m.group(1).strip()]
            return items[0]
        # Replace individual bullet items, mark them temporarily
        text = re.sub(r'^[\s]*[-*+]\s+(.+)', r'\1', text, flags=re.MULTILINE)

        # Numbered lists: "1. cat" → "cat"
        text = re.sub(r'^\s*\d+[.)]\s+', '', text, flags=re.MULTILINE)

        # Code fences (block): ```…``` → (empty)
        text = re.sub(r'```[\s\S]*?```', '', text)

        # Inline code: `text` → text
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Horizontal rules: --- or *** or ___
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

        if text != original:
            logging.info(
                "[LLM_TextAutoClean] Pass 3 (markdown_strip): "
                "stripped markdown syntax — output length %d (was %d)",
                len(text), len(original)
            )
        return text

    # ──────────────────────────────────────────────────────────────
    # Pass 4: Conversational Prefix (medium-high risk, aggressive only)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _pass_conversational_prefix(text: str) -> str:
        """Strip common conversational prefixes from the start of text."""
        if not text:
            return text

        original = text
        changed = False
        for pattern in LLM_TextAutoClean._PREFIX_PATTERNS:
            match = pattern.match(text)
            if match:
                prefix = match.group(0)
                text = text[len(prefix):]
                changed = True
                logging.info(
                    "[LLM_TextAutoClean] Pass 4 (conversational_prefix): "
                    "removed prefix %r",
                    prefix[:60]
                )

        # Also strip common suffixes at the end
        _suffix_patterns = [
            re.compile(r'\s*(I hope this helps[.!]?)$', re.IGNORECASE),
            re.compile(r'\s*(Hope this helps[.!]?)$', re.IGNORECASE),
            re.compile(r'\s*(Let me know if you need anything else[.!]?)$', re.IGNORECASE),
            re.compile(r'\s*(Feel free to ask if you have questions[.!]?)$', re.IGNORECASE),
        ]
        for pattern in _suffix_patterns:
            match = pattern.search(text)
            if match:
                suffix = match.group(0)
                text = text[:match.start()]
                changed = True
                logging.info(
                    "[LLM_TextAutoClean] Pass 4 (conversational_suffix): "
                    "removed suffix %r",
                    suffix[:60]
                )

        if changed:
            return text
        return original

    # ──────────────────────────────────────────────────────────────
    # Pass 5: Stray Quotes & Brackets (low-medium risk, always runs)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _pass_stray_punctuation(text: str) -> str:
        """Remove leading/trailing stray quotes, brackets that aren't paired."""
        if not text:
            return text

        original = text

        # Leading/trailing stray quotes
        text = re.sub(r'^[\'"`]+', '', text)
        text = re.sub(r'[\'"`]+$', '', text)

        # Leading unmatched opening brackets
        text = re.sub(r'^[\[({]+(?![^\]})]*[\]})])', '', text)

        # Trailing unmatched closing brackets
        text = re.sub(r'(?<![\[({])[\]})]+$', '', text)

        if text != original:
            logging.info(
                "[LLM_TextAutoClean] Pass 5 (stray_punctuation): "
                "removed stray punctuation — output length %d (was %d)",
                len(text), len(original)
            )
        return text

    # ──────────────────────────────────────────────────────────────
    # Pass 6: Whitespace Normalize (zero risk, always runs)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _pass_whitespace_normalize(text: str) -> str:
        """Collapse runs of blank lines and strip leading/trailing whitespace."""
        if not text:
            return text

        original = text

        # 3+ consecutive blank lines → 1 blank line
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        if text != original:
            logging.info(
                "[LLM_TextAutoClean] Pass 6 (whitespace_normalize): "
                "normalized whitespace — output length %d (was %d)",
                len(text), len(original)
            )
        return text

    # ──────────────────────────────────────────────────────────────
    # Pipeline Orchestrator
    # ──────────────────────────────────────────────────────────────
    def clean(self, text="", aggressive=False):
        """
        Run the cascading pipeline on the input text.

        Args:
            text: Input text to clean.
            aggressive: When True, also runs markdown strip and
                        conversational prefix removal.

        Returns:
            tuple: (cleaned_text,)
        """
        if not text:
            return ("",)

        # Pass 1: Model Artifacts (always)
        text = self._pass_model_artifacts(text)

        # Pass 2: JSON Extraction (always)
        text = self._pass_json_extraction(text)

        # Pass 3: Markdown Strip (aggressive only)
        if aggressive:
            text = self._pass_markdown_strip(text)

        # Pass 4: Conversational Prefix (aggressive only)
        if aggressive:
            text = self._pass_conversational_prefix(text)

        # Pass 5: Stray Quotes & Brackets (always)
        text = self._pass_stray_punctuation(text)

        # Pass 6: Whitespace Normalize (always)
        text = self._pass_whitespace_normalize(text)

        return (text,)


# ──────────────────────────────────────────────────────────────────────
# Node Registration
# ──────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LLM_TextInput": LLM_TextInput,
    "LLM_TextJoiner": LLM_TextJoiner,
    "LLM_TextReplacer": LLM_TextReplacer,
    "LLM_TextExtractor": LLM_TextExtractor,
    "LLM_TextLimiter": LLM_TextLimiter,
    "LLM_TextWhitespaceCleaner": LLM_TextWhitespaceCleaner,
    "LLM_TextDuplicateRemover": LLM_TextDuplicateRemover,
    "LLM_TextAutoClean": LLM_TextAutoClean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_TextInput": "📝 EasyLLM Text Input",
    "LLM_TextJoiner": "🔗 EasyLLM Text Joiner",
    "LLM_TextReplacer": "🧹 EasyLLM Text Replacer",
    "LLM_TextExtractor": "✂️ EasyLLM Text Extractor",
    "LLM_TextLimiter": "📏 EasyLLM Text Limiter",
    "LLM_TextWhitespaceCleaner": "🧽 EasyLLM Whitespace Cleaner",
    "LLM_TextDuplicateRemover": "🗃️ EasyLLM Text Duplicate Remover",
    "LLM_TextAutoClean": "🧼 EasyLLM Auto Clean (no settings)",
}
