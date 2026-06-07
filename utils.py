"""
Shared utilities for model detection, prompt formatting, and text cleaning.

Provides helper functions for:
- Detection of whether a loaded CLIP object supports text generation
- Chat template formatting for different model types
- Direct access to inner tokenizers
- Safe token ID decoding
- Model size detection for auto-tuning sampling parameters
- Image tensor to base64 PNG conversion
"""

import json
import logging
import os
import re
import struct
from functools import lru_cache

# Pre-compiled regex patterns for _extract_prefix
_RE_STRIP_QUANT = re.compile(
    r"[.-](?:Q[0-9]_[A-Z0-9_]+|F16|F32|BF16|IQ[0-9]_[A-Z0-9_]+)$",
    re.IGNORECASE,
)
_RE_STRIP_FP = re.compile(
    r"[.-](?:fp16|fp32|bf16)$",
    re.IGNORECASE,
)
import io
import base64
import hashlib
import threading

import torch
from PIL import Image

# Qwen chat template format (legacy, kept for backward compatibility)
QWEN_CHAT_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

# ── Multi-Model Chat Template Registry ──────────────────────────────
#
# Maps model family -> template format for {system, user, assistant, stop}.
# "system" is omitted entirely when system_prompt is empty.
# Mistral embeds system inside [INST]; its "system" key includes {user_text}
# and replaces "user" when system_prompt is provided.

CHAT_TEMPLATES = {
    "qwen": {
        "system": "<|im_start|>system\n{system_prompt}<|im_end|>\n",
        "user": "<|im_start|>user\n{user_text}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n",
        "stop": ["<|im_end|>"],
    },
    "llama": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "stop": ["<|eot_id|>", "<|start_header_id|>"],
    },
    "mistral": {
        # Mistral embeds system prompt inside the [INST] block.
        # When system_prompt is provided, the "system" key (which includes {user_text}) replaces
        # the user message entirely — system and user are wrapped together in one [INST] block.
        "system": "[INST] {system_prompt}\n\n{user_text} [/INST]",
        "user": "[INST] {user_text} [/INST]",
        "assistant": "",
        "stop": ["</s>", "[INST]"],
    },
    "phi": {
        "system": "<|system|>\n{system_prompt}<|end|>\n",
        "user": "<|user|>\n{user_text}<|end|>\n",
        "assistant": "<|assistant|>\n",
        "stop": ["<|end|>"],
    },
    "deepseek": {
        "system": "{system_prompt}\n\n",
        "user": "User: {user_text}\n\n",
        "assistant": "Assistant: ",
        "stop": ["<｜end▁of▁sentence｜>"],
    },
    "gemma": {
        "system": "<start_of_turn>user\n{system_prompt}\n{user_text}<end_of_turn>\n",
        "user": "<start_of_turn>user\n{user_text}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n",
        "stop": ["<end_of_turn>"],
    },
}

# ── Architecture-to-Template Mapping ─────────────────────────────────────
# Maps GGUF general.architecture values to CHAT_TEMPLATES keys.
# Fallback used when tokenizer.chat_template metadata is missing.
ARCHITECTURE_TO_TEMPLATE = {
    "llama": "llama",
    "llama3": "llama",
    "llama2": "llama",
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2vl": "qwen",
    "mistral": "mistral",
    "mixtral": "mistral",
    "phi": "phi",
    "phi3": "phi",
    "phi4": "phi",
    "deepseek": "deepseek",
    "deepseek2": "deepseek",
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "gemma4": "gemma",
}

# Default template name when lookup fails
_DEFAULT_TEMPLATE = "qwen"


def format_prompt_by_template(
    user_text: str,
    system_prompt: str = "",
    template_name: str = "qwen",
    history: list | None = None,
) -> str:
    """Format a prompt using a named chat template.

    Supports both single-turn (no history) and multi-turn (with history) formatting.
    When system_prompt is empty/blank, the system block is omitted entirely.

    For Mistral-style templates where system_prompt is embedded in the [INST] block,
    the system key (which includes {user_text}) replaces the user key when a system
    prompt is provided.

    Args:
        user_text: The user's message to respond to.
        system_prompt: Optional system prompt. Empty = no system block.
        template_name: Key into CHAT_TEMPLATES dict. Falls back to "qwen" if unknown.
        history: Optional list of {"role": "user"/"assistant", "message": "..."} dicts
                 for multi-turn conversation context.

    Returns:
        str: The fully formatted prompt string, ready to feed to the model.
    """
    # Look up template, fall back to qwen on unknown name
    template = CHAT_TEMPLATES.get(template_name)
    if template is None:
        logging.warning(
            f"[LLM Chat] Unknown chat template '{template_name}' — "
            f"falling back to '{_DEFAULT_TEMPLATE}'"
        )
        template = CHAT_TEMPLATES[_DEFAULT_TEMPLATE]

    # Determine if this is an embedded template (like Mistral) where the system
    # key includes {user_text} and replaces the user message entirely.
    is_embedded = "{user_text}" in template["system"]

    has_system = bool(system_prompt and system_prompt.strip())
    has_history = bool(history)

    # ── Multi-turn: format system + history + current user ──────────────
    if has_history:
        parts = []

        # System block
        if has_system:
            if is_embedded:
                # Embedded (Mistral): system key contains {user_text}, so it
                # replaces the user key for this turn. Append assistant prefix
                # and return — the whole turn (system+user) is in one block,
                # and history entries precede it.
                for entry in history:
                    role = entry.get("role", "user")
                    msg = entry.get("message", "")
                    if role == "user":
                        parts.append(template["user"].format(user_text=msg))
                    elif role == "assistant":
                        parts.append(template["assistant"])
                        parts.append(msg)
                # Current turn: system with embedded user_text
                parts.append(template["system"].format(
                    system_prompt=system_prompt, user_text=user_text
                ))
                parts.append(template["assistant"])
                return "".join(parts)
            else:
                # Standard templates: separate system block
                parts.append(template["system"].format(system_prompt=system_prompt))

        # History entries
        for entry in history:
            role = entry.get("role", "user")
            msg = entry.get("message", "")
            if role == "user":
                parts.append(template["user"].format(user_text=msg))
            elif role == "assistant":
                parts.append(template["assistant"])
                parts.append(msg)

        # Current user message
        parts.append(template["user"].format(user_text=user_text))
        # Append assistant prefix (model generates from here)
        parts.append(template["assistant"])

        return "".join(parts)

    # ── Single-turn: format directly ───────────────────────────────────
    if has_system:
        if is_embedded:
            # Mistral-style: system key includes {user_text}, replaces user key
            formatted = template["system"].format(
                system_prompt=system_prompt, user_text=user_text
            )
        else:
            # Standard templates: separate system block + user block
            formatted = (
                template["system"].format(system_prompt=system_prompt)
                + template["user"].format(user_text=user_text)
            )
        formatted += template["assistant"]
        return formatted

    # No system prompt — user only
    return template["user"].format(user_text=user_text) + template["assistant"]


# ── Chat Template Detection from GGUF Metadata ──────────────────────────

# Pattern-based recognition for GGUF metadata tokenizer.chat_template field.
# Maps Jinja template patterns found in GGUF model metadata to
# CHAT_TEMPLATES keys.
_TEMPLATE_DETECTION_PATTERNS = [
    (r"<\|im_start\|>", "qwen"),
    (r"<\|start_header_id\|>.*?<\|eot_id\|>", "llama"),
    (r"\[INST\]", "mistral"),
    (r"<\|system\|>.*?<\|end\|>", "phi"),
    (r"User:\s*\{", "deepseek"),
    (r"<start_of_turn>", "gemma"),
]


def detect_chat_template_from_metadata(metadata: dict) -> str | None:
    """Try to detect the matching chat template name from GGUF model metadata.

    Reads 'tokenizer.chat_template' (or 'tokenizer.ggml.chat_template') from
    GGUF metadata and matches known Jinja patterns to our CHAT_TEMPLATES names.

    Args:
        metadata: Dict from LlamaCppModel.metadata (e.g. {'tokenizer.chat_template': '...'}).

    Returns:
        str: One of CHAT_TEMPLATES keys (qwen, llama, mistral, etc.),
             or None if no match found.

    Auto-detection only applies when user has not explicitly overridden the default template.
    """
    if not metadata or not isinstance(metadata, dict):
        return None

    # Try common metadata keys for chat template (older GGUF files use
    # 'tokenizer.ggml.chat_template', newer ones use 'tokenizer.chat_template')
    chat_template = metadata.get("tokenizer.chat_template", "")
    if not chat_template:
        chat_template = metadata.get("tokenizer.ggml.chat_template", "")
    if not chat_template:
        return None

    # Pattern matching (order matters: most specific first)
    for pattern, template_name in _TEMPLATE_DETECTION_PATTERNS:
        if re.search(pattern, chat_template, re.DOTALL):
            logging.info(
                f"[LLM Chat] Auto-detected chat template '{template_name}' "
                f"from GGUF metadata (matched pattern: {pattern})"
            )
            return template_name

    logging.debug(
        f"[LLM Chat] Could not auto-detect chat template from metadata — "
        f"template string: {chat_template[:100]}..."
    )
    return None


def detect_template_from_architecture(metadata: dict) -> str | None:
    """Fallback: map ``general.architecture`` to a CHAT_TEMPLATES key.

    Used when ``tokenizer.chat_template`` metadata is missing or unreadable.
    The ``general.architecture`` field is almost always present in GGUF files
    and is more reliable than filename heuristics.

    Args:
        metadata: Dict from LlamaCppModel.metadata or GGUF reader.

    Returns:
        str: One of CHAT_TEMPLATES keys, or None if architecture is unknown.
    """
    if not metadata or not isinstance(metadata, dict):
        return None

    arch = metadata.get("general.architecture", "")
    if not arch:
        return None

    arch = arch.strip().lower()
    template = ARCHITECTURE_TO_TEMPLATE.get(arch)
    if template:
        logging.info(
            f"[LLM Chat] Detected chat template '{template}' "
            f"from model architecture '{arch}'"
        )
        return template

    logging.debug(
        f"[LLM Chat] Unknown model architecture '{arch}' — "
        f"no template mapping available"
    )
    return None


# ── Filename Heuristics for Template Detection ───────────────────────────
# Regex patterns matched against the model's filename (after stripping
# quantization suffix). Fallback when metadata and architecture lookups fail.
_FILENAME_TEMPLATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Llama 3 / 3.1 family
    (re.compile(r"(?:llama[.-]?3|v13[.-]?ki|gqa[.-]?l10)", re.IGNORECASE), "llama"),
    # Llama 2 (legacy)
    (re.compile(r"(?:llama[.-]?2)", re.IGNORECASE), "llama"),
    # Qwen family
    (re.compile(r"(?:^|[\W_])qwen", re.IGNORECASE), "qwen"),
    # DeepSeek family
    (re.compile(r"(?:^|[\W_])deepseek", re.IGNORECASE), "deepseek"),
    # Gemma family
    (re.compile(r"(?:^|[\W_])gemma", re.IGNORECASE), "gemma"),
    # Phi family
    (re.compile(r"(?:^|[\W_])phi[.-]?[34]|\bphi\b", re.IGNORECASE), "phi"),
    # Mistral family
    (re.compile(r"(?:^|[\W_])mistral", re.IGNORECASE), "mistral"),
]


def heuristic_template_from_filename(model_path: str) -> str | None:
    """Fallback: detect chat template from model filename using regex heuristics.

    Used as last resort when both GGUF metadata and architecture lookups fail.
    Strips quantization suffixes and GGUF extension before matching.

    Args:
        model_path: Absolute or relative path to the .gguf file.

    Returns:
        str: One of CHAT_TEMPLATES keys, or None if no pattern matches.
    """
    if not model_path:
        return None

    # Extract clean filename without path or extension
    import os
    filename = os.path.basename(model_path)
    if not filename:
        return None

    # Strip .gguf extension and quantization suffixes
    clean_name = _extract_prefix(filename)

    for pattern, template_name in _FILENAME_TEMPLATE_PATTERNS:
        if pattern.search(clean_name):
            logging.info(
                f"[LLM Chat] Heuristic filename match: '{template_name}' "
                f"for '{filename}' (matched pattern: {pattern.pattern})"
            )
            return template_name

    logging.debug(
        f"[LLM Chat] No heuristic filename match for '{filename}'"
    )
    return None


def resolve_system_prompt(
    system_prompt: str,
    prompt_template: str,
    system_prompt_text: str,
) -> str:
    """Resolve effective system prompt from up to 4 sources.

    Priority (highest to lowest):
    1. system_prompt: Connected from another node (forceInput socket)
    2. prompt_template: Selected dropdown value (looked up via get_prompt_by_name())
    3. system_prompt_text: Manual text input widget
    4. Empty string fallback (no system prompt)

    Args:
        system_prompt: Value from the forceInput socket (may be empty string).
        prompt_template: Selected template name (e.g. "Art Style Descriptor" or "Custom").
        system_prompt_text: Manual text typed in the system_prompt_text widget.

    Returns:
        str: The resolved system prompt text, or empty string if all sources are empty.
    """
    # Priority 1: Connected from another node (forceInput socket)
    if system_prompt and system_prompt.strip():
        return system_prompt

    # Priority 2: Template selected from dropdown
    if prompt_template and prompt_template != "Custom":
        # Import here to avoid circular imports at module level
        from .prompt_manager import get_prompt_by_name
        template_text = get_prompt_by_name(prompt_template)
        if template_text:
            return template_text

    # Priority 3: Custom text typed in widget
    if system_prompt_text and system_prompt_text.strip():
        return system_prompt_text

    # Priority 4: Empty — no system prompt
    return ""


def resolve_text(text_input: str, text: str) -> str:
    """Resolve effective input text from available sources.

    Priority:
    1. text_input: Connected from another node's STRING output (forceInput socket)
    2. text: Widget value (typed on canvas or set by popup)

    Logs a warning when both sources have values, since the chain input
    silently takes priority and the widget value is ignored.

    Args:
        text_input: Value from the forceInput socket (may be empty string).
        text: Value from the canvas text widget.

    Returns:
        str: The resolved text, or empty string if both sources are empty.
    """
    # Warn if both sources have values (silent priority conflict)
    if text_input and text_input.strip() and text and text.strip():
        logging.warning(
            "[LLM Chat] Both text_input (chain) and text widget have values. "
            "Chain input takes priority. Widget value '%s...' will be ignored.",
            text[:50]
        )

    # Priority 1: Connected from another node
    if text_input and text_input.strip():
        return text_input

    # Priority 2: Widget value
    return text


# Garbled Unicode filter pattern for cleaning generated text
# Small models (0.6B) can generate high token IDs that decode to
# unusual Unicode symbols. These ranges contain non-textual characters:
#
#   U+2300-23FF  Miscellaneous Technical
#   U+2500-257F  Box Drawing
#   U+2580-259F  Block Elements
#   U+25A0-25FF  Geometric Shapes
#   U+2600-26FF  Miscellaneous Symbols  (⚙ U+2699, ⚐ U+2690)
#   U+2700-27BF  Dingbats
#   U+2B00-2BFF  Miscellaneous Symbols and Arrows
#   U+FFFD       Replacement Character (�)
#   U+FE00-FE0F  Variation Selectors
#   U+200B-200F  Zero-width spaces (invisible formatting)
#   U+2060-206F  Invisible operators
#   U+FEFF       BOM / Zero-width no-break space
GARBLED_UNICODE_PATTERN = re.compile(
    r'[\u2300-\u23ff\u2500-\u27bf\u2b00-\u2bff\ufffd\ufffe\uffff'
    r'\U0001f000-\U0001f02f\ufe00-\ufe0f\u200b-\u200f\u2060-\u206f\ufeff]'
)

# Think tag pattern: strips <think>...</think> blocks from model output.
# Qwen3 and similar models output reasoning/thinking in these tags.
# We strip them from the cleaned text output but keep them in raw_text.
THINK_TAG_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)

# DeepSeek think pattern: strips "Thinking...\n...\n\n" blocks from model output.
# DeepSeek models start generation with "Thinking...\n" followed by reasoning,
# then a blank line before the final response.
DEEPSEEK_THINK_PATTERN = re.compile(r'^Thinking\.\.\.\n.*?\n\n', re.DOTALL | re.MULTILINE)

# Gemma channel pattern: strips everything from <|channel> to <channel|>.
# Gemma 3/4 models use official thinking format:
#   <|channel>thought\n{thinking}\n<channel|>{response}
GEMMA_CHANNEL_PATTERN = re.compile(r'<\|channel>.*?<channel\|>', re.DOTALL)

# Chat template leak pattern: strips standalone USER:/ASSISTANT: markers
# that models sometimes generate as raw text when the model's built-in
# Jinja template produces a generic "USER:/ASSISTANT:" format and no
# proper stop tokens are configured. This is a safety-net — the primary
# fix is forwarding stop tokens from CHAT_TEMPLATES to create_chat_completion.
# Matches lines like "USER:\n", "User:\n", "ASSISTANT:\n", "Assistant:\n"
# appearing as standalone markers (preceded by newline/start, followed by
# newline/end).
_CHAT_TEMPLATE_LEAK_PATTERN = re.compile(
    r'(?:^|\n)[ \t]*(?:USER|User|ASSISTANT|Assistant)\s*:\s*(?:\n|$)',
    re.MULTILINE,
)

# Universal plain-text role markers appended to template-specific stop tokens
# in the vision/multimodal path. Catches models whose built-in Jinja template
# renders to a generic "USER:/ASSISTANT:" plain-text format (e.g., JoyCaption,
# Vicuna variants) rather than special tokens like <|eot_id|>.
# These are only added in the vision path where create_chat_completion() is used.
_UNIVERSAL_ROLE_STOP_TOKENS = [
    "USER:",
    "ASSISTANT:",
]


def auto_seed(seed: int) -> int:
    """If seed is 0, return a random seed. Otherwise return seed as-is.

    Args:
        seed: Input seed value. 0 means "auto-randomize".

    Returns:
        int: A valid random seed in range [1, 0xFFFFFFFF].
    """
    if seed == 0:
        import random
        return random.randint(1, 0xFFFFFFFF)
    return seed




def supports_generation(clip):
    """
    Check if the loaded CLIP model supports text generation.

    We detect generation capability by looking for a 'generate' method
    on the inner SDClipModel (the actual transformer wrapper), not on
    the outermost SD1ClipModel (which always has generate due to delegation).

    Args:
        clip: A comfy.sd.CLIP object

    Returns:
        bool: True if the model supports generation, False otherwise
    """
    if clip is None:
        return False

    try:
        cond_stage = clip.cond_stage_model  # SD1ClipModel / SDXLClipModel

        # Check for generate method on inner models
        if hasattr(cond_stage, "clip_name"):
            clip_name = cond_stage.clip_name
            inner = getattr(cond_stage, clip_name, None)
            if inner is None:
                inner = getattr(cond_stage, f"clip_{clip_name}", None)
            if inner is not None and hasattr(inner, "generate"):
                return True

        # Fallback: check if the cond_stage itself has generate
        if hasattr(cond_stage, "generate"):
            return True

        return False
    except Exception as e:
        logging.warning(f"[LLM Chat] Error checking generation support: {e}")
        return False


def format_prompt(user_text: str, system_prompt: str = "") -> str:
    """Format a prompt using the Qwen chat template.

    When system_prompt is non-empty, wraps it in the standard
    system/user/assistant template. When empty/blank, the system block
    is omitted entirely — the model receives only the user message
    with no system preamble.

    Args:
        user_text: The user's message or prompt to enhance
        system_prompt: Optional system prompt. Empty = no system prompt.

    Returns:
        str: The formatted chat prompt
    """
    if system_prompt and system_prompt.strip():
        return QWEN_CHAT_TEMPLATE.format(system=system_prompt, user=user_text)
    # No system prompt — omit system block entirely
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def format_prompt_with_history(
    user_text: str,
    history: list,
    system_prompt: str = "",
) -> str:
    """Format a prompt with full conversation history using Qwen chat template.

    Builds the prompt by concatenating system + history + current user message:

      <|im_start|>system\n{system}<|im_end|>\n        (if system_prompt provided)
      <|im_start|>user\n{msg_1}<|im_end|>\n
      <|im_start|>assistant\n{resp_1}<|im_end|>\n
      ...
      <|im_start|>user\n{current_msg}<|im_end|>\n
      <|im_start|>assistant\n

    Args:
        user_text: The current user message to respond to.
        history: List of {"role": "user"/"assistant", "message": "..."} dicts.
        system_prompt: Optional system prompt. Empty = no system block.

    Returns:
        str: The fully formatted chat prompt with history.
    """
    parts = []
    if system_prompt and system_prompt.strip():
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")

    for entry in history:
        role = entry.get("role", "user")
        msg = entry.get("message", "")
        if role == "user":
            parts.append(f"<|im_start|>user\n{msg}<|im_end|>\n")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{msg}<|im_end|>\n")

    # Current user message
    parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")

    return "".join(parts)


def _get_tokenizer(clip):
    """
    Get the inner tokenizer from the CLIP object, bypassing the outer
    tokenizer wrapper to avoid extra start/end/padding tokens.

    Searches clip.tokenizer, clip.cond_stage_model.<clip_name>.tokenizer,
    and clip.tokenizer.<clip_name> depending on the model architecture.

    Args:
        clip: A comfy.sd.CLIP object

    Returns:
        A tokenizer-like object with .tokenizer() and .decode() methods,
        or None if it cannot be found
    """
    if clip is None:
        return None

    try:
        # --- Wrapper adapter for tokenizers that lack .tokenizer attribute ---
        class _TokenizerAdapter:
            """Wraps a tokenizer to expose .tokenizer() + .decode() interface."""
            __slots__ = ('_tok',)
            def __init__(self, tok):
                self._tok = tok
            def tokenizer(self, text, truncation=True):
                return self._tok(text, truncation=truncation)
            def decode(self, token_ids, skip_special_tokens=False):
                return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

        def _ensure_tokenizer_interface(obj):
            """If obj has decode()/__call__() but no .tokenizer, wrap it."""
            if obj is None:
                return None
            if hasattr(obj, "tokenizer"):
                return obj  # already has the expected interface
            if hasattr(obj, "decode") and hasattr(obj, "__call__"):
                logging.debug(
                    f"[LLM Chat] _get_tokenizer: wrapping tokenizer "
                    f"type={type(obj).__name__} (no .tokenizer attr)"
                )
                return _TokenizerAdapter(obj)
            return None

        # === DEBUG LOGGING ===
        te_attrs = []
        if hasattr(clip, "tokenizer"):
            te = clip.tokenizer
            te_type = type(te).__name__
            te_attrs = [a for a in dir(te) if not a.startswith('_')]
            if hasattr(te, "clip_name"):
                try:
                    logging.debug(
                        f"[LLM Chat] _get_tokenizer: te type={te_type}, "
                        f"clip_name={te.clip_name}, "
                        f"attrs={te_attrs}"
                    )
                except Exception:
                    logging.debug(f"[LLM Chat] _get_tokenizer: te type={te_type}, attrs={te_attrs}")
            else:
                logging.debug(f"[LLM Chat] _get_tokenizer: te type={te_type}, no clip_name, attrs={te_attrs}")

        # === Path 1: Standard location on clip.tokenizer ===
        if hasattr(clip, "tokenizer"):
            te = clip.tokenizer

            # --- Sub-path 1a: te itself IS a usable tokenizer ---
            # Some models store the tokenizer directly as clip.tokenizer
            # (e.g., a PreTrainedTokenizerFast with decode() and __call__)
            wrapped = _ensure_tokenizer_interface(te)
            if wrapped is not None:
                return wrapped

            # --- Sub-path 1b: te has clip_name -> look up inner tokenizer ---
            if hasattr(te, "clip_name"):
                inner_name = te.clip_name
                # Try direct attribute name (e.g., te.l, te.qwen3_8b)
                inner_tokenizer = getattr(te, inner_name, None)
                wrapped = _ensure_tokenizer_interface(inner_tokenizer)
                if wrapped is not None:
                    return wrapped
                # Try with "clip_" prefix (e.g., te.clip_l)
                inner_tokenizer = getattr(te, f"clip_{inner_name}", None)
                wrapped = _ensure_tokenizer_interface(inner_tokenizer)
                if wrapped is not None:
                    return wrapped

            # --- Sub-path 1c: Scan te for ANY attribute that looks like a tokenizer ---
            # This catches models where the attribute name doesn't match clip_name
            for attr_name in te_attrs:
                if attr_name in ("clip_name", "tokenizer", "component", "inner"):
                    continue
                attr = getattr(te, attr_name, None)
                wrapped = _ensure_tokenizer_interface(attr)
                if wrapped is not None:
                    logging.debug(
                        f"[LLM Chat] _get_tokenizer: found tokenizer at "
                        f"clip.tokenizer.{attr_name} (type={type(attr).__name__})"
                    )
                    return wrapped

        # === Path 2: Legacy — look inside cond_stage_model ===
        cond_stage = clip.cond_stage_model
        if hasattr(cond_stage, "clip_name"):
            clip_name = cond_stage.clip_name
            inner = getattr(cond_stage, clip_name, None)
            if inner is None:
                inner = getattr(cond_stage, f"clip_{clip_name}", None)
            wrapped = _ensure_tokenizer_interface(inner)
            if wrapped is not None:
                return wrapped

        # === Path 3: Direct tokenizer on cond_stage ===
        if hasattr(cond_stage, "tokenizer"):
            wrapped = _ensure_tokenizer_interface(cond_stage.tokenizer)
            if wrapped is not None:
                return wrapped

        # === Path 4: Scan cond_stage attributes ===
        cs_attrs = [a for a in dir(cond_stage) if not a.startswith('_') and a not in ("clip_name", "tokenizer", "model")]
        for attr_name in cs_attrs:
            attr = getattr(cond_stage, attr_name, None)
            if attr is not None and hasattr(attr, "decode") and hasattr(attr, "__call__"):
                logging.debug(
                    f"[LLM Chat] _get_tokenizer: found tokenizer at "
                    f"clip.cond_stage_model.{attr_name} (type={type(attr).__name__})"
                )
                return _TokenizerAdapter(attr) if not hasattr(attr, "tokenizer") else attr

        logging.warning(
            f"[LLM Chat] _get_tokenizer: FAILED — no tokenizer found. "
            f"clip.tokenizer type={type(clip.tokenizer).__name__ if hasattr(clip, 'tokenizer') else 'N/A'}, "
            f"te_attrs={te_attrs}"
        )
        return None
    except Exception as e:
        logging.warning(f"[LLM Chat] Error getting tokenizer: {e}", exc_info=True)
        return None


def build_token_dict(clip, text: str) -> dict:
    """
    Tokenize text using the inner tokenizer, returning a dict suitable for generate().

    This bypasses the outer tokenizer wrapper to avoid extra padding tokens.

    Args:
        clip: A comfy.sd.CLIP object
        text: The text to tokenize

    Returns:
        dict: A token dict mapping clip_name -> list of (token_id, weight) pairs
    """
    tokenizer = _get_tokenizer(clip)
    if tokenizer is None:
        raise RuntimeError("Could not find tokenizer for CLIP model")

    token_ids = tokenizer.tokenizer(text, truncation=True)["input_ids"]

    # Determine clip_name key expected by generate()
    clip_name = None
    if hasattr(clip, "tokenizer") and hasattr(clip.tokenizer, "clip_name"):
        clip_name = clip.tokenizer.clip_name
    if clip_name is None:
        cond_stage = clip.cond_stage_model
        if hasattr(cond_stage, "clip_name"):
            clip_name = cond_stage.clip_name
        else:
            clip_name = "l"

    # Build the expected dict format: {clip_name: [[(id, weight), ...]]}
    # We wrap in an extra list to match the batch dimension expected by generate()
    token_list = [(tid, 1.0) for tid in token_ids]
    return {clip_name: [token_list]}


def decode_token_ids(clip, token_ids: list) -> str:
    """
    Decode a list of generated token IDs back to text.

    Args:
        clip: A comfy.sd.CLIP object
        token_ids: List of integer token IDs from generate_text()

    Returns:
        str: The decoded text
    """
    tokenizer = _get_tokenizer(clip)
    if tokenizer is None:
        return f"[decoding failed: {token_ids}]"
    result = tokenizer.decode(token_ids, skip_special_tokens=False)
    if not result and token_ids:
        import logging
        logging.warning(
            f"[LLM Chat] decode_token_ids returned empty string "
            f"for {len(token_ids)} tokens — first 10 IDs: {token_ids[:10]}"
        )
    return result


def clean_generated_text(text: str) -> str:
    """Clean up common artifacts from generated text.

    Removes:
    - <think>...</think> reasoning blocks (Qwen)
    - "Thinking...\n...\n\n" reasoning blocks (DeepSeek)
    - <|channel>...<channel|> reasoning blocks (Gemma 3/4)
    - Leading/trailing whitespace and partial sentences
    - Garbled Unicode characters (box drawing, symbols, zero-width spaces)
    - Repetitive punctuation runs like ".;;\n.;;\n.;;"

    Args:
        text: Raw generated text

    Returns:
        str: Cleaned text
    """
    if not text:
        return text

    # Step 1a: Strip <think>...</think> reasoning blocks (Qwen)
    text = THINK_TAG_PATTERN.sub("", text)

    # Step 1b: Strip "Thinking...\n...\n\n" reasoning blocks (DeepSeek)
    text = DEEPSEEK_THINK_PATTERN.sub("", text)

    # Step 1c: Strip <|channel>...<channel|> reasoning blocks (Gemma 3/4)
    text = GEMMA_CHANNEL_PATTERN.sub("", text)

    # Step 1d: Strip leaked USER:/ASSISTANT: chat template markers
    # Safety-net for models whose built-in Jinja template produces a
    # generic "USER:/ASSISTANT:" plain-text format. The primary fix is
    # forwarding proper stop tokens from CHAT_TEMPLATES, but this
    # catches any residual leakage.
    text = _CHAT_TEMPLATE_LEAK_PATTERN.sub("\n", text)

    # Step 2: Remove garbled Unicode
    text = GARBLED_UNICODE_PATTERN.sub("", text)

    # Step 3: Remove repetitive punctuation runs (3+ repetitions)
    text = re.sub(r'([^\w\s])\s*\n?\s*(?:\1\s*\n?\s*){2,}', '', text)

    # Step 4: Strip leading/trailing whitespace
    text = text.strip()

    # Step 5: Return empty if only punctuation/symbols remain
    if re.match(r'^[\W_]+$', text):
        return ""

    return text


def get_model_size(clip) -> int:
    """
    Detect the approximate model size in parameters.
    Used to auto-tune sampling parameters.

    Args:
        clip: A comfy.sd.CLIP object

    Returns:
        int: Approximate parameter count (0 if unknown)
    """
    try:
        cond_stage = clip.cond_stage_model
        if hasattr(cond_stage, "clip_name"):
            clip_name = cond_stage.clip_name
            inner = getattr(cond_stage, clip_name, None)
            if inner is None:
                inner = getattr(cond_stage, f"clip_{clip_name}", None)
            if inner is not None and hasattr(inner, "transformer"):
                transformer = inner.transformer
                if hasattr(transformer, "model") and hasattr(transformer.model, "config"):
                    config = transformer.model.config
                    hidden = config.hidden_size
                    layers = config.num_hidden_layers
                    return hidden * hidden * layers  # rough proxy
    except Exception:
        pass
    return 0


def get_optimal_sampling_params(clip, temperature_override: str = "auto") -> dict:
    """
    Get sampling parameters optimized for the detected model size.

    Larger models → higher temperature / more creative sampling.
    Smaller models → more conservative parameters.

    Args:
        clip: A comfy.sd.CLIP object
        temperature_override: "auto" to auto-detect, or "0.3"/"0.5"/"0.7"/"0.9"

    Returns:
        dict: {"temperature": float, "top_k": int, "top_p": float, "repetition_penalty": float}
    """
    # Defaults for medium models (4B range)
    params = {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }

    # Only auto-tune if no manual override
    if temperature_override == "auto":
        size = get_model_size(clip)
        if size > 0:
            if size < 1_000_000_000:  # 0.6B class
                params.update({
                    "temperature": 0.5,
                    "top_k": 40,
                    "top_p": 0.85,
                    "repetition_penalty": 1.25,
                })
            elif size > 5_000_000_000:  # 7B+ class
                params.update({
                    "temperature": 0.8,
                    "top_k": 60,
                    "top_p": 0.92,
                    "repetition_penalty": 1.15,
                })
            # else keep defaults (4B range)
    else:
        params["temperature"] = float(temperature_override)
        # Adjust other params based on temperature
        if params["temperature"] <= 0.3:
            params.update({"top_k": 30, "top_p": 0.8, "repetition_penalty": 1.3})
        elif params["temperature"] >= 0.9:
            params.update({"top_k": 80, "top_p": 0.95, "repetition_penalty": 1.1})
    return params


# ── GGUF Model Path Resolution ─────────────────────────────────────
# Searches ComfyUI's registered folder paths and user-added custom directories
# for GGUF model files.
# ============================================================================

# ── Persistence file for browsed directories ──────────────────────
_BROWSED_DIRS_FILE: str | None = None  # Set on first use


def _get_browsed_dirs_path() -> str:
    """Get path to browsed dirs persistence file."""
    global _BROWSED_DIRS_FILE
    if _BROWSED_DIRS_FILE is None:
        import tempfile
        _BROWSED_DIRS_FILE = os.path.join(
            tempfile.gettempdir(), "easyllm_browsed_gguf_dirs.json"
        )
    return _BROWSED_DIRS_FILE


def _load_browsed_dirs() -> set[str]:
    """Restore browsed directories from disk, collapsing nested entries.

    If a directory is a subdirectory of another loaded directory, it is
    removed since the parent already covers it via os.walk.
    """
    path = _get_browsed_dirs_path()
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                # Only keep directories that still exist
                dirs = {d for d in data if os.path.isdir(d)}
                # Remove subdirectories of other loaded dirs (cleanup)
                sorted_dirs = sorted(dirs)
                cleaned: set[str] = set()
                for d in sorted_dirs:
                    d_norm = os.path.normpath(d) + os.sep
                    is_child = any(
                        os.path.normpath(e) + os.sep != d_norm
                        and d_norm.startswith(os.path.normpath(e) + os.sep)
                        for e in cleaned
                    )
                    if not is_child:
                        cleaned.add(d)
                return cleaned
        except Exception as e:
            logging.debug(f"[LLM Chat GGUF] Failed to load browsed dirs: {e}")
    return set()


def _save_browsed_dirs(dirs: set[str]) -> None:
    """Persist browsed directories to disk."""
    path = _get_browsed_dirs_path()
    try:
        with open(path, "w") as f:
            json.dump(sorted(dirs), f)
    except Exception as e:
        logging.debug(f"[LLM Chat GGUF] Failed to save browsed dirs: {e}")


# ── Persistence file for excluded directories ───────────────────
_EXCLUDED_DIRS_FILE: str | None = None  # Set on first use


def _get_excluded_dirs_path() -> str:
    """Get path to excluded dirs persistence file."""
    global _EXCLUDED_DIRS_FILE
    if _EXCLUDED_DIRS_FILE is None:
        import tempfile
        _EXCLUDED_DIRS_FILE = os.path.join(
            tempfile.gettempdir(), "easyllm_excluded_gguf_dirs.json"
        )
    return _EXCLUDED_DIRS_FILE


def _load_excluded_dirs() -> set[str]:
    """Restore excluded directories from disk on startup."""
    path = _get_excluded_dirs_path()
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return set(data)  # keep even if dir no longer exists
        except Exception as e:
            logging.debug(f"[LLM Chat GGUF] Failed to load excluded dirs: {e}")
    return set()


def _save_excluded_dirs(dirs: set[str]) -> None:
    """Persist excluded directories to disk."""
    path = _get_excluded_dirs_path()
    try:
        with open(path, "w") as f:
            json.dump(sorted(dirs), f)
    except Exception as e:
        logging.debug(f"[LLM Chat GGUF] Failed to save excluded dirs: {e}")


# ── Model index cache (persistent GGUF metadata cache) ──────────
_MODEL_INDEX_FILE: str | None = None  # Set on first use
_model_index_lock = threading.Lock()
MAX_FILES = 500


def _get_model_index_path() -> str:
    """Get path to the persistent model index cache JSON file."""
    global _MODEL_INDEX_FILE
    if _MODEL_INDEX_FILE is None:
        import tempfile
        _MODEL_INDEX_FILE = os.path.join(
            tempfile.gettempdir(), "easyllm_model_index.json"
        )
    return _MODEL_INDEX_FILE


def _load_model_index() -> dict | None:
    """Load and validate the persistent model index cache.

    Returns:
        The cached index dict if valid, or ``None`` if missing / corrupt.
    """
    path = _get_model_index_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        # Validate required keys
        required = {"version", "generated_at", "search_dirs_hash", "files"}
        if not required.issubset(data.keys()):
            return None
        # Validate files list structure
        files = data.get("files", [])
        if not isinstance(files, list):
            return None
        for entry in files:
            if not isinstance(entry, dict):
                return None
            if "path" not in entry or "name" not in entry:
                return None
            # Validate field types to catch schema-drift corruption
            # (e.g. mtime stored as string, file_size as bool).
            # Invalid types won't crash the frontend but would show
            # wrong metadata — better to discard and rebuild.
            mtime = entry.get("mtime")
            if mtime is not None and not isinstance(mtime, (int, float)):
                return None
            file_size = entry.get("file_size")
            if file_size is not None and not isinstance(file_size, int):
                return None
            has_mmproj = entry.get("has_mmproj")
            if has_mmproj is not None and not isinstance(has_mmproj, bool):
                return None
        return data
    except Exception as e:
        logging.debug(f"[LLM Chat GGUF] Failed to load model index: {e}")
        return None


def _save_model_index(index: dict) -> None:
    """Persist the model index cache to disk (thread-safe).

    Uses an atomic write pattern: writes to a temporary file first, then
    renames to the final path.  This prevents partial/corrupt cache files
    if the process crashes mid-write.
    """
    path = _get_model_index_path()
    tmp_path = path + ".tmp"
    try:
        with _model_index_lock:
            with open(tmp_path, "w") as f:
                json.dump(index, f, indent=2)
            os.replace(tmp_path, path)
    except Exception as e:
        logging.debug(f"[LLM Chat GGUF] Failed to save model index: {e}")
        # Clean up temp file if rename failed
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _compute_search_dirs_hash() -> str:
    """Compute a SHA-256 hash of the current search directory list.

    This hash is used to detect when the set of search directories has
    changed (add, remove, exclude toggle), which triggers a hard cache
    invalidation.

    Returns:
        Hex digest string.
    """
    dirs = _get_gguf_search_dirs()
    serialized = json.dumps(sorted(dirs), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


def _skip_gguf_value(f, val_type: int) -> None:
    """Skip a single GGUF metadata value at the current file position.

    GGUF value types:
        0=uint8, 1=int8, 2=uint16, 3=int16,
        4=uint32, 5=int32, 6=float32, 7=bool,
        8=string, 9=array, 10=uint64, 11=int64,
        12=float64, 13=float16
    """
    if val_type == 8:  # string
        length = struct.unpack("<Q", f.read(8))[0]
        f.seek(length, 1)
    elif val_type == 9:  # array
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        for _ in range(arr_len):
            _skip_gguf_value(f, arr_type)
    elif val_type in (0, 1):       # uint8, int8
        f.seek(1, 1)
    elif val_type in (2, 3, 13):   # uint16, int16, float16
        f.seek(2, 1)
    elif val_type in (4, 5, 6, 7): # uint32, int32, float32, bool
        f.seek(4, 1)
    elif val_type in (10, 11, 12): # uint64, int64, float64
        f.seek(8, 1)
    else:
        f.seek(4, 1)  # unknown — guess 4 bytes


def _read_gguf_architecture_only(filepath: str) -> str | None:
    """Read only ``general.architecture`` from a GGUF file's binary header.

    Minimal parser: reads magic, version, counts, then scans metadata
    KV pairs for ``general.architecture``. Skips all tensor information.
    Much faster than full GGUFReader which parses the entire header.

    Returns:
        Lowercase architecture string or ``None`` if unreadable/invalid.
    """
    GGUF_MAGIC = b"GGUF"
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                return None

            version = struct.unpack("<I", f.read(4))[0]

            if version == 1:
                # GGUF v1: metadata_kv_count is uint32 at offset 16
                f.read(8)   # skip tensor_count int64
                kv_count = struct.unpack("<I", f.read(4))[0]
            else:
                # GGUF v2/v3: tensor_count and metadata_kv_count are uint64
                f.read(8)   # skip tensor_count int64
                kv_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                val_type = struct.unpack("<I", f.read(4))[0]

                if key == "general.architecture":
                    # String value: uint64 length + UTF-8 data
                    val_len = struct.unpack("<Q", f.read(8))[0]
                    val = f.read(val_len).decode("utf-8", errors="replace")
                    val = val.strip().lower()
                    # Reject obviously bogus values (e.g. "[20]")
                    if val and not val.startswith("[") and not val.isdigit():
                        return val
                    return None
                else:
                    _skip_gguf_value(f, val_type)
            return None
    except Exception:
        return None


@lru_cache(maxsize=512)
def _extract_prefix(filename: str) -> str:
    """Strip quantization suffix and .gguf extension to get canonical name.

    Handles patterns like:
      - model-name-Q4_K_M.gguf  -> model-name
      - model-name.Q8_0.gguf    -> model-name
      - model-name-F16.gguf     -> model-name
      - model-name.gguf         -> model-name

    Results are cached (``lru_cache``) since filenames are processed
    repeatedly across rebuilds.

    Args:
        filename: The original filename (e.g. ``"qwen2-vl-7b-Q4_K_M.gguf"``).

    Returns:
        Cleaned base name (e.g. ``"qwen2-vl-7b"``).
    """
    base = filename
    if base.lower().endswith(".gguf"):
        base = base[:-5]
    # Strip quantization suffixes: -Q4_K_M, .Q8_0, -F16, -IQ1_S, etc.
    base = _RE_STRIP_QUANT.sub("", base)
    # Also strip fp16, fp32, bf16 variants
    base = _RE_STRIP_FP.sub("", base)
    return base.strip()


def _build_model_index() -> dict:
    """Perform a full scan of all GGUF search directories and build a
    fresh model index cache.

    Walks every search directory, separates mmproj files from model
    files, reads architecture metadata from GGUF binary headers (all
    model files, up to ``MAX_FILES``), and returns the complete cache
    dict ready for serialisation.

    Reads binary headers. Result should be cached and only re-run
    when invalidated.

    Returns:
        dict with keys ``version``, ``generated_at``, ``search_dirs_hash``,
        ``files`` (list of file entries).
    """
    import os
    import re

    search_dirs = _get_gguf_search_dirs()
    if not search_dirs:
        now = __import__("datetime", fromlist=["datetime"]).datetime
        return {
            "version": 1,
            "generated_at": now.utcnow().isoformat() + "Z",
            "search_dirs_hash": _compute_search_dirs_hash(),
            "files": [],
        }

    model_files: list[str] = []
    mmproj_files: list[str] = []

    from .debug_profiler import profiler

    # Single-pass walk: separate mmproj from model files
    # NOTE: os.walk is wrapped in try/except to handle PermissionError
    # and other OS-level errors when scanning directories (e.g., disconnected
    # network drives, system-protected folders). A single unreadable directory
    # should not crash the entire cache rebuild.
    profiler.begin_sub("cache_build_total", "dir_walk")
    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        try:
            for root, _dirs, files in os.walk(base_dir):
                for fname in files:
                    if not fname.lower().endswith(".gguf"):
                        continue
                    full_path = os.path.join(root, fname)
                    if "mmproj" in fname.lower():
                        mmproj_files.append(full_path)
                    else:
                        model_files.append(full_path)
                    # Safety cap (total files combined)
                    if len(model_files) + len(mmproj_files) > MAX_FILES * 2:
                        break
                if len(model_files) + len(mmproj_files) > MAX_FILES * 2:
                    break
        except Exception as e:
            logging.warning(
                f"[LLM Chat GGUF] Skipping unreadable directory "
                f"{base_dir}: {e}"
            )
            continue
    profiler.end_sub("cache_build_total", "dir_walk")

    # ── Deduplicate model files by filename (case-insensitive) ──
    seen_names: set[str] = set()
    deduped_models: list[str] = []
    for mf in model_files:
        name_lower = os.path.basename(mf).lower()
        if name_lower not in seen_names:
            seen_names.add(name_lower)
            deduped_models.append(mf)
    model_files = deduped_models

    # Build mmproj prefix set + architecture map from reading GGUF headers
    # NOTE: Uses a minimal binary scanner (_read_gguf_architecture_only) instead
    # of GGUFReader to avoid CPU-heavy parsing of tensor info and full metadata.
    # Parallelized with ThreadPoolExecutor (same pattern as model files below).
    mmproj_prefixes: set[str] = set()
    mmproj_architectures: dict[str, list[str]] = {}

    def _read_single_mmproj(mf: str) -> tuple[str | None, str | None, str]:
        """Read prefix + architecture for a single mmproj file."""
        prefix = _extract_prefix(os.path.basename(mf))
        try:
            m_arch = _read_gguf_architecture_only(mf)
            return prefix, m_arch, mf
        except Exception:
            return prefix, None, mf

    from concurrent.futures import ThreadPoolExecutor, as_completed
    profiler.begin_sub("cache_build_total", "mmproj_arch_read")
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_path = {
            executor.submit(_read_single_mmproj, mf): mf
            for mf in mmproj_files
        }
        for future in as_completed(future_to_path):
            try:
                prefix, m_arch, mf = future.result()
                if prefix:
                    mmproj_prefixes.add(prefix)
                if m_arch:
                    mmproj_architectures.setdefault(m_arch, []).append(mf)
            except Exception as exc:
                path = future_to_path[future]
                logging.warning(
                    f"[LLM Chat GGUF] Failed to read mmproj metadata for "
                    f"{os.path.basename(path)}: {exc}"
                )
    profiler.end_sub("cache_build_total", "mmproj_arch_read")

    def _find_companion_mmproj_path(
        model_path: str, model_arch: str | None
    ) -> str | None:
        """Find companion mmproj path for a model, or ``None``.

        Only mmproj files in the **same directory** as the model are
        considered — cross-directory detection is unreliable and produces
        false positives.

        Uses the same multi-tier strategy as runtime auto-detect
        but operates entirely on pre-built metadata (no GGUF header reads):

        - Tier 1: Architecture-based match (fast, accurate) — the mmproj's
          GGUF ``general.architecture`` must match the model's architecture.
        - Tier 2: Scored filename matching — uses shared prefix length +
          substring containment (minimum prefix >= 4, min substring >= 4).

        Returns the **highest-scored** mmproj path, or ``None``.
        """
        _MIN_PREFIX = 4
        _MIN_SUBSTRING = 4

        def _score_mmproj_for_model(mp_prefix: str) -> int:
            """Score an mmproj prefix against the model stem.

            Same algorithm as ``EasyLLMGGUF._mmproj_score`` to ensure
            consistency between cache build and runtime auto-detect.
            Returns 0 if below minimum thresholds to prevent false positives.
            """
            m_stem = mp_prefix.lower().replace(
                "mmproj-", ""
            ).replace("mmproj_", "")
            if not m_stem:
                return 0
            # 1) Shared prefix length
            prefix_score = 0
            for i in range(min(len(model_stem), len(m_stem)), 0, -1):
                if model_stem.startswith(m_stem[:i]) or m_stem.startswith(model_stem[:i]):
                    prefix_score = i
                    break
            if prefix_score < _MIN_PREFIX:
                prefix_score = 0
            # 2) Substring containment bonus
            substring_bonus = 0
            if m_stem and (m_stem in model_stem or model_stem in m_stem):
                match_len = min(len(model_stem), len(m_stem))
                if match_len >= _MIN_SUBSTRING:
                    substring_bonus = match_len  # 1x multiplier (was 2x max)
            return prefix_score + substring_bonus

        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        model_stem = os.path.splitext(model_name)[0].lower()

        # Tier 1: Architecture-based match (same-directory only)
        if model_arch and model_arch in mmproj_architectures:
            candidates = mmproj_architectures[model_arch]
            # Filter to same-directory only
            for c in candidates:
                if os.path.dirname(c) == model_dir:
                    return c

        # Tier 2: Scored filename matching (same-directory only)
        # Filter mmproj files to same directory as the model
        local_mmproj_files = [
            mf for mf in mmproj_files
            if os.path.dirname(mf) == model_dir
        ]
        if not local_mmproj_files:
            return None
        best_score = 0
        best_path = None
        for mf in local_mmproj_files:
            mp = _extract_prefix(os.path.basename(mf))
            score = _score_mmproj_for_model(mp)
            if score > best_score:
                best_score = score
                best_path = mf
        # Only return if best score > 0 — prevents false positives
        # from zero-score matches (e.g. no shared prefix, no substring)
        if best_score > 0:
            return best_path
        return None

    # Import prefix fallback from streaming module
    def _get_arch_from_prefix_or_metadata(filepath: str, fname: str) -> tuple[str | None, str]:
        """Try GGUF metadata first, then prefix fallback.

        Returns:
            Tuple of (architecture_string, source_label).
            source_label is ``"metadata"`` or ``"prefix"``.
        """
        try:
            from .chat_node import EasyLLMGGUF
            arch = EasyLLMGGUF._get_gguf_architecture(filepath)
            if arch:
                return arch, "metadata"
        except Exception:
            pass
        # Fallback: import from streaming module
        from .streaming import _get_architecture_from_prefix
        arch = _get_architecture_from_prefix(fname)
        return arch, "prefix"

    # ── Parallel architecture reads ─────────────────────────────────
    # Reading GGUF headers sequentially is I/O-bound.
    # ThreadPoolExecutor with 8 workers provides near-linear speedup.
    def _read_single_model(model_path: str) -> dict:
        """Read architecture + file metadata for a single model file."""
        fname = os.path.basename(model_path)
        arch, arch_source = _get_arch_from_prefix_or_metadata(
            model_path, fname
        )
        # Find companion mmproj path from cache (no extra GGUF reads)
        mmproj_path = _find_companion_mmproj_path(model_path, arch)
        has_mmproj = mmproj_path is not None
        try:
            file_size = os.path.getsize(model_path)
            mtime = os.path.getmtime(model_path)
        except OSError:
            file_size = 0
            mtime = 0.0
        entry = {
            "path": model_path,
            "name": fname,
            "architecture": arch or "",
            "architecture_source": arch_source,
            "has_mmproj": has_mmproj,
            "file_size": file_size,
            "mtime": mtime,
        }
        if mmproj_path:
            entry["mmproj_path"] = mmproj_path
        return entry

    from concurrent.futures import ThreadPoolExecutor, as_completed
    model_subset = model_files[:MAX_FILES]
    profiler.begin_sub("cache_build_total", "model_arch_read")
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_path = {
            executor.submit(_read_single_model, mf): mf
            for mf in model_subset
        }
        result_files = []
        for future in as_completed(future_to_path):
            try:
                entry = future.result()
                result_files.append(entry)
            except Exception as exc:
                # Should not happen — _read_single_model catches all exceptions
                path = future_to_path[future]
                logging.warning(
                    f"[LLM Chat GGUF] Failed to read model metadata for "
                    f"{os.path.basename(path)}: {exc}"
                )
    profiler.end_sub("cache_build_total", "model_arch_read")

    # Sort alphabetically by name for consistent ordering
    result_files.sort(key=lambda f: f["name"].lower())

    from datetime import datetime
    now = datetime.utcnow()

    return {
        "version": 1,
        "generated_at": now.isoformat() + "Z",
        "search_dirs_hash": _compute_search_dirs_hash(),
        "files": result_files,
    }


# ── Cache Update Helper ─────────────────────────────────────────────


def _mmproj_exists_for(model_path: str, architecture: str | None) -> bool:
    """Check if a companion mmproj file exists for a given model.

    First checks the model index cache (fast, zero I/O). Falls back to
    directory scan + GGUF header read only if cache doesn't have the answer.

    Args:
        model_path: Absolute path to the .gguf model file.
        architecture: The model's architecture string, or ``None``.

    Returns:
        ``True`` if a likely companion mmproj was found.
    """
    import os

    # ── Fast path: check model index cache first ──────────────────────
    try:
        index = _load_model_index()
        if index is not None:
            for entry in index.get("files", []):
                if entry.get("path") == model_path:
                    # Cache has mmproj_path field → immediate answer
                    if entry.get("mmproj_path"):
                        return True
                    # Cache has has_mmproj field → immediate answer
                    if entry.get("has_mmproj") is False:
                        return False
                    # Fall through to disk scan (shouldn't happen for
                    # cached entries, but handle gracefully)
                    break
    except Exception:
        pass

    # ── Slow path: scan directory + read GGUF headers ────────────────
    model_dir = os.path.dirname(model_path)
    model_stem = os.path.splitext(os.path.basename(model_path))[0].lower()
    try:
        for fname in os.listdir(model_dir):
            if "mmproj" in fname.lower() and fname.lower().endswith(".gguf"):
                mmproj_path = os.path.join(model_dir, fname)
                # Tier 1: Architecture match (100% accurate)
                if architecture:
                    try:
                        from .chat_node import EasyLLMGGUF
                        mmproj_arch = EasyLLMGGUF._get_gguf_architecture(
                            mmproj_path
                        )
                        if mmproj_arch == architecture:
                            return True
                    except Exception:
                        pass
                # Tier 2: Filename substring match (fallback)
                # Minimum 4 chars to prevent false positives from
                # accidental short matches (e.g. "t5" in "t5-danbooru").
                mmproj_stem = (
                    os.path.splitext(fname)[0]
                    .lower()
                    .replace("mmproj-", "")
                    .replace("mmproj_", "")
                )
                if mmproj_stem and (
                    mmproj_stem in model_stem or model_stem in mmproj_stem
                ) and min(len(model_stem), len(mmproj_stem)) >= 4:
                    return True
    except Exception:
        pass
    return False


def _update_cached_architecture(
    model_path: str,
    architecture: str,
    context_length: str = "",
    description: str = "",
    model_name: str = "",
) -> bool:
    """Update or insert a model entry's architecture in the persistent cache.

    Called after a model is successfully loaded in ``chat_node.py``.
    If the model path already exists in the cache, its architecture,
    context length, description, and name are updated to the now-known
    values.  If it doesn't exist, a new entry is inserted so the cache
    stays consistent with what's actually loadable.

    This is thread-safe (uses ``_model_index_lock``) and non-critical —
    failures are silently logged and return ``False``.

    Args:
        model_path: Absolute path to the .gguf model file.
        architecture: The architecture string (e.g. ``"qwen2vl"``).
        context_length: The model's context window size (string).
        description: The model's description string.
        model_name: The model's name (e.g. ``"Qwen2.5 7B"``).

    Returns:
        ``True`` if the cache was updated, ``False`` on any failure.
    """
    if not model_path or not architecture:
        return False
    try:
        index = _load_model_index()
        if index is None:
            return False  # no cache yet; rebuild will pick it up

        name = os.path.basename(model_path)
        found = False
        for entry in index.get("files", []):
            if entry.get("path") == model_path:
                entry["architecture"] = architecture
                entry["architecture_source"] = "loaded"
                if context_length:
                    entry["context_length"] = context_length
                if description:
                    entry["description"] = description
                if model_name:
                    entry["model_name"] = model_name
                found = True
                break

        if not found:
            # Model not in cache — add a new entry
            try:
                file_size = os.path.getsize(model_path)
                mtime = os.path.getmtime(model_path)
            except OSError:
                file_size = 0
                mtime = 0.0
            # _mmproj_exists_for now checks cache first (fast path)
            has_mmproj = _mmproj_exists_for(model_path, architecture)
            new_entry = {
                "path": model_path,
                "name": name,
                "architecture": architecture,
                "architecture_source": "loaded",
                "has_mmproj": has_mmproj,
                "file_size": file_size,
                "mtime": mtime,
            }
            # If mmproj known, store its path for instant lookup later
            if has_mmproj:
                # Try to find mmproj_path from existing cache entries
                for existing in index.get("files", []):
                    if existing.get("mmproj_path") and existing.get("architecture") == architecture:
                        new_entry["mmproj_path"] = existing["mmproj_path"]
                        break
            index["files"].append(new_entry)
            # Keep sorted for consistent display
            index["files"].sort(
                key=lambda f: f.get("name", "").lower()
            )

        _save_model_index(index)
        return True
    except Exception as e:
        logging.debug(
            f"[LLM Chat GGUF] Failed to update cached architecture for "
            f"{os.path.basename(model_path)}: {e}"
        )
        return False


# Runtime set: directories excluded from GGUF search.
# Populated by the Model Browser UI "toggle off" action.
# Persisted to disk for survival across restarts.
_excluded_gguf_dirs: set[str] = _load_excluded_dirs()


def exclude_gguf_search_dir(directory: str) -> bool:
    """Add a directory to the exclusion set.

    Excluded directories are skipped during GGUF file search.
    Returns True if the directory was newly excluded.
    """
    if not directory:
        return False
    abs_dir = os.path.abspath(directory)
    if abs_dir not in _excluded_gguf_dirs:
        _excluded_gguf_dirs.add(abs_dir)
        _save_excluded_dirs(_excluded_gguf_dirs)
        logging.info(
            f"[LLM Chat GGUF] Excluded search directory: {abs_dir}"
        )
        return True
    return False


def unexclude_gguf_search_dir(directory: str) -> bool:
    """Remove a directory from the exclusion set.

    Returns True if the directory was found and unexcluded.
    """
    if not directory:
        return False
    abs_dir = os.path.abspath(directory)
    if abs_dir in _excluded_gguf_dirs:
        _excluded_gguf_dirs.discard(abs_dir)
        _save_excluded_dirs(_excluded_gguf_dirs)
        logging.info(
            f"[LLM Chat GGUF] Removed exclusion for directory: {abs_dir}"
        )
        return True
    return False


# Runtime set: directories learned from successful browse operations.
# Populated by validate_gguf_path / validate_mmproj_path APIs.
# Persisted to disk for survival across restarts.
_browsed_gguf_dirs: set[str] = _load_browsed_dirs()


def register_browsed_gguf_dir(directory: str) -> None:
    """Register a directory learned from a successful browse.

    Accepts both model directory paths and mmproj directory paths.
    Persists to disk so directories survive ComfyUI restarts.

    When a new directory is registered, any existing entries that are
    subdirectories of it are removed (reverse cleanup). Then, if the
    new directory is a subdirectory of an already-registered parent,
    registration is skipped — the parent already covers it via os.walk.
    """
    if not directory or not os.path.isdir(directory):
        return
    abs_dir = os.path.normpath(os.path.abspath(directory))

    # ── Remove existing entries that are subdirectories of this newly
    #    registered directory (reverse cleanup).
    for existing in list(_browsed_gguf_dirs):
        existing_norm = os.path.normpath(existing) + os.sep
        if existing_norm.startswith(abs_dir + os.sep):
            _browsed_gguf_dirs.discard(existing)

    # Skip if this dir is already a subdirectory of a known custom dir
    for existing in list(_browsed_gguf_dirs):
        existing_norm = os.path.normpath(existing) + os.sep
        if (abs_dir + os.sep).startswith(existing_norm):
            logging.debug(
                f"[LLM Chat GGUF] Skip registering {abs_dir} — "
                f"already covered by existing dir: {existing}"
            )
            return

    _browsed_gguf_dirs.add(abs_dir)
    _save_browsed_dirs(_browsed_gguf_dirs)
    logging.info(
        f"[LLM Chat GGUF] Learned GGUF directory from browse: "
        f"{abs_dir}"
    )


def unregister_browsed_gguf_dir(directory: str) -> bool:
    """Remove a directory from the browsed list.

    Returns True if the directory was found and removed.
    """
    if directory:
        abs_dir = os.path.abspath(directory)
        if abs_dir in _browsed_gguf_dirs:
            _browsed_gguf_dirs.discard(abs_dir)
            _save_browsed_dirs(_browsed_gguf_dirs)
            logging.info(
                f"[LLM Chat GGUF] Removed browsed GGUF directory: {abs_dir}"
            )
            return True
    return False


def _get_external_gguf_dirs() -> list[str]:
    """Return user-added (browsed) directories for GGUF model search.

    Users add directories manually via the Custom Directories section
    in the Model Browser popup. No OS-specific auto-detection is performed
    (no Ollama, LM Studio registry/PATH scan, etc.) — users explicitly
    add whatever paths they need.

    Returns:
        list[str]: Absolute paths of user-added directories to search.
    """
    return sorted(_browsed_gguf_dirs)


def _collapse_nested_dirs(dirs: list[str]) -> list[str]:
    """Remove any directory that is a subdirectory of another in the list.

    If a parent directory is already present (e.g. ``ComfyUI/models``),
    its children (e.g. ``ComfyUI/models/text_encoders``) are removed
    since ``os.walk`` on the parent already covers subdirectories.

    Args:
        dirs: List of absolute directory paths.

    Returns:
        Collapsed list with nested children removed.
    """
    if not dirs:
        return []

    sorted_dirs = sorted(dirs)  # parents sort before children
    collapsed: list[str] = []
    for d in sorted_dirs:
        d_norm = os.path.normpath(d) + os.sep
        is_child = any(
            os.path.normpath(a) + os.sep != d_norm
            and d_norm.startswith(os.path.normpath(a) + os.sep)
            for a in collapsed
        )
        if not is_child:
            collapsed.append(d)
    return collapsed


def _get_gguf_search_dirs() -> list[str]:
    """Collect all directories to search for GGUF model files.

    Uses ComfyUI's folder_paths module when available; falls back to
    common default locations. Also searches user-added custom directories.

    Returns:
        list[str]: Absolute paths of directories to search, with nested
        children collapsed (parents already cover subdirectories).
    """
    import os
    dirs: list[str] = []
    folder_paths_available = False

    try:
        import folder_paths
        folder_paths_available = True

        # Only the main models directory — individual folder types
        # (text_encoders, llm, gguf, clip) are added below.
        if hasattr(folder_paths, "models_dir") and folder_paths.models_dir:
            dirs.append(os.path.abspath(folder_paths.models_dir))

        # Only search folder types likely to contain GGUF models or mmproj files.
        # This reduces the search surface from ~60 dirs to ~5-10, making error
        # messages much cleaner when a model isn't found.
        GGUF_RELEVANT_FOLDERS = {
            "text_encoders", "llm", "LLM", "gguf", "GGUF", "clip",
        }
        if hasattr(folder_paths, "folder_names_and_paths"):
            for folder_name, (paths_list, _) in folder_paths.folder_names_and_paths.items():
                if folder_name not in GGUF_RELEVANT_FOLDERS:
                    continue  # skip irrelevant types (controlnet, vae, embeddings, etc.)
                for base_dir in paths_list:
                    abs_dir = os.path.abspath(base_dir)
                    if abs_dir not in dirs:
                        dirs.append(abs_dir)
    except ImportError:
        logging.warning(
            "[LLM Chat] folder_paths module not available — "
            "cannot search ComfyUI model directories. "
            "Will fall back to common GGUF locations."
        )
    except Exception as e:
        logging.warning(
            f"[LLM Chat] folder_paths lookup failed: {e}"
        )

    if folder_paths_available:
        logging.info(
            f"[LLM Chat GGUF] Searched {len(dirs)} ComfyUI model directories "
            f"via folder_paths"
        )

    # Add user-added custom directories (lower priority)
    for ext_dir in _get_external_gguf_dirs():
        if ext_dir not in dirs:
            dirs.append(ext_dir)
            logging.info(
                f"[LLM Chat GGUF] Added custom GGUF directory: {ext_dir}"
            )

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    # ── Filter out excluded directories (user-toggled off in UI) ──
    if _excluded_gguf_dirs:
        filtered: list[str] = []
        for d in unique:
            # Exclude if this exact directory is in the exclusion set
            if d in _excluded_gguf_dirs:
                logging.debug(
                    f"[LLM Chat GGUF] Skipping excluded directory: {d}"
                )
                continue
            filtered.append(d)
        unique = filtered

    # ── Collapse nested directories (parents cover children) ──
    unique = _collapse_nested_dirs(unique)

    # Log all search directories at startup for debugging
    logging.info(
        f"[LLM Chat GGUF] GGUF search directories ({len(unique)}):\n"
        + "\n".join(f"  • {d}" for d in unique)
    )

    return unique


def _get_searched_paths(model_path: str) -> list[str]:
    """Build the list of all candidate paths for a given model_path.

    Args:
        model_path: The user-provided model path string.

    Returns:
        list[str]: All absolute paths that were or would be checked.
    """
    import os
    searched: list[str] = []
    filename = os.path.basename(model_path)
    dirname = os.path.dirname(model_path)

    # Tier 1: As-is
    searched.append(os.path.abspath(model_path))

    search_dirs = _get_gguf_search_dirs()

    # Tiers 2-3: Try relative to each search directory
    for base_dir in search_dirs:
        # Full path relative to base dir
        candidate = os.path.join(base_dir, model_path)
        if candidate not in searched:
            searched.append(candidate)

        # Just the filename
        candidate_fn = os.path.join(base_dir, filename)
        if candidate_fn not in searched:
            searched.append(candidate_fn)

        # If dirname was provided (e.g. "GGUF"), try base_dir/dirname/filename
        if dirname:
            candidate_sub = os.path.join(base_dir, dirname, filename)
            if candidate_sub not in searched:
                searched.append(candidate_sub)
            # Also try lowercase dirname
            if dirname != dirname.lower():
                candidate_lower = os.path.join(base_dir, dirname.lower(), filename)
                if candidate_lower not in searched:
                    searched.append(candidate_lower)

    return searched


def resolve_gguf_path(model_path: str) -> str | None:
    """Resolve a GGUF model file path using multi-tier search.

    Searches ComfyUI's registered folder paths and user-added custom directories.

    Resolution tiers:
        1. As-is (absolute or CWD-relative)
        2. Relative to each registered directory
        3. Filename-only in each registered directory
        4. With subdirectory name under each base
        5. Recursive walk for bare filenames (up to 4 levels deep)

    Args:
        model_path: The user-provided model path (may be relative or absolute).

    Returns:
        The resolved absolute path if found, or None if not found.
    """
    import os
    if not model_path or not model_path.strip():
        logging.warning("[LLM Chat] resolve_gguf_path called with empty path")
        return None

    model_path = model_path.strip()

    # Tier 1: Try as-is (handles absolute paths and CWD-relative)
    if os.path.isfile(model_path):
        resolved = os.path.abspath(model_path)
        logging.info(
            f"[LLM Chat GGUF] Model found (tier 1, as-is): {resolved}"
        )
        return resolved

    filename = os.path.basename(model_path)
    dirname = os.path.dirname(model_path)

    # Collect search directories
    search_dirs = _get_gguf_search_dirs()

    if not search_dirs:
        logging.debug(
            "[LLM Chat GGUF] No search directories available from folder_paths"
        )
        return None

    # Tiers 2-4: Search through registered directories
    for base_dir in search_dirs:
        # Tier 2: Full relative path under base dir
        candidate = os.path.join(base_dir, model_path)
        if os.path.isfile(candidate):
            resolved = os.path.abspath(candidate)
            logging.info(
                f"[LLM Chat GGUF] Model found (tier 2, under {os.path.basename(base_dir)}): "
                f"{resolved}"
            )
            return resolved

        # Tier 3: Just the filename under base dir
        candidate_fn = os.path.join(base_dir, filename)
        if os.path.isfile(candidate_fn):
            resolved = os.path.abspath(candidate_fn)
            logging.info(
                f"[LLM Chat GGUF] Model found (tier 3, filename in "
                f"{os.path.basename(base_dir)}): {resolved}"
            )
            return resolved

        # Tier 4: If dirname was provided, try base_dir/dirname/filename
        if dirname:
            candidate_sub = os.path.join(base_dir, dirname, filename)
            if os.path.isfile(candidate_sub):
                resolved = os.path.abspath(candidate_sub)
                logging.info(
                    f"[LLM Chat GGUF] Model found (tier 4, subdir "
                    f"'{dirname}' in {os.path.basename(base_dir)}): {resolved}"
                )
                return resolved

            # Also try lowercase dirname (for case-insensitive FS compatibility)
            if dirname != dirname.lower():
                candidate_lower = os.path.join(base_dir, dirname.lower(), filename)
                if os.path.isfile(candidate_lower):
                    resolved = os.path.abspath(candidate_lower)
                    logging.info(
                        f"[LLM Chat GGUF] Model found (tier 4, subdir "
                        f"'{dirname.lower()}' in {os.path.basename(base_dir)}): {resolved}"
                    )
                    return resolved

    # ── Tier 5: Recursive subdirectory search (last resort) ──────────────
    # Only used for bare filenames (no dirname component). Walks up to 4
    # levels deep into each search directory to find the file. This handles
    # deeply nested model structures common to LM Studio, Ollama, and
    # HuggingFace downloads.
    if not dirname:
        for base_dir in search_dirs:
            try:
                found = _find_gguf_recursive(base_dir, filename, max_depth=4)
                if found:
                    resolved = os.path.abspath(found)
                    logging.info(
                        f"[LLM Chat GGUF] Model found (tier 5, recursive in "
                        f"{os.path.basename(base_dir)}): {resolved}"
                    )
                    return resolved
            except Exception:
                continue

    logging.warning(
        f"[LLM Chat GGUF] Model not found: {model_path}\n"
        f"Searched directories ({len(search_dirs)}):\n"
        + "\n".join(f"  • {d}" for d in search_dirs)
    )
    return None


def _find_gguf_recursive(
    directory: str, filename: str, max_depth: int = 4
) -> str | None:
    """Recursively search for a filename under a directory, up to max_depth levels.

    Used by resolve_gguf_path() Tier 5 for nested subdirectory structures.

    Args:
        directory: Root directory to start searching from.
        filename: The filename to search for (e.g., 'tame-tiger.Q6_K.gguf').
        max_depth: Maximum directory depth to recurse into (default: 4).
                   Depth 0 = only the directory itself, no recursion.

    Returns:
        The full path to the file if found, or None if not found.
    """
    import os

    if max_depth < 0 or not os.path.isdir(directory):
        return None

    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.lower() == filename.lower():
                return entry.path
            elif entry.is_dir() and max_depth > 0:
                try:
                    result = _find_gguf_recursive(
                        entry.path, filename, max_depth - 1
                    )
                    if result is not None:
                        return result
                except PermissionError:
                    continue
                except OSError:
                    continue
    except PermissionError:
        pass
    except OSError:
        pass

    return None


# ── Image Conversion (Vision-Language / Multimodal Support) ──────────────


def tensor_to_base64_png(image_tensor: torch.Tensor, max_size: int | None = 1024) -> str:
    """Convert a ComfyUI IMAGE tensor to a base64-encoded PNG data URI.

    Used by the multimodal vision-language generation path in EasyLLMGGUF.
    Converts float32 [0,1] tensor to PNG, optionally resizing the longest edge
    to reduce token usage in the context window.

    Args:
        image_tensor: Tensor of shape (1, H, W, 3) in float32 [0, 1].
        max_size: Optional max dimension (longest edge). None = no resize.
                  Default 1024.

    Returns:
        str: Base64-encoded PNG data URI (data:image/png;base64,...).
    """
    # Remove batch dimension if present (4D -> 3D)
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]

    # Convert float32 [0,1] tensor to uint8 PIL Image
    img = Image.fromarray(
        (image_tensor.cpu().numpy() * 255).astype('uint8')
    )

    # Optional resize to limit token usage in the vision encoder
    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

    # Save to PNG bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode as base64 and return as data URI
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def get_comfyui_input_dir() -> str:
    """Get the absolute path to ComfyUI's input/ directory.

    Uses folder_paths if available (normal ComfyUI environment),
    falls back to a relative path from custom_nodes.
    """
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except ImportError:
        # Fallback: relative to custom_nodes/llm-chat
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "input")
        )


def load_image_to_base64(filename: str, max_size: int | None = 1024) -> str:
    """Load an image from ComfyUI's input/ directory and convert to base64 PNG data URI.

    Used by the chat image upload path in EasyLLMGGUF.generate().
    Handles file-based images (uploaded via /upload/image endpoint)
    as opposed to tensor_to_base64_png which handles IMAGE tensors.

    Args:
        filename: The image filename (e.g., "my_photo.png") as stored in ComfyUI's input/.
        max_size: Optional max dimension for resize (default 1024).
                  None = no resize.

    Returns:
        str: Base64-encoded PNG data URI (data:image/png;base64,...).

    Raises:
        FileNotFoundError: If the file does not exist in ComfyUI's input/ directory.
    """
    input_dir = get_comfyui_input_dir()
    filepath = os.path.join(input_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Uploaded image not found: {filepath}"
        )

    img = Image.open(filepath)

    # Resize if needed to limit token usage
    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

    # Convert to PNG bytes then base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

