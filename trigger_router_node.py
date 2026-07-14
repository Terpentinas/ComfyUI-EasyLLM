"""
🎛️ EasyLLM Trigger Router Node
=================================

Parses structured JSON from the LLM node's ``trigger_prompt`` output socket
and decomposes it into individual STRING sockets for downstream pipeline nodes.

Input:
    trigger_prompt (STRING): JSON string from LLM node's trigger_prompt socket.

Outputs:
    prompt          (STRING): Extracted prompt text for image generation.
                               Empty string if action is ``just_chat``.
    negative_prompt (STRING): Extracted negative prompt (if present in JSON).
                               Empty string otherwise.
    session_uuid    (STRING): Session UUID for image capture reconciliation.
                               Empty string if not present in JSON.

Architecture:
    This is the second step in the Trigger Prompt pipeline:
        LLM → trigger_prompt JSON → Trigger Router → STRING sockets → downstream nodes

    The LLM generates structured JSON (e.g. ``{"action": "generate_image",
    "prompt": "cat wearing a hat"}``), which this node parses into
    usable string sockets for CLIP Text Encode, KSampler, etc.

    Design principle: pure STRING processor — no IMAGE tensors, no side effects.
    See ``plans/trigger-prompt-router-architecture.md`` for full architecture.
"""

import json
import logging

logger = logging.getLogger(__name__)


class LLM_TriggerRouter:
    """
    🎛️ EasyLLM Trigger Router — Parse trigger_prompt JSON into usable sockets.

    Takes the JSON string from an EasyLLM node's ``trigger_prompt`` output
    and decomposes it into ``prompt``, ``negative_prompt``, and ``session_uuid``
    string outputs for downstream pipeline nodes.

    Pure string processor — no image tensors, no side effects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_prompt": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": (
                        "JSON string from EasyLLM node's trigger_prompt output socket. "
                        "Contains structured instructions for image generation."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "session_uuid")
    OUTPUT_TOOLTIPS = (
        "Extracted prompt text for image generation. Empty if action is just_chat.",
        "Extracted negative prompt (if present in JSON). Empty string otherwise.",
        "Session UUID for image capture reconciliation. Empty string if not present.",
    )
    FUNCTION = "route"
    CATEGORY = "EasyLLM/🎛️ Advanced"
    DESCRIPTION = (
        "Parses the trigger_prompt JSON from an EasyLLM node into individual "
        "STRING sockets (prompt, negative_prompt, session_uuid) for CLIP Text Encode, "
        "KSampler, and other downstream image generation nodes.\n\n"
        "Connect: EasyLLM trigger_prompt → this node's trigger_prompt input.\n"
        "Output: prompt → CLIP Text Encode (positive), negative_prompt → CLIP (negative)."
    )

    def route(self, trigger_prompt: str):
        """
        Parse the trigger_prompt JSON and return decomposed string outputs.

        Args:
            trigger_prompt: JSON string from LLM node's trigger_prompt socket.
                            Expected format:
                            ``{"action": "generate_image", "prompt": "...", "negative": "...",
                              "session_uuid": "..."}``

        Returns:
            tuple: (prompt, negative_prompt, session_uuid)

        Note:
            The ``action`` field from the JSON is parsed internally to gate
            whether ``prompt`` is extracted (only for ``generate_image`` and
            ``edit_image``), but is no longer exposed as an output socket.
        """
        prompt = ""
        negative_prompt = ""
        session_uuid = ""

        if not trigger_prompt or not trigger_prompt.strip():
            logger.debug("[LLM_TriggerRouter] Empty trigger_prompt input")
            return (prompt, negative_prompt, session_uuid)

        try:
            parsed = json.loads(trigger_prompt)
        except json.JSONDecodeError as e:
            logger.warning(
                f"[LLM_TriggerRouter] Failed to parse trigger_prompt JSON: {e}"
            )
            return (prompt, negative_prompt, session_uuid)

        # Extract session_uuid (present in any trigger action including just_chat)
        session_uuid = parsed.get("session_uuid", "")

        # Extract action (internal-only — no longer exposed as a socket)
        parsed_action = parsed.get("action", "")
        if parsed_action not in ("generate_image", "edit_image", "just_chat"):
            logger.warning(
                f"[LLM_TriggerRouter] Unknown action '{parsed_action}' — "
                f"expected generate_image, edit_image, or just_chat"
            )
            return (prompt, negative_prompt, session_uuid)

        action = parsed_action

        # Extract prompt (for generate_image and edit_image)
        if action in ("generate_image", "edit_image"):
            prompt = parsed.get("prompt", "")
            if not prompt:
                logger.warning(
                    f"[LLM_TriggerRouter] Action is '{action}' but prompt is empty"
                )

        # Extract negative prompt (optional field)
        negative_prompt = parsed.get("negative", "")

        logger.debug(
            f"[LLM_TriggerRouter] Parsed: action={action}, "
            f"prompt_len={len(prompt)}, negative_len={len(negative_prompt)}, "
            f"session_uuid={'set' if session_uuid else 'empty'}"
        )

        return (prompt, negative_prompt, session_uuid)


# ── Node Registration ─────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LLM_TriggerRouter": LLM_TriggerRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_TriggerRouter": "🎛️ EasyLLM Trigger Router",
}
