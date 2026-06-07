"""
Module-level GenerationState singleton for LLM Chat.

Consolidates generation-related state previously scattered across
``streaming.py`` and ``cuda_optimizations.py`` into a single
``GenerationState`` dataclass with a module-level singleton instance.

Usage:

    from .generation_state import get_state

    state = get_state()
    state.streaming_callback = my_callback
    if state.progress_enabled:
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationState:
    """Centralised namespace for all module-level generation state.

    Attributes
    ----------
    popup_active_nodes : set[str]
        Node IDs with active popups.
    abort_flags : dict[str, bool]
        Per-node abort signals.
    streaming_callback : Callable[[int], None] | None
        Per-token streaming callback fired from GPU-native generation loop.
    progress_enabled : bool
        Whether progress WebSocket events are emitted.
    progress_node_id : str | None
        Node ID for progress event routing.
    """

    # ── Popup tracking ──────────────────────────────────
    popup_active_nodes: set = field(default_factory=set)
    abort_flags: dict = field(default_factory=dict)

    # ── Streaming callback ─────────────────────────────
    streaming_callback: Optional[Callable[[int], None]] = None

    # ── Progress event state ────────────────────────────
    progress_enabled: bool = False
    progress_node_id: Optional[str] = None

    # ── Convenience helpers ────────────────────────────────────────

    def is_popup_mode(self, unique_id: str) -> bool:
        """Check if the given node has an active popup."""
        return unique_id in self.popup_active_nodes

    def set_popup_active(self, unique_id: str) -> None:
        """Mark a node as having an open popup."""
        self.popup_active_nodes.add(unique_id)
        # Clear stale abort flag from a previous interrupted run
        self.abort_flags.pop(unique_id, None)

    def clear_popup_active(self, unique_id: str) -> None:
        """Mark a node's popup as closed."""
        self.popup_active_nodes.discard(unique_id)
        self.abort_flags.pop(unique_id, None)

    # -- Abort flags --

    def set_abort(self, unique_id: str) -> None:
        """Signal abort for a node's generation."""
        self.abort_flags[unique_id] = True

    def is_aborted(self, unique_id: str) -> bool:
        """Atomically check and consume the abort flag for a node.

        Returns True if an abort was requested (consumes the flag so
        subsequent generations are not affected).
        """
        if self.abort_flags.get(unique_id, False):
            self.abort_flags.pop(unique_id, None)
            return True
        return False

    def cleanup_abort(self, unique_id: str) -> None:
        """Remove the abort flag for a node without reading it."""
        self.abort_flags.pop(unique_id, None)

    # -- Streaming callback --

    def set_streaming_callback(self, cb: Optional[Callable[[int], None]]) -> None:
        """Register (or clear) the per-token streaming callback."""
        self.streaming_callback = cb

    def clear_streaming_callback(self) -> None:
        """Clear the streaming callback."""
        self.streaming_callback = None

    def has_streaming_callback(self) -> bool:
        """Check if a streaming callback is currently registered."""
        return self.streaming_callback is not None

    # -- Progress events --

    def enable_progress(self, node_id: str) -> None:
        """Enable progress event emission for a specific node."""
        self.progress_enabled = True
        self.progress_node_id = node_id

    def disable_progress(self) -> None:
        """Disable progress event emission."""
        self.progress_enabled = False
        self.progress_node_id = None


# ── Module-level singleton ──────────────────────────────────────────

_STATE = GenerationState()


def get_state() -> GenerationState:
    """Return the module-level GenerationState singleton."""
    return _STATE
