"""
Centralized version configuration for llama-cpp-python management.

Both install.py and llama_cpp_backend.py import from this single source,
so version changes propagate to both files automatically.

Usage:
    # In install.py (standalone script):
    from _version_config import SAFE_BASELINE_VERSION, LATEST_VERSION, MIN_VISION_VERSION

    # In llama_cpp_backend.py (package module):
    from ._version_config import SAFE_BASELINE_VERSION, LATEST_VERSION, MIN_VISION_VERSION
"""

# ── SAFE_BASELINE_VERSION ─────────────────────────────────────────────────
# Default version installed for ALL users. This is the thoroughly tested,
# stable release that works on all supported hardware, including CPUs
# that lack AVX2 (Ivy Bridge, Sandy Bridge, AMD FX/Phenom).
#
# The CPU-only wheel is compiled without AVX2 requirement, so it works
# on every x86-64 CPU.
SAFE_BASELINE_VERSION = "0.3.26"

# ── LATEST_VERSION ────────────────────────────────────────────────────────
# Newest upstream release. Users can opt-in to this version via the
# --try-latest flag or LLM_CHAT_TRY_LATEST env var. This is the
# future-proofing mechanism: when a newer release arrives, update this
# constant first so adventurous users can test it. Only promote it to
# SAFE_BASELINE_VERSION after thorough validation.
#
# Currently matches SAFE_BASELINE_VERSION since v0.3.26 is both the
# latest and the validated safe baseline. These will diverge when a
# newer version is released but not yet fully validated.
LATEST_VERSION = "0.3.26"

# ── MIN_VISION_VERSION ────────────────────────────────────────────────────
# Minimum llama-cpp-python version for vision/image_url support in
# create_chat_completion(). Vision works on v0.3.23 and newer.
# The backend uses a try/except fallback (see generate_chat methods) so
# even versions below this will attempt vision first and degrade gracefully
# to text-only on failure.
MIN_VISION_VERSION = "0.3.23"
