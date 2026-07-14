#!/usr/bin/env python3
"""
EasyLLM for ComfyUI -- llama.cpp Installation Helper
====================================================

Auto-detects OS, GPU/CUDA version, and ComfyUI Python environment,
then installs the matching pre-built llama-cpp-python wheel.

Pre-built wheels supported:
  - CUDA 12.x (cu124, cu121) / CUDA 11.x (cu118)
  - Apple Silicon / Metal
  - CPU only / AMD ROCm / Vulkan

Detection order:
  1. Environment variable overrides (LLM_CHAT_PYTHON, LLM_CHAT_BACKEND)
  2. CLI arguments (--python, --backend)
  3. Apple Silicon (macOS ARM64) -> Metal backend
  4. PyTorch CUDA probe via torch.cuda.is_available + torch.version.cuda
  5. nvidia-smi driver version -> CUDA toolkit mapping
  6. Interactive user selection (fallback)

Fallback chain:
  1. Official abetlen wheel index (pre-compiled CUDA/Metal/CPU/ROCm/Vulkan)
  2. --no-binary source compilation from PyPI (requires C++ compiler)

Compatibility: Manual terminal, ComfyUI Manager, Windows Portable,
Stability Matrix / Pinokio, manual venv, Apple Silicon, Linux.

Usage examples:
    python install.py                          # Auto-detect & install
    python install.py --check                  # Diagnostic mode (no install)
    python install.py --python <path>          # Force a specific Python
    python install.py --backend cpu            # Force a specific backend
    python install.py --force                  # Reinstall (skip version check)
"""
import shutil
import textwrap

import argparse
import os
import platform
import re
import sys
import subprocess
import threading
import time
from pathlib import Path
from packaging.version import Version


# -- Configuration --------------------------------------------------------

# ── Version Management ──────────────────────────────────────────────────
# Version constants in _version_config.py, shared by install.py and
# llama_cpp_backend.py.
from _version_config import (
    SAFE_BASELINE_VERSION,
    LATEST_VERSION,
    MIN_VISION_VERSION,
)


class CPUSeverity:
    """CPU compatibility severity level for pre-built wheels.

    COMPATIBLE      — CPU has AVX2 (Haswell and newer, or unknown / undetected).
                      Latest pre-built wheel should work fine.
    INCOMPATIBLE    — CPU lacks AVX2 entirely. Cannot run pre-built wheels
                      without source compilation with -DGGML_AVX2=OFF.
    """
    COMPATIBLE = "compatible"
    WARNING = "warning"
    INCOMPATIBLE = "incompatible"


WHEEL_INDEX_URLS = {
    'cu124':  'https://abetlen.github.io/llama-cpp-python/whl/cu124',
    'cu121':  'https://abetlen.github.io/llama-cpp-python/whl/cu121',
    'cu118':  'https://abetlen.github.io/llama-cpp-python/whl/cu118',
    'metal':  'https://abetlen.github.io/llama-cpp-python/whl/metal',
    'cpu':    'https://abetlen.github.io/llama-cpp-python/whl/cpu',
    'rocm5':  'https://abetlen.github.io/llama-cpp-python/whl/rocm5',
    'vulkan': 'https://abetlen.github.io/llama-cpp-python/whl/vulkan',
}

# Fallback chain: try each backend in order, per GPU family.
FALLBACK_CHAIN = {
    'cu124': ['cu124', 'cu121', 'cu118', 'cpu'],
    'cu121': ['cu121', 'cu124', 'cu118', 'cpu'],
    'cu118': ['cu118', 'cu121', 'cu124', 'cpu'],
    'metal': ['metal', 'cpu'],
    'rocm5': ['rocm5', 'cpu'],
    'vulkan': ['vulkan', 'cpu', 'cu118'],
    'cpu':    ['cpu'],
}

# NVIDIA Driver Version -> CUDA Toolkit mapping
# Source: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# A driver supports ALL CUDA toolkits up to its max version.
CUDA_DRIVER_MAP = [
    (550, 'cu124'),   # Driver 550+ -> CUDA 12.4 wheels
    (525, 'cu121'),   # Driver 525+ -> CUDA 12.1 wheels
    (470, 'cu118'),   # Driver 470+ -> CUDA 11.8 wheels
    (0,   'cu118'),   # Fallback: oldest supported CUDA
]

BACKEND_LABELS = {
    'cu124':  'NVIDIA CUDA 12.x (RTX 30/40/50 series)',
    'cu121':  'NVIDIA CUDA 12.1',
    'cu118':  'NVIDIA CUDA 11.x (older GPUs)',
    'metal':  'Apple Silicon / Metal (M1/M2/M3/M4)',
    'cpu':    'CPU only (works on any system)',
    'rocm5':  'AMD ROCm 5.x',
    'vulkan': 'Vulkan (any GPU, experimental)',
}

# ComfyUI Manager environment variables for detection robustness
MANAGER_ENV_VARS = [
    'COMFYUI_FOLDERS_BASE_PATH',
    'COMFYUI_MANAGER',
    'COMFYUI_PATH',
]

PREFIX = '[EasyLLM GGUF Install]'

# ── CPU Compatibility ──────────────────────────────────────────────────────────
# Known CPU families grouped by AVX2 support. Used to warn users when a
# pre-built wheel may not work on their CPU.
_KNOWN_COMPATIBLE_CPUS: list[tuple[str, str]] = [
    # Intel Core (12th gen+ has AVX-VNNI which some wheels may require)
    ("13th Gen Intel", "modern"),
    ("12th Gen Intel", "modern"),
    ("11th Gen Intel", "modern"),
    ("10th Gen Intel", "modern"),
    ("9th Gen Intel", "modern"),
    ("8th Gen Intel", "modern"),
    ("7th Gen Intel", "modern"),
    ("6th Gen Intel", "modern"),
    # Intel Xeon Scalable (Skylake-SP and newer)
    ("Intel(R) Xeon(R) Platinum", "modern"),
    ("Intel(R) Xeon(R) Gold", "modern"),
    ("Intel(R) Xeon(R) Silver", "modern"),
    ("Intel(R) Xeon(R) Bronze", "modern"),
    # AMD Ryzen (all generations support AVX2)
    ("AMD Ryzen 9", "modern"),
    ("AMD Ryzen 7", "modern"),
    ("AMD Ryzen 5", "modern"),
    ("AMD Ryzen 3", "modern"),
    ("AMD EPYC", "modern"),
]

# CPUs that lack AVX2 entirely — cannot run pre-built wheels without
# source compilation with -DGGML_AVX2=OFF.
_OLDER_CPU_PATTERNS: list[str] = [
    "E7-",        # Xeon E7 (Ivy Bridge / older)
    "i7-3",       # Ivy Bridge (3rd gen)
    "i5-3",       # Ivy Bridge
    "i3-3",       # Ivy Bridge
    "i7-2",       # Sandy Bridge (2nd gen)
    "i5-2",
    "i3-2",
    "FX(tm)",     # AMD FX series
    "Phenom",     # AMD Phenom
    "A10-",       # AMD A-series (some)
    "A8-",
    "A6-",
]


def get_cpu_name() -> str | None:
    """Get the CPU name string on Windows via wmic.

    Returns:
        CPU name string (e.g., 'Intel(R) Xeon(R) CPU E5-2696 v3 @ 2.30GHz')
        or None if detection fails.
    """
    if sys.platform != 'win32':
        return None
    try:
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'name'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and line.lower() != 'name' and not line.startswith('Name'):
                    return line
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    # Fallback: PROCESSOR_IDENTIFIER env var
    return os.environ.get('PROCESSOR_IDENTIFIER')


def check_cpu_compatibility() -> CPUSeverity:
    """Check if the detected CPU lacks AVX2 and may be incompatible with
    pre-built wheels (which are compiled with AVX2 by default).

    Returns:
        CPUSeverity.COMPATIBLE     — CPU has AVX2 or couldn't be detected
        CPUSeverity.INCOMPATIBLE   — CPU lacks AVX2 entirely (Ivy Bridge,
                                     Sandy Bridge, AMD FX/Phenom)
    """
    cpu_name = get_cpu_name()
    if not cpu_name:
        return CPUSeverity.COMPATIBLE  # Can't detect — proceed silently

    cpu_upper = cpu_name.upper()

    # First: check if it's a known-modern CPU
    for pattern, _ in _KNOWN_COMPATIBLE_CPUS:
        if pattern.upper() in cpu_upper:
            return CPUSeverity.COMPATIBLE  # Known modern CPU — should be fine

    # Check if it's a known-older CPU pattern (all lack AVX2)
    for pattern in _OLDER_CPU_PATTERNS:
        if pattern.upper() in cpu_upper:
            print()
            print(f"  +{'-' * 66}+")
            print(f"  |  [!!] CPU INCOMPATIBLE{' ' * 48}|")
            print(f"  |                                                           |")
            print(f"  |  Detected CPU: {cpu_name:<49}|")
            print(f"  |                                                           |")
            print(f"  |  This CPU does NOT support AVX2 instruction set.           |")
            print(f"  |  Pre-built GPU-accelerated wheels require AVX2.            |")
            print(f"  |                                                           |")
            print(f"  |  v{SAFE_BASELINE_VERSION} has CPU-only fallback wheels that  " +
                  "work      |")
            print(f"  |  without AVX2. If the install fails, re-run with:          |")
            print(f"  |    python install.py --backend cpu                         |")
            print(f"  |                                                           |")
            print(f"  |  Or compile from source with:                              |")
            print(f"  |    set CMAKE_ARGS=-DGGML_AVX2=OFF                          |")
            print(f"  |    pip install llama-cpp-python --force-reinstall          |")
            print(f"  +{'-' * 66}+")
            print()
            return CPUSeverity.INCOMPATIBLE

    # Unknown CPU — treat as compatible (most likely has AVX2)
    return CPUSeverity.COMPATIBLE


# ── GPU Architecture Detection ───────────────────────────────────────────
# Older GPU architectures that should use CUDA 11.x (cu118) wheels
# instead of CUDA 12.x, even if the driver version would map to cu124.
# This prevents binary incompatibility on older hardware.
_GPU_ARCH_MAP: list[tuple[str, str]] = [
    # Kepler (GKxxx) — GTX 600/700 series
    ("GeForce GTX 680",  "cu118"),
    ("GeForce GTX 690",  "cu118"),
    ("GeForce GTX 760",  "cu118"),
    ("GeForce GTX 770",  "cu118"),
    ("GeForce GTX 780",  "cu118"),
    ("GeForce GTX 780 Ti", "cu118"),
    ("GeForce GTX Titan", "cu118"),
    ("GeForce GTX 750",  "cu118"),  # Maxwell GM107, but same safe zone
    # Maxwell (GM2xx) — GTX 900 series
    ("GeForce GTX 950",  "cu118"),
    ("GeForce GTX 960",  "cu118"),
    ("GeForce GTX 970",  "cu118"),
    ("GeForce GTX 980",  "cu118"),
    ("GeForce GTX 980 Ti", "cu118"),
    ("GeForce GTX Titan X", "cu118"),  # Maxwell Titan X
    # Pascal (GPxxx) — GTX 10xx series (very large install base)
    ("GeForce GTX 1050", "cu118"),
    ("GeForce GTX 1060", "cu118"),
    ("GeForce GTX 1070", "cu118"),
    ("GeForce GTX 1080", "cu118"),
    ("GeForce GTX 1080 Ti", "cu118"),
    ("TITAN X",          "cu118"),  # Pascal Titan X
    ("TITAN Xp",         "cu118"),
    # Volta — GV100
    ("TITAN V",          "cu118"),
    ("Tesla V100",       "cu118"),
    # Turing (TUxxx) — GTX 16xx, RTX 20xx
    ("GeForce GTX 1650", "cu118"),
    ("GeForce GTX 1660", "cu118"),
    ("GeForce GTX 1660 Super", "cu118"),
    ("GeForce GTX 1660 Ti", "cu118"),
    ("GeForce RTX 2060", "cu118"),
    ("GeForce RTX 2070", "cu118"),
    ("GeForce RTX 2080", "cu118"),
    ("GeForce RTX 2080 Ti", "cu118"),
    ("TITAN RTX",        "cu118"),
    ("Quadro RTX",       "cu118"),
    ("Tesla T4",         "cu118"),
    # Older Quadro / Tesla workstation cards
    ("Quadro P",         "cu118"),  # Pascal-based Quadro
    ("Quadro M",         "cu118"),  # Maxwell-based Quadro
    ("Quadro K",         "cu118"),  # Kepler-based Quadro
    ("M4000",            "cu118"),  # Quadro M4000 etc.
    ("P4000",            "cu118"),  # Quadro P4000 etc.
    ("Tesla P",          "cu118"),  # Pascal-based Tesla
    ("Tesla M",          "cu118"),  # Maxwell-based Tesla
    ("Tesla K",          "cu118"),  # Kepler-based Tesla
]

# GPU name substrings that are modern enough for CUDA 12.x auto-detection
# (these fall through to standard driver-version-based detection).
_MODERN_GPU_PATTERNS: list[str] = [
    'GeForce RTX 30',   # Ampere
    'GeForce RTX 40',   # Ada Lovelace
    'GeForce RTX 50',   # Blackwell
    'RTX A',            # RTX A-series workstation
    'RTX Ada',          # Ada Generation workstation
    'Tesla A',          # Ampere-based Tesla
]


# -- CLI Argument Parsing -------------------------------------------------
def parse_args():
    """Parse CLI arguments for manual invocation.

    Returns:
        argparse.Namespace with fields:
            python: str | None      -- explicit Python path override
            backend: str | None     -- explicit backend override
            check: bool             -- dry-run / diagnostic mode only
            force: bool             -- skip version check, always reinstall
            no_upgrade_pip: bool    -- skip automatic pip self-upgrade
            try_latest: bool        -- on older CPUs, install latest version
                                       (safe baseline used by default)
    """
    parser = argparse.ArgumentParser(
        description='Install llama-cpp-python for EasyLLM in ComfyUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Environment variable overrides:\n'
            '  LLM_CHAT_PYTHON     Override Python executable path\n'
            '  LLM_CHAT_BACKEND    Override backend (cu124, cu118, metal, cpu, rocm5, vulkan)\n'
            '  LLM_CHAT_TRY_LATEST Set to 1 to install latest version on older CPUs\n'
            '\n'
            'Flags:\n'
            '  --check               Diagnostic mode (no install)\n'
            '  --force               Reinstall even if version is current\n'
            '  --try-latest          On older CPUs, install latest version instead\n'
            '                        of the safe baseline (experimental)\n'
            '\n'
            'Examples:\n'
            '  python install.py --check\n'
            '  python install.py --python python_embeded\\python.exe --backend cpu\n'
            '  python install.py --force\n'
            '  python install.py --try-latest\n'
        ),
    )
    parser.add_argument(
        '--python', type=str, default=None,
        help='Explicit Python executable path (skip auto-detection)',
    )
    parser.add_argument(
        '--backend', type=str, default=None,
        choices=list(WHEEL_INDEX_URLS.keys()),
        help='Backend to install for (skip GPU auto-detection)',
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Diagnostic mode: detect and print info, do NOT install',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Skip version check, always reinstall llama-cpp-python',
    )
    parser.add_argument(
        '--no-upgrade-pip', action='store_true',
        help='Skip automatic pip self-upgrade before install',
    )
    parser.add_argument(
        '--try-latest', action='store_true',
        help='On older CPUs, install the latest version instead of the '
             'safe baseline (experimental — may crash on old hardware)',
    )
    # Filter out args ComfyUI Manager might unintentionally pass
    args, _ = parser.parse_known_args()
    return args


# -- Environment Variable Overrides ---------------------------------------

def get_try_latest_from_env() -> bool:
    """Check LLM_CHAT_TRY_LATEST env var for opt-in to latest version.

    Returns:
        True if the env var is set to '1', 'true', or 'yes'.
    """
    val = os.environ.get('LLM_CHAT_TRY_LATEST', '').strip().lower()
    return val in ('1', 'true', 'yes')


def get_python_from_env() -> str | None:
    """Check LLM_CHAT_PYTHON env var for explicit Python path override.

    Returns:
        The path if the env var is set and the file exists, None otherwise.
    """
    override = os.environ.get('LLM_CHAT_PYTHON')
    if override:
        path = Path(override)
        if path.exists():
            print(f"  {PREFIX} [OK] Using Python from LLM_CHAT_PYTHON env var: {override}")
            return str(path.resolve())
        else:
            print(
                f"  {PREFIX} [WARNING] LLM_CHAT_PYTHON='{override}' "
                f"does not exist -- ignoring"
            )
    return None


def get_backend_from_env() -> str | None:
    """Check LLM_CHAT_BACKEND env var for explicit backend override.

    Returns:
        Backend key if valid, None otherwise.
    """
    override = os.environ.get('LLM_CHAT_BACKEND')
    if override:
        key = override.strip().lower()
        if key in WHEEL_INDEX_URLS:
            print(f"  {PREFIX} [OK] Using backend from LLM_CHAT_BACKEND env var: {key}")
            return key
        else:
            print(
                f"  {PREFIX} [WARNING] LLM_CHAT_BACKEND='{override}' is not a valid "
                f"backend (valid: {', '.join(WHEEL_INDEX_URLS.keys())}) -- ignoring"
            )
    return None


# -- Detection Helpers ----------------------------------------------------

def is_manager_mode() -> bool:
    """Detect if running under ComfyUI Manager (vs manual terminal).
    """
    for var in MANAGER_ENV_VARS:
        if var in os.environ:
            return True
    return False


def get_python(cli_python: str | None = None) -> str:
    """Get the correct Python executable for ComfyUI.

    Priority:
      1. CLI --python argument
      2. LLM_CHAT_PYTHON env var
      3. ComfyUI Python auto-detection (Manager mode or parent walk)
      4. sys.executable fallback

    Args:
        cli_python: Optional explicit Python path from --python CLI arg.

    Returns:
        Path to the Python executable to use.
    """
    # Priority 1: CLI argument
    if cli_python:
        path = Path(cli_python)
        if path.exists():
            print(f"  {PREFIX} [OK] Using Python from --python argument: {cli_python}")
            return str(path.resolve())
        else:
            print(
                f"  {PREFIX} [ERROR] Specified Python path does not exist: "
                f"{cli_python}\n"
                f"  {PREFIX} Falling back to auto-detection..."
            )

    # Priority 2: Environment variable
    env_python = get_python_from_env()
    if env_python:
        return env_python

    # Priority 3: Manager mode (Manager already routes correctly)
    if is_manager_mode():
        return sys.executable

    # Priority 4: Auto-detect ComfyUI's Python
    detected = _detect_comfyui_python()
    if detected:
        return detected

    # Priority 5: Fallback with strong warning
    _print_fallback_warning(sys.executable)
    return sys.executable


def _print_fallback_warning(fallback_python: str):
    """Print a highly visible warning when falling back to sys.executable.
    """
    border = "-" * 66
    print()
    print(f"  +{border}+")
    print(f"  |  !! WARNING: Could not detect ComfyUI's Python automatically!  |")
    print(f"  |                                                                 |")
    print(f"  |  Installation will target:                                      |")
    print(f"  |    {fallback_python}")
    print(f"  |                                                                 |")
    print(f"  |  If this is NOT the Python that ComfyUI uses, the install       |")
    print(f"  |  will have NO EFFECT. Re-run with:                              |")
    print(f"  |                                                                 |")
    print(f"  |    python_embeded\\python.exe install.py                         |")
    print(f"  |                                                                 |")
    print(f"  |  Or use the --python flag:                                      |")
    print(f"  |    python install.py --python <path\\to\\comfyui\\python.exe>     |")
    print(f"  |                                                                 |")
    print(f"  |  Or set the LLM_CHAT_PYTHON env var.                            |")
    print(f"  +{'-' * 66}+")
    print()


def _detect_comfyui_python() -> str | None:
    """Search for ComfyUI's Python in common installation layouts.

    Detection strategies in order:
    1. Walk up from this script's location to find ComfyUI root
    2. Check for python_embeded/ (Windows Portable) inside or adjacent to root
    3. Check for venv/ (Manual git clone)
    4. Check for Stability Matrix layout (python_embeded above Packages/)
    5. Fall back to sys.executable
    """
    script_dir = Path(__file__).resolve().parent

    # -- Step 1: Find ComfyUI root by walking up parent chain --
    # ComfyUI root has main.py or ComfyUI_main.py
    comfy_root: Path | None = None
    for parent in [script_dir] + list(script_dir.parents):
        if (parent / 'main.py').exists() or (parent / 'ComfyUI_main.py').exists():
            comfy_root = parent
            break

    if comfy_root is not None:
        # -- Step 2a: Windows Portable: python_embeded/ inside ComfyUI root --
        embedded = comfy_root / 'python_embeded' / 'python.exe'
        if embedded.exists():
            print(f"  {PREFIX} [OK] Detected ComfyUI Windows Portable at: {comfy_root}")
            return str(embedded)

        # -- Step 2b: Stability Matrix layout: python_embeded/ above ComfyUI --
        # e.g. StabilityMatrix/Data/python_embeded/python.exe
        #       StabilityMatrix/Data/Packages/ComfyUI/
        sm_embedded = comfy_root.parent / 'python_embeded' / 'python.exe'
        if sm_embedded.exists():
            print(f"  {PREFIX} [OK] Detected Stability Matrix / shared python_embeded")
            return str(sm_embedded.resolve())

        # -- Step 3: Manual venv: ComfyUI/venv/Scripts/python.exe --
        venv = comfy_root / 'venv' / 'Scripts' / 'python.exe'
        if venv.exists():
            print(f"  {PREFIX} [OK] Detected ComfyUI venv at: {comfy_root}")
            return str(venv)

        # -- Step 4: Fallback to sys.executable --
        print(f"  {PREFIX} [OK] Detected ComfyUI at: {comfy_root}")
        print(f"  {PREFIX}     (using current Python: {sys.executable})")
        return sys.executable

    # -- Step 5: ComfyUI root not found -- try known alternative layouts --
    # Stability Matrix layout where python_embeded is at a different relative path
    sm_python = script_dir.parents[3] / 'python_embeded' / 'python.exe'
    if sm_python.exists():
        print(f"  {PREFIX} [OK] Detected Stability Matrix environment (alternate layout)")
        return str(sm_python.resolve())

    return None


def _detect_gpu_name_from_driver() -> str | None:
    """Get the GPU product name via nvidia-smi (internal helper).

    Returns:
        GPU name string (e.g., 'NVIDIA GeForce GTX 1660 Ti')
        or None if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            name = result.stdout.strip()
            if name:
                return name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def detect_backend() -> str | None:
    """Detect the best backend (CUDA version, ROCm, or CPU).

    Detection priority:
      1. GPU architecture / model name (via nvidia-smi)
         — maps known older GPUs to CUDA 11.x (cu118)
         — modern GPUs fall through to driver version detection
      2. nvidia-smi driver version → CUDA toolkit mapping
      3. nvcuda.dll existence check (Windows, no nvidia-smi)
      4. AMD ROCm detection

    Returns:
        Backend key (e.g., 'cu124', 'cu118', 'cpu', 'rocm5', 'vulkan')
        or None if NVIDIA GPU exists but version can't be determined.
    """
    # ── Check 0: GPU architecture / model name detection ──────────────
    # This is more precise than driver-version mapping. Older GPU
    # architectures (Kepler, Maxwell, Pascal, Volta, Turing) should use
    # CUDA 11.x wheels regardless of driver version.
    gpu_name = _detect_gpu_name_from_driver()
    if gpu_name:
        gpu_upper = gpu_name.upper()
        for pattern, recommended in _GPU_ARCH_MAP:
            if pattern.upper() in gpu_upper:
                print(f"  {PREFIX} [OK] Detected GPU: {gpu_name}")
                print(f"  {PREFIX}     -> Architecture maps to: {BACKEND_LABELS[recommended]}")
                return recommended
        # If it matches a known-modern pattern, fall through to driver version
        is_modern = any(p.upper() in gpu_upper for p in _MODERN_GPU_PATTERNS)
        if not is_modern:
            # Unknown GPU — assume older/safer CUDA 11.x
            print(f"  {PREFIX} [OK] Detected GPU: {gpu_name}")
            print(f"  {PREFIX}     -> Unknown architecture, using safe fallback: cu118")
            return 'cu118'

    # ── Check 1: nvidia-smi driver version ────────────────────────────
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            version_str = result.stdout.strip()
            match = re.search(r'(\d+)\.', version_str)
            if match:
                major = int(match.group(1))
                print(f"  {PREFIX} [OK] Detected NVIDIA driver version: {version_str.strip()}")
                for threshold, backend in CUDA_DRIVER_MAP:
                    if major >= threshold:
                        print(f"  {PREFIX}     -> Compatible with: {BACKEND_LABELS[backend]}")
                        return backend
                return 'cu118'  # Oldest supported fallback
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ── Check 2: nvcuda.dll exists (NVIDIA driver but no nvidia-smi) ──
    try:
        import ctypes
        ctypes.WinDLL('nvcuda.dll')
        print(
            f"  {PREFIX} [OK] NVIDIA driver detected (nvcuda.dll found)\n"
            f"  {PREFIX} [WARNING] Could not determine exact CUDA version "
            "(nvidia-smi not found)\n"
            f"  {PREFIX}     Please select your CUDA version manually below."
        )
        return None  # Signal to ask user
    except OSError:
        pass

    # ── Check 3: AMD ROCm ─────────────────────────────────────────────
    try:
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            print(f"  {PREFIX} [OK] Detected AMD GPU with ROCm drivers")
            return 'rocm5'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ── No GPU detected ───────────────────────────────────────────────
    print(f"  {PREFIX} [INFO] No dedicated GPU detected -- will use CPU-only wheel")
    return 'cpu'


def detect_cuda_via_torch(python_path: str) -> tuple[bool, str | None]:
    """Query PyTorch directly for CUDA availability and version.

    More reliable than nvidia-smi driver version mapping because PyTorch
    reports the CUDA toolkit version it was compiled against, which is
    exactly what matters for wheel compatibility.

    Args:
        python_path: Path to the target Python executable

    Returns:
        (cuda_available, cuda_version_string)
        e.g. (True, '12.1') or (False, None)
    """
    code = (
        'import torch; '
        'print(torch.cuda.is_available(), torch.version.cuda or "None")'
    )
    try:
        result = subprocess.run(
            [python_path, '-c', code],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) == 2:
                available = parts[0] == 'True'
                version = parts[1] if parts[1] != 'None' else None
                if available and version:
                    print(
                        f"  {PREFIX} [OK] PyTorch reports CUDA "
                        f"available (version {version})"
                    )
                    return True, version
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  {PREFIX} PyTorch CUDA probe failed: {e}")
    return False, None


def detect_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (ARM64 Mac).

    Returns:
        True if running on Darwin/ARM64, False otherwise.
    """
    is_apple_silicon = sys.platform == 'darwin' and platform.machine() == 'arm64'
    if is_apple_silicon:
        print(
            f"  {PREFIX} [OK] Detected Apple Silicon (ARM64) -- "
            "will use Metal-accelerated wheel"
        )
    return is_apple_silicon


def cuda_version_to_backend(cuda_version: str) -> str | None:
    """Map a CUDA version string (e.g. '12.1', '12.4') to a backend key.

    Args:
        cuda_version: CUDA version string like '12.1' or '12.4'

    Returns:
        Backend key like 'cu121' or 'cu124', or None if unsupported.
    """
    # Normalize: remove dots, take first 3 chars (e.g. '12.1' -> '121', '12.4' -> '124')
    digits = cuda_version.replace('.', '')[:3]
    key = f'cu{digits}'
    if key in WHEEL_INDEX_URLS:
        return key
    # Fallback: try matching just the major version
    major = cuda_version.split('.')[0]
    if major == '12':
        return 'cu124'  # Latest CUDA 12.x
    elif major == '11':
        return 'cu118'  # Latest CUDA 11.x
    return None


# -- Requirements.txt Scan ------------------------------------------------

def check_requirements_txt():
    """Scan the node's own requirements.txt for uncommented llama-cpp-python pins.

    If a version pin (==, >=, <=) is found uncommented, warn the user that
    ComfyUI Manager's startup check may enforce the pinned version and
    undo the install.py upgrade.

    """
    req_path = Path(__file__).parent / 'requirements.txt'
    if not req_path.exists():
        return

    lines = req_path.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue
        # Check for uncommented llama-cpp-python with version specifier
        if 'llama-cpp-python' in stripped and any(op in stripped for op in ['==', '>=', '<=', '~=', '!=', '>', '<']):
            print()
            print(f"  +{'-' * 66}+")
            print(f"  |  !! WARNING: Version pin found in requirements.txt!          |")
            print(f"  |                                                               |")
            print(f"  |  Line {i}: {stripped}")
            print(f"  |                                                               |")
            print(f"  |  ComfyUI Manager may enforce this pin during startup,         |")
            print(f"  |  potentially downgrading the version we just installed.        |")
            print(f"  |                                                               |")
            print(f"  |  To prevent this, comment out or remove the version pin       |")
            print(f"  |  from requirements.txt, or use '>={MIN_VISION_VERSION}'.      |")
            print(f"  +{'-' * 66}+")
            print()
            return  # Only warn once


# -- Installation ---------------------------------------------------------

def get_installed_version(python_path: str) -> str | None:
    """Check if llama-cpp-python is already installed and return its version.

    Runs a quick import test.

    Returns:
        Version string (e.g., '0.3.23') if installed, None if not installed.
    """
    try:
        result = subprocess.run(
            [python_path, '-c', 'import llama_cpp; print(llama_cpp.__version__)'],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  {PREFIX} Version check subprocess failed: {e}")
    return None


def get_installed_backend(python_path: str) -> str | None:
    """Detect the GGML backend of the currently installed llama-cpp-python.

    Probes for backend-specific C extension attributes to determine which
    GGML backend the wheel was compiled for. This catches the false-positive
    case where a Vulkan wheel is installed but the script thinks it's CUDA
    because the version number matches.

    Returns:
        Backend key ('cu118', 'vulkan', 'cpu', etc.) or None if undetectable.
    """
    code = textwrap.dedent("""\
        import llama_cpp
        try:
            from llama_cpp import llama_cpp
            if hasattr(llama_cpp, 'ggml_cuda_init'):
                print('CUDA')
            elif hasattr(llama_cpp, 'ggml_vulkan_init'):
                print('VULKAN')
            elif hasattr(llama_cpp, 'ggml_metal_init'):
                print('METAL')
            else:
                print('CPU')
        except Exception:
            print('UNKNOWN')
    """)
    try:
        result = subprocess.run(
            [python_path, '-c', code],
            capture_output=True, text=True, timeout=15,
        )
        backend_str = result.stdout.strip()
        # Map detected backend strings to our backend keys
        backend_map = {
            'CUDA': 'cu118',     # Can't distinguish CUDA 11 vs 12 easily
            'VULKAN': 'vulkan',
            'METAL': 'metal',
            'CPU': 'cpu',
        }
        detected = backend_map.get(backend_str)
        if detected:
            return detected
        return None
    except Exception:
        return None


def upgrade_pip(python_path: str) -> bool:
    """Upgrade pip itself before installing llama-cpp-python.

    Embedded Python (Windows Portable) often ships with an old pip that
    may not handle --extra-index-url correctly. This ensures we have a
    recent pip before proceeding.

    Args:
        python_path: Path to the target Python executable.

    Returns:
        True if pip upgrade succeeded or was skipped, False on error.
    """
    print(f"  {PREFIX} [INFO] Checking pip version...")
    cmd = [python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--no-cache-dir']
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            # Extract the new pip version from output if available
            pip_version = "updated"
            for line in result.stdout.splitlines():
                if 'pip' in line and ('install' in line.lower() or 'updated' in line.lower()):
                    pip_version = line.strip()
            print(f"  {PREFIX} [OK] pip {pip_version}")
            return True
        else:
            # pip upgrade failure is non-fatal — proceed anyway
            err = result.stderr.strip()[-200:]
            print(f"  {PREFIX} [WARNING] pip upgrade failed (non-fatal): {err}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  {PREFIX} [WARNING] pip upgrade skipped ({e})")
        return True


# ── Safe Baseline Installation (for older CPUs) ─────────────────────────

def install_safe_baseline(
    python_path: str,
    force: bool = False,
    backend: str | None = None,
) -> bool:
    """Install the safe baseline version of llama-cpp-python.

    First tries the abetlen wheel index (pre-built wheels) if a backend
    is provided. Falls back to PyPI source build if the pre-built wheel
    is not available (requires C++ compiler).

    Uses --no-deps to avoid accidentally downgrading other packages.

    Args:
        python_path: Path to the target Python executable
        force: If True, always reinstall even if already installed
        backend: Optional backend key (e.g., 'cu124') to use the abetlen
                 wheel index for pre-built wheels

    Returns:
        True if installation succeeded, False otherwise
    """
    version = SAFE_BASELINE_VERSION

    # Check if safe baseline is already installed
    current_version = get_installed_version(python_path)
    if current_version is not None and not force:
        try:
            cv = Version(current_version)
            bv = Version(version)
            if cv == bv:
                print(
                    f"  {PREFIX} [OK] Safe baseline v{SAFE_BASELINE_VERSION} "
                    "is already installed, skipping."
                )
                return True
        except Exception:
            pass  # Can't parse — reinstall to be safe

    # Only print progress in non-Manager mode
    show_progress = not is_manager_mode()
    if show_progress:
        print()
        print(f"  {PREFIX} Installing safe baseline v{version}...")
        if backend and backend in WHEEL_INDEX_URLS:
            print(f"  {PREFIX} Using abetlen wheel index ({backend}) for pre-built wheels.")
        else:
            print(f"  {PREFIX} Using PyPI (source build; requires C++ compiler if no pre-built wheel).")
        print()

    cmd = [
        python_path, '-m', 'pip', 'install',
        '--force-reinstall',
        '--no-deps',            # Don't touch other packages
        '--no-cache-dir',
        '--default-timeout', '300',
        f'llama-cpp-python=={version}',
    ]

    # Add abetlen wheel index URL if backend is known — provides pre-built
    # wheels for the safe baseline version on supported platforms.
    if backend and backend in WHEEL_INDEX_URLS:
        cmd.append('--extra-index-url')
        cmd.append(WHEEL_INDEX_URLS[backend])

    return _run_install(cmd, show_progress=show_progress)


def install_llama_cpp(
    python_path: str,
    backend: str,
    force: bool = False,
    try_latest: bool = False,
    version: str | None = None,
) -> bool:
    """Install llama-cpp-python with the matching pre-built wheel.

    Version selection: defaults to SAFE_BASELINE_VERSION for all users;
    pass try_latest=True to install LATEST_VERSION (experimental, for
    testing future releases before they become the safe baseline).
    Pass an explicit version string to override all version logic
    (used by the recovery menu for version fallback).

    On failure, falls through the backend fallback chain, then tries
    SAFE_BASELINE_VERSION from PyPI as a final alternative source.

    Args:
        python_path: Target Python executable path
        backend: Backend key (e.g., 'cu124', 'cpu', 'metal')
        force: If True, skip version check and always reinstall
        try_latest: If True, install LATEST_VERSION instead of SAFE_BASELINE_VERSION
        version: Explicit version override (e.g. '0.3.23') for recovery fallback

    Returns:
        True if installation succeeded, False otherwise
    """
    index_url = WHEEL_INDEX_URLS.get(backend)
    if index_url is None:
        print(f"  {PREFIX} [FAIL] Unknown backend: {backend}")
        return False

    # ── Version selection ──────────────────────────────────────────────
    if version:
        # User-specified version override (e.g., from recovery menu)
        package_spec = f'llama-cpp-python=={version}'
        print(f"  {PREFIX} [INFO] Using specified version: {version}")
        # Skip CPU compatibility check — user explicitly chose this
    else:
        cpu_severity = check_cpu_compatibility()
        is_old_cpu = cpu_severity == CPUSeverity.INCOMPATIBLE

        if is_old_cpu and not try_latest:
            # Old CPU, no --try-latest → install safe baseline via PyPI for
            # best compatibility with non-AVX2 CPUs.
            print()
            print(f"  {PREFIX} [INFO] Detected older CPU profile (no AVX2).")
            print(f"  {PREFIX} [INFO] Installing v{SAFE_BASELINE_VERSION} — the CPU-only "
                  "fallback wheel in this version")
            print(f"  {PREFIX} [INFO] works without AVX2. Installing via PyPI for best "
                  "compatibility...")
            print()
            safe_success = install_safe_baseline(python_path, force=force, backend=backend)
            if safe_success:
                return True
            # Safe baseline failed — fall through to abetlen index path
            print()
            print(f"  {PREFIX} [INFO] PyPI install failed (network issue?). "
                  "Falling back to abetlen wheel index...")
            print()

        if is_old_cpu and try_latest:
            # Old CPU + --try-latest → experimental mode (LATEST_VERSION may
            # differ from SAFE_BASELINE_VERSION in future releases).
            print()
            print(f"  {PREFIX} [INFO] --try-latest: installing v{LATEST_VERSION} "
                  "(experimental mode)")
            if LATEST_VERSION != SAFE_BASELINE_VERSION:
                print(f"  {PREFIX} [INFO] This is ahead of the safe baseline "
                      f"v{SAFE_BASELINE_VERSION}. If it crashes at model load,")
                print(f"  {PREFIX} [INFO] re-run without --try-latest to get the "
                      f"stable safe baseline.")
            else:
                print(f"  {PREFIX} [INFO] (Currently matches safe baseline v{SAFE_BASELINE_VERSION} — "
                      "no behavioral change.)")
            print()

        # ── Determine package_spec from version policy ────────────────────
        if is_old_cpu and not try_latest:
            package_spec = f'llama-cpp-python=={SAFE_BASELINE_VERSION}'
        else:
            package_spec = 'llama-cpp-python'

    # ── Set CMAKE_ARGS fallback for source builds ────────────────────
    # Ensures pip's source-build fallback uses the correct accelerator
    # instead of defaulting to CPU.
    if backend.startswith('cu'):
        # CUDA backend
        os.environ.setdefault("CMAKE_ARGS", "-DGGML_CUDA=on")
        print(
            f"  {PREFIX} [SET] CMAKE_ARGS=-DGGML_CUDA=on "
            "(fallback for source compilation)"
        )
    elif backend == 'metal':
        # Apple Silicon backend
        os.environ.setdefault("CMAKE_ARGS", "-DGGML_METAL=on")
        print(
            f"  {PREFIX} [SET] CMAKE_ARGS=-DGGML_METAL=on "
            "(fallback for source compilation)"
        )

    # ── Version-aware install ─────────────────────────────────────────
    # Check if already installed; upgrade only if too old for vision.
    current_version = get_installed_version(python_path)
    needs_upgrade = False

    if current_version is not None:
        try:
            cv = Version(current_version)
            mv = Version(MIN_VISION_VERSION)
            needs_upgrade = cv < mv

            # ── Backend mismatch detection ────────────────────────────────
            # Check if the installed wheel was compiled for a different
            # backend than the one we're about to install. This prevents the
            # false-positive case where a Vulkan (or other backend) binary is
            # cached but the script thinks nothing needs to change.
            if not needs_upgrade and not force:
                installed_backend = get_installed_backend(python_path)
                if installed_backend and installed_backend != backend:
                    print(
                        f"  {PREFIX} llama-cpp-python {current_version} "
                        f"is installed, but backend differs: "
                        f"installed={installed_backend}, "
                        f"requested={backend}"
                    )
                    print(
                        f"  {PREFIX} [INFO] Backend mismatch detected — "
                        "forcing reinstall"
                    )
                    needs_upgrade = True

            print(
                f"  {PREFIX} llama-cpp-python {current_version} is installed"
                f"{' (vision requires >=' + MIN_VISION_VERSION + ')' if needs_upgrade and cv < mv else ''}"
                f"{' (backend mismatch' + ')' if needs_upgrade and not (cv < mv) else ''}"
            )
        except Exception:
            # Can't parse version — treat as too old to be safe
            needs_upgrade = True
            print(
                f"  {PREFIX} llama-cpp-python {current_version} is installed "
                "(cannot determine version — will upgrade to be safe)"
            )

        if not needs_upgrade and not force:
            # Version is recent enough — no action needed
            print(
                f"  {PREFIX} [OK] Version requirement satisfied, "
                "skipping installation"
            )
            return True

        if not force:
            # Version is too old — ask user before upgrading
            print()
            print(
                f"  {PREFIX} [INFO] "
                f"llama-cpp-python {current_version} is too old for vision/multimodal"
            )
            print(
                f"  {PREFIX} [INFO] "
                f"The 'image_url' content type in create_chat_completion() "
                f"requires >= {MIN_VISION_VERSION}."
            )
            print()
            if not is_manager_mode():
                if not confirm("Upgrade to the latest version?"):
                    print(
                        f"\n  {PREFIX} [WARNING] Skipping upgrade. "
                        "Image input will NOT work.\n"
                        "    To upgrade manually later:\n"
                        f"      pip install llama-cpp-python --upgrade "
                        f"--extra-index-url {index_url}\n"
                    )
                    return True  # Don't fail — text-only mode still works
    else:
        print(f"  {PREFIX} [INFO] llama-cpp-python is not installed yet")
        needs_upgrade = True  # Fresh install

    # ── Build install/upgrade command ─────────────────────────────────
    cmd = [
        python_path, '-m', 'pip', 'install',
        '--force-reinstall',
        '--prefer-binary',
        '--no-cache-dir',
        '--default-timeout', '300',
        package_spec,
        '--extra-index-url', index_url,
    ]
    if current_version is not None and package_spec == 'llama-cpp-python':
        # Only add --upgrade for unversioned specs; version-pinned installs
        # implicitly handle the upgrade via --force-reinstall
        upgrade_idx = cmd.index(package_spec)
        cmd.insert(upgrade_idx, '--upgrade')

    if is_manager_mode():
        success = _run_install(cmd)
    else:
        success = _run_install(cmd, show_progress=True)

    # ── Post-install: if wheel version is too old, try source compilation ─
    if success:
        post_version = get_installed_version(python_path)
        if post_version is not None:
            try:
                pv = Version(post_version)
                mv = Version(MIN_VISION_VERSION)
                if pv < mv:
                    print()
                    print(
                        f"  {PREFIX} [WARNING] Pre-built wheel installed version "
                        f"{post_version}, but vision support requires "
                        f">= {MIN_VISION_VERSION}."
                    )
                    print()

                    # ── Fall through to source compilation ────────────────
                    print(
                        f"  {PREFIX} [INFO] Attempting source compilation "
                        "to get the latest version from PyPI..."
                    )
                    print(
                        f"  {PREFIX} [INFO] This requires a C/C++ compiler "
                        "(Visual Studio Build Tools on Windows)."
                    )
                    print()
                    if not is_manager_mode() and confirm(
                        "Try source compilation to get a newer version?"
                    ):
                        source_cmd = [
                            python_path, '-m', 'pip', 'install',
                            '--force-reinstall',
                            '--no-cache-dir',
                            '--default-timeout', '600',
                            '--no-binary', 'llama-cpp-python',
                            'llama-cpp-python',
                        ]
                        os.environ['CMAKE_ARGS'] = os.environ.get(
                            'CMAKE_ARGS',
                            '-DGGML_CUDA=on' if backend.startswith('cu')
                            else '-DGGML_VULKAN=on' if backend == 'vulkan'
                            else '-DGGML_METAL=on' if backend == 'metal'
                            else ''
                        )

                        print()
                        print(
                            f"  {PREFIX} [SET] CMAKE_ARGS="
                            f"{os.environ['CMAKE_ARGS']} "
                            "(source compilation)"
                        )
                        source_success = _run_install(source_cmd, show_progress=True)
                        if source_success:
                            # Re-verify version after source install
                            final_version = get_installed_version(python_path)
                            if final_version:
                                try:
                                    fv = Version(final_version)
                                    if fv >= mv:
                                        print(
                                            f"  {PREFIX} [OK] Source compilation "
                                            f"succeeded: version {final_version}"
                                        )
                                    else:
                                        print(
                                            f"  {PREFIX} [WARNING] Source "
                                            f"compilation installed "
                                            f"{final_version}, still below "
                                            f"{MIN_VISION_VERSION}."
                                        )
                                except Exception:
                                    pass
                            return source_success
                        else:
                            print(
                                f"  {PREFIX} [INFO] Source compilation failed. "
                                f"Keeping wheel version {post_version} -- "
                                "text-only mode will still work."
                            )
            except Exception:
                pass

    # Graceful fallback chain on failure
    if not success and not is_manager_mode():
        fallback_list = FALLBACK_CHAIN.get(backend, ['cpu'])
        fallbacks = [b for b in fallback_list if b != backend]
        for fallback_backend in fallbacks:
            fallback_index = WHEEL_INDEX_URLS.get(fallback_backend)
            if fallback_index is None:
                continue
            print()
            print(
                f"  {PREFIX} [INFO] Primary install failed. "
                f"Try fallback: {BACKEND_LABELS.get(fallback_backend, fallback_backend)}"
            )
            if not confirm(f"Try {BACKEND_LABELS.get(fallback_backend, fallback_backend)} instead?"):
                continue

            # Rebuild command with new backend
            fallback_cmd = [
                python_path, '-m', 'pip', 'install',
                '--force-reinstall',
                '--prefer-binary',
                '--no-cache-dir',
                '--default-timeout', '300',
                'llama-cpp-python',
                '--extra-index-url', fallback_index,
            ]
            if current_version is not None:
                upgrade_idx = fallback_cmd.index('llama-cpp-python')
                fallback_cmd.insert(upgrade_idx, '--upgrade')

            success = _run_install(fallback_cmd, show_progress=True)
            if success:
                print(
                    f"  {PREFIX} [OK] Installation succeeded with "
                    f"fallback backend: {BACKEND_LABELS.get(fallback_backend, fallback_backend)}"
                )
                globals()['_actual_backend'] = fallback_backend
                return True

    # ── Version fallback: if abetlen index failed AND CPU is old, try PyPI directly ──
    if not success and is_old_cpu:
        print()
        print(f"  {PREFIX} [INFO] Abetlen wheel index install failed for "
              f"v{SAFE_BASELINE_VERSION}.")
        print(f"  {PREFIX} [INFO] Trying v{SAFE_BASELINE_VERSION} directly from PyPI "
              "as fallback...")
        success = install_safe_baseline(python_path, force=True)
        if success:
            print(f"  {PREFIX} [OK] Installed v{SAFE_BASELINE_VERSION} from PyPI.")

    return success


def _run_install(cmd: list[str], show_progress: bool = False) -> bool:
    """Run the pip install command.

    Args:
        cmd: The pip install command list.
        show_progress: When True, prints a header block + progress dots while
                       the process runs (for interactive terminal use). When
                       False, runs silently (for ComfyUI Manager context).

    Returns:
        True if installation succeeded, False otherwise.
    """
    # Extract python_path from the command (first element)
    python_path = cmd[0] if cmd else sys.executable

    if show_progress:
        print(f"\n  +- Installation Command ----------------------------------------------")
        print(f"  | {' '.join(cmd)}")
        print(f"  +-----------------------------------------------------------------------")
        print(f"  {PREFIX} Installing... (this may take 5-15 minutes depending on your system and internet)", end='', flush=True)

    stdout = ""
    stderr = ""
    return_code = -1

    if show_progress:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Print progress dots every 10 seconds while the process runs
        stop_dots = threading.Event()

        def _print_dots():
            while not stop_dots.is_set():
                if stop_dots.wait(10):
                    break
                if not stop_dots.is_set():
                    print('.', end='', flush=True)

        dot_thread = threading.Thread(target=_print_dots, daemon=True)
        dot_thread.start()

        try:
            stdout, stderr = process.communicate(timeout=600)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return_code = -1
        finally:
            stop_dots.set()
            dot_thread.join(timeout=2)

        print()  # Newline after dots
    else:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return_code = result.returncode
            stderr = result.stderr
            stdout = result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"\n  {PREFIX} [FAIL] Subprocess error: {e}")
            return False

    if return_code == 0:
        if show_progress:
            print(f"  {PREFIX} [OK] Installation completed successfully")
        return True
    else:
        if show_progress:
            print(f"  {PREFIX} [FAIL] Installation failed")
        else:
            print(f"  {PREFIX} [FAIL] Installation failed:\n{stderr[-500:]}")
        stderr_out = stderr.strip() if stderr else ''
        if stderr_out and show_progress:
            print(f"\n  {PREFIX} Error details:\n{stderr_out[-500:]}")

        # Save pip failure log for diagnostics
        _save_install_failure_log(python_path, cmd, stderr_out, stdout)

        # Smarter error context
        _print_error_guidance(stderr_out, cmd)
        return False


def _print_error_guidance(stderr: str, cmd: list[str]):
    """Print actionable next-step suggestions based on error text.

    Args:
        stderr: The stderr output from the failed pip command.
        cmd: The pip command that was run (for context).
    """
    if not stderr:
        return

    stderr_lower = stderr.lower()
    suggestions = []

    if 'no matching distribution' in stderr_lower or 'could not find a version' in stderr_lower:
        suggestions.append(
            "  • The wheel may not be available for your Python version.\n"
            "  • Try 'python install.py --backend cpu' for a CPU-only fallback.\n"
            "  • Or try 'python install.py --force' to force a fresh download."
        )
    if 'ssl' in stderr_lower or 'certificate' in stderr_lower:
        suggestions.append(
            "  • SSL/TLS error — check your internet connection / proxy settings.\n"
            "  • Try: pip install --trusted-host abetlen.github.io llama-cpp-python"
        )
    if 'permission denied' in stderr_lower or 'access is denied' in stderr_lower:
        suggestions.append(
            "  • Permission denied — files may be locked by running ComfyUI.\n"
            "  • Close ComfyUI completely and try again.\n"
            "  • Or run as Administrator (Windows) / with 'sudo' (Linux/macOS)."
        )
    if 'timeout' in stderr_lower or 'connection' in stderr_lower:
        suggestions.append(
            "  • Network timeout — check your internet connection.\n"
            "  • The --default-timeout 300 flag is already set; try again."
        )
    if 'visual studio' in stderr_lower or 'c++' in stderr_lower or 'compiler' in stderr_lower:
        suggestions.append(
            "  • pip fell back to source compilation but no C++ compiler found.\n"
            "  • Ensure --prefer-binary is working, or install Visual Studio Build Tools."
        )

    if suggestions:
        print(f"\n  {PREFIX} Suggested next steps:")
        for s in suggestions:
            print(s)
    else:
        print(f"\n  {PREFIX} No specific guidance for this error.")
        print(f"  {PREFIX} Try 'python install.py --backend cpu' as a fallback.")


# -- Error Classification & Diagnostics ----------------------------------

# Known process crash exit codes and their human-readable names.
# These are used to provide useful diagnostic information when a subprocess
# crashes without producing Python traceback output (e.g. access violation).
_CRASH_EXIT_CODES: dict[int, str] = {
    # Windows NT status codes (negative in subprocess returncode)
    -1073741819: "ACCESS_VIOLATION (0xC0000005)",   # SIGSEGV equivalent
    -1073740791: "STATUS_STACK_BUFFER_OVERRUN (0xC0000409)",
    -1073741676: "STATUS_STACK_OVERFLOW (0xC00000FD)",
    -1073741510: "DLL_NOT_FOUND (0xC0000135)",      # Import DLL missing
    -1073740940: "STATUS_FLOAT_INEXACT_RESULT (0xC000008C)",
    -1073741502: "CONTROL_C_EXIT (0xC000013A)",
    # Unix signals (positive exit code = signal number)
    -6:  "SIGABRT (signal 6)",
    -11: "SIGSEGV (signal 11) — invalid memory reference",
}


def _interpret_exit_code(exit_code: int) -> str:
    """Interpret a subprocess exit code into a human-readable description.

    On Windows, process crashes produce large negative NTSTATUS codes
    (e.g. -1073741819 = 0xC0000005 = ACCESS_VIOLATION). On Unix,
    negative values indicate signal termination.

    Args:
        exit_code: The returncode from subprocess.run() or Popen.

    Returns:
        A human-readable string describing the exit condition.
    """
    if exit_code == 0:
        return "Success"
    if exit_code in _CRASH_EXIT_CODES:
        return _CRASH_EXIT_CODES[exit_code]
    # Generic interpretation
    if exit_code < 0:
        return f"Process terminated by signal ({exit_code})"
    return f"Process exited with code {exit_code}"


def _classify_import_error(stderr: str) -> str:
    """Classify the import error stderr into a known category.

    Returns one of:
        'dll_not_found:<dll_name>', 'cuda_mismatch', 'no_module',
        'crt_missing', 'avx2_incompatible', 'unknown'
    """
    lower = stderr.lower()

    # DLL not found errors (most common on Windows)
    if 'dll' in lower and ('not found' in lower or 'load' in lower):
        for dll_pattern in ['cublas', 'cufft', 'curand', 'cusolver', 'cusparse',
                           'cuda', 'ggml', 'python', 'vcruntime', 'msvcp',
                           'api-ms-win-crt']:
            if dll_pattern in lower:
                return f'dll_not_found:{dll_pattern}'
        return 'dll_not_found:unknown'

    # CUDA version mismatch
    if 'cuda' in lower and ('version' in lower or 'unsupported' in lower):
        return 'cuda_mismatch'

    # Module not found
    if 'no module' in lower:
        return 'no_module'

    # CRT/VC++ runtime issues
    if 'api-ms-win-crt' in lower or 'vcruntime' in lower or 'msvcp' in lower:
        return 'crt_missing'

    # AVX2 / CPU incompatibility
    if 'illegal instruction' in lower or 'invalid instruction' in lower or 'avx' in lower:
        return 'avx2_incompatible'

    return 'unknown'


def _print_error_tips(error_type: str, stderr: str):
    """Print actionable tips based on the classified import error type."""
    tips = {
        'dll_not_found:cublas': [
            "CUDA runtime DLL cublas not found.",
            "The nvidia-cublas-cuXX package may not have installed correctly.",
            "Try recovery option for CUDA 12.x cu121 index instead.",
            "Or try CPU-only backend as fallback.",
        ],
        'dll_not_found:cuda': [
            "General CUDA DLL load failure.",
            "Your NVIDIA driver may be too old for this wheel version.",
            "Try recovery option with older wheel v0.3.23.",
            "Or try CUDA 12.x backend (cu121 index).",
        ],
        'dll_not_found:ggml': [
            "Core ggml DLL failed to load.",
            "This can indicate a corrupted download or incompatible wheel.",
            "Try recovery option with older wheel v0.3.23.",
            "Or try Vulkan backend which uses a different code path.",
        ],
        'dll_not_found:unknown': [
            "An unknown DLL failed to load (see full error above).",
            "Try CPU-only backend as a quick test.",
            "If CPU works, the issue is CUDA-related.",
        ],
        'crt_missing': [
            "Visual C++ Runtime not found or too old.",
            "The wheel requires a newer VC++ redistributable.",
            "Install from: https://aka.ms/vs/17/release/vc_redist.x64.exe",
            "Or try older version v0.3.23 which may use an older CRT.",
        ],
        'cuda_mismatch': [
            "CUDA version mismatch between wheel and runtime.",
            "Try a different CUDA backend in the recovery menu.",
        ],
        'avx2_incompatible': [
            "CPU lacks AVX2 instruction set.",
            "Pre-built wheels require AVX2. Compile from source:",
            "  set CMAKE_ARGS=-DGGML_AVX2=OFF -DGGML_CUDA=on",
            "  pip install llama-cpp-python --force-reinstall",
        ],
        'no_module': [
            "The package wasn't installed correctly.",
            "Try 'python install.py --force' to reinstall.",
        ],
        'unknown': [
            "Could not classify the import error automatically.",
            "An error log has been saved — please share it for support.",
            "Try CPU-only backend as a quick test.",
        ],
    }

    tips_list = tips.get(error_type, tips['unknown'])
    print(f"  {PREFIX} [DIAGNOSIS] {tips_list[0]}")
    for tip in tips_list[1:]:
        print(f"  {PREFIX}             {tip}")


def _get_disk_space(path: str) -> str:
    """Get available disk space for the given path (in human-readable format).

    Uses shutil.disk_usage() as the primary method (cross-platform), with a
    ctypes GetDiskFreeSpaceExW fallback for edge cases.

    Args:
        path: A file or directory path to check disk space for.

    Returns:
        Human-readable disk space string including the drive letter
        (e.g. '124.5 GB free (D:)'), or 'N/A' if detection fails.
    """
    drive = Path(path).drive or path
    # Method 1: shutil.disk_usage (cross-platform, reliable)
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        return f"{free_gb:.1f} GB free ({drive})"
    except Exception:
        pass

    # Method 2: ctypes fallback for edge cases (Windows only)
    try:
        import ctypes

        free_bytes = ctypes.c_ulonglong(0)
        result = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(path),
            None, None, ctypes.pointer(free_bytes),
        )
        if result != 0:  # BOOL: non-zero = success
            free_gb = free_bytes.value / (1024 ** 3)
            return f"{free_gb:.1f} GB free ({drive})"
    except Exception:
        pass

    return "N/A"


def _check_disk_space(python_path: str) -> None:
    """Check available disk space on the Python drive and warn if low.

    Prints a clear warning if free space is below thresholds:
    - < 1 GB: CRITICAL — installation will almost certainly fail.
    - < 5 GB: WARNING — may succeed but disk is tight.

    Args:
        python_path: Path to the Python executable (to determine the drive).
    """
    try:
        usage = shutil.disk_usage(python_path)
        free_gb = usage.free / (1024 ** 3)
        drive = Path(python_path).drive or python_path

        if free_gb < 1.0:
            print()
            print(f"  {'!' * 66}")
            print(f"  !!  CRITICAL: Low disk space on {drive}")
            print(f"  !!  Only {free_gb:.1f} GB free — installation will almost certainly fail.")
            print(f"  !!  Free up space or install to a different drive.")
            print(f"  {'!' * 66}")
            print()
            if not confirm("Continue anyway?"):
                print(f"\n  {PREFIX} Installation cancelled.\n")
                sys.exit(1)
        elif free_gb < 5.0:
            print()
            print(f"  {PREFIX} [WARNING] Low disk space on {drive}: "
                  f"{free_gb:.1f} GB free")
            print(f"  {PREFIX} Installation may still succeed, but consider freeing space.")
            print()
    except Exception:
        pass  # Non-critical check; proceed even if check fails


def _save_error_log(
    python_path: str,
    backend: str,
    stderr: str,
    exit_code: int = -1,
    pip_cmd: list[str] | None = None,
    recovery_attempts: list[dict] | None = None,
    label: str = "",
):
    """Save diagnostic information to a timestamped file for troubleshooting.

    Args:
        python_path: Path to the target Python executable.
        backend: Backend key (e.g. 'cu118', 'vulkan').
        stderr: Stderr output from the failed operation. For process crashes,
                this should contain the crash diagnostics string.
        exit_code: Subprocess exit code (helps detect crashes vs Python errors).
        pip_cmd: The pip install command that was run (if applicable).
        recovery_attempts: List of dicts describing previous recovery attempts.
        label: Optional label for the log (e.g. 'recovery_2' or 'install_failure').
    """
    log_dir = Path(__file__).parent / 'install_logs'
    log_dir.mkdir(exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    suffix = f'_{label}' if label else ''
    log_path = log_dir / f'install_error_{timestamp}{suffix}.txt'

    # Crash interpretation
    crash_info = _interpret_exit_code(exit_code) if exit_code != 0 else "N/A — process exited normally"
    is_crash = exit_code != 0 and not stderr.strip()

    lines = [
        "EasyLLM Install Error Log",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python: {python_path}",
        f"Backend: {backend}",
        f"Version: {get_installed_version(python_path) or 'N/A'}",
        f"OS: {platform.platform()}",
        f"CPU: {get_cpu_name() or 'Unknown'}",
        f"Disk space: {_get_disk_space(python_path)}",
        "",
        f"--- Exit Status ---",
        f"Exit code: {exit_code}",
        f"Crash signal: {crash_info}",
        f"Process crash detected: {'Yes' if is_crash else 'No'}",
        "",
    ]

    if pip_cmd:
        lines.append("--- Pip Install Command ---")
        lines.append(' '.join(pip_cmd))
        lines.append("")

    if recovery_attempts:
        lines.append("--- Recovery Attempts ---")
        for i, attempt in enumerate(recovery_attempts, start=1):
            outcome = attempt.get('outcome', 'unknown')
            lines.append(
                f"  Attempt {i}: backend={attempt.get('backend', '?')}, "
                f"version={attempt.get('version', '?')}, "
                f"outcome={outcome}"
            )
        lines.append("")

    lines.append("--- Import Error ---")
    lines.append(stderr if stderr else "(no error output captured)")
    lines.append("")

    lines.append("--- GPU Info ---")
    # Append nvidia-smi output if available
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines.append(result.stdout.strip())
        else:
            lines.append("(nvidia-smi not available)")
    except Exception:
        lines.append("(nvidia-smi query failed)")

    log_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  {PREFIX} [INFO] Error log saved to: {log_path}")
    print(f"  {PREFIX} [INFO] Share this file when asking for help.")
    return log_path


def _save_install_failure_log(
    python_path: str,
    cmd: list[str],
    stderr: str,
    stdout: str = "",
):
    """Save a log specifically for pip install failures.

    Unlike _save_error_log which covers import/verification failures, this
    captures the full pip output when pip itself fails to install the package.

    Args:
        python_path: Path to the target Python executable.
        cmd: The pip install command that was run.
        stderr: Full stderr from the pip process.
        stdout: Full stdout from the pip process (useful for debugging).
    """
    log_dir = Path(__file__).parent / 'install_logs'
    log_dir.mkdir(exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'install_error_{timestamp}_pip_failure.txt'

    lines = [
        "EasyLLM Pip Install Failure Log",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python: {python_path}",
        f"OS: {platform.platform()}",
        f"CPU: {get_cpu_name() or 'Unknown'}",
        f"Disk space: {_get_disk_space(python_path)}",
        "",
        "--- Pip Command ---",
        ' '.join(cmd),
        "",
        "--- Pip stdout ---",
        stdout if stdout else "(no stdout captured)",
        "",
        "--- Pip stderr ---",
        stderr if stderr else "(no stderr captured)",
        "",
    ]

    log_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  {PREFIX} [INFO] Pip failure log saved to: {log_path}")
    print(f"  {PREFIX} [INFO] Share this file when asking for help.")


# -- Verification & Post-Install -----------------------------------------

def verify_installation(python_path: str) -> tuple[bool, str | None, str, int]:
    """Verify llama-cpp-python is importable after installation.

    Args:
        python_path: Path to the target Python executable.

    Returns:
        (success, version_string, raw_stderr, exit_code) tuple.
        raw_stderr is the full stderr output for downstream error classification.
        exit_code is the subprocess return code (useful for detecting process
        crashes where stderr may be empty).
    """
    print(f"\n  {PREFIX} -- Verifying installation --")
    cmd = [
        python_path, '-c',
        'import llama_cpp; '
        'print(f"Version: {llama_cpp.__version__}")',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  {PREFIX} [FAIL] Verification subprocess error: {e}")
        return False, None, "", -1

    if result.returncode == 0:
        version_str = result.stdout.strip()
        print(f"  {PREFIX} [OK] {version_str}")
        # Extract just the version number
        version_match = re.search(r'Version: (.+)', version_str)
        actual_version = version_match.group(1) if version_match else version_str
        return True, actual_version, "", 0
    else:
        stderr = result.stderr.strip()
        exit_code = result.returncode
        print(f"  {PREFIX} [FAIL] Verification failed! (exit code {exit_code})")

        # Detect process crash: empty stderr + non-zero exit = likely crash signal
        if not stderr and exit_code != 0:
            crash_desc = _interpret_exit_code(exit_code)
            stderr = (
                f"[PROCESS CRASH] Python subprocess exited with code {exit_code}\n"
                f"  Exit code interpretation: {crash_desc}\n"
                f"\n"
                f"  No Python traceback was captured. This typically means the\n"
                f"  import of llama_cpp crashed the Python interpreter directly\n"
                f"  (e.g. an access violation or DLL loading failure in a C\n"
                f"  extension module).\n"
                f"\n"
                f"  Common causes:\n"
                f"  - CUDA DLL version mismatch (ggml-cuda.dll vs nvidia-* DLLs)\n"
                f"  - Missing Visual C++ Runtime redistributable\n"
                f"  - Corrupted wheel download / incomplete installation\n"
                f"  - CPU instruction set incompatibility (e.g. AVX2 required)\n"
            )
            print()
            print(f"  +{'=' * 66}+")
            print(f"  |  PROCESS CRASH DETECTED{' ' * 40}|")
            print(f"  +{'=' * 66}+")
            print(f"  |  The Python subprocess crashed with exit code {exit_code}.")
            print(f"  |  {crash_desc}")
            print(f"  |")
            print(f"  |  No Python traceback is available — the crash occurred")
            print(f"  |  inside a C extension module (likely llama_cpp or CUDA DLL).")
            print(f"  +{'=' * 66}+")
            print()
        elif stderr:
            print()
            print(f"  +{'=' * 66}+")
            print(f"  |  VERIFICATION ERROR{' ' * 47}|")
            print(f"  +{'=' * 66}+")
            # Print each line of the error in full (no truncation)
            for line in stderr.splitlines():
                print(f"  | {line}")
            print(f"  +{'=' * 66}+")
            print()

            # Use error classification for smarter diagnostic tips
            error_type = _classify_import_error(stderr)
            _print_error_tips(error_type, stderr)
            print()

        return False, None, stderr, exit_code


def verify_comfyui_runtime(python_path: str, comfy_root: Path | None) -> bool:
    """Verify that ComfyUI itself can see the installed package.

    Runs a quick import check from ComfyUI's root directory to simulate
    what happens when ComfyUI loads the custom node at startup.

    Args:
        python_path: Path to the target Python executable.
        comfy_root: Path to ComfyUI root (if found during detection).

    Returns:
        True if verification passed or was skipped (no comfy_root).
    """
    if comfy_root is None or not comfy_root.exists():
        print(f"  {PREFIX} [SKIP] ComfyUI root not detected — skipping runtime verification")
        return True  # Can't verify — skip silently

    print(f"\n  {PREFIX} -- Verifying ComfyUI can find the package --")
    test_code = (
        'import sys; '
        f'sys.path.insert(0, {str(comfy_root)!r}); '
        'import llama_cpp; '
        'print(f"OK {llama_cpp.__version__}")'
    )
    cmd = [python_path, '-c', test_code]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip().startswith('OK'):
            version = result.stdout.strip()[3:]  # Strip "OK " prefix
            print(f"  {PREFIX} [OK] ComfyUI can see llama-cpp-python {version}")
            return True
        else:
            print(
                f"  {PREFIX} [WARNING] ComfyUI may not find the package at runtime.\n"
                f"  {PREFIX}     Import error: {result.stderr.strip()[-200:]}"
            )
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  {PREFIX} [WARNING] Runtime verification skipped: {e}")
        return True


def _print_version_summary(
    python_path: str,
    backend: str,
    version: str | None,
    success: bool,
):
    """Print a clear, boxed summary of the installation result.
    """
    border = "-" * 66
    print()
    print(f"  +{border}+")
    if success and version:
        print(f"  |  [OK] INSTALLATION SUCCESSFUL!{' ' * 33}|")
        print(f"  |{' ' * 66}|")
        print(f"  |  Installed in: {python_path}")
        print(f"  |  Package:      llama-cpp-python")
        print(f"  |  Version:      {version}  (required: >= {MIN_VISION_VERSION}){' ' * max(0, 1)}|")
        print(f"  |  Backend:      {BACKEND_LABELS.get(backend, backend)}")
        print(f"  |{' ' * 66}|")
        print(f"  |  Next step:   Restart ComfyUI{' ' * 38}|")
        print(f"  |  The 'EasyLLM GGUF' node is now ready.{' ' * 22}|")
    elif success:
        print(f"  |  [OK] Installation completed (version check pending restart){' ' * 6}|")
        print(f"  |{' ' * 66}|")
        print(f"  |  Installed in: {python_path}")
        print(f"  |{' ' * 66}|")
        print(f"  |  Next step:   Restart ComfyUI{' ' * 38}|")
    else:
        print(f"  |  [FAIL] INSTALLATION FAILED{' ' * 35}|")
        print(f"  |{' ' * 66}|")
        print(f"  |  Possible causes:{' ' * 50}|")
        print(f"  |  1. No internet connection{' ' * 39}|")
        print(f"  |  2. Pre-built wheel not available for your platform{' ' * 19}|")
        print(f"  |  3. Python environment issue{' ' * 33}|")
        print(f"  |{' ' * 66}|")
        print(f"  |  Next steps:{' ' * 54}|")
        print(f"  |  * Check your internet and try again{' ' * 28}|")
        print(f"  |  * Try 'python install.py --backend cpu' as fallback{' ' * 15}|")
        print(f"  |  * Or set CMAKE_ARGS and compile from source (see README){' ' * 9}|")
    print(f"  +{'-' * 66}+")
    print()


# -- CUDA Runtime Libraries ----------------------------------------------


def _get_cuda_packages(backend: str) -> list[str]:
    """Return the correct NVIDIA CUDA library packages for the backend.

    The llama-cpp-python CUDA wheel ships with ggml-cuda.dll but NOT
    the supporting CUDA library DLLs (cublas, cusparse, etc.).
    These must be installed separately from nvidia's pip index.
    The package suffix must match the CUDA toolkit version the wheel
    was compiled against.

    Args:
        backend: Backend key ('cu124', 'cu121', or 'cu118')

    Returns:
        List of nvidia-* package names for the matching CUDA version
    """
    cuda_ver = 'cu12' if backend in ('cu124', 'cu121') else 'cu11'
    return [
        f'nvidia-cublas-{cuda_ver}',
        f'nvidia-cufft-{cuda_ver}',
        f'nvidia-curand-{cuda_ver}',
        f'nvidia-cusolver-{cuda_ver}',
        f'nvidia-cusparse-{cuda_ver}',
    ]


def install_cuda_libraries(python_path: str, backend: str) -> bool:
    """Install NVIDIA CUDA runtime library packages needed by the CUDA wheel.

    The llama-cpp-python CUDA wheel ships with ggml-cuda.dll but NOT
    the supporting CUDA library DLLs (cublas, cusparse, etc.).
    These must be installed separately from nvidia's pip index.

    Automatically selects the correct CUDA version:
      - cu124 / cu121  →  nvidia-*-cu12  (CUDA 12.x runtime)
      - cu118          →  nvidia-*-cu11  (CUDA 11.x runtime)

    After installation, the DLLs are copied into llama_cpp/lib/
    so that the library loader can find them at runtime.

    Args:
        python_path: Path to the target Python executable
        backend: Backend key (e.g., 'cu124', 'cu118')

    Returns:
        True if CUDA libraries are ready, False on failure
    """
    if backend not in ('cu124', 'cu121', 'cu118'):
        return True  # Not a CUDA backend, nothing to do

    nvidia_packages = _get_cuda_packages(backend)
    cuda_label = 'CUDA 12' if backend in ('cu124', 'cu121') else 'CUDA 11'

    print()
    print(f"  {PREFIX} -- Installing {cuda_label} runtime libraries --")
    print(f"  {PREFIX} (required by the pre-built CUDA wheel)")

    cmd = [
        python_path, '-m', 'pip', 'install',
        '--no-cache-dir',
        '--force-reinstall',
        '--default-timeout', '300',
        *nvidia_packages,
        '--extra-index-url', 'https://pypi.nvidia.com',
    ]

    if is_manager_mode():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"  {PREFIX} [FAIL] {cuda_label} library installation failed:\n{result.stderr[-300:]}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  {PREFIX} [FAIL] {cuda_label} library subprocess error: {e}")
            return False
    else:
        print(f"\n  {PREFIX} Installing {cuda_label} libraries (this may take 2-5 minutes)...")
        if not _run_install(cmd, show_progress=True):
            return False

    # Copy CUDA DLLs into llama_cpp/lib/ so the loader finds them
    _copy_cuda_dlls(python_path)
    return True

def _copy_cuda_dlls(python_path: str):
    """Copy CUDA DLLs from nvidia packages into llama_cpp/lib/.

    The llama_cpp loader (_ctypes_extensions.py) only adds llama_cpp/lib/
    to the DLL search path. By copying the CUDA DLLs there, they are
    automatically found when importing llama_cpp.

    Copies ALL DLLs from nvidia/*/bin/ since we cannot predict exactly
    which ones ggml-cuda.dll will need at load time.
    """
    site_packages = _get_site_packages(python_path)
    if not site_packages:
        return

    llama_lib = site_packages / 'llama_cpp' / 'lib'
    nvidia_dir = site_packages / 'nvidia'

    if not llama_lib.exists() or not nvidia_dir.exists():
        return

    # Copy ALL DLLs from all nvidia/*/bin/ directories
    # ggml-cuda.dll may need any of them at load time
    cuda_dlls_copied = 0
    for pkg_dir in nvidia_dir.iterdir():
        bin_dir = pkg_dir / 'bin'
        if bin_dir.exists():
            for dll in bin_dir.glob('*.dll'):
                dest = llama_lib / dll.name
                if not dest.exists():
                    try:
                        shutil.copy2(str(dll), str(dest))
                        cuda_dlls_copied += 1
                    except (shutil.Error, OSError):
                        pass

    if cuda_dlls_copied > 0:
        print(f"  {PREFIX} [OK] Copied {cuda_dlls_copied} CUDA DLLs to llama_cpp/lib/")


def _get_site_packages(python_path: str) -> Path | None:
    """Get the site-packages directory for the given Python.

    Uses sys.path instead of site.getsitepackages() because
    embedded Python builds (e.g., Stability Matrix) may return
    incorrect paths from site.getsitepackages().
    """
    try:
        result = subprocess.run(
            [python_path, '-c',
             'import sys; '
             'paths = [p for p in sys.path if "site-packages" in p.lower()]; '
             'print(paths[0] if paths else "")'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            if path:
                return Path(path)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def fix_dll_search_path(python_path: str) -> bool:
    """Fix DLL search path issues on Windows.

    Two known issues:
    1. api-ms-win-crt-*.dll files may be in System32\\downlevel\\ instead
       of directly in System32 (Windows 10 quirk). Copy them to
       llama_cpp/lib/ so the loader finds them.
    2. The CUDA runtime DLLs were already copied by _copy_cuda_dlls().

    Returns:
        True if fix applied or not needed, False on error
    """
    if sys.platform != 'win32':
        return True

    site_packages = _get_site_packages(python_path)
    if not site_packages:
        return False

    llama_lib = site_packages / 'llama_cpp' / 'lib'
    if not llama_lib.exists():
        return False

    # Check if api-ms-win-crt-*.dll exists directly in System32
    system32 = Path(os.environ.get('SystemRoot', 'C:\\Windows')) / 'System32'
    has_crt_in_system32 = len(list(system32.glob('api-ms-win-crt-*.dll'))) > 0

    if not has_crt_in_system32:
        # Try downlevel directory
        downlevel = system32 / 'downlevel'
        if downlevel.exists():
            crt_files = list(downlevel.glob('api-ms-win-crt-*.dll'))
            if crt_files:
                copied = 0
                for dll in crt_files:
                    dest = llama_lib / dll.name
                    if not dest.exists():
                        try:
                            shutil.copy2(str(dll), str(dest))
                            copied += 1
                        except (shutil.Error, OSError):
                            pass
                if copied > 0:
                    print(f"  {PREFIX} [OK] Copied {copied} api-ms-win-crt DLLs to llama_cpp/lib/")
                return True

    return True


# -- Interactive Prompts (Manual mode only) -------------------------------

def ask_for_backend() -> str:
    """Let user choose a backend when auto-detection fails."""
    print("\n  +- Backend Selection ---------------------------------------------------")
    print("  | Could not detect GPU automatically.")
    print("  | Select your backend:")
    print("  |")
    options = [
        ('1', 'cu124', 'NVIDIA CUDA 12.x (recommended for RTX 30/40/50 series)'),
        ('2', 'cu118', 'NVIDIA CUDA 11.x (older GPUs)'),
        ('3', 'cpu',   'CPU only (works on any system, slower)'),
        ('4', 'rocm5', 'AMD ROCm 5.x'),
        ('5', 'vulkan','Vulkan (any GPU, experimental)'),
        ('6', 'metal', 'Apple Silicon / Metal (M1/M2/M3/M4)'),
    ]
    for key, _, label in options:
        print(f"  |   [{key}] {label}")
    print("  +-----------------------------------------------------------------------")

    choices = {key: backend for key, backend, _ in options}
    while True:
        choice = input("\n  Enter number (1-6): ").strip()
        if choice in choices:
            return choices[choice]
        print("  Invalid choice. Enter 1-6.")


def confirm(message: str) -> bool:
    """Ask for user confirmation. Auto-confirms if non-interactive."""
    try:
        response = input(f"\n  {message} [Y/n]: ").strip().lower()
        return response in ('', 'y', 'yes')
    except EOFError:
        # Non-interactive terminal (e.g., run from script, double-click, CI)
        print(f"  {message} [Y/n]: y (non-interactive, auto-confirming)")
        return True


def print_manual_instructions():
    """Print manual installation instructions as fallback."""
    print("""
  +- Manual Installation ----------------------------------------------------
  |
  |  Pick the command for your GPU:
  |
  |  NVIDIA CUDA 12.x (RTX 30/40/50 series):
  |    pip install llama-cpp-python --extra-index-url
  |    https://abetlen.github.io/llama-cpp-python/whl/cu124
  |
  |  NVIDIA CUDA 11.x (older GPUs):
  |    pip install llama-cpp-python --extra-index-url
  |    https://abetlen.github.io/llama-cpp-python/whl/cu118
  |
  |  Apple Silicon / Metal (M1/M2/M3/M4):
  |    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
  |    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
  |
  |  CPU only:
  |    pip install llama-cpp-python --extra-index-url
  |    https://abetlen.github.io/llama-cpp-python/whl/cpu
  |
  |  For ComfyUI Windows Portable, prefix with:
  |    python_embeded\\python.exe -m pip install ...
  |
  +-------------------------------------------------------------------------
  """)


def _show_recovery_menu(
    python_path: str,
    failed_backend: str,
    error_stderr: str = "",
    recovery_attempts: list[dict] | None = None,
) -> bool:
    """Interactive recovery menu with version + backend fallback options.

    Presents 8 retry options including CUDA version fallbacks (cu121),
    older wheel version (v0.3.23), and Vulkan backend. Automatically
    suggests the best first option based on error classification.
    Loops back to the menu if a retry also fails.

    Args:
        python_path: Path to the target Python executable.
        failed_backend: The backend key that just failed (e.g. 'cu118').
        error_stderr: Raw stderr from the failed import (for classification).

    Returns:
        True if the user eventually succeeded with a retry backend,
        False if the user chose to exit.
    """
    # Track recovery attempts for error logging
    attempts = list(recovery_attempts) if recovery_attempts else []

    # Classify error for smart suggestion ordering
    error_type = _classify_import_error(error_stderr) if error_stderr else 'unknown'

    # Build weighted suggestion order based on error type.
    # Order: Vulkan (GPU) > CPU latest > cu118 latest > cu121 latest > old versions
    # v0.3.23 options are at the bottom since wheels may disappear from the index.
    if 'dll_not_found' in error_type or 'crt_missing' in error_type:
        # DLL/CRT issue → try Vulkan first (completely different code path),
        # then old versions (older toolchain), then other CUDA backends
        suggestion_order = ['6', '4', '5', '2', '3', '1']
    elif 'cuda_mismatch' in error_type:
        # CUDA mismatch → try cu121 first, then cu118 old
        suggestion_order = ['3', '4', '2', '6', '5', '1']
    elif 'avx2' in error_type:
        # AVX2 issue → old CPU-only wheel
        suggestion_order = ['5', '1', '4', '2', '3', '6']
    else:
        # Unknown → try Vulkan first (GPU acceleration > CPU-only)
        suggestion_order = ['6', '1', '2', '3', '4', '5']

    backends = {
        '1': ('cpu',    LATEST_VERSION,  'CPU-only (latest v{} )'),
        '2': ('cu118',  LATEST_VERSION,  'CUDA 11.x cu118 (latest v{} )'),
        '3': ('cu121',  LATEST_VERSION,  'CUDA 12.x cu121 (latest v{} )'),
        '4': ('cu118',  '0.3.23',        'CUDA 11.x cu118 OLDER v0.3.23'),
        '5': ('cpu',    '0.3.23',        'CPU-only OLDER v0.3.23'),
        '6': ('vulkan', LATEST_VERSION,  'Vulkan backend (latest v{}, any GPU)'),
    }

    while True:
        print()
        border = "-" * 66
        print(f"  +{border}+")
        print(f"  |  Installation did not complete successfully.                 |")
        print(f"  |  Choose a recovery option:                                  |")
        print(f"  +{'-' * 66}+")
        print()
        print("  [1] CPU-only backend (latest version, guaranteed)")
        print("  [2] CUDA 11.x cu118 index (latest version)")
        print("  [3] CUDA 12.x cu121 index (latest version)")
        print("  [4] CUDA 11.x cu118 index + OLDER v0.3.23 wheel")
        print("  [5] CPU-only + OLDER v0.3.23 wheel")
        print("  [6] Vulkan backend (any GPU, experimental)")
        print("  [7] Show manual installation guide")
        print("  [8] Exit")
        print()
        print(f"  [TIP] Recommended first try: option [{suggestion_order[0]}]")
        print()

        choice = input("  Enter number (1-8): ").strip()

        if choice in backends:
            selected_backend, selected_version, label_template = backends[choice]
            label = label_template.format(selected_version)
            print()
            print(f"  {PREFIX} Retrying: {label}")
            print()
            success = install_llama_cpp(
                python_path, selected_backend,
                force=True,
                version=selected_version,
            )
            if success:
                # Install CUDA libraries only for CUDA backends
                if selected_backend.startswith('cu'):
                    install_cuda_libraries(python_path, selected_backend)
                fix_dll_search_path(python_path)
                verified, version, error_stderr, exit_code = verify_installation(python_path)
                _print_version_summary(python_path, selected_backend, version, verified)
                if verified:
                    print()
                    print(f"  {PREFIX} [OK] Recovery succeeded with {label}!")
                    print(f"  {PREFIX} Next step: Restart ComfyUI.")
                    print()
                    return True
                else:
                    print()
                    print(f"  {PREFIX} [FAIL] Verification still failed with {label}.")
                    attempts.append({
                        'backend': selected_backend,
                        'version': selected_version,
                        'outcome': 'verification_failed',
                        'exit_code': exit_code,
                    })
                    _save_error_log(
                        python_path, selected_backend, error_stderr,
                        exit_code=exit_code,
                        recovery_attempts=attempts,
                        label=f"recovery_{len(attempts)}",
                    )
            else:
                print()
                print(f"  {PREFIX} [FAIL] Installation failed with {label}.")
                attempts.append({
                    'backend': selected_backend,
                    'version': selected_version,
                    'outcome': 'install_failed',
                })
            # Loop back to menu
        elif choice == '7':
            print_manual_instructions()
        elif choice == '8':
            print()
            # Note: error log was already saved in main() before the recovery
            # menu appeared, so we don't save it again here.
            print(f"  {PREFIX} Exiting recovery menu.")
            print(f"  {PREFIX} You can re-run with a specific backend:")
            print(f"  {PREFIX}   python install.py --backend cpu")
            print(f"  {PREFIX}   python install.py --backend cu118")
            print(f"  {PREFIX}   python install.py --backend cu121")
            print(f"  {PREFIX}   python install.py --backend vulkan")
            print()
            return False
        else:
            print("  Invalid choice. Enter 1-8.")


# -- Diagnostic / Check Mode (--check) ------------------------------------

def run_diagnostics(python_path: str):
    """Run in diagnostic mode: detect everything, print report, NO installation.

    Args:
        python_path: The Python path to use for diagnostics.
    """
    border = "-" * 66
    print()
    print(f"  +{border}+")
    print(f"  |  [DIAGNOSTIC MODE] --check detected -- no installation will run{' ' * 4}|")
    print(f"  +{'-' * 66}+")
    print()

    # -- Python info --
    print(f"  {PREFIX} -- Python Environment --")
    print(f"  {PREFIX}     Script path:  {Path(__file__).resolve()}")
    print(f"  {PREFIX}     Python:       {python_path}")

    # Get Python version
    try:
        result = subprocess.run(
            [python_path, '--version'],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  {PREFIX}     Version:      {result.stdout.strip()}")
    except Exception as e:
        print(f"  {PREFIX}     Version:      (failed to check: {e})")

    # Check if pip is available
    try:
        result = subprocess.run(
            [python_path, '-m', 'pip', '--version'],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  {PREFIX}     pip:          {result.stdout.strip()}")
    except Exception as e:
        print(f"  {PREFIX}     pip:          (not available: {e})")

    # Check for installed llama-cpp-python
    installed_version = get_installed_version(python_path)
    if installed_version:
        try:
            cv = Version(installed_version)
            mv = Version(MIN_VISION_VERSION)
            status = "[OK]" if cv >= mv else f"[TOO OLD] (need >= {MIN_VISION_VERSION})"
        except Exception:
            status = "[UNKNOWN]"
        print(f"  {PREFIX}     llama-cpp-python: {installed_version} {status}")
    else:
        print(f"  {PREFIX}     llama-cpp-python: [NOT INSTALLED]")

    # -- ComfyUI detection --
    print()
    print(f"  {PREFIX} -- ComfyUI Detection --")
    script_dir = Path(__file__).resolve().parent
    comfy_root = None
    for parent in [script_dir] + list(script_dir.parents):
        if (parent / 'main.py').exists() or (parent / 'ComfyUI_main.py').exists():
            comfy_root = parent
            break
    if comfy_root:
        print(f"  {PREFIX}     ComfyUI root:  {comfy_root}")
        embedded = comfy_root / 'python_embeded' / 'python.exe'
        sm_embedded = comfy_root.parent / 'python_embeded' / 'python.exe'
        venv = comfy_root / 'venv' / 'Scripts' / 'python.exe'
        if embedded.exists():
            print(f"  {PREFIX}     Python type:   Windows Portable (python_embeded)")
            print(f"  {PREFIX}     Python path:   {embedded}")
            if str(python_path) != str(embedded.resolve()):
                print(f"  {PREFIX}     !! MISMATCH: install target is DIFFERENT from ComfyUI's Python!")
        elif sm_embedded.exists():
            print(f"  {PREFIX}     Python type:   Stability Matrix (shared python_embeded)")
            print(f"  {PREFIX}     Python path:   {sm_embedded.resolve()}")
            if str(python_path) != str(sm_embedded.resolve()):
                print(f"  {PREFIX}     !! MISMATCH: install target is DIFFERENT from ComfyUI's Python!")
        elif venv.exists():
            print(f"  {PREFIX}     Python type:   Manual venv")
            print(f"  {PREFIX}     Python path:   {venv}")
            if str(python_path) != str(venv.resolve()):
                print(f"  {PREFIX}     !! MISMATCH: install target is DIFFERENT from ComfyUI's Python!")
        else:
            print(f"  {PREFIX}     Python type:   Using current Python (no embedded/venv found)")
    else:
        print(f"  {PREFIX}     ComfyUI root:  [NOT FOUND] (cannot detect ComfyUI)")

    # -- GPU / Backend detection --
    print()
    print(f"  {PREFIX} -- GPU / Backend Detection --")

    # Apple Silicon
    if detect_apple_silicon():
        print(f"  {PREFIX}     Platform:      Apple Silicon -> Metal backend")
    else:
        # PyTorch probe
        cuda_avail, cuda_ver = detect_cuda_via_torch(python_path)
        if cuda_avail and cuda_ver:
            bk = cuda_version_to_backend(cuda_ver)
            print(f"  {PREFIX}     CUDA (PyTorch): {cuda_ver} -> {BACKEND_LABELS.get(bk, bk) if bk else 'unsupported'}")
        else:
            print(f"  {PREFIX}     CUDA (PyTorch): Not detected (torch not available or no CUDA)")

        # nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"  {PREFIX}     nvidia-smi:    Driver {result.stdout.strip()}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  {PREFIX}     nvidia-smi:    Not available")

        backend = detect_backend()
        print(f"  {PREFIX}     Auto-detected backend: {BACKEND_LABELS.get(backend, str(backend)) if backend else 'Manual selection needed'}")

    # -- requirements.txt check --
    print()
    print(f"  {PREFIX} -- Requirements Check --")
    check_requirements_txt()
    print(f"  {PREFIX}     requirements.txt: No version pin conflicts detected")

    # -- Summary --
    print()
    print(f"  +{border}+")
    print(f"  |  [DIAGNOSTIC COMPLETE] No installation was performed.{' ' * 18}|")
    print(f"  |{' ' * 66}|")
    if installed_version:
        print(f"  |  Currently installed: llama-cpp-python {installed_version}{' ' * 17}|")
    print(f"  |  Run without --check to install.{' ' * 34}|")
    print(f"  +{'-' * 66}+")
    print()


# -- Main ------------------------------------------------------------------

def main():
    """Main entry point for the install helper script."""
    # ── Parse CLI arguments ─────────────────────────────────────────────
    args = parse_args()

    print()
    print("+-----------------------------------------------------------------------+")
    print("|     EasyLLM for ComfyUI -- llama.cpp Install Helper                   |")
    print("+-----------------------------------------------------------------------+")

    if is_manager_mode():
        print(f"  {PREFIX} [Running via ComfyUI Manager -- automatic mode]")
    else:
        print(f"  {PREFIX} [Running manually from terminal]")
    print()

    # ── Friendly warning: close ComfyUI first ─────────────────────────
    if not is_manager_mode():
        print(f"  {'!' * 66}")
        print(f"  !!  IMPORTANT: Please close ComfyUI before continuing.")
        print(f"  !!  If ComfyUI is running, files may be locked and")
        print(f"  !!  installation may fail with permission errors.")
        print(f"  {'!' * 66}")
        print()

    # ── Step 1: Detect Python ────────────────────────────────────────────
    print(f"  {PREFIX} -- Step 1/3: Detecting Python environment --")
    python_path = get_python(cli_python=args.python)
    print(f"  {PREFIX}     Python: {python_path}")
    print()

    # ── Pre-flight: Check disk space before install ─────────────────────
    _check_disk_space(python_path)

    # ── Diagnostic mode (--check) ────────────────────────────────────────
    if args.check:
        run_diagnostics(python_path)
        return

    # ── Step 2: Detect Backend ───────────────────────────────────────────
    print(f"  {PREFIX} -- Step 2/3: Detecting OS / GPU / CUDA --")

    # Check environment variable override first
    backend = get_backend_from_env()
    if backend:
        print(f"  {PREFIX}     Selected backend: {BACKEND_LABELS[backend]}")
        print()
    elif args.backend:
        # CLI argument override
        backend = args.backend
        print(f"  {PREFIX}     Using backend from --backend argument: {BACKEND_LABELS[backend]}")
        print()
    else:
        # 2a: Apple Silicon check (macOS)
        if detect_apple_silicon():
            backend = 'metal'
            print(f"  {PREFIX}     Selected backend: {BACKEND_LABELS[backend]}")
            print()
        else:
            # 2b: PyTorch CUDA probe (more reliable than nvidia-smi)
            cuda_available, cuda_version = detect_cuda_via_torch(python_path)
            if cuda_available and cuda_version:
                backend_key = cuda_version_to_backend(cuda_version)
                if backend_key:
                    backend = backend_key
                    print(
                        f"  {PREFIX}     Mapped CUDA {cuda_version} "
                        f"to backend: {BACKEND_LABELS[backend]}"
                    )
                else:
                    # CUDA version not in our wheel map — fall through to nvidia-smi
                    print(
                        f"  {PREFIX} [INFO] CUDA {cuda_version} not directly "
                        f"supported -- falling back to nvidia-smi detection"
                    )
                    backend = detect_backend()
            else:
                # 2c: Fall back to nvidia-smi / legacy detection
                backend = detect_backend()

            if backend is None:
                backend = ask_for_backend()
            print(f"  {PREFIX}     Selected backend: {BACKEND_LABELS.get(backend, backend)}")
            print()

    # ── Step 3: Install ──────────────────────────────────────────────────
    print(f"  {PREFIX} -- Step 3/3: Installing llama-cpp-python --")

    # Scan requirements.txt for version pins
    check_requirements_txt()

    # In Manager mode, skip confirmation and install silently
    if not is_manager_mode():
        print(f"  {PREFIX}     Target:  {python_path}")
        print(f"  {PREFIX}     Backend: {BACKEND_LABELS.get(backend, backend)}")
        print(f"  {PREFIX}     Package: llama-cpp-python (pre-built wheel)")
        print()

        if not confirm("Proceed with installation?"):
            print(f"\n  {PREFIX} Installation cancelled.\n")
            sys.exit(1)

    # ── Automatic pip self-upgrade ──────────────────────────────────────
    if not args.no_upgrade_pip:
        upgrade_pip(python_path)
        print()

    # ── Resolve version preference ──────────────────────────────────────
    try_latest = args.try_latest or get_try_latest_from_env()

    # ── Main install ─────────────────────────────────────────────────────
    success = install_llama_cpp(
        python_path, backend,
        force=args.force,
        try_latest=try_latest,
    )

    # Track the actual backend used (may have been changed by fallback chain)
    actual_backend = globals().get('_actual_backend', backend)

    if success:
        # Post-install: install CUDA 12 libraries if needed
        install_cuda_libraries(python_path, actual_backend)

        # Post-install: fix DLL search path issues (Windows only)
        fix_dll_search_path(python_path)

        verified, version, error_stderr, exit_code = verify_installation(python_path)

        # Post-install: ComfyUI runtime verification
        script_dir = Path(__file__).resolve().parent
        comfy_root = None
        for parent in [script_dir] + list(script_dir.parents):
            if (parent / 'main.py').exists() or (parent / 'ComfyUI_main.py').exists():
                comfy_root = parent
                break
        verify_comfyui_runtime(python_path, comfy_root)

        # Clear version summary
        _print_version_summary(python_path, actual_backend, version, verified)

        if not verified:
            # ── Save error log immediately — before recovery menu ─────────
            # This preserves the root cause even if the user eventually
            # succeeds via a recovery option (e.g. Vulkan).
            _save_error_log(
                python_path, actual_backend, error_stderr,
                exit_code=exit_code,
            )

            print()
            print("+-----------------------------------------------------------------------+")
            print("|    [WARNING] Install command ran but verification failed              |")
            print("+-----------------------------------------------------------------------+")
            print()
            print("  This is unusual. Try running the install again, or see")
            print("  the troubleshooting section in README.md.")
            print()

            # ── Interactive recovery menu (not in Manager mode) ────────────
            if not is_manager_mode():
                recovered = _show_recovery_menu(
                    python_path, actual_backend,
                    error_stderr=error_stderr,
                )
                if recovered:
                    # User succeeded via recovery — exit cleanly
                    pass
                else:
                    # User chose to exit recovery menu — log already saved above
                    if not is_manager_mode():
                        input("  Press Enter to exit...")
                    sys.exit(1)
            else:
                # Manager mode: no interactivity, just exit with code
                sys.exit(1)
    else:
        print()
        print("+-----------------------------------------------------------------------+")
        print("|                  [FAIL] INSTALLATION FAILED                            |")
        print("+-----------------------------------------------------------------------+")
        print()
        print("  Possible causes:")
        print("  1. No internet connection")
        print("  2. Pre-built wheel not available for your platform")
        print("  3. Python environment issue")
        print()
        print("  Next steps:")
        print("  * Check your internet and try again")
        print("  * If you have Visual Studio Build Tools, compile from source:")
        print("      set CMAKE_ARGS=-DGGML_CUDA=on")
        print("      pip install llama-cpp-python")
        print()

        # ── Interactive recovery menu (not in Manager mode) ────────────────
        if not is_manager_mode():
            recovered = _show_recovery_menu(python_path, backend)
            if recovered:
                # User succeeded via recovery — exit cleanly
                pass
            else:
                # User chose to exit recovery menu
                if not is_manager_mode():
                    input("  Press Enter to exit...")
                sys.exit(1)
        else:
            sys.exit(1)

    # ── Terminal pause for non-Manager mode ─────────────────────────────
    # Keeps the window open so the user can read/copy the final output.
    if not is_manager_mode():
        input("\n  Press Enter to exit...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Installation cancelled.\n")
        input("  Press Enter to exit...")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [FAIL] Unexpected error: {e}")
        print("  Please report this issue with the error details above.")
        input("  Press Enter to exit...")
        sys.exit(1)
