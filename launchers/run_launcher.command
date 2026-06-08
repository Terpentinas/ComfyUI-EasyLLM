#!/bin/bash
# =============================================================================
#  run_launcher.command — macOS Interactive Launcher for ComfyUI-EasyLLM Install
# =============================================================================
#
#  macOS uses the .command extension so that Finder opens the script in
#  Terminal.app when double-clicked, giving Mac users the same "double-click
#  to run" ease-of-use that Windows users get with a .bat file.
#
#  This script provides a simple numbered menu for running the install.py
#  script with different profiles:
#
#    [1] Normal Mode         — Smart auto-selection (safe baseline on old CPUs)
#    [2] Experimental Mode   — Forces latest wheels (sets LLM_CHAT_TRY_LATEST)
#    [3] Force CUDA 12.x     — Modern GPUs (RTX 30/40/50 series)
#    [4] Force Vulkan        — Older GPUs, AMD, any-GPU fallback
#    [5] Force CPU-only      — No GPU acceleration (compatible with everything)
#    [6] Exit                — Close this terminal
#
#  If the installation fails (non-zero exit code), a recovery menu appears:
#
#    [1] Retry with recovery menu (launches the installer's built-in 8-option
#        recovery menu with CPU, CUDA 11/12, Vulkan, and version fallbacks)
#    [2] Show manual installation guide
#    [3] Exit
#
#  The launcher now delegates all backend-specific retries to install.py's
#  own comprehensive recovery menu. This avoids the confusing double-menu
#  nesting that existed in previous versions.
#
#  Uses the native Bash 'select' construct which automatically renders a
#  numbered list and validates user input. Invalid choices are gracefully
#  re-prompted instead of crashing.
#
#  The script resolves its own directory via $(dirname "$0"), then navigates
#  to the parent to find install.py. This works regardless of where the
#  script is launched from.
#
#  IMPORTANT: This file must be executable for macOS to recognize it as a
#  runnable .command file. In Terminal, run once:
#    chmod +x run_launcher.command
#
#  Usage:
#    Double-click run_launcher.command in Finder, OR run from Terminal:
#    ./run_launcher.command
# =============================================================================

clear
echo "==================================================="
echo "       ComfyUI-Easy-LLM Launcher Mode"
echo "==================================================="
echo ""
echo "  !!! IMPORTANT: Please close ComfyUI before installing."
echo "  !!! If ComfyUI is running, files may be locked"
echo "  !!! and installation may fail."
echo ""
echo "==================================================="
echo ""

# PS3 is the native prompt string used by the 'select' construct
PS3="Select an execution profile (1-6): "

options=(
    "Launch Normal Mode (Smart Auto-Selection)"
    "Launch Experimental Mode (Force Latest Wheels)"
    "Force NVIDIA CUDA 12.x Backend (Modern GPUs)"
    "Force Vulkan Backend (Older GPUs / AMD / Fallback)"
    "Force CPU-only Mode (No GPU Acceleration)"
    "Exit Launcher"
)

select opt in "${options[@]}"
do
    case $opt in
        "Launch Normal Mode (Smart Auto-Selection)")
            echo ""
            echo "[Easy-LLM] Booting with safe automatic fallback tracking..."
            sleep 2
            cd "$(dirname "$0")/.."
            python3 install.py
            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                show_recovery_menu
            fi
            break
            ;;
        "Launch Experimental Mode (Force Latest Wheels)")
            echo ""
            echo "[Easy-LLM] WARNING: Forcing cutting-edge llama-cpp-python packages!"
            export LLM_CHAT_TRY_LATEST=true
            sleep 2
            cd "$(dirname "$0")/.."
            python3 install.py
            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                show_recovery_menu
            fi
            break
            ;;
        "Force NVIDIA CUDA 12.x Backend (Modern GPUs)")
            echo ""
            echo "[Easy-LLM] Forcing NVIDIA CUDA 12.x backend (RTX 30/40/50 series)..."
            sleep 2
            cd "$(dirname "$0")/.."
            python3 install.py --backend cu124 --force
            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                show_recovery_menu
            fi
            break
            ;;
        "Force Vulkan Backend (Older GPUs / AMD / Fallback)")
            echo ""
            echo "[Easy-LLM] Forcing Vulkan backend (older GPUs / AMD / fallback)..."
            sleep 2
            cd "$(dirname "$0")/.."
            python3 install.py --backend vulkan --force
            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                show_recovery_menu
            fi
            break
            ;;
        "Force CPU-only Mode (No GPU Acceleration)")
            echo ""
            echo "[Easy-LLM] Forcing CPU-only mode (no GPU acceleration)..."
            sleep 2
            cd "$(dirname "$0")/.."
            python3 install.py --backend cpu --force
            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                show_recovery_menu
            fi
            break
            ;;
        "Exit Launcher")
            echo "Exiting launcher..."
            break
            ;;
        *)
            echo "Invalid selection. Please choose an option from 1 to 6."
            ;;
    esac
done

# =============================================================================
#  Recovery Menu — shown when install.py exits with a non-zero code
#
#  Delegates all backend retries to install.py's own comprehensive 8-option
#  recovery menu (CPU, CUDA 11/12, Vulkan, version fallbacks, manual guide).
#  This avoids the confusing double-menu nesting of previous versions.
# =============================================================================
function show_recovery_menu() {
    echo ""
    echo "==================================================="
    echo "  Installation did not complete successfully."
    echo "  Choose a recovery option:"
    echo "==================================================="
    echo ""
    PS3="Select a recovery option (1-3): "
    recovery_options=(
        "Retry with installer recovery menu"
        "Show manual installation guide"
        "Exit"
    )
    select recovery_opt in "${recovery_options[@]}"; do
        case $recovery_opt in
            "Retry with installer recovery menu")
                echo ""
                echo "[Easy-LLM] Re-launching installer with recovery menu..."
                sleep 1
                cd "$(dirname "$0")/.."
                python3 install.py
                if [ $? -eq 0 ]; then
                    echo ""
                    echo "[Easy-LLM] Installation succeeded!"
                else
                    echo ""
                    echo "[Easy-LLM] Installer's built-in recovery menu was shown."
                fi
                break
                ;;
            "Show manual installation guide")
                echo ""
                echo "[Easy-LLM] Showing manual installation guide..."
                sleep 1
                cd "$(dirname "$0")/.."
                python3 install.py --check
                echo ""
                echo "==================================================="
                echo "  Manual Installation Guide:"
                echo "==================================================="
                echo ""
                echo "  For GPU-accelerated install, run one of these from"
                echo "  the ComfyUI-EasyLLM folder:"
                echo ""
                echo "    python3 install.py --backend cpu"
                echo "    python3 install.py --backend cu118"
                echo "    python3 install.py --backend cu121"
                echo "    python3 install.py --backend vulkan"
                echo ""
                echo "  Or compile from source (requires build tools):"
                echo "    CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall"
                echo ""
                break
                ;;
            "Exit")
                break
                ;;
            *)
                echo "Invalid selection. Please choose an option from 1 to 3."
                ;;
        esac
    done
}
