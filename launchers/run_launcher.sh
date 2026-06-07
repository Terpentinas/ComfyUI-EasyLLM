#!/bin/bash
# =============================================================================
#  run_launcher.sh — Linux Interactive Launcher for LLM Chat Install
# =============================================================================
#
#  This script provides a simple numbered menu for running the install.py
#  script with different profiles:
#
#    [1] Normal Mode      — Smart auto-selection (safe baseline on old CPUs)
#    [2] Experimental Mode — Forces latest wheels (sets LLM_CHAT_TRY_LATEST)
#    [3] Exit             — Close this terminal
#
#  If the installation fails (non-zero exit code), a recovery menu appears:
#
#    [1] Retry with CPU-only backend  (guaranteed to work, slower)
#    [2] Retry with CUDA 11.x backend (for older GPUs like GTX 1660)
#    [3] Show manual installation guide
#    [4] Exit
#
#  Uses the native Bash 'select' construct which automatically renders a
#  numbered list and validates user input. Invalid choices are gracefully
#  re-prompted instead of crashing.
#
#  The script resolves its own directory via $(dirname "$0"), then navigates
#  to the parent to find install.py. This works regardless of where the
#  script is launched from.
#
#  Usage:
#    chmod +x run_launcher.sh
#    ./run_launcher.sh
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
PS3="Select an execution profile (1-3): "

options=(
    "Launch Normal Mode (Smart Auto-Selection)"
    "Launch Experimental Mode (Force Latest Wheels)"
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
        "Exit Launcher")
            echo "Exiting launcher..."
            break
            ;;
        *)
            echo "Invalid selection. Please choose an option from 1 to 3."
            ;;
    esac
done

# =============================================================================
#  Recovery Menu — shown when install.py exits with a non-zero code
# =============================================================================
function show_recovery_menu() {
    echo ""
    echo "==================================================="
    echo "  Installation did not complete successfully."
    echo "  Choose a recovery option:"
    echo "==================================================="
    echo ""
    PS3="Select a recovery option (1-4): "
    recovery_options=(
        "Retry with CPU-only backend (guaranteed to work)"
        "Retry with CUDA 11.x backend (older GPUs)"
        "Show manual installation guide"
        "Exit"
    )
    select recovery_opt in "${recovery_options[@]}"; do
        case $recovery_opt in
            "Retry with CPU-only backend (guaranteed to work)")
                echo ""
                echo "[Easy-LLM] Retrying with CPU-only backend..."
                sleep 1
                cd "$(dirname "$0")/.."
                python3 install.py --backend cpu --force
                if [ $? -eq 0 ]; then
                    echo ""
                    echo "[Easy-LLM] CPU-only installation succeeded!"
                else
                    echo ""
                    echo "[Easy-LLM] CPU-only installation also failed."
                    echo "See the manual installation guide for next steps."
                fi
                break
                ;;
            "Retry with CUDA 11.x backend (older GPUs)")
                echo ""
                echo "[Easy-LLM] Retrying with CUDA 11.x backend (older GPUs)..."
                sleep 1
                cd "$(dirname "$0")/.."
                python3 install.py --backend cu118 --force
                if [ $? -eq 0 ]; then
                    echo ""
                    echo "[Easy-LLM] CUDA 11.x installation succeeded!"
                else
                    echo ""
                    echo "[Easy-LLM] CUDA 11.x installation also failed."
                    echo "See the manual installation guide for next steps."
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
                echo "  the llm-chat folder:"
                echo ""
                echo "    python3 install.py --backend cpu"
                echo "    python3 install.py --backend cu118"
                echo "    python3 install.py --backend cu124"
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
                echo "Invalid selection. Please choose an option from 1 to 4."
                ;;
        esac
    done
}
