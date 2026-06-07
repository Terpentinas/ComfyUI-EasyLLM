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
            break
            ;;
        "Launch Experimental Mode (Force Latest Wheels)")
            echo ""
            echo "[Easy-LLM] WARNING: Forcing cutting-edge llama-cpp-python packages!"
            export LLM_CHAT_TRY_LATEST=true
            sleep 2
            cd "$(dirname "$0")/.."
            python3 install.py
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
