@echo off
cls
title ComfyUI Easy-LLM Mode Selector

rem ============================================================================
rem  run_launcher.bat — Windows Interactive Launcher for LLM Chat Install
rem ============================================================================
rem
rem  This batch file provides a simple numbered menu for running the
rem  install.py script with different profiles:
rem
rem    [1] Normal Mode      — Smart auto-selection (safe baseline on old CPUs)
rem    [2] Experimental Mode — Forces latest wheels (sets LLM_CHAT_TRY_LATEST)
rem    [3] Exit             — Close this window
rem
rem  Uses the native CHOICE command so the user just presses 1, 2, or 3
rem  (no Enter key needed, and invalid keys are automatically blocked).
rem
rem  The script resolves its own directory via %~dp0, then navigates to the
rem  parent to find install.py. This works regardless of where the launcher
rem  is double-clicked from.
rem
rem ============================================================================

echo ===================================================
echo       ComfyUI-Easy-LLM Launcher Mode
echo ===================================================
echo.
echo  [1] Launch Normal Mode (Smart Auto-Selection)
echo  [2] Launch Experimental Mode (Force Latest Wheels)
echo  [3] Exit Launcher
echo.
echo ===================================================

rem /C specifies the valid keys. /M sets the prompt text.
rem ERRORLEVEL is set to the index of the key pressed (1 = first key).
choice /C:123 /M "Select an execution profile (1-3):"

rem IMPORTANT: Windows ERRORLEVEL must be checked in DECREASING order!
if errorlevel 3 goto :exit
if errorlevel 2 goto :experimental
if errorlevel 1 goto :normal

:normal
echo.
echo [Easy-LLM] Booting with safe automatic fallback tracking...
timeout /t 2 >nul
cd /d "%~dp0.."
python install.py
goto :end

:experimental
echo.
echo [Easy-LLM] WARNING: Forcing cutting-edge llama-cpp-python packages!
set LLM_CHAT_TRY_LATEST=true
timeout /t 2 >nul
cd /d "%~dp0.."
python install.py
goto :end

:exit
echo Exiting launcher...

:end
pause
