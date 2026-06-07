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
rem  If the installation fails (non-zero exit code), a recovery menu appears:
rem
rem    [1] Retry with CPU-only backend  (guaranteed to work, slower)
rem    [2] Retry with CUDA 11.x backend (for older GPUs like GTX 1660)
rem    [3] Show manual installation guide
rem    [4] Exit
rem
rem  Uses the native CHOICE command so the user just presses a number key
rem  (no Enter key needed, and invalid keys are automatically blocked).
rem
rem  The script resolves its own directory via %~dp0, then navigates to the
rem  parent to find install.py. This works regardless of where the launcher
rem  is double-clicked from.
rem
rem ============================================================================

:menu
cls
echo ===================================================
echo       ComfyUI-Easy-LLM Launcher Mode
echo ===================================================
echo.
echo  !!! IMPORTANT: Please close ComfyUI before installing.
echo  !!! If ComfyUI is running, files may be locked
echo  !!! and installation may fail.
echo.
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
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% NEQ 0 goto :recovery_menu
goto :end

:experimental
echo.
echo [Easy-LLM] WARNING: Forcing cutting-edge llama-cpp-python packages!
set LLM_CHAT_TRY_LATEST=true
timeout /t 2 >nul
cd /d "%~dp0.."
python install.py
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% NEQ 0 goto :recovery_menu
goto :end

rem ============================================================================
rem  Recovery Menu — shown when install.py exits with a non-zero code
rem ============================================================================
:recovery_menu
echo.
echo ===================================================
echo   Installation did not complete successfully.
echo   Choose a recovery option:
echo ===================================================
echo.
echo  [1] Retry with CPU-only backend (guaranteed to work)
echo  [2] Retry with CUDA 11.x backend (older GPUs)
echo  [3] Show manual installation guide
echo  [4] Exit
echo.
choice /C:1234 /M "Select a recovery option (1-4): "

if errorlevel 4 goto :exit
if errorlevel 3 goto :manual
if errorlevel 2 goto :retry_cu118
if errorlevel 1 goto :retry_cpu

:retry_cpu
echo.
echo [Easy-LLM] Retrying with CPU-only backend...
timeout /t 1 >nul
cd /d "%~dp0.."
python install.py --backend cpu --force
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [Easy-LLM] CPU-only installation succeeded!
) else (
    echo.
    echo [Easy-LLM] CPU-only installation also failed.
    echo See the manual installation guide for next steps.
    timeout /t 3 >nul
)
goto :end

:retry_cu118
echo.
echo [Easy-LLM] Retrying with CUDA 11.x backend (older GPUs)...
timeout /t 1 >nul
cd /d "%~dp0.."
python install.py --backend cu118 --force
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [Easy-LLM] CUDA 11.x installation succeeded!
) else (
    echo.
    echo [Easy-LLM] CUDA 11.x installation also failed.
    echo See the manual installation guide for next steps.
    timeout /t 3 >nul
)
goto :end

:manual
echo.
echo [Easy-LLM] Showing manual installation guide...
timeout /t 1 >nul
cd /d "%~dp0.."
python install.py --check
echo.
echo ===================================================
echo   Manual Installation Guide:
echo ===================================================
echo.
echo   For GPU-accelerated install, run one of these from
echo   the llm-chat folder:
echo.
echo     python install.py --backend cpu
echo     python install.py --backend cu118
echo     python install.py --backend cu124
echo.
echo   Or compile from source (requires Visual Studio):
echo     set CMAKE_ARGS=-DGGML_CUDA=on
echo     pip install llama-cpp-python --force-reinstall
echo.
goto :end

:exit
echo Exiting launcher...

:end
echo.
pause
