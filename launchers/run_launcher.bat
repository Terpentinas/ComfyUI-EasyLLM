@echo off
cls
title ComfyUI Easy-LLM Mode Selector

rem ============================================================================
rem  run_launcher.bat — Windows Interactive Launcher for ComfyUI-EasyLLM Install
rem ============================================================================
rem
rem  This batch file provides a simple numbered menu for running the
rem  install.py script with different profiles:
rem
rem    [1] Normal Mode         — Smart auto-selection (safe baseline on old CPUs)
rem    [2] Experimental Mode   — Forces latest wheels (sets LLM_CHAT_TRY_LATEST)
rem    [3] Force CUDA 12.x     — Modern GPUs (RTX 30/40/50 series)
rem    [4] Force Vulkan        — Older GPUs, AMD, any-GPU fallback
rem    [5] Force CPU-only      — No GPU acceleration (compatible with everything)
rem    [6] Exit                — Close this window
rem
rem  If the installation fails (non-zero exit code), a recovery menu appears:
rem
rem    [1] Retry with recovery menu (launches the installer's built-in 8-option
rem        recovery menu with CPU, CUDA 11/12, Vulkan, and version fallbacks)
rem    [2] Show manual installation guide
rem    [3] Exit
rem
rem  The launcher now delegates all backend-specific retries to install.py's
rem  own comprehensive recovery menu. This avoids the confusing double-menu
rem  nesting that existed in previous versions.
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
echo  [3] Force NVIDIA CUDA 12.x Backend (Modern GPUs)
echo  [4] Force Vulkan Backend (Older GPUs / AMD / Fallback)
echo  [5] Force CPU-only Mode (No GPU Acceleration)
echo  [6] Exit Launcher
echo.
echo ===================================================

rem /C specifies the valid keys. /M sets the prompt text.
rem ERRORLEVEL is set to the index of the key pressed (1 = first key).
choice /C:123456 /M "Select an execution profile (1-6):"

rem IMPORTANT: Windows ERRORLEVEL must be checked in DECREASING order!
if errorlevel 6 goto :exit
if errorlevel 5 goto :cpuonly
if errorlevel 4 goto :vulkan
if errorlevel 3 goto :cuda12
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

:cuda12
echo.
echo [Easy-LLM] Forcing NVIDIA CUDA 12.x backend (RTX 30/40/50 series)...
timeout /t 2 >nul
cd /d "%~dp0.."
python install.py --backend cu124 --force
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% NEQ 0 goto :recovery_menu
goto :end

:vulkan
echo.
echo [Easy-LLM] Forcing Vulkan backend (older GPUs / AMD / fallback)...
timeout /t 2 >nul
cd /d "%~dp0.."
python install.py --backend vulkan --force
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% NEQ 0 goto :recovery_menu
goto :end

:cpuonly
echo.
echo [Easy-LLM] Forcing CPU-only mode (no GPU acceleration)...
timeout /t 2 >nul
cd /d "%~dp0.."
python install.py --backend cpu --force
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% NEQ 0 goto :recovery_menu
goto :end

rem ============================================================================
rem  Recovery Menu — shown when install.py exits with a non-zero code
rem
rem  Delegates all backend retries to install.py's own comprehensive 8-option
rem  recovery menu (CPU, CUDA 11/12, Vulkan, version fallbacks, manual guide).
rem  This avoids the confusing double-menu nesting of previous versions.
rem ============================================================================
:recovery_menu
echo.
echo ===================================================
echo   Installation did not complete successfully.
echo   Choose a recovery option:
echo ===================================================
echo.
echo  [1] Retry with recovery menu (launches install.py's built-in
echo      8-option menu with CPU, CUDA 11/12, Vulkan fallbacks)
echo  [2] Show manual installation guide
echo  [3] Exit
echo.
choice /C:123 /M "Select a recovery option (1-3): "

if errorlevel 3 goto :exit
if errorlevel 2 goto :manual
if errorlevel 1 goto :retry

:retry
echo.
echo [Easy-LLM] Re-launching installer with recovery menu...
timeout /t 1 >nul
cd /d "%~dp0.."
python install.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [Easy-LLM] Installation succeeded!
) else (
    echo.
    echo [Easy-LLM] Installer's built-in recovery menu was shown.
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
echo   the ComfyUI-EasyLLM folder:
echo.
echo     python install.py --backend cpu
echo     python install.py --backend cu118
echo     python install.py --backend cu121
echo     python install.py --backend vulkan
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
