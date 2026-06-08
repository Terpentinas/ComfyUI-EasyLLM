# LLM Chat Launcher Files

These launcher scripts provide a simple interactive menu to run the
[`install.py`](../install.py) script with different execution profiles —
no terminal commands needed.

## Main Menu Options

| Option | Description |
|--------|-------------|
| **1 — Normal Mode** | Smart auto-selection. On older CPUs, the safe baseline version is installed for guaranteed compatibility. On modern CPUs, the latest version is used. |
| **2 — Experimental Mode** | Forces the latest llama-cpp-python wheel even on older CPUs. Sets the `LLM_CHAT_TRY_LATEST=true` environment variable before running `install.py`. |
| **3 — Force CUDA 12.x** | Installs with NVIDIA CUDA 12.x backend for modern GPUs (RTX 30/40/50 series). Runs `python install.py --backend cu124 --force`. |
| **4 — Force Vulkan** | Installs with Vulkan backend for older GPUs (GTX 900/1000/1600 series), AMD GPUs, or as a stable fallback when CUDA fails. Runs `python install.py --backend vulkan --force`. |
| **5 — Force CPU-only** | Installs a CPU-only wheel with no GPU acceleration. Compatible with every system. Runs `python install.py --backend cpu --force`. |
| **6 — Exit** | Closes the launcher. |

## Recovery Menu (shown on failure)

If the installation fails, a **recovery menu** appears automatically with
backup options — no need to re-run the launcher:

| Option | Description |
|--------|-------------|
| **1 — Retry with recovery menu** | Re-launches `install.py` which shows its own comprehensive 8-option recovery menu with CPU, CUDA 11/12, Vulkan, older version (v0.3.23), and manual guide options. |
| **2 — Show manual guide** | Prints manual installation instructions you can copy and run yourself. |
| **3 — Exit** | Closes the launcher. |

*Note: The launcher delegates all backend-specific retries to `install.py`'s
built-in recovery menu, which offers 8 options with smart suggestion ordering
based on the error type. This avoids confusing double-menu nesting.*

## Flow Diagram

```
You (double-click launcher)
    │
    ▼
┌──────────────────────────────────┐
│  WARNING: Close ComfyUI first!   │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Main Menu                       │
│  [1] Normal                      │
│  [2] Experimental                │
│  [3] Force CUDA 12.x             │
│  [4] Force Vulkan                │
│  [5] Force CPU-only              │
│  [6] Exit                        │
└──────────┬───────────────────────┘
           │ choice
           ▼
┌──────────────────────────────────────┐
│  Runs: python install.py             │
│  (with --backend + --force for       │
│   options 3, 4, 5)                   │
└──────────┬───────────────────────────┘
           │
           ▼
    ┌──────┴──────┐
    ▼              ▼
┌────────┐  ┌──────────────────────────┐
│ SUCCESS│  │ FAILURE                  │
│ (exit) │  │ Launcher Recovery Menu:  │
│        │  │ [1] Re-launch install.py │
│        │  │     (which shows 8-      │
│        │  │      option recovery)     │
│        │  │ [2] Manual guide         │
│        │  │ [3] Exit                 │
└────────┘  └──────────────────────────┘
```

---

## ⚠ Important: Close ComfyUI First

Before running any installation, **make sure ComfyUI is closed**.
If ComfyUI is running, its Python process may lock the package files,
causing `pip install` to fail with "Permission denied" errors.

All launcher scripts show this warning at startup, and [`install.py`](../install.py)
also prints it before proceeding.

---

## Platform Instructions

### 🪟 Windows — [`run_launcher.bat`](run_launcher.bat)

1. **Double-click** `run_launcher.bat` in File Explorer
2. A terminal window opens showing the interactive menu
3. Press **1**, **2**, or **3** on your keyboard (no Enter needed)
4. The launcher runs `install.py` with your chosen profile
5. If installation fails, the recovery menu appears automatically

> **Note:** Windows may show a SmartScreen warning. Click "More info" → "Run anyway" — the script is safe.

### 🐧 Linux — [`run_launcher.sh`](run_launcher.sh)

1. Open a terminal in this directory
2. Make the script executable (one-time):
   ```bash
   chmod +x run_launcher.sh
   ```
3. Run the launcher:
   ```bash
   ./run_launcher.sh
   ```
4. Type **1**, **2**, or **3** and press Enter
5. If installation fails, the recovery menu appears automatically

> **Tip:** You can create a desktop shortcut pointing to `run_launcher.sh` for one-click access.

### 🍎 macOS — [`run_launcher.command`](run_launcher.command)

1. Open **Terminal.app** and navigate to this directory
2. Make the script executable (one-time):
   ```bash
   chmod +x run_launcher.command
   ```
3. After `chmod +x`, you can **double-click** `run_launcher.command` in Finder
   — macOS automatically opens Terminal.app and runs the script
4. Type **1**, **2**, or **3** and press Enter
5. If installation fails, the recovery menu appears automatically

> **Important:** The `.command` file **must** be made executable with `chmod +x`
> before double-clicking works. This is a one-time setup step.

---

## How It Works

```mermaid
flowchart TD
    A[User launches script] --> B[WARNING: Close ComfyUI first]
    B --> C{Show main menu}

    C -->|1 Normal| D[Run: python install.py]
    C -->|2 Experimental| E[Set LLM_CHAT_TRY_LATEST=true<br>Run: python install.py]
    C -->|3 CUDA 12.x| F[Run: python install.py<br>--backend cu124 --force]
    C -->|4 Vulkan| G[Run: python install.py<br>--backend vulkan --force]
    C -->|5 CPU-only| H[Run: python install.py<br>--backend cpu --force]
    C -->|6 Exit| I[Exit]

    D --> J{Install succeeded?}
    E --> J
    F --> J
    G --> J
    H --> J

    J -->|Yes| K[Show success summary<br>Exit]
    J -->|No| L{Show launcher recovery menu}

    L -->|1 Retry| M[Re-run: python install.py<br>→ shows 8-option recovery]
    L -->|2 Manual| N[Run: python install.py --check]
    L -->|3 Exit| I

    M --> O{Success?}
    O -->|Yes| K
    O -->|No| P[install.py's 8-option menu<br>offers CPU, cu118, cu121,<br>v0.3.23, Vulkan fallbacks]
    P --> Q{User picks option}
    Q -->|Try backend| R[install_llama_cpp with<br>selected backend+version]
    R --> S{Success?}
    S -->|Yes| K
    S -->|No| P

    N --> I
```

The launcher resolves its own directory to find `install.py` in the parent
folder, so it works regardless of where the launcher is double-clicked from.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **"python is not recognized"** (Windows) | Python is not in your PATH. Run `python_embeded\python.exe install.py` directly instead. |
| **"python3: command not found"** (Linux/Mac) | Try `python` instead of `python3`, or install Python 3. |
| **Double-click on Mac just opens a text editor** | Run `chmod +x run_launcher.command` in Terminal first. |
| **Permission denied** (Linux/Mac) | Run `chmod +x run_launcher.sh` (or `.command`) first. |
| **Install keeps failing even with recovery options** | Your system may need a C++ compiler. See the [README](../README.md) for source compilation instructions. |
| **"Access is denied" / "Permission denied" during install** | You didn't close ComfyUI! Close it completely and try again. |
