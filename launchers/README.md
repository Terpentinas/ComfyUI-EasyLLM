# LLM Chat Launcher Files

These launcher scripts provide a simple interactive menu to run the
[`install.py`](../install.py) script with different execution profiles —
no terminal commands needed.

## Menu Options

| Option | Description |
|--------|-------------|
| **1 — Normal Mode** | Smart auto-selection. On older CPUs, the safe baseline version is installed for guaranteed compatibility. On modern CPUs, the latest version is used. |
| **2 — Experimental Mode** | Forces the latest llama-cpp-python wheel even on older CPUs. Sets the `LLM_CHAT_TRY_LATEST=true` environment variable before running `install.py`. |
| **3 — Exit** | Closes the launcher. |

---

## Platform Instructions

### 🪟 Windows — [`run_launcher.bat`](run_launcher.bat)

1. **Double-click** `run_launcher.bat` in File Explorer
2. A terminal window opens showing the interactive menu
3. Press **1**, **2**, or **3** on your keyboard (no Enter needed)
4. The launcher runs `install.py` with your chosen profile

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

> **Important:** The `.command` file **must** be made executable with `chmod +x`
> before double-clicking works. This is a one-time setup step.

---

## How It Works

```
You (double-click launcher)
    │
    ▼
┌──────────────────────────────┐
│  Interactive Menu            │
│  [1] Normal                  │
│  [2] Experimental            │
│  [3] Exit                    │
└──────────┬───────────────────┘
           │ choice
           ▼
┌──────────────────────────────┐
│  Sets LLM_CHAT_TRY_LATEST    │
│  (only for option 2)         │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Runs: python install.py     │
│  (or python3 on Linux/Mac)   │
└──────────────────────────────┘
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
