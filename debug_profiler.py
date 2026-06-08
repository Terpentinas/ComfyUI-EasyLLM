"""
First-Load Smart Debug Profiler for EasyLLM.

Usage:
    from .debug_profiler import profiler
    profiler.begin("phase_name")
    # ... work ...
    profiler.end("phase_name")

Activation:
    - Environment variable: LLM_CHAT_DEBUG=1
    - Or touch file:       .debug in the EasyLLM custom_node directory

Output:
    - Structured table at logging.INFO when first load completes
    - JSON file at llm_chat_debug_profile.json in custom_nodes dir
"""

import json
import logging
import os
import time
from typing import Optional


class StartupProfiler:
    """Singleton profiler for first-load diagnostics."""

    _instance: Optional["StartupProfiler"] = None

    def __new__(cls) -> "StartupProfiler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._enabled = self._check_enabled()
        self.phases: dict[str, float] = {}
        self._start_times: dict[str, float] = {}
        self._sub_phases: dict[str, dict[str, float]] = {}
        self._sub_starts: dict[str, dict[str, float]] = {}
        self._module_times: dict[str, float] = {}
        self._report_printed = False
        self._model_count = 0

        if self._enabled:
            logging.info(
                "[LLM Chat Debug] StartupProfiler enabled "
                "(LLM_CHAT_DEBUG=1 or .debug file present)"
            )

    @staticmethod
    def _check_enabled() -> bool:
        """Check if debug mode is enabled via env var or .debug file."""
        if os.environ.get("LLM_CHAT_DEBUG", "").lower() in ("1", "true", "yes"):
            return True
        debug_file = os.path.join(os.path.dirname(__file__), ".debug")
        if os.path.isfile(debug_file):
            return True
        return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def begin(self, phase: str) -> None:
        if not self._enabled:
            return
        self._start_times[phase] = time.perf_counter()

    def end(self, phase: str) -> None:
        if not self._enabled:
            return
        start = self._start_times.pop(phase, None)
        if start is not None:
            elapsed = time.perf_counter() - start
            self.phases[phase] = elapsed
            if elapsed > 0.1:  # Only log phases >100ms
                logging.info(
                    f"[LLM Chat Debug] Phase '{phase}' took {elapsed*1000:.1f}ms"
                )

    def begin_sub(self, phase: str, sub: str) -> None:
        if not self._enabled:
            return
        if phase not in self._sub_starts:
            self._sub_starts[phase] = {}
        self._sub_starts[phase][sub] = time.perf_counter()

    def end_sub(self, phase: str, sub: str) -> None:
        if not self._enabled:
            return
        start = self._sub_starts.get(phase, {}).get(sub)
        if start is not None:
            elapsed = time.perf_counter() - start
            if phase not in self._sub_phases:
                self._sub_phases[phase] = {}
            self._sub_phases[phase][sub] = elapsed
            if elapsed > 0.05:  # Log sub-phases >50ms
                logging.info(
                    f"[LLM Chat Debug] Phase '{phase}.{sub}' took {elapsed*1000:.1f}ms"
                )

    def record_module_import(self, module_name: str, elapsed: float) -> None:
        if not self._enabled:
            return
        self._module_times[module_name] = elapsed
        if elapsed > 0.01:
            logging.info(
                f"[LLM Chat Debug] Module import '{module_name}' took {elapsed*1000:.1f}ms"
            )

    def set_model_count(self, count: int) -> None:
        self._model_count = count

    def print_report(self) -> None:
        """Print structured report to logging.info()."""
        if not self._enabled or self._report_printed:
            return
        self._report_printed = True

        total = sum(self.phases.values())

        log = logging.info
        log("")
        log("╔══════════════════════════════════════════════════════════════╗")
        log("║     LLM Chat First-Load Debug Report                        ║")
        log("╠══════════════════════════════════════════════════════════════╣")

        if self._model_count > 0:
            log(f"║ Models found: {self._model_count:<51d}║")
            log("╠══════════════════════════════════════════════════════════════╣")

        log("║ Phase                      │ Duration     │ % of Total       ║")
        log("╠════════════════════════════╪══════════════╪══════════════════╣")

        # Module imports
        if self._module_times:
            mod_total = sum(self._module_times.values())
            pct = (mod_total / total * 100) if total > 0 else 0
            log(f"║ Module Import Total         │ {mod_total:>8.2f}s  │ {pct:>5.1f}%            ║")
            for mod_name, mod_time in sorted(self._module_times.items()):
                pct_m = (mod_time / total * 100) if total > 0 else 0
                log(f"║   {mod_name:<27s} │ {mod_time:>8.3f}s  │ {pct_m:>5.1f}%            ║")

        # Top-level phases
        for phase_name in sorted(self.phases.keys()):
            elapsed = self.phases[phase_name]
            pct = (elapsed / total * 100) if total > 0 else 0
            label = phase_name.replace("_", " ").title()
            log(f"║ {label:<26s} │ {elapsed:>8.2f}s  │ {pct:>5.1f}%            ║")

            # Sub-phases
            subs = self._sub_phases.get(phase_name, {})
            for sub_name, sub_time in sorted(subs.items()):
                pct_s = (sub_time / total * 100) if total > 0 else 0
                log(f"║   {sub_name:<24s} │ {sub_time:>8.3f}s  │ {pct_s:>5.1f}%            ║")

        log("╠════════════════════════════╧══════════════╧══════════════════╣")
        log(f"║ Total First-Load Time      │ {total:>8.2f}s                    ║")
        log("╚══════════════════════════════════════════════════════════════╝")
        log("")

        # Save JSON report
        self.save_report()

    def save_report(self) -> None:
        """Save structured data to JSON for later analysis."""
        if not self._enabled:
            return
        report = {
            "version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "model_count": self._model_count,
            "phases": dict(self.phases),
            "sub_phases": dict(self._sub_phases),
            "module_imports": dict(self._module_times),
            "total_seconds": sum(self.phases.values()),
        }
        try:
            report_path = os.path.join(
                os.path.dirname(__file__),
                "llm_chat_debug_profile.json"
            )
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logging.info(
                f"[LLM Chat Debug] Report saved to {report_path}"
            )
        except Exception as e:
            logging.warning(
                f"[LLM Chat Debug] Failed to save report: {e}"
            )


# Module-level singleton
profiler = StartupProfiler()
