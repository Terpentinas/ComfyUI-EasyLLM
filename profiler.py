"""
High-precision generation profiler.

Tracks per-phase durations via time.perf_counter() and produces a
structured performance dashboard showing:
  - Phase breakdown (model_setup, generation_loop, etc.)
  - Time to First Token (TTFT)
  - Active / overall Tokens Per Second (TPS)
  - VRAM allocation state

Usage:
    profiler = GenerationProfiler()
    profiler.begin_phase("model_setup")
    ...
    profiler.end_phase("model_setup")
    profiler.mark_first_token()   # once after first token sampled
    profiler.record_token()       # every generated token
    profiler.print_summary_table()
"""

import logging
import time
from typing import Optional


class GenerationProfiler:
    """Tracks generation phase timings with microsecond precision.

    Wraps GPU-native generation in streaming.generate_text() and
    cuda_optimizations.generate_text_gpu(). All timing uses
    time.perf_counter(). Formatting and VRAM queries only happen
    at print_summary_table() time, not during the generation loop.

    Args:
        model_name: Optional model name/description for the report header.
        max_tokens: Optional max_tokens setting for the report.
    """

    PHASES = [
        "model_setup",
        "embedding_setup",
        "generation_loop",
        "output_transfer",
        "decode_cleanup",
    ]

    def __init__(self, model_name: str = "", max_tokens: int = 0):
        self.model_name = model_name
        self.max_tokens = max_tokens

        # Phase tracking: phase_name -> elapsed seconds
        self.phases: dict[str, float] = {}

        # Internal timing state
        self._start_times: dict[str, float] = {}

        # Token tracking
        self._loop_start: float = 0.0
        self._first_token_time: float = 0.0
        self._has_first_token: bool = False
        self._token_count: int = 0
        self._total_tokens_requested: int = max_tokens

        # Error tracking
        self._errors: list[str] = []

    # ── Phase Timing ──────────────────────────────────────────────────

    def begin_phase(self, name: str) -> None:
        """Start timing a phase.

        Args:
            name: Phase name (e.g. 'model_setup', 'generation_loop').
                  Must be called before the corresponding end_phase().
        """
        self._start_times[name] = time.perf_counter()

        # If this is the generation_loop phase, record loop start
        if name == "generation_loop":
            self._loop_start = time.perf_counter()

    def end_phase(self, name: str) -> None:
        """End timing a phase and store its duration.

        Args:
            name: Phase name matching a previous begin_phase() call.
        """
        start = self._start_times.pop(name, None)
        if start is not None:
            elapsed = time.perf_counter() - start
            self.phases[name] = elapsed

    # ── Per-Token Tracking ────────────────────────────────────────────

    def mark_first_token(self) -> None:
        """Record the moment the first token is produced.

        Must be called exactly once, immediately after the first token
        is sampled in the generation loop.
        """
        if not self._has_first_token:
            self._first_token_time = time.perf_counter()
            self._has_first_token = True

    def record_token(self) -> None:
        """Increment the per-token counter.

        Must be called once per generated token in the generation loop.
        """
        self._token_count += 1

    # ── Error Tracking ────────────────────────────────────────────────

    def record_error(self, message: str) -> None:
        """Record a non-fatal error that occurred during generation.

        Args:
            message: Error description string.
        """
        self._errors.append(message)

    # ── Metrics Computations ──────────────────────────────────────────

    def get_ttft(self) -> float:
        """Time to First Token in seconds.

        Returns:
            float: Seconds from generation_loop start to first token,
                   or 0.0 if not yet measured.
        """
        if not self._has_first_token or self._loop_start == 0.0:
            return 0.0
        return self._first_token_time - self._loop_start

    def get_total_generation_time(self) -> float:
        """Total time of generation_loop phase in seconds.

        Returns:
            float: Seconds, or 0.0 if generation_loop hasn't ended.
        """
        return self.phases.get("generation_loop", 0.0)

    def get_active_tps(self) -> float:
        """Tokens per second during the active generation loop.

        Calculated as: (total_tokens - 1) / (total_time - ttft)
        i.e. excludes the TTFT period to measure active generation rate.

        Returns:
            float: Tokens per second, or 0.0 if not computable.
        """
        total_time = self.get_total_generation_time()
        ttft = self.get_ttft()
        active_time = total_time - ttft
        if active_time <= 0 or self._token_count <= 1:
            return 0.0
        return (self._token_count - 1) / active_time

    def get_overall_tps(self) -> float:
        """Overall tokens per second including TTFT.

        Returns:
            float: Tokens per second over the entire generation_loop.
        """
        total_time = self.get_total_generation_time()
        if total_time <= 0 or self._token_count == 0:
            return 0.0
        return self._token_count / total_time

    def get_vram_info(self) -> dict:
        """Query GPU VRAM allocation state.

        Uses the centralized query_vram_state() from memory_manager to avoid
        duplicating torch.cuda.memory_* calls.

        Returns:
            dict with keys:
                - allocated_mb: Currently allocated CUDA memory in MB
                - reserved_mb: Reserved CUDA memory in MB
                - utilization_pct: Estimated GPU utilization (0.0 if unknown)
        """
        try:
            from .memory_manager import query_vram_state
            state = query_vram_state()
            if state["total_bytes"] > 0:
                return {
                    "allocated_mb": round(state["allocated_bytes"] / (1024 * 1024), 1),
                    "reserved_mb": round(state["reserved_bytes"] / (1024 * 1024), 1),
                    "utilization_pct": self._estimate_utilization(),
                }
        except Exception:
            pass
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "utilization_pct": 0.0}

    @staticmethod
    def _estimate_utilization() -> float:
        """Query GPU utilization via nvidia-smi.

        Falls back to 0.0 if nvidia-smi is unavailable or the query fails.

        Returns:
            float: GPU utilization percentage (0-100).
        """
        try:
            import subprocess
            import sys

            # Query nvidia-smi for GPU utilization
            result = subprocess.run(
                [sys.executable, "-c", """
import subprocess, re
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5
    )
    print(out.stdout.strip().split("\\n")[0])
except Exception:
    print("0")
"""],
                capture_output=True, text=True, timeout=10,
            )
            val = result.stdout.strip()
            if val and val != "0":
                return float(val)
        except Exception:
            pass
        return 0.0

    # ── Summary Table ─────────────────────────────────────────────────

    def print_summary_table(self, logger: Optional[logging.Logger] = None) -> None:
        """Print a structured performance report to the terminal.

        Outputs a clean, box-drawn table with:
        - Phase breakdown with durations and % of total
        - Key metrics (TTFT, TPS, VRAM)
        - Any recorded errors/non-fatal issues

        Args:
            logger: Optional logger instance. If None, uses print().
        """
        total = sum(self.phases.values())
        log = logger.info if logger else print

        # ── Header ──
        log("")
        log("╔══════════════════════════════════════════════════════════╗")
        log("║           LLM Chat Performance Report                    ║")
        log("╠══════════════════════════════════════════════════════════╣")

        if self.model_name:
            log(f"║ Model: {self.model_name:<49s}║")
        if self.max_tokens > 0:
            log(f"║ Max Tokens: {self.max_tokens:<46d}║")
        if self.model_name or self.max_tokens > 0:
            log("╠══════════════════════════════════════════════════════════╣")

        # ── Phase Breakdown ──
        log("║ Phase                    │ Duration     │ % of Total     ║")
        log("╠──────────────────────────┼──────────────┼────────────────╣")

        phase_labels = {
            "model_setup": "Model Setup",
            "embedding_setup": "Embedding Setup",
            "generation_loop": "Generation Loop",
            "output_transfer": "Output Transfer",
            "decode_cleanup": "Decode + Cleanup",
        }

        for phase_name in self.PHASES:
            elapsed = self.phases.get(phase_name, 0.0)
            label = phase_labels.get(phase_name, phase_name.replace("_", " ").title())
            pct = (elapsed / total * 100) if total > 0 else 0.0
            log(
                f"║ {label:<25s} │ {elapsed:>8.2f}s  │ "
                f"{pct:>5.1f}%          ║"
            )

        # ── Metrics ──
        log("╠══════════════════════════╧══════════════╧════════════════╣")
        log("║ Metric                   │ Value                         ║")
        log("╠══════════════════════════╪═══════════════════════════════╣")

        ttft = self.get_ttft()
        tps_active = self.get_active_tps()
        tps_overall = self.get_overall_tps()
        vram = self.get_vram_info()

        log(f"║ Time to First Token      │ {ttft:>8.2f}s                    ║")
        log(f"║ Active Tokens Generated  │ {self._token_count:<28d} ║")
        if tps_active > 0:
            log(f"║ Active Token Rate        │ {tps_active:>8.2f} tok/s               ║")
        log(f"║ Overall Token Rate       │ {tps_overall:>8.2f} tok/s               ║")
        log(f"║ Total Generation Time    │ {total:>8.2f}s                    ║")

        if vram["allocated_mb"] > 0:
            log(f"║ VRAM Allocated           │ {vram['allocated_mb']:>8.1f} MB                 ║")
            log(f"║ VRAM Reserved            │ {vram['reserved_mb']:>8.1f} MB                 ║")
        if vram["utilization_pct"] > 0:
            log(f"║ GPU Utilization          │ {vram['utilization_pct']:>5.1f}%                     ║")

        # ── Errors / Non-Fatal Issues ──
        if self._errors:
            log("╠══════════════════════════╧═══════════════════════════════╣")
            log("║ Issues Detected:                                       ║")
            for err in self._errors:
                log(f"║   • {err:<57s}║")

        log("╚══════════════════════════════════════════════════════════╝")
        log("")
