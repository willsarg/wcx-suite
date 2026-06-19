"""Safe GPU probes: spawn the worker subprocess, parse its one-line JSON, never crash.

All real GPU work lives in ``probe_worker`` (a separate process) so that an out-of-memory
or a driver fault kills the worker, not the suite. This module only orchestrates and reads
back results — it never imports torch or touches CUDA itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    device: str
    overhead_gb: float    # fixed VRAM the CUDA context costs before any model weights


def _run_worker(args: list[str], timeout: float = 120) -> dict | None:
    """Run the probe worker in a subprocess; return its parsed JSON, or None on any failure."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "wcx_suite.probe_worker", *args],
            capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return None


def calibrate(timeout: float = 120) -> Calibration | None:
    """Measure the CUDA context's fixed VRAM overhead on this machine, or None if it can't."""
    data = _run_worker(["calibrate"], timeout=timeout)
    if not data or not data.get("ok"):
        return None
    return Calibration(device=data["device"], overhead_gb=float(data["overhead_gb"]))
