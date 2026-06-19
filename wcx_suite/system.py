"""GPU memory facts for this machine (NVIDIA / CUDA).

The crash wall is the GPU's **VRAM**. Unlike Apple's unified memory — where crossing the
working-set wall can hard-lock the whole machine — a CUDA over-allocation usually raises an
out-of-memory error and kills the *workload*, not the OS. But on a GPU that also drives the
display, VRAM exhaustion can still freeze the desktop or trip a driver reset (TDR). Same
discipline either way: find each model's safe context ceiling by staying under the VRAM
wall, never probing into it. VRAM is read LIVE per machine and treated as exact.

Barebones reads it via ``nvidia-smi`` (always present with the driver) — zero deps.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .config import DEFAULT_MARGIN_GB

MIB_PER_GB = 1024  # nvidia-smi reports MiB; 1 GiB = 1024 MiB


@dataclass(frozen=True)
class GPULimits:
    device: str
    total_gb: float       # total VRAM on the card
    wall_gb: float        # the crash wall — total VRAM (allocating past free OOMs)
    free_gb: float        # VRAM free right now
    used_gb: float        # VRAM in use right now (display / other processes)

    def safe_threshold_gb(self, margin_gb: float = DEFAULT_MARGIN_GB) -> float:
        """The line we never let predicted peak cross — the wall minus a cushion."""
        return self.wall_gb - margin_gb


def _smi_query() -> str | None:
    """Raw ``nvidia-smi`` VRAM query stdout, or None if it can't be run."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return out.stdout
    except Exception:
        return None


def read_limits() -> GPULimits | None:
    """The first NVIDIA GPU's VRAM facts, or None when there's no CUDA GPU here."""
    out = _smi_query()
    if not out or not out.strip():
        return None
    first = out.strip().splitlines()[0]
    try:
        name, total, used, free = (x.strip() for x in first.split(","))
        total_gb = float(total) / MIB_PER_GB
        return GPULimits(
            device=name,
            total_gb=total_gb,
            wall_gb=total_gb,
            free_gb=float(free) / MIB_PER_GB,
            used_gb=float(used) / MIB_PER_GB,
        )
    except (ValueError, IndexError):
        return None
