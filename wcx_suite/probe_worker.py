"""The process that actually touches the GPU — run in isolation so a crash/OOM kills it,
not the suite. Emits exactly one JSON line on stdout.

    python -m wcx_suite.probe_worker calibrate

torch is imported lazily inside each mode, so the module loads without the (heavy,
NVIDIA-only) ``[cuda]`` extras installed — only running a GPU mode needs them.
"""
from __future__ import annotations

import json
import sys

from . import system


def _used_mb() -> float | None:
    """VRAM in use right now, in MiB (via nvidia-smi), or None."""
    lim = system.read_limits()
    return None if lim is None else lim.used_gb * system.MIB_PER_GB


def calibrate() -> dict:
    """Fixed VRAM the CUDA context itself costs — the nvidia-smi delta across torch CUDA
    init. Every model pays this before its own weights. Tiny allocation; safe."""
    import torch

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}
    before = _used_mb()
    torch.cuda.init()
    torch.zeros(1, device="cuda")      # force the context to materialize
    torch.cuda.synchronize()
    after = _used_mb()
    if before is None or after is None:
        return {"ok": False, "error": "nvidia-smi unavailable"}
    return {
        "ok": True,
        "device": torch.cuda.get_device_name(0),
        "overhead_gb": max(0.0, after - before) / system.MIB_PER_GB,
    }


_MODES = {"calibrate": calibrate}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    mode = argv[0] if argv else ""
    fn = _MODES.get(mode)
    if fn is None:
        result: dict = {"ok": False, "error": f"unknown mode: {mode!r}"}
    else:
        try:
            result = fn()
        except Exception as exc:   # never let a GPU fault escape as a traceback
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
