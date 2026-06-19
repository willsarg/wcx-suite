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
    # cuBLAS/cuDNN load lazily on first use — but every real model run pays for them,
    # so a tiny matmul forces them in now. Otherwise we'd undercount overhead (unsafe).
    _m = torch.randn(64, 64, device="cuda")
    (_m @ _m).sum().item()
    torch.cuda.synchronize()
    after = _used_mb()
    if before is None or after is None:
        return {"ok": False, "error": "nvidia-smi unavailable"}
    return {
        "ok": True,
        "device": torch.cuda.get_device_name(0),
        "overhead_gb": max(0.0, after - before) / system.MIB_PER_GB,
    }


def characterize(model_id: str, contexts_csv: str) -> dict:
    """Load *model_id* on the GPU and measure total VRAM at each context in *contexts_csv*.

    Probes small→large; stops at the first context that errors/OOMs, keeping the safe points
    below it. Total VRAM is nvidia-smi ``used`` after a synchronized forward — it captures the
    display baseline, the CUDA context, and torch's (retained) reserved pool, so it's the
    conservative number to compare against the wall.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}
    device = torch.cuda.get_device_name(0)
    contexts = [int(c) for c in contexts_csv.split(",") if c]
    try:
        AutoTokenizer.from_pretrained(model_id)   # validate + warm the cache
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16).to("cuda")
        model.eval()
    except Exception as exc:
        return {"ok": False, "error": f"load failed: {type(exc).__name__}: {exc}"}

    points: list[dict] = []
    for ctx in contexts:
        try:
            ids = torch.randint(0, int(model.config.vocab_size), (1, ctx), device="cuda")
            with torch.no_grad():
                model(ids)
            torch.cuda.synchronize()
        except Exception:
            break   # OOM/error at this context — stop, keep the safe points below it
        used = _used_mb()
        if used is None:
            return {"ok": False, "error": "nvidia-smi unavailable"}
        points.append({"context": ctx, "used_gb": used / system.MIB_PER_GB})

    if not points:
        return {"ok": False, "error": "no contexts measured"}
    return {"ok": True, "device": device, "model": model_id, "points": points}


_MODES = {"calibrate": calibrate, "characterize": characterize}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    mode = argv[0] if argv else ""
    fn = _MODES.get(mode)
    if fn is None:
        result: dict = {"ok": False, "error": f"unknown mode: {mode!r}"}
    else:
        try:
            result = fn(*argv[1:])
        except Exception as exc:   # never let a GPU fault escape as a traceback
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
