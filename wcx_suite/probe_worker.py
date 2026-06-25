# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The process that actually touches the GPU — run in isolation so a crash/OOM kills it,
not the suite. Emits exactly one JSON line on stdout.

    python -m wcx_suite.probe_worker calibrate
    python -m wcx_suite.probe_worker measure <hf_id> <ctx> <abort_gb>

torch is imported lazily inside each mode, so the module loads without the (heavy,
NVIDIA-only) ``[cuda]`` extras installed — only running a GPU mode needs them.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

from . import system

WATCHDOG_INTERVAL_S = 0.25   # how often the L5 watchdog re-reads live VRAM


def _used_mb() -> float | None:
    """VRAM in use right now, in MiB (via nvidia-smi), or None."""
    lim = system.read_limits()
    return None if lim is None else lim.used_gb * system.MIB_PER_GB


def _used_gb() -> float | None:
    """VRAM in use right now, in GB (via nvidia-smi), or None."""
    mb = _used_mb()
    return None if mb is None else mb / system.MIB_PER_GB


def _poll_abort(abort_gb: float, *, sampler=_used_gb, on_breach=lambda: None) -> bool:
    """One watchdog tick: if live VRAM has reached *abort_gb*, fire *on_breach* and report True.

    ``>=`` is intentional — the budget is the line live VRAM must never reach (RULE #1). An
    unreadable sampler (None) is treated as 'can't confirm a breach', so it never false-fires.
    """
    used = sampler()
    if used is not None and used >= abort_gb:
        on_breach()
        return True
    return False


def _start_watchdog(abort_gb: float, *, interval: float = WATCHDOG_INTERVAL_S) -> None:
    """Spawn a daemon thread that hard-kills this process (``os._exit(3)``) the instant live VRAM
    reaches *abort_gb* — the last line of defence (L5) if the pre-flight gate mis-predicted."""
    def loop() -> None:
        while not _poll_abort(abort_gb, on_breach=lambda: os._exit(3)):
            time.sleep(interval)
    threading.Thread(target=loop, daemon=True).start()


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


def _fill_kv(model, ids, kv_bits, torch) -> None:
    """Populate the KV cache to *ids* length at the requested precision, so the VRAM read reflects
    how it'll actually run. fp16 is a plain forward; quantized goes through ``generate`` (one step)
    so transformers builds the quantized cache via its public ``cache_implementation`` API."""
    from . import models

    with torch.no_grad():
        if kv_bits is None:
            model(ids)
        else:
            model.generate(ids, max_new_tokens=1, do_sample=False,
                           **models.kv_cache_kwargs(kv_bits))
    torch.cuda.synchronize()


def characterize(model_id: str, contexts_csv: str, kv_bits_str: str | None = None) -> dict:
    """Load *model_id* on the GPU and measure total VRAM at each context in *contexts_csv*.

    Probes small→large; stops at the first context that errors/OOMs, keeping the safe points
    below it. Total VRAM is nvidia-smi ``used`` after a synchronized forward — it captures the
    display baseline, the CUDA context, and torch's (retained) reserved pool, so it's the
    conservative number to compare against the wall. *kv_bits_str* (when given) measures with a
    quantized KV cache.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}
    device = torch.cuda.get_device_name(0)
    contexts = [int(c) for c in contexts_csv.split(",") if c]
    kv_bits = int(kv_bits_str) if kv_bits_str is not None else None
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
            _fill_kv(model, ids, kv_bits, torch)
        except Exception:
            break   # OOM/error at this context — stop, keep the safe points below it
        used = _used_mb()
        if used is None:
            return {"ok": False, "error": "nvidia-smi unavailable"}
        points.append({"context": ctx, "used_gb": used / system.MIB_PER_GB})

    if not points:
        return {"ok": False, "error": "no contexts measured"}
    return {"ok": True, "device": device, "model": model_id, "points": points}


def measure(model_id: str, ctx_str: str, abort_gb_str: str | None = None,
            kv_bits_str: str | None = None) -> dict:
    """Load *model_id* on the GPU at one context and return its VRAM delta over the launch baseline.

    Single-context primitive ARA's driver drives (via :func:`probe.measure_once`). Refuses before
    importing torch if no abort limit is given — running without the L5 watchdog would mean
    probing toward the wall with no backstop (RULE #1, fail-closed). Reports ``baseline_gb`` (live
    VRAM before the model loads, in this fresh process) and ``used_gb`` (after a synchronized
    forward); their difference is the model's own footprint, which ARA re-bases on the live wall.
    *kv_bits_str* (when given) measures with a quantized KV cache, so the certified ceiling matches
    how ``run`` will execute.
    """
    if abort_gb_str is None:
        return {"ok": False, "error": "refusing to probe without an abort limit (L5)"}
    ctx, abort_gb = int(ctx_str), float(abort_gb_str)
    kv_bits = int(kv_bits_str) if kv_bits_str is not None else None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}
    baseline = _used_gb()
    if baseline is None:
        return {"ok": False, "error": "nvidia-smi unavailable"}

    _start_watchdog(abort_gb)        # L5 armed before a single byte of weights loads
    device = torch.cuda.get_device_name(0)
    try:
        AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16).to("cuda")
        model.eval()
        ids = torch.randint(0, int(model.config.vocab_size), (1, ctx), device="cuda")
        _fill_kv(model, ids, kv_bits, torch)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    used = _used_gb()
    if used is None:
        return {"ok": False, "error": "nvidia-smi unavailable"}
    return {"ok": True, "device": device, "used_gb": used, "baseline_gb": baseline}


_MODES = {"calibrate": calibrate, "characterize": characterize, "measure": measure}


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
