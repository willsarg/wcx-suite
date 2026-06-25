# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""measure_one.py — the engine-side single-context primitive: no-load preflight + gated run."""
from __future__ import annotations

import json

from wcx_suite import measure_one, models, system


def _info(**over):
    base = dict(hf_id="org/m", weights_gb=0.3, n_layers=30, growing_layers=30,
                kv_heads=3, head_dim=64, hidden_size=576, max_context=8192,
                can_quantize_kv=True, is_causal=True)
    base.update(over)
    return models.ModelInfo(**base)


def _limits(used_gb=2.0):
    return system.GPULimits(device="RTX 2070", total_gb=8.0, wall_gb=8.0,
                            free_gb=8.0 - used_gb, used_gb=used_gb)


def _patch(monkeypatch, info=None, limits=None):
    monkeypatch.setattr(models, "describe", lambda hf: info if info is not None else _info())
    monkeypatch.setattr(system, "read_limits", lambda: limits if limits is not None else _limits())


# ---- preflight ----------------------------------------------------------------------------

def test_preflight_returns_estimate(monkeypatch):
    _patch(monkeypatch)
    est = measure_one.preflight("org/m", margin_gb=1.0, overhead_gb=0.6)
    assert est["ref_baseline_gb"] == 2.0                    # live VRAM in use now
    assert est["base_gb"] == round(2.0 + 0.3 + 0.6, 4)      # baseline + weights + overhead
    assert est["budget_gb"] == 7.0                          # wall 8 − margin 1
    assert est["max_context"] == 8192
    assert est["slope_gb_per_k"] > 0


def test_preflight_error_when_not_cached(monkeypatch):
    _patch(monkeypatch, info=None)
    monkeypatch.setattr(models, "describe", lambda hf: None)
    assert "error" in measure_one.preflight("missing/m", margin_gb=1.0, overhead_gb=0.6)


def test_preflight_error_when_not_causal(monkeypatch):
    _patch(monkeypatch, info=_info(is_causal=False))
    assert "error" in measure_one.preflight("org/m", margin_gb=1.0, overhead_gb=0.6)


def test_preflight_error_when_no_gpu(monkeypatch):
    monkeypatch.setattr(models, "describe", lambda hf: _info())
    monkeypatch.setattr(system, "read_limits", lambda: None)
    assert "error" in measure_one.preflight("org/m", margin_gb=1.0, overhead_gb=0.6)


# ---- run: L4 refuse-before-load ----------------------------------------------------------

def test_run_refuses_when_base_alone_exceeds_budget(monkeypatch):
    # weights 7 + overhead 0.6 + baseline 2 ≫ budget 7 → base estimate refuses, never spawns
    _patch(monkeypatch, info=_info(weights_gb=7.0))
    spawned = []
    monkeypatch.setattr(measure_one, "_spawn_worker", lambda *a, **k: spawned.append(a) or {})
    r = measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6)
    assert r["refused"] is True and "won't load" in r["reason"]
    assert spawned == []


def test_run_refuses_when_predicted_at_ctx_exceeds_budget(monkeypatch):
    # base fits, but a steep slope pushes the predicted footprint at ctx over budget
    _patch(monkeypatch, info=_info(weights_gb=3.0, head_dim=4096))
    spawned = []
    monkeypatch.setattr(measure_one, "_spawn_worker", lambda *a, **k: spawned.append(a) or {})
    r = measure_one.run("org/m", 8000, margin_gb=1.0, overhead_gb=0.6)
    assert r["refused"] is True and "predicted" in r["reason"]
    assert spawned == []


# ---- run: safe path measures the DELTA ---------------------------------------------------

def test_run_measures_delta_over_launch_baseline(monkeypatch):
    _patch(monkeypatch)
    # worker reports absolute used 4.0 over a launch baseline of 2.0 → model delta 2.0
    monkeypatch.setattr(measure_one, "_spawn_worker",
                        lambda hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none": {"status": "ok", "used_gb": 4.0,
                                                   "baseline_gb": 2.0})
    r = measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6, repeats=1)
    assert r == {"context": 2000, "mem_gb": 2.0}


def test_run_passes_safe_budget_as_abort_limit(monkeypatch):
    _patch(monkeypatch)
    seen = {}

    def spy(hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none"):
        seen["abort_gb"] = abort_gb
        return {"status": "ok", "used_gb": 3.0, "baseline_gb": 2.0}

    monkeypatch.setattr(measure_one, "_spawn_worker", spy)
    measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6, repeats=1)
    assert seen["abort_gb"] == 7.0          # the L5 watchdog gets the absolute safe budget


def test_run_median_over_repeats(monkeypatch):
    _patch(monkeypatch)
    deltas = iter([2.0, 5.0, 2.2])          # median delta = 2.2
    monkeypatch.setattr(measure_one, "_spawn_worker",
                        lambda hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none": {"status": "ok",
                                                   "used_gb": 2.0 + next(deltas), "baseline_gb": 2.0})
    r = measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6, repeats=3)
    assert r["mem_gb"] == 2.2


def test_run_refused_when_worker_not_ok(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(measure_one, "_spawn_worker",
                        lambda hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none": {"status": "error", "note": "OOM"})
    r = measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6)
    assert r["refused"] is True and "OOM" in r["reason"]


def test_run_refused_when_model_missing(monkeypatch):
    monkeypatch.setattr(models, "describe", lambda hf: None)
    r = measure_one.run("missing/m", 2000, margin_gb=1.0, overhead_gb=0.6)
    assert r["refused"] is True


# ---- weight-quant lever -------------------------------------------------------------------

def test_model_base_scales_with_weight_quant(monkeypatch):
    monkeypatch.setattr(models, "is_prequantized", lambda hf: False)
    info = _info(weights_gb=4.0)
    full = measure_one._model_base_gb(info, 0.6, "none")
    q4 = measure_one._model_base_gb(info, 0.6, "int4")
    assert q4 < full and q4 == 4.0 * models.weight_bytes_factor("int4") + 0.6


def test_model_base_prequantized_ignores_weight_quant(monkeypatch):
    monkeypatch.setattr(models, "is_prequantized", lambda hf: True)
    info = _info(weights_gb=4.0)
    # on-disk weights already small → factor 1.0 regardless of the flag
    assert measure_one._model_base_gb(info, 0.6, "int4") == measure_one._model_base_gb(info, 0.6, "none")


def test_run_threads_weight_quant_to_worker(monkeypatch):
    _patch(monkeypatch)
    seen = {}

    def spy(hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none"):
        seen["wq"] = weight_quant
        return {"status": "ok", "used_gb": 4.0, "baseline_gb": 2.0}

    monkeypatch.setattr(measure_one, "_spawn_worker", spy)
    measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6, repeats=1, weight_quant="int4")
    assert seen["wq"] == "int4"


def test_main_threads_weight_quant(monkeypatch):
    _patch(monkeypatch)
    seen = {}
    monkeypatch.setattr(measure_one, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "mem_gb": 1.0})
    measure_one.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6",
                      "--weight-quant", "int8"])
    assert seen["weight_quant"] == "int8"


# ---- KV-quant lever -----------------------------------------------------------------------

def test_effective_kv_bits_forced_fp16_when_not_quantizable():
    assert measure_one._effective_kv_bits(_info(can_quantize_kv=True), 4) == 4
    assert measure_one._effective_kv_bits(_info(can_quantize_kv=False), 4) is None
    assert measure_one._effective_kv_bits(_info(can_quantize_kv=True), None) is None


def test_run_measures_at_requested_kv_bits(monkeypatch):
    _patch(monkeypatch)                                     # default model can quantize
    seen = {}

    def spy(hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none"):
        seen["kv_bits"] = kv_bits
        return {"status": "ok", "used_gb": 4.0, "baseline_gb": 2.0}

    monkeypatch.setattr(measure_one, "_spawn_worker", spy)
    measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6, repeats=1, kv_bits=4)
    assert seen["kv_bits"] == 4            # certified at the precision run will use


def test_run_forces_fp16_for_non_quantizable_model(monkeypatch):
    _patch(monkeypatch, info=_info(can_quantize_kv=False))
    seen = {}

    def spy(hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none"):
        seen["kv_bits"] = kv_bits
        return {"status": "ok", "used_gb": 4.0, "baseline_gb": 2.0}

    monkeypatch.setattr(measure_one, "_spawn_worker", spy)
    measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6, repeats=1, kv_bits=4)
    assert seen["kv_bits"] is None        # sliding-window cache → never quantized


def test_gate_slope_is_kv_aware(monkeypatch):
    _patch(monkeypatch)
    info, limits = _info(), _limits()
    # quant lowers the predicted footprint, so a context that the fp16 gate refuses can pass at q4
    fp16 = measure_one.safety_gate(info, limits, 200000, margin_gb=1.0, overhead_gb=0.6,
                                   live_base=2.0, kv_bits=None)
    q4 = measure_one.safety_gate(info, limits, 200000, margin_gb=1.0, overhead_gb=0.6,
                                 live_base=2.0, kv_bits=4)
    assert fp16 is not None and q4 is None


def test_preflight_slope_reflects_kv_bits(monkeypatch):
    _patch(monkeypatch)
    f16 = measure_one.preflight("org/m", margin_gb=1.0, overhead_gb=0.6)["slope_gb_per_k"]
    q4 = measure_one.preflight("org/m", margin_gb=1.0, overhead_gb=0.6, kv_bits=4)["slope_gb_per_k"]
    assert q4 < f16


def test_run_threads_prefer_flash_to_worker(monkeypatch):
    _patch(monkeypatch)
    seen = {}

    def spy(hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none"):
        seen["prefer_flash"] = prefer_flash
        return {"status": "ok", "used_gb": 4.0, "baseline_gb": 2.0}

    monkeypatch.setattr(measure_one, "_spawn_worker", spy)
    measure_one.run("org/m", 2000, margin_gb=1.0, overhead_gb=0.6, repeats=1, prefer_flash=True)
    assert seen["prefer_flash"] is True


def test_main_threads_flash_attn(monkeypatch):
    _patch(monkeypatch)
    seen = {}
    monkeypatch.setattr(measure_one, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "mem_gb": 1.0})
    measure_one.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6", "--flash-attn"])
    assert seen["prefer_flash"] is True


def test_main_threads_kv_bits(monkeypatch):
    _patch(monkeypatch)
    seen = {}
    monkeypatch.setattr(measure_one, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "mem_gb": 1.0})
    measure_one.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6", "--kv-bits", "8"])
    assert seen["kv_bits"] == 8


# ---- main ---------------------------------------------------------------------------------

def test_main_preflight_prints_estimate(monkeypatch, capsys):
    _patch(monkeypatch)
    measure_one.main(["org/m", "0", "--margin", "1.0", "--overhead", "0.6", "--preflight"])
    assert "base_gb" in json.loads(capsys.readouterr().out.strip())


def test_main_measure_prints_result(monkeypatch, capsys):
    _patch(monkeypatch)
    monkeypatch.setattr(measure_one, "_spawn_worker",
                        lambda hf, ctx, abort_gb, kv_bits=None, prefer_flash=False, weight_quant="none": {"status": "ok", "used_gb": 4.0, "baseline_gb": 2.0})
    measure_one.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6", "--repeats", "1"])
    assert json.loads(capsys.readouterr().out.strip())["context"] == 2000
