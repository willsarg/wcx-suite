# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""probe.py — orchestrate the GPU worker subprocess (the heavy lifting is isolated there)."""
from __future__ import annotations

import json
import types

from wcx_suite import probe


def test_calibrate_returns_overhead(monkeypatch):
    monkeypatch.setattr(probe, "_run_worker",
                        lambda args, timeout=120: {"ok": True, "device": "RTX 2070",
                                                   "overhead_gb": 0.42})
    cal = probe.calibrate()
    assert cal is not None
    assert cal.device == "RTX 2070"
    assert cal.overhead_gb == 0.42


def test_calibrate_none_when_worker_fails(monkeypatch):
    monkeypatch.setattr(probe, "_run_worker", lambda args, timeout=120: None)
    assert probe.calibrate() is None


def test_calibrate_none_when_worker_not_ok(monkeypatch):
    monkeypatch.setattr(probe, "_run_worker",
                        lambda args, timeout=120: {"ok": False, "error": "CUDA not available"})
    assert probe.calibrate() is None


def test_characterize_fits_and_extrapolates(monkeypatch):
    seen = {}

    def fake_worker(args, timeout=600):
        seen["args"] = args
        return {"ok": True, "device": "RTX 2070",
                "points": [{"context": 512, "used_gb": 2.0},
                           {"context": 2048, "used_gb": 2.6}]}

    monkeypatch.setattr(probe, "_run_worker", fake_worker)
    c = probe.characterize("smol-model", budget_gb=7.0, contexts=(512, 2048))
    assert c is not None
    assert c.device == "RTX 2070" and c.model == "smol-model"
    assert c.points == [(512, 2.0), (2048, 2.6)]
    assert c.safe_context in (13311, 13312)
    assert seen["args"] == ["characterize", "smol-model", "512,2048"]


def test_characterize_none_on_worker_failure(monkeypatch):
    monkeypatch.setattr(probe, "_run_worker", lambda args, timeout=600: {"ok": False, "error": "OOM"})
    assert probe.characterize("smol", budget_gb=7.0) is None


def test_characterize_none_when_no_output(monkeypatch):
    monkeypatch.setattr(probe, "_run_worker", lambda args, timeout=600: None)
    assert probe.characterize("smol", budget_gb=7.0) is None


def _capture_args(monkeypatch, result):
    seen = {}
    monkeypatch.setattr(probe, "_run_worker",
                        lambda args, timeout=600: seen.update(args=args) or result)
    return seen


def test_measure_once_passes_kv_bits_positionally(monkeypatch):
    seen = _capture_args(monkeypatch, {"ok": True, "used_gb": 4.0, "baseline_gb": 2.0})
    probe.measure_once("m", 2000, abort_gb=7.0, kv_bits=4)
    assert seen["args"] == ["measure", "m", "2000", "7.0", "4"]


def test_measure_once_omits_kv_bits_for_fp16(monkeypatch):
    seen = _capture_args(monkeypatch, {"ok": True, "used_gb": 4.0, "baseline_gb": 2.0})
    probe.measure_once("m", 2000, abort_gb=7.0)
    assert seen["args"] == ["measure", "m", "2000", "7.0"]


def test_measure_once_appends_flash_attn(monkeypatch):
    seen = _capture_args(monkeypatch, {"ok": True, "used_gb": 4.0, "baseline_gb": 2.0})
    probe.measure_once("m", 2000, abort_gb=7.0, prefer_flash=True)
    assert "--flash-attn" in seen["args"]
    seen2 = _capture_args(monkeypatch, {"ok": True, "used_gb": 4.0, "baseline_gb": 2.0})
    probe.measure_once("m", 2000, abort_gb=7.0)
    assert "--flash-attn" not in seen2["args"]          # SDPA default


def test_measure_once_appends_weight_quant(monkeypatch):
    seen = _capture_args(monkeypatch, {"ok": True, "used_gb": 4.0, "baseline_gb": 2.0})
    probe.measure_once("m", 2000, abort_gb=7.0, weight_quant="int4")
    assert seen["args"][-2:] == ["--weight-quant", "int4"]
    seen2 = _capture_args(monkeypatch, {"ok": True, "used_gb": 4.0, "baseline_gb": 2.0})
    probe.measure_once("m", 2000, abort_gb=7.0)            # default none → omitted
    assert "--weight-quant" not in seen2["args"]


def test_characterize_appends_kv_bits(monkeypatch):
    seen = _capture_args(monkeypatch, {"ok": True, "device": "X",
                                       "points": [{"context": 512, "used_gb": 2.0},
                                                  {"context": 2048, "used_gb": 2.6}]})
    probe.characterize("smol", budget_gb=7.0, contexts=(512, 2048), kv_bits=8)
    assert seen["args"] == ["characterize", "smol", "512,2048", "8"]


def test_run_worker_parses_json_line(monkeypatch):
    def fake_run(cmd, capture_output, text, timeout):
        return types.SimpleNamespace(stdout='noise\n{"ok": true, "overhead_gb": 0.5}\n',
                                     stderr="", returncode=0)
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    assert probe._run_worker(["calibrate"]) == {"ok": True, "overhead_gb": 0.5}


def test_run_worker_none_on_empty_output(monkeypatch):
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="", stderr="", returncode=1))
    assert probe._run_worker(["calibrate"]) is None


def test_run_worker_none_on_bad_json(monkeypatch):
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="not json", stderr="", returncode=0))
    assert probe._run_worker(["calibrate"]) is None


def test_run_worker_none_on_subprocess_error(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no python")
    monkeypatch.setattr(probe.subprocess, "run", boom)
    assert probe._run_worker(["calibrate"]) is None
