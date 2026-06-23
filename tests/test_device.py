# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""device.py — VRAM limits + CUDA-overhead calibration as JSON workers (for ARA's driver)."""
from __future__ import annotations

import json

from wcx_suite import device, probe, system


def _fake_limits(monkeypatch):
    lim = system.GPULimits(device="RTX 2070", total_gb=8.0, wall_gb=8.0,
                           free_gb=6.0, used_gb=2.0)
    monkeypatch.setattr(system, "read_limits", lambda: lim)


def test_limits_returns_engine_facts(monkeypatch):
    _fake_limits(monkeypatch)
    m = device.limits(margin_gb=1.0)
    assert m["device"] == "RTX 2070"
    assert m["total_gb"] == 8.0 and m["wall_gb"] == 8.0
    assert m["safe_budget_gb"] == 7.0          # wall 8 − margin 1
    assert m["margin_gb"] == 1.0
    assert m["headroom_gb"] == 5.0             # safe 7 − used 2 (live display VRAM)
    assert m["swap_free_gb"] is None           # VRAM has no swap


def test_limits_error_when_no_gpu(monkeypatch):
    monkeypatch.setattr(system, "read_limits", lambda: None)
    assert device.limits() == {"error": "no NVIDIA GPU visible to nvidia-smi"}


def test_calibrate_surfaces_measured_and_default(monkeypatch):
    monkeypatch.setattr(probe, "calibrate",
                        lambda: probe.Calibration(device="RTX 2070", overhead_gb=0.8))
    m = device.calibrate(None)
    assert m["device"] == "RTX 2070"
    assert m["measured_overhead_gb"] == 0.8
    assert m["default_overhead_gb"] == device.DEFAULT_OVERHEAD_GB
    assert m["n_points"] == 1


def test_calibrate_ignores_model_arg(monkeypatch):
    # CUDA-context overhead is model-independent; ARA still passes a model in the argv.
    monkeypatch.setattr(probe, "calibrate",
                        lambda: probe.Calibration(device="RTX 2070", overhead_gb=0.8))
    assert device.calibrate("org/whatever")["measured_overhead_gb"] == 0.8


def test_calibrate_error_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(probe, "calibrate", lambda: None)
    assert "error" in device.calibrate(None)


def test_main_limits_prints_one_json_line(monkeypatch, capsys):
    _fake_limits(monkeypatch)
    device.main(["limits"])
    out = capsys.readouterr().out.strip()
    assert json.loads(out)["device"] == "RTX 2070"


def test_main_calibrate_prints_one_json_line(monkeypatch, capsys):
    monkeypatch.setattr(probe, "calibrate",
                        lambda: probe.Calibration(device="RTX 2070", overhead_gb=0.8))
    device.main(["calibrate", "org/model"])
    assert json.loads(capsys.readouterr().out.strip())["measured_overhead_gb"] == 0.8
