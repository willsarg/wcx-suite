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
