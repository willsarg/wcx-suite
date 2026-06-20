"""The L5 VRAM watchdog + measure normalisation — the crash-safety logic that runs without a GPU.

The actual load-and-measure path needs CUDA and is verified on hardware; these tests pin the
safety-critical decisions that don't: refuse-before-probe when no abort limit is given (L5
fail-closed), the breach decision the watchdog acts on, and the worker→driver result shape.
"""
from __future__ import annotations

from wcx_suite import probe, probe_worker


# ---- L5 fail-closed: never probe without an abort limit -----------------------------------

def test_measure_refuses_without_abort_limit():
    # abort_gb omitted → must refuse BEFORE importing torch / touching the GPU
    out = probe_worker.measure("org/m", "2000")
    assert out["ok"] is False and "abort limit" in out["error"]


# ---- the watchdog's breach decision -------------------------------------------------------

def test_poll_abort_fires_on_breach():
    fired = []
    breached = probe_worker._poll_abort(7.0, sampler=lambda: 7.2, on_breach=lambda: fired.append(1))
    assert breached is True and fired == [1]


def test_poll_abort_at_limit_is_a_breach():
    fired = []
    # >= the limit is unsafe (the budget is the line peak must never reach), per RULE #1
    breached = probe_worker._poll_abort(7.0, sampler=lambda: 7.0, on_breach=lambda: fired.append(1))
    assert breached is True and fired == [1]


def test_poll_abort_silent_when_safe():
    fired = []
    breached = probe_worker._poll_abort(7.0, sampler=lambda: 3.0, on_breach=lambda: fired.append(1))
    assert breached is False and fired == []


def test_poll_abort_ignores_unreadable_sampler():
    fired = []
    breached = probe_worker._poll_abort(7.0, sampler=lambda: None, on_breach=lambda: fired.append(1))
    assert breached is False and fired == []


# ---- probe.measure_once: worker {ok} → driver {status} ------------------------------------

def test_measure_once_normalizes_ok(monkeypatch):
    monkeypatch.setattr(probe, "_run_worker",
                        lambda args, timeout=600: {"ok": True, "device": "RTX 2070",
                                                   "used_gb": 4.0, "baseline_gb": 2.0})
    assert probe.measure_once("org/m", 2000, abort_gb=7.0) == {
        "status": "ok", "used_gb": 4.0, "baseline_gb": 2.0}


def test_measure_once_passes_measure_argv(monkeypatch):
    seen = {}
    monkeypatch.setattr(probe, "_run_worker",
                        lambda args, timeout=600: seen.update(args=args) or {"ok": True, "used_gb": 4.0, "baseline_gb": 2.0})
    probe.measure_once("org/m", 2000, abort_gb=7.0)
    assert seen["args"] == ["measure", "org/m", "2000", "7.0"]


def test_measure_once_error_on_worker_failure(monkeypatch):
    monkeypatch.setattr(probe, "_run_worker",
                        lambda args, timeout=600: {"ok": False, "error": "OOM on load"})
    out = probe.measure_once("org/m", 2000, abort_gb=7.0)
    assert out["status"] == "error" and "OOM on load" in out["note"]


def test_measure_once_error_on_no_output(monkeypatch):
    monkeypatch.setattr(probe, "_run_worker", lambda args, timeout=600: None)
    assert probe.measure_once("org/m", 2000, abort_gb=7.0)["status"] == "error"
