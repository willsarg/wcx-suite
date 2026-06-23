# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""system.py — read the GPU's VRAM facts (the crash wall) from nvidia-smi."""
from __future__ import annotations

from wcx_suite import system


def test_read_limits_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(system, "_smi_query",
                        lambda: "NVIDIA GeForce RTX 2070, 8192, 1024, 7168\n")
    lim = system.read_limits()
    assert lim.device == "NVIDIA GeForce RTX 2070"
    assert lim.total_gb == 8.0      # 8192 MiB / 1024
    assert lim.wall_gb == 8.0       # the wall is total VRAM
    assert lim.used_gb == 1.0
    assert lim.free_gb == 7.0


def test_read_limits_none_when_no_gpu(monkeypatch):
    monkeypatch.setattr(system, "_smi_query", lambda: None)
    assert system.read_limits() is None


def test_read_limits_none_on_blank_output(monkeypatch):
    monkeypatch.setattr(system, "_smi_query", lambda: "   \n")
    assert system.read_limits() is None


def test_read_limits_none_on_garbage(monkeypatch):
    monkeypatch.setattr(system, "_smi_query", lambda: "no commas here\n")
    assert system.read_limits() is None


def test_smi_query_returns_none_when_nvidia_smi_absent(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr(system.subprocess, "run", boom)
    assert system._smi_query() is None


def test_safe_threshold_subtracts_margin():
    lim = system.GPULimits(device="x", total_gb=8.0, wall_gb=8.0, free_gb=7.0, used_gb=1.0)
    assert lim.safe_threshold_gb(1.0) == 7.0
    assert lim.safe_threshold_gb() == 7.0   # default margin = 1 GB
