# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""system.py — read the GPU's VRAM facts (the crash wall) via NVML, with nvidia-smi as fallback."""
from __future__ import annotations

import types

import pytest

from wcx_suite import system


@pytest.fixture(autouse=True)
def _no_pynvml(monkeypatch):
    """Default for these tests: NVML unavailable, so ``read_limits()`` exercises the nvidia-smi
    fallback (matching a box without ``nvidia-ml-py``). Native tests override this with a fake."""
    def _raise():
        raise ModuleNotFoundError("pynvml not installed")
    monkeypatch.setattr(system, "_pynvml", _raise, raising=False)


def _fake_nvml(total_gib, used_gib, free_gib, name="NVIDIA GeForce RTX 2070"):
    """A stand-in pynvml module exposing just the calls _native_limits uses (bytes in, like NVML)."""
    mem = types.SimpleNamespace(total=int(total_gib * 1024 ** 3),
                                used=int(used_gib * 1024 ** 3),
                                free=int(free_gib * 1024 ** 3))
    return types.SimpleNamespace(nvmlDeviceGetMemoryInfo=lambda h: mem,
                                 nvmlDeviceGetName=lambda h: name)


def test_native_limits_via_pynvml(monkeypatch):
    # Native NVML read is the primary path — nvidia-smi must NOT be spawned when it succeeds.
    monkeypatch.setattr(system, "_pynvml", lambda: _fake_nvml(8.0, 1.0, 7.0), raising=False)
    monkeypatch.setattr(system, "_nvml_handle", lambda: object(), raising=False)
    monkeypatch.setattr(system, "_smi_query",
                        lambda: (_ for _ in ()).throw(AssertionError("smi should not run")))
    lim = system.read_limits()
    assert lim.device == "NVIDIA GeForce RTX 2070"
    assert lim.total_gb == 8.0 and lim.wall_gb == 8.0
    assert lim.used_gb == 1.0 and lim.free_gb == 7.0


def test_native_name_decodes_bytes(monkeypatch):
    # nvmlDeviceGetName returns bytes on older pynvml; decode it.
    nv = _fake_nvml(8.0, 1.0, 7.0, name=b"NVIDIA L4")
    monkeypatch.setattr(system, "_pynvml", lambda: nv, raising=False)
    monkeypatch.setattr(system, "_nvml_handle", lambda: object(), raising=False)
    assert system.read_limits().device == "NVIDIA L4"


def test_read_limits_falls_back_to_smi_on_native_error(monkeypatch):
    # _no_pynvml makes the native read raise → nvidia-smi fallback (same value, never less safe).
    monkeypatch.setattr(system, "_smi_query",
                        lambda: "NVIDIA GeForce RTX 2070, 8192, 1024, 7168\n")
    lim = system.read_limits()
    assert lim.device == "NVIDIA GeForce RTX 2070" and lim.total_gb == 8.0


def test_read_limits_falls_back_when_native_implausible(monkeypatch):
    # A native read that returns an implausible total (0) must not be trusted — fall back to smi.
    monkeypatch.setattr(system, "_native_limits",
                        lambda: system.GPULimits("X", 0.0, 0.0, 0.0, 0.0), raising=False)
    monkeypatch.setattr(system, "_smi_query",
                        lambda: "NVIDIA GeForce RTX 2070, 8192, 1024, 7168\n")
    assert system.read_limits().device == "NVIDIA GeForce RTX 2070"


def test_native_matches_nvidia_smi_on_real_gpu():
    # On a real NVIDIA box both sources should agree; skips everywhere else.
    pynvml = pytest.importorskip("pynvml")
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    except Exception:
        pytest.skip("no NVIDIA GPU / NVML here")
    smi = system._read_limits_smi()
    if smi is None:
        pytest.skip("nvidia-smi unavailable")
    assert abs(mem.total / (1024 ** 3) - smi.total_gb) < 0.1


def test_flash_attn_capable_true_when_package_and_ampere(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n: object() if n == "flash_attn" else None)
    monkeypatch.setattr(system, "_pynvml", lambda: types.SimpleNamespace(
        nvmlDeviceGetCudaComputeCapability=lambda h: (8, 6)), raising=False)
    monkeypatch.setattr(system, "_nvml_handle", lambda: object(), raising=False)
    assert system.flash_attn_capable() is True


def test_flash_attn_capable_false_without_package(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: None)
    assert system.flash_attn_capable() is False


def test_flash_attn_capable_false_on_pre_ampere(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: object())
    monkeypatch.setattr(system, "_pynvml", lambda: types.SimpleNamespace(
        nvmlDeviceGetCudaComputeCapability=lambda h: (7, 5)), raising=False)   # Turing (RTX 2070)
    monkeypatch.setattr(system, "_nvml_handle", lambda: object(), raising=False)
    assert system.flash_attn_capable() is False


def test_flash_attn_capable_false_when_nvml_errors(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: object())
    # autouse _no_pynvml makes _pynvml raise → can't read capability → conservatively False
    assert system.flash_attn_capable() is False


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
