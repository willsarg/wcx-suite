# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""probe_worker.py — the subprocess dispatch + JSON contract (GPU modes verified on hardware)."""
from __future__ import annotations

import json
import types

from wcx_suite import probe_worker


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _fake_torch():
    return types.SimpleNamespace(no_grad=_NullCtx,
                                 cuda=types.SimpleNamespace(synchronize=lambda: None))


def test_fill_kv_fp16_uses_plain_forward():
    calls = []

    class M:
        def __call__(self, ids): calls.append(("forward", ids))
        def generate(self, *a, **k): calls.append(("generate", a, k))

    probe_worker._fill_kv(M(), "IDS", None, _fake_torch())
    assert calls == [("forward", "IDS")]            # fp16 → a plain forward, never generate


def test_fill_kv_quant_uses_generate_with_hqq_cache():
    calls = []

    class M:
        def __call__(self, ids): calls.append(("forward", ids))
        def generate(self, ids, max_new_tokens=None, do_sample=None, **kw):
            calls.append(("generate", max_new_tokens, kw))

    probe_worker._fill_kv(M(), "IDS", 4, _fake_torch())
    assert calls[0][0] == "generate" and calls[0][1] == 1
    assert calls[0][2]["cache_implementation"] == "quantized"
    assert calls[0][2]["cache_config"]["nbits"] == 4


def test_worker_unknown_mode(capsys):
    rc = probe_worker.main(["bogus"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["ok"] is False and "unknown mode" in out["error"]


def test_worker_no_mode(capsys):
    rc = probe_worker.main([])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["ok"] is False


def test_main_strips_flash_attn_into_prefer_flash_kwarg(monkeypatch, capsys):
    seen = {}
    monkeypatch.setitem(probe_worker._MODES, "fake",
                        lambda *a, **k: seen.update(args=a, kw=k) or {"ok": True})
    probe_worker.main(["fake", "model", "--flash-attn", "2000"])
    # --flash-attn becomes a kwarg; the remaining args stay positional (never shifts kv_bits)
    assert seen["kw"] == {"prefer_flash": True} and seen["args"] == ("model", "2000")


def test_main_no_flash_attn_means_no_prefer_kwarg(monkeypatch, capsys):
    seen = {}
    monkeypatch.setitem(probe_worker._MODES, "fake",
                        lambda *a, **k: seen.update(kw=k) or {"ok": True})
    probe_worker.main(["fake", "model"])
    assert seen["kw"] == {}


def test_worker_runs_mode_and_emits_json(monkeypatch, capsys):
    monkeypatch.setitem(probe_worker._MODES, "fake", lambda: {"ok": True, "x": 1})
    rc = probe_worker.main(["fake"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out == {"ok": True, "x": 1}


def test_worker_catches_mode_exception(monkeypatch, capsys):
    def boom():
        raise RuntimeError("cuda exploded")
    monkeypatch.setitem(probe_worker._MODES, "boom", boom)
    rc = probe_worker.main(["boom"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["ok"] is False and "cuda exploded" in out["error"]
