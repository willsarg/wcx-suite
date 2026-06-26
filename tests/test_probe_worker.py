# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""probe_worker.py — the subprocess dispatch + JSON contract (GPU modes verified on hardware)."""
from __future__ import annotations

import json
import sys
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


class _FakeIds:
    """Minimal stand-in for a (1, N) token tensor: knows its length, device, and slices."""
    def __init__(self, n): self.n = n; self.device = "cuda"
    @property
    def shape(self): return (1, self.n)
    def __getitem__(self, key):
        sl = key[1]
        return f"ids[{sl.start}:{sl.stop}]"


def _arange_torch():
    return types.SimpleNamespace(arange=lambda s, e, device=None: f"arange({s},{e})",
                                 cuda=types.SimpleNamespace(empty_cache=lambda: None))


class _CacheModel:
    """Records each forward; returns logits + threads a growing cache via .past_key_values."""
    def __init__(self): self.calls = []
    def __call__(self, ids, past_key_values=None, cache_position=None,
                 use_cache=None, logits_to_keep=None):
        self.calls.append({"ids": ids, "cache_position": cache_position,
                           "cache_in": past_key_values, "logits_to_keep": logits_to_keep})
        return types.SimpleNamespace(logits=f"logits@{ids}",
                                     past_key_values=f"{past_key_values}+{ids}")


def test_chunked_prefill_feeds_one_chunk_per_forward_with_cache_position():
    model = _CacheModel()
    _cache, _last = probe_worker._chunked_prefill(
        model, _FakeIds(2500), _arange_torch(), chunk=1000, cache="C0")
    # 2500 over chunk=1000 → exactly three forwards, the last a short tail
    assert [c["ids"] for c in model.calls] == ["ids[0:1000]", "ids[1000:2000]", "ids[2000:2500]"]
    assert [c["cache_position"] for c in model.calls] == [
        "arange(0,1000)", "arange(1000,2000)", "arange(2000,2500)"]


def test_chunked_prefill_returns_last_logits_and_threaded_cache():
    model = _CacheModel()
    cache, last = probe_worker._chunked_prefill(
        model, _FakeIds(2500), _arange_torch(), chunk=1000, cache="C0")
    assert last == "logits@ids[2000:2500]"                  # only the final chunk's logits
    assert cache == "C0+ids[0:1000]+ids[1000:2000]+ids[2000:2500]"   # cache threaded through all


def test_chunked_prefill_empties_cache_after_each_chunk():
    # Releasing the per-chunk transient scratch keeps torch's reserved pool ~= the live working set,
    # so it never bloats past dedicated VRAM and spills to shared RAM (the ~4x-slowdown cliff).
    model = _CacheModel()
    empties = []
    torch = types.SimpleNamespace(
        arange=lambda s, e, device=None: f"arange({s},{e})",
        cuda=types.SimpleNamespace(empty_cache=lambda: empties.append(1)))
    probe_worker._chunked_prefill(model, _FakeIds(2500), torch, chunk=1000, cache="C0")
    assert len(empties) == 3            # 2500/1000 → 3 chunks, one empty_cache per chunk


def test_chunked_prefill_single_pass_when_prompt_fits_one_chunk():
    model = _CacheModel()
    probe_worker._chunked_prefill(model, _FakeIds(400), _arange_torch(), chunk=512, cache="C0")
    assert len(model.calls) == 1                            # ≤ chunk ⇒ exactly one forward (= today)
    assert model.calls[0]["ids"] == "ids[0:400]"
    assert model.calls[0]["logits_to_keep"] == 1


def _fake_cache_utils(monkeypatch):
    cu = types.SimpleNamespace(DynamicCache=lambda: "DYN",
                               QuantizedCache=lambda **k: ("QC", k))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(cache_utils=cu))
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", cu)
    return cu


def test_new_cache_fp16_builds_dynamic_cache(monkeypatch):
    _fake_cache_utils(monkeypatch)
    model = types.SimpleNamespace(config="CFG")
    assert probe_worker._new_cache(None, _fake_torch(), model) == "DYN"


def test_new_cache_quantized_builds_hqq_quantized_cache(monkeypatch):
    _fake_cache_utils(monkeypatch)
    model = types.SimpleNamespace(config="CFG")
    kind, kw = probe_worker._new_cache(4, _fake_torch(), model)
    assert kind == "QC"
    # transformers 5.x QuantizedCache(backend, config, nbits, q_group_size) — config is the model's
    assert kw["backend"] == "hqq" and kw["config"] == "CFG"
    assert kw["nbits"] == 4 and kw["q_group_size"] == probe_worker.models.KV_GROUP_SIZE


def test_fill_kv_chunk_routes_fp16_through_chunked_prefill(monkeypatch):
    seen = {}
    monkeypatch.setattr(probe_worker, "_new_cache", lambda kv_bits, torch, model: f"cache(kv={kv_bits})")
    monkeypatch.setattr(probe_worker, "_chunked_prefill",
                        lambda model, ids, torch, *, chunk, cache:
                        seen.update(chunk=chunk, cache=cache) or ("C", "L"))

    class M:
        def __call__(self, ids): seen["forward"] = True
        def generate(self, *a, **k): seen["generate"] = True

    probe_worker._fill_kv(M(), "IDS", None, _fake_torch(), chunk=512)
    assert seen["chunk"] == 512 and seen["cache"] == "cache(kv=None)"
    assert "forward" not in seen and "generate" not in seen   # chunked, never single-shot


def test_fill_kv_chunk_quantized_falls_back_to_single_shot_on_incompat(monkeypatch):
    calls = []
    monkeypatch.setattr(probe_worker, "_new_cache", lambda kv_bits, torch, model: "QCACHE")

    def boom(*a, **k): raise NotImplementedError("HQQ cache can't take a multi-token update")
    monkeypatch.setattr(probe_worker, "_chunked_prefill", boom)

    class M:
        def __call__(self, ids): calls.append(("forward", ids))
        def generate(self, ids, max_new_tokens=None, do_sample=None, **kw):
            calls.append(("generate", max_new_tokens, kw.get("cache_implementation")))

    probe_worker._fill_kv(M(), "IDS", 4, _fake_torch(), chunk=512)
    # chunked path raised → fall back to the existing single-shot quantized generate, not a crash
    assert calls == [("generate", 1, "quantized")]


def test_fill_kv_chunk_does_not_swallow_oom(monkeypatch):
    monkeypatch.setattr(probe_worker, "_new_cache", lambda kv_bits, torch, model: "C")

    def oom(*a, **k): raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(probe_worker, "_chunked_prefill", oom)

    class M:
        def __call__(self, ids): raise AssertionError("must not fall back on OOM")
        def generate(self, *a, **k): raise AssertionError("must not fall back on OOM")

    try:
        probe_worker._fill_kv(M(), "IDS", None, _fake_torch(), chunk=512)
        raise AssertionError("OOM should propagate, not be swallowed")
    except RuntimeError as e:
        assert "out of memory" in str(e)


def _mock_gpu_for_orchestrator(monkeypatch):
    """Mock the heavy GPU/transformers deps so characterize/measure run host-side; returns the dict
    that records what `_fill_kv` was handed (the measure↔run lockstep seam)."""
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True,
                                   get_device_name=lambda i: "RTX 2070"),
        randint=lambda lo, hi, shape, device=None: "IDS")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda mid: None)))
    monkeypatch.setattr(probe_worker, "_load_model",
                        lambda mid, torch, **k: types.SimpleNamespace(
                            config=types.SimpleNamespace(vocab_size=32000)))
    monkeypatch.setattr(probe_worker, "_used_mb", lambda: 4096.0)
    seen = {}
    monkeypatch.setattr(probe_worker, "_fill_kv",
                        lambda model, ids, kv_bits, torch, **k: seen.update(k))
    return seen


def test_characterize_threads_chunk_into_fill_kv(monkeypatch):
    seen = _mock_gpu_for_orchestrator(monkeypatch)
    out = probe_worker.characterize("org/m", "2000", chunk=256)
    assert out["ok"] is True and seen.get("chunk") == 256   # chunked measurement, certified at 256


def test_measure_threads_chunk_into_fill_kv(monkeypatch):
    seen = _mock_gpu_for_orchestrator(monkeypatch)
    monkeypatch.setattr(probe_worker, "_used_gb", lambda: 4.0)
    monkeypatch.setattr(probe_worker, "_start_watchdog", lambda abort: None)
    out = probe_worker.measure("org/m", "2000", "7.0", chunk=256)
    assert out["ok"] is True and seen.get("chunk") == 256   # run will prefill the same way


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


def test_main_strips_weight_quant_value_into_kwarg(monkeypatch, capsys):
    seen = {}
    monkeypatch.setitem(probe_worker._MODES, "fake",
                        lambda *a, **k: seen.update(args=a, kw=k) or {"ok": True})
    probe_worker.main(["fake", "model", "--weight-quant", "int8", "2000"])
    assert seen["kw"] == {"weight_quant": "int8"} and seen["args"] == ("model", "2000")


def test_load_model_device_map_for_quantized_else_to_cuda(monkeypatch):
    seen = {}

    class FakeModel:
        def eval(self): return self
        def to(self, d): seen["to"] = d; return self

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda mid, **k: (seen.update(kw=k) or FakeModel()))))
    monkeypatch.setattr(probe_worker.models, "is_prequantized", lambda hf: False)
    monkeypatch.setattr(probe_worker.models, "attn_implementation", lambda pf: "sdpa")
    monkeypatch.setattr(probe_worker.models, "weight_quant_kwargs",
                        lambda wq, prequantized=False: {"quantization_config": "BNB"} if wq != "none" else {})
    ft = types.SimpleNamespace(float16="f16")
    probe_worker._load_model("m", ft, weight_quant="int4")            # quantized → device_map, no .to
    assert seen["kw"].get("device_map") == "cuda" and "to" not in seen
    seen.clear()
    probe_worker._load_model("m", ft, weight_quant="none")            # fp16 → .to("cuda")
    assert seen.get("to") == "cuda" and "device_map" not in seen["kw"]


def test_main_strips_prefill_chunk_value_into_int_kwarg(monkeypatch, capsys):
    seen = {}
    monkeypatch.setitem(probe_worker._MODES, "fake",
                        lambda *a, **k: seen.update(args=a, kw=k) or {"ok": True})
    probe_worker.main(["fake", "model", "--prefill-chunk", "256", "2000"])
    assert seen["kw"] == {"chunk": 256} and seen["args"] == ("model", "2000")


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
