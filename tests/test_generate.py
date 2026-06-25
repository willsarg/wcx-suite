# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""generate.py — the engine-side one-shot inference verb: refuse-before-load, then generate.

Everything that touches a GPU or downloads weights is mocked, so these run on any box (no CUDA,
no real model). The CUDA sibling of wmx_suite.generate: same contract, torch/transformers instead
of mlx_lm. We gate on the EFFECTIVE context (prompt_tokens + max_tokens, capped at ctx), never the
raw ceiling — the hard-won lesson from the MLX verb.
"""
from __future__ import annotations

import json
import sys
import types

from wcx_suite import generate, measure_one, models, system


def _info(**over):
    base = dict(hf_id="org/m", weights_gb=0.3, n_layers=30, growing_layers=30,
                kv_heads=3, head_dim=64, hidden_size=576, max_context=40960,
                can_quantize_kv=True, is_causal=True)
    base.update(over)
    return models.ModelInfo(**base)


def _limits(used_gb=2.0):
    return system.GPULimits(device="RTX 2070", total_gb=8.0, wall_gb=8.0,
                            free_gb=8.0 - used_gb, used_gb=used_gb)


class _FakeTokenizer:
    """Minimal stand-in for transformers.AutoTokenizer — encode + decode + call + .to-able ids."""

    def __init__(self, n_prompt_tokens):
        self._n = n_prompt_tokens

    @classmethod
    def make(cls, n_prompt_tokens):
        tok = cls(n_prompt_tokens)
        return types.SimpleNamespace(from_pretrained=lambda hf_id: tok)

    def encode(self, prompt):
        return list(range(self._n))

    def __call__(self, prompt, return_tensors=None):
        # transformers returns a BatchEncoding with input_ids; mimic the .to(device) + indexing.
        ids = list(range(self._n))
        return types.SimpleNamespace(
            input_ids=_FakeIds([ids]),
            to=lambda device: types.SimpleNamespace(input_ids=_FakeIds([ids])),
        )

    def decode(self, ids, skip_special_tokens=True):
        return "generated text"


class _FakeIds:
    """Stand-in for a torch tensor of token ids: knows its shape and supports slicing."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, item):
        return _FakeIds([self.rows[0]])

    def to(self, device):
        return self


def _patch_engine(monkeypatch, *, info=None, limits=None, n_prompt_tokens=6,
                  gate_reason=None, generated="generated text"):
    """Mock the engine collaborators + torch/transformers so nothing real is loaded."""
    monkeypatch.setattr(models, "describe", lambda hf: info if info is not None else _info())
    monkeypatch.setattr(system, "read_limits",
                        lambda: limits if limits is not None else _limits())

    gate_seen = {}

    def fake_gate(info_, limits_, ctx_, *, margin_gb, overhead_gb, live_base, kv_bits=None, weight_quant="none"):
        gate_seen["ctx"] = ctx_
        gate_seen["margin_gb"] = margin_gb
        gate_seen["overhead_gb"] = overhead_gb
        gate_seen["live_base"] = live_base
        gate_seen["kv_bits"] = kv_bits
        return gate_reason

    monkeypatch.setattr(measure_one, "safety_gate", fake_gate)

    # ---- fake torch + transformers via sys.modules (lazy-imported inside generate) ----------
    fake_torch = types.SimpleNamespace(
        float16="float16",
        cuda=types.SimpleNamespace(is_available=lambda: True,
                                   get_device_name=lambda i: "RTX 2070"),
        no_grad=_nullctx,
    )

    class _FakeModel:
        config = types.SimpleNamespace(vocab_size=1000)

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, input_ids=None, max_new_tokens=None, **kw):
            gate_seen["max_new_tokens"] = max_new_tokens
            gate_seen["gen_kw"] = kw
            prompt_len = input_ids.shape[1]
            # return prompt tokens followed by max_new_tokens new ones
            return _FakeIds([list(range(prompt_len + max_new_tokens))])

    tok_obj = _FakeTokenizer(n_prompt_tokens)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda hf_id: tok_obj),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return gate_seen


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---- happy path ---------------------------------------------------------------------------

def test_generate_happy_path(monkeypatch):
    _patch_engine(monkeypatch)
    r = generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256)
    assert r["context"] == 40960
    assert r["completion"] == "generated text"
    assert "refused" not in r


# ---- KV-quant lever -----------------------------------------------------------------------

def test_run_quantizes_kv_when_requested(monkeypatch):
    seen = _patch_engine(monkeypatch)               # default model can quantize
    generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                 prompt="hi", kv_bits=4)
    assert seen["kv_bits"] == 4                      # the gate saw the effective precision
    assert seen["gen_kw"].get("cache_implementation") == "quantized"
    assert seen["gen_kw"]["cache_config"]["nbits"] == 4


def test_run_forces_fp16_for_non_quantizable(monkeypatch):
    seen = _patch_engine(monkeypatch, info=_info(can_quantize_kv=False))
    generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                 prompt="hi", kv_bits=4)
    assert seen["kv_bits"] is None                   # sliding-window → fp16 for gate AND generate
    assert "cache_implementation" not in seen["gen_kw"]


def test_main_threads_kv_bits(monkeypatch):
    seen = {}
    monkeypatch.setattr(generate, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "completion": "x"})
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(read=lambda: "hi"))
    generate.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6",
                   "--max-tokens", "8", "--kv-bits", "8"])
    assert seen["kv_bits"] == 8


def test_run_threads_prefer_flash_to_generate(monkeypatch):
    _patch_engine(monkeypatch)
    seen = {}
    monkeypatch.setattr(generate, "_generate",
                        lambda hf, p, mt, kv=None, pf=False, wq="none": seen.update(pf=pf) or "out")
    generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                 prompt="hi", prefer_flash=True)
    assert seen["pf"] is True


def test_main_threads_flash_attn(monkeypatch):
    seen = {}
    monkeypatch.setattr(generate, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "completion": "x"})
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(read=lambda: "hi"))
    generate.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6",
                   "--max-tokens", "8", "--flash-attn"])
    assert seen["prefer_flash"] is True


def test_run_threads_weight_quant_to_generate(monkeypatch):
    _patch_engine(monkeypatch)
    seen = {}
    monkeypatch.setattr(generate, "_generate",
                        lambda hf, p, mt, kv=None, pf=False, wq="none": seen.update(wq=wq) or "out")
    generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                 prompt="hi", weight_quant="int4")
    assert seen["wq"] == "int4"


def test_main_threads_weight_quant(monkeypatch):
    seen = {}
    monkeypatch.setattr(generate, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "completion": "x"})
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(read=lambda: "hi"))
    generate.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6",
                   "--max-tokens", "8", "--weight-quant", "int8"])
    assert seen["weight_quant"] == "int8"


# ---- refusing safety gate -----------------------------------------------------------------

def test_generate_refused_by_safety_gate(monkeypatch):
    _patch_engine(monkeypatch, gate_reason="predicted 9.00GB at 262 tok >= safe budget 7.00GB")
    r = generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256)
    assert r["refused"] is True
    assert r["context"] == 40960            # refusal reports the ceiling, not the effective ctx
    assert "safe budget" in r["reason"]


# ---- no GPU -------------------------------------------------------------------------------

def test_generate_refused_when_no_gpu(monkeypatch):
    monkeypatch.setattr(models, "describe", lambda hf: _info())
    monkeypatch.setattr(system, "read_limits", lambda: None)
    r = generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256)
    assert r["refused"] is True
    assert "NVIDIA GPU" in r["reason"]


def test_generate_refused_when_model_missing(monkeypatch):
    monkeypatch.setattr(models, "describe", lambda hf: None)
    r = generate.run("missing/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256)
    assert r["refused"] is True


def test_generate_refused_when_not_causal(monkeypatch):
    monkeypatch.setattr(models, "describe", lambda hf: _info(is_causal=False))
    monkeypatch.setattr(system, "read_limits", lambda: _limits())
    r = generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256)
    assert r["refused"] is True


# ---- gates on EFFECTIVE context, not the raw ceiling --------------------------------------

def test_generate_gates_on_effective_context(monkeypatch):
    # prompt 6 tokens + 256 max_tokens = 262 effective; ceiling 40960 must NOT reach the gate.
    seen = _patch_engine(monkeypatch, n_prompt_tokens=6)
    generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256, prompt="hi")
    assert seen["ctx"] == 262
    assert seen["ctx"] != 40960


def test_generate_gates_capped_at_ceiling(monkeypatch):
    # prompt 6 + max_tokens 100000 would exceed the ceiling → gate receives ctx (40960).
    seen = _patch_engine(monkeypatch, n_prompt_tokens=6)
    generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=100000, prompt="hi")
    assert seen["ctx"] == 40960


def test_generate_passes_margin_overhead_and_max_tokens(monkeypatch):
    seen = _patch_engine(monkeypatch, n_prompt_tokens=6)
    generate.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256, prompt="hi")
    assert seen["margin_gb"] == 1.0
    assert seen["overhead_gb"] == 0.6
    assert seen["live_base"] == 2.0            # mirrors measure_one.run: live_base = used_gb
    assert seen["max_new_tokens"] == 256       # model.generate got max_new_tokens=max_tokens


# ---- main: prompt from stdin, one JSON line -----------------------------------------------

def test_main_reads_prompt_from_stdin_and_prints_one_json_line(monkeypatch, capsys):
    _patch_engine(monkeypatch)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(read=lambda: "hello world"))
    generate.main(["org/m", "40960", "--margin", "1.0", "--overhead", "0.6",
                   "--max-tokens", "256"])
    out = capsys.readouterr().out.strip()
    assert out.count("\n") == 0               # exactly one line
    parsed = json.loads(out)
    assert parsed["context"] == 40960
    assert parsed["completion"] == "generated text"


def test_main_empty_prompt_short_circuits_token_count(monkeypatch, capsys):
    # Empty prompt → 0 prompt tokens, no tokenizer import needed for the count.
    seen = _patch_engine(monkeypatch, n_prompt_tokens=6)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(read=lambda: ""))
    generate.main(["org/m", "40960", "--margin", "1.0", "--overhead", "0.6",
                   "--max-tokens", "256"])
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    # effective ctx = min(40960, 0 + 256) = 256
    assert seen["ctx"] == 256
    assert parsed["context"] == 40960
