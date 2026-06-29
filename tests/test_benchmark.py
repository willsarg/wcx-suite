# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""benchmark.py — load-once governed multi-prompt inference worker.

Everything that touches a GPU or downloads weights is mocked, so these run on any box (no CUDA,
no real model). The key behavioral property is load-once: the model is loaded exactly once
regardless of prompt count — the whole point vs calling generate N times.

Per-prompt gating is also tested: each prompt's effective context (min(ctx, tokens+max_tokens))
is independently gated, and a per-prompt refusal does NOT abort the remaining prompts.
"""
from __future__ import annotations

import json
import sys
import types

from wcx_suite import benchmark, measure_one, models, probe_worker, system


def _info(**over):
    base = dict(hf_id="org/m", weights_gb=0.3, n_layers=30, growing_layers=30,
                kv_heads=3, head_dim=64, hidden_size=576, max_context=40960,
                can_quantize_kv=True, is_causal=True)
    base.update(over)
    return models.ModelInfo(**base)


def _limits(used_gb=2.0):
    return system.GPULimits(device="RTX 2070", total_gb=8.0, wall_gb=8.0,
                            free_gb=8.0 - used_gb, used_gb=used_gb)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


class _FakeIds:
    """Stand-in for a torch tensor of token ids: knows its shape and supports indexing."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, item):
        return _FakeIds([self.rows[0]])

    def to(self, device):
        return self


class _FakeTokenizer:
    """Minimal stand-in for transformers.AutoTokenizer."""

    eos_token_id = None
    chat_template = "fake_template"      # non-None → instruct model by default

    def __init__(self, n_prompt_tokens):
        self._n = n_prompt_tokens
        self._apply_chat_template_calls = []
        self._raw_call_count = 0
        self._last_call_text = None

    def encode(self, prompt):
        return list(range(self._n))

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        self._apply_chat_template_calls.append({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        })
        # tokenize=False renders a STRING (not a tensor/BatchEncoding); __call__ then tokenises it.
        return f"<TPL>{messages[0]['content']}"

    def __call__(self, prompt, return_tensors=None):
        self._raw_call_count += 1
        self._last_call_text = prompt
        ids = list(range(self._n))
        return types.SimpleNamespace(
            input_ids=_FakeIds([ids]),
            to=lambda device: types.SimpleNamespace(input_ids=_FakeIds([ids])),
        )

    def decode(self, ids, skip_special_tokens=True):
        return "generated text"


class _FakeModel:
    """Minimal model stand-in: generate returns prompt-plus-new-token ids."""

    config = types.SimpleNamespace(vocab_size=1000)

    def to(self, device): return self
    def eval(self): return self

    def generate(self, input_ids=None, max_new_tokens=None, **kw):
        prompt_len = input_ids.shape[1]
        return _FakeIds([list(range(prompt_len + max_new_tokens))])


def _patch_engine(monkeypatch, *, info=None, limits=None, n_prompt_tokens=6,
                  gate_reasons=None):
    """Mock engine collaborators. Returns a tracking dict with gate_calls, load_count, load_kwargs.

    gate_reasons: list of per-call gate results (None = pass); if shorter than actual calls,
    trailing calls all pass. Default None → all pass.
    """
    tracking = {"gate_calls": [], "load_count": [0], "load_kwargs": []}
    gate_reasons = gate_reasons or []
    call_idx = [0]

    monkeypatch.setattr(models, "describe", lambda hf: info if info is not None else _info())
    monkeypatch.setattr(system, "read_limits",
                        lambda: limits if limits is not None else _limits())

    def fake_gate(info_, limits_, ctx_, *, margin_gb, overhead_gb, live_base,
                  kv_bits=None, weight_quant="none"):
        tracking["gate_calls"].append({
            "ctx": ctx_, "kv_bits": kv_bits, "margin_gb": margin_gb,
            "overhead_gb": overhead_gb, "live_base": live_base, "weight_quant": weight_quant,
        })
        idx = call_idx[0]
        call_idx[0] += 1
        return gate_reasons[idx] if idx < len(gate_reasons) else None

    monkeypatch.setattr(measure_one, "safety_gate", fake_gate)

    # Fake torch + transformers via sys.modules (lazy-imported inside benchmark)
    fake_torch = types.SimpleNamespace(
        float16="float16",
        cuda=types.SimpleNamespace(is_available=lambda: True,
                                   get_device_name=lambda i: "RTX 2070"),
        no_grad=_nullctx,
    )

    tok_obj = _FakeTokenizer(n_prompt_tokens)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda hf_id: tok_obj),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    def fake_load_model(model_id, torch_, *, prefer_flash=False, weight_quant="none"):
        tracking["load_count"][0] += 1
        tracking["load_kwargs"].append({"prefer_flash": prefer_flash, "weight_quant": weight_quant})
        return _FakeModel()

    monkeypatch.setattr(probe_worker, "_load_model", fake_load_model)
    tracking["tok"] = tok_obj
    return tracking


# ---- happy path: N prompts → N completions, context echoed --------------------------------

def test_benchmark_happy_path_three_prompts(monkeypatch):
    _patch_engine(monkeypatch)
    r = benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                      prompts=["a", "b", "c"])
    assert r["context"] == 40960
    assert "refused" not in r
    results = r["results"]
    assert len(results) == 3
    for i, res in enumerate(results):
        assert res["prompt_index"] == i
        assert res["completion"] == "generated text"
        assert "refused" not in res


# ---- LOAD-ONCE: model loader is called exactly once regardless of prompt count ------------

def test_benchmark_loads_model_exactly_once(monkeypatch):
    """The raison d'être of this verb: single load amortised across all prompts."""
    tr = _patch_engine(monkeypatch)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                  prompts=["a", "b", "c", "d", "e"])
    assert tr["load_count"][0] == 1


# ---- per-prompt gate refusal: one prompt refused, others still complete -------------------

def test_benchmark_per_prompt_refusal_does_not_abort_run(monkeypatch):
    """Middle prompt trips the gate → that index is refused; before and after still complete."""
    tr = _patch_engine(monkeypatch,
                       gate_reasons=[None, "predicted 9.00GB >= safe budget 7.00GB", None])
    r = benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                      prompts=["a", "b", "c"])
    assert "refused" not in r                    # whole-run did NOT refuse
    results = r["results"]
    assert len(results) == 3
    assert results[0] == {"prompt_index": 0, "completion": "generated text"}
    assert results[1]["refused"] is True
    assert results[1]["prompt_index"] == 1
    assert "safe budget" in results[1]["reason"]
    assert results[2] == {"prompt_index": 2, "completion": "generated text"}


def test_benchmark_all_prompts_refused_individually_does_not_whole_refuse(monkeypatch):
    """All per-prompt refusals → results full of refused entries; no top-level refused."""
    tr = _patch_engine(monkeypatch,
                       gate_reasons=["oom", "oom"])
    r = benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                      prompts=["a", "b"])
    assert "refused" not in r
    assert all(res["refused"] is True for res in r["results"])


# ---- effective-context gating: gate sees min(ctx, tokens+max_tokens), not the ceiling ----

def test_benchmark_gates_on_effective_context(monkeypatch):
    """6 prompt tokens + 256 max_tokens = 262 effective; the gate must never see 40960."""
    tr = _patch_engine(monkeypatch, n_prompt_tokens=6)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                  prompts=["hi"])
    assert len(tr["gate_calls"]) == 1
    assert tr["gate_calls"][0]["ctx"] == 262
    assert tr["gate_calls"][0]["ctx"] != 40960


def test_benchmark_gates_capped_at_ceiling(monkeypatch):
    """When prompt+max_tokens would exceed ctx, gate receives ctx (the ceiling)."""
    tr = _patch_engine(monkeypatch, n_prompt_tokens=6)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=100000,
                  prompts=["hi"])
    assert tr["gate_calls"][0]["ctx"] == 40960


def test_benchmark_effective_context_per_prompt(monkeypatch):
    """Each prompt gets its own effective-ctx gate call (not a shared single gate)."""
    tr = _patch_engine(monkeypatch, n_prompt_tokens=6)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                  prompts=["a", "b", "c"])
    assert len(tr["gate_calls"]) == 3
    for call in tr["gate_calls"]:
        assert call["ctx"] == 14        # min(40960, 6+8) = 14


# ---- KV-quant / flash / weight-quant / chunk threading reaches loader and gate -----------

def test_benchmark_kv_bits_reaches_gate(monkeypatch):
    tr = _patch_engine(monkeypatch)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                  prompts=["hi"], kv_bits=4)
    assert tr["gate_calls"][0]["kv_bits"] == 4


def test_benchmark_kv_bits_forced_fp16_for_non_quantizable(monkeypatch):
    """Sliding-window model (can_quantize_kv=False) forces fp16 at the gate — mirrors generate."""
    tr = _patch_engine(monkeypatch, info=_info(can_quantize_kv=False))
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                  prompts=["hi"], kv_bits=4)
    assert tr["gate_calls"][0]["kv_bits"] is None


def test_benchmark_prefer_flash_reaches_loader(monkeypatch):
    tr = _patch_engine(monkeypatch)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                  prompts=["hi"], prefer_flash=True)
    assert tr["load_kwargs"][0]["prefer_flash"] is True


def test_benchmark_weight_quant_reaches_loader(monkeypatch):
    tr = _patch_engine(monkeypatch)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                  prompts=["hi"], weight_quant="int4")
    assert tr["load_kwargs"][0]["weight_quant"] == "int4"


def test_benchmark_weight_quant_reaches_gate(monkeypatch):
    tr = _patch_engine(monkeypatch)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                  prompts=["hi"], weight_quant="int8")
    assert tr["gate_calls"][0]["weight_quant"] == "int8"


def test_benchmark_chunk_reaches_generate_one(monkeypatch):
    """chunk threads through to _generate_one — verified by patching _generate_one directly."""
    _patch_engine(monkeypatch)
    seen = {}
    monkeypatch.setattr(benchmark, "_generate_one",
                        lambda model, tok, prompt, mt, kv=None, ch=None:
                        seen.update(ch=ch) or "out")
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=8,
                  prompts=["hi"], chunk=256)
    assert seen["ch"] == 256


# ---- whole-run refusals: same three guards as generate.run --------------------------------

def test_benchmark_whole_run_refused_model_missing(monkeypatch):
    monkeypatch.setattr(models, "describe", lambda hf: None)
    r = benchmark.run("missing/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                      prompts=["hi"])
    assert r["refused"] is True
    assert r["context"] == 40960
    assert "not found" in r["reason"]


def test_benchmark_whole_run_refused_not_causal(monkeypatch):
    monkeypatch.setattr(models, "describe", lambda hf: _info(is_causal=False))
    monkeypatch.setattr(system, "read_limits", lambda: _limits())
    r = benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                      prompts=["hi"])
    assert r["refused"] is True
    assert "results" not in r


def test_benchmark_whole_run_refused_no_gpu(monkeypatch):
    monkeypatch.setattr(models, "describe", lambda hf: _info())
    monkeypatch.setattr(system, "read_limits", lambda: None)
    r = benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                      prompts=["hi"])
    assert r["refused"] is True
    assert "NVIDIA GPU" in r["reason"]


# ---- main: stdin JSON array parsing and one JSON line out ---------------------------------

def test_main_reads_prompts_from_stdin_as_json_array(monkeypatch, capsys):
    _patch_engine(monkeypatch)
    prompts = ["hello", "world"]
    monkeypatch.setattr(sys, "stdin",
                        types.SimpleNamespace(read=lambda: json.dumps(prompts)))
    benchmark.main(["org/m", "40960", "--margin", "1.0", "--overhead", "0.6",
                    "--max-tokens", "256"])
    out = capsys.readouterr().out.strip()
    assert out.count("\n") == 0         # exactly one JSON line
    parsed = json.loads(out)
    assert parsed["context"] == 40960
    assert len(parsed["results"]) == 2


def test_main_threads_kv_bits(monkeypatch):
    seen = {}
    monkeypatch.setattr(benchmark, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "results": []})
    monkeypatch.setattr(sys, "stdin",
                        types.SimpleNamespace(read=lambda: '["hi"]'))
    benchmark.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6",
                    "--max-tokens", "8", "--kv-bits", "8"])
    assert seen["kv_bits"] == 8


def test_main_threads_flash_attn(monkeypatch):
    seen = {}
    monkeypatch.setattr(benchmark, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "results": []})
    monkeypatch.setattr(sys, "stdin",
                        types.SimpleNamespace(read=lambda: '["hi"]'))
    benchmark.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6",
                    "--max-tokens", "8", "--flash-attn"])
    assert seen["prefer_flash"] is True


def test_main_threads_weight_quant(monkeypatch):
    seen = {}
    monkeypatch.setattr(benchmark, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "results": []})
    monkeypatch.setattr(sys, "stdin",
                        types.SimpleNamespace(read=lambda: '["hi"]'))
    benchmark.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6",
                    "--max-tokens", "8", "--weight-quant", "int8"])
    assert seen["weight_quant"] == "int8"


def test_main_threads_prefill_chunk(monkeypatch):
    seen = {}
    monkeypatch.setattr(benchmark, "run",
                        lambda *a, **k: seen.update(k) or {"context": a[1], "results": []})
    monkeypatch.setattr(sys, "stdin",
                        types.SimpleNamespace(read=lambda: '["hi"]'))
    benchmark.main(["org/m", "2000", "--margin", "1.0", "--overhead", "0.6",
                    "--max-tokens", "8", "--prefill-chunk", "256"])
    assert seen["chunk"] == 256


# ---- chat-template tokenisation (instruct model fix) -------------------------------------

def test_generate_one_uses_chat_template_for_instruct_model(monkeypatch):
    """_generate_one must call apply_chat_template for models that have one (e.g. gemma-4).

    Raw prompts on instruct models yield empty/garbage output; the template wraps the user
    turn so the model sees the instruction format it was fine-tuned on.
    """
    tr = _patch_engine(monkeypatch, n_prompt_tokens=6)
    tok = tr["tok"]                             # tok.chat_template = "fake_template" (default)
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                  prompts=["hello"])
    assert len(tok._apply_chat_template_calls) == 1, "apply_chat_template was not called"
    call = tok._apply_chat_template_calls[0]
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False, "must render to a string (tokenize=False), not a tensor"
    assert tok._last_call_text == "<TPL>hello"   # the templated string is what got tokenised


def test_generate_one_falls_back_to_raw_tokenize_for_base_model(monkeypatch):
    """When tokenizer.chat_template is None (base model), _generate_one falls back to the
    raw tokenizer call — apply_chat_template would raise with a missing template."""
    tr = _patch_engine(monkeypatch, n_prompt_tokens=6)
    tok = tr["tok"]
    tok.chat_template = None                    # simulate a base (non-instruct) model
    benchmark.run("org/m", 40960, margin_gb=1.0, overhead_gb=0.6, max_tokens=256,
                  prompts=["hello"])
    assert len(tok._apply_chat_template_calls) == 0, "apply_chat_template called on base model"
    assert tok._raw_call_count == 1, "raw __call__ was not used as fallback"
    assert tok._last_call_text == "hello"       # the RAW prompt was tokenised, untemplated
