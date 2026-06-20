"""models.py — classify a model's VRAM behaviour from its HF-cache config (no torch, no GPU)."""
from __future__ import annotations

import json
import os

import pytest

from wcx_suite import models


def _make_model(tmp_path, hf_id, config, *, blob_bytes=0):
    """Materialise a fake HF-cache entry for *hf_id*: a snapshot config.json plus an optional
    weight blob, and point models.HUB at it. Returns the cache root."""
    root = tmp_path / ("models--" + hf_id.replace("/", "--"))
    snap = root / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(config))
    if blob_bytes:
        blobs = root / "blobs"
        blobs.mkdir()
        (blobs / "weight0").write_bytes(b"\0" * blob_bytes)
    return root


_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "hidden_size": 576,
    "head_dim": 64,
    "max_position_embeddings": 8192,
}


def test_describe_none_when_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    assert models.describe("nope/missing") is None


def test_describe_parses_llama_style_config(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    _make_model(tmp_path, "org/smol", _LLAMA, blob_bytes=270_000_000)
    info = models.describe("org/smol")
    assert info.hf_id == "org/smol"
    assert info.n_layers == 30
    assert info.growing_layers == 30          # no layer_types → all layers grow
    assert info.kv_heads == 3
    assert info.head_dim == 64
    assert info.max_context == 8192
    assert info.is_causal is True
    assert info.can_quantize_kv is True        # no sliding window
    assert info.weights_gb == pytest.approx(0.27, abs=0.01)


def test_kv_slope_matches_analytic_fp16(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    _make_model(tmp_path, "org/smol", _LLAMA)
    info = models.describe("org/smol")
    # fp16 KV bytes/token = layers × kv_heads × head_dim × 2(K,V) × 2(bytes)
    assert info.fp16_kv_bytes_per_token() == 30 * 3 * 64 * 2 * 2
    expected = info.fp16_kv_bytes_per_token() * 1000 / 1e9 * models.PREFILL_SPIKE_MULT
    assert info.estimated_slope_gb_per_k() == pytest.approx(expected)


def test_head_dim_falls_back_to_hidden_over_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    cfg = {**_LLAMA}
    del cfg["head_dim"]                          # 576 / 9 = 64
    _make_model(tmp_path, "org/nohd", cfg)
    assert models.describe("org/nohd").head_dim == 64


def test_kv_heads_falls_back_to_attention_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    cfg = {**_LLAMA}
    del cfg["num_key_value_heads"]               # no GQA → MHA, kv_heads == attn heads
    _make_model(tmp_path, "org/mha", cfg)
    assert models.describe("org/mha").kv_heads == 9


def test_sliding_window_blocks_kv_quant(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    cfg = {**_LLAMA, "sliding_window": 4096, "use_sliding_window": True}
    _make_model(tmp_path, "org/gemma", cfg)
    assert models.describe("org/gemma").can_quantize_kv is False


def test_non_causal_architecture_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    cfg = {**_LLAMA, "architectures": ["BertForSequenceClassification"]}
    _make_model(tmp_path, "org/bert", cfg)
    assert models.describe("org/bert").is_causal is False


def test_missing_architectures_assumed_causal(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    cfg = {k: v for k, v in _LLAMA.items() if k != "architectures"}
    _make_model(tmp_path, "org/bare", cfg)
    assert models.describe("org/bare").is_causal is True


def test_layer_types_count_only_growing_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    cfg = {**_LLAMA,
           "layer_types": ["full_attention", "sliding_attention", "full_attention"]}
    _make_model(tmp_path, "org/mixed", cfg)
    info = models.describe("org/mixed")
    assert info.growing_layers == 2             # only the two full_attention layers grow

