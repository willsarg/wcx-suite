# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""One-shot governed inference — the engine-side `run` verb, under the same hard veto as L4.

ARA drives this out-of-process for a single completion: given a model, a context ceiling, and a
prompt on stdin, refuse-before-loading if the run wouldn't be safe, else load on CUDA, generate,
and emit one JSON line. The CUDA sibling of ``wmx_suite.generate`` — identical contract and safety
discipline, but torch + transformers (imported LAZILY) instead of mlx_lm.

    success:  {"context": <int>, "completion": "<text>"}
    refusal:  {"context": <int>, "refused": true, "reason": "<why>"}

Crucially we gate the EFFECTIVE context, not the raw ceiling. transformers/torch grow the KV cache
dynamically during generation, so a one-shot from a short prompt only reaches
``prompt_tokens + max_tokens`` of context — not the full ceiling. Gating the ceiling over-predicts
memory and would refuse runs that ``characterize`` already certified. The token count uses the
tokenizer ONLY (no model weights), so the gate runs before a single byte of weights loads (RULE #1).

Usage: ``python -m wcx_suite.generate <hf_id> <ctx> --margin G --overhead G --max-tokens N``
       (PROMPT is read from stdin, never argv.)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import measure_one, models, system

# Reuse the canonical refusal shape so every engine verb speaks the same dialect.
_refused = measure_one._refused


def _count_prompt_tokens(hf_id: str, prompt: str) -> int:
    """Prompt length in tokens via the tokenizer ONLY — no model weights touched.

    Empty prompt short-circuits to 0 before any import, so the no-prompt case never pays the
    transformers import cost (and never needs the cache warm)."""
    if not prompt:
        return 0
    from transformers import AutoTokenizer  # lazy: heavy, CUDA-box-only extra

    return len(AutoTokenizer.from_pretrained(hf_id).encode(prompt))


def _generate(hf_id: str, prompt: str, max_tokens: int) -> str:
    """Load *hf_id* on CUDA, generate up to *max_tokens* new tokens, return the NEW text only."""
    import torch                                            # lazy (NVIDIA-only [cuda] extra)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16).to("cuda")        # mirror probe_worker's load
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(input_ids=inputs.input_ids, max_new_tokens=max_tokens)
    new_tokens = out[0][prompt_len:]                         # decode only the newly generated span
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run(hf_id: str, ctx: int, *, margin_gb: float, overhead_gb: float, max_tokens: int,
        prompt: str = "") -> dict:
    """Gate on the effective context, then (if safe) load + generate. Returns the result dict.

    The refusal/success both report the ceiling *ctx* (consistent with the wmx verb and the cpu
    worker), even though the gate is evaluated against the smaller effective context.
    """
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")
    limits = system.read_limits()
    if limits is None:
        return _refused(ctx, "no NVIDIA GPU visible to nvidia-smi")

    # A one-shot only reaches prompt + new tokens of context; never gate the raw ceiling.
    prompt_tokens = _count_prompt_tokens(hf_id, prompt)
    effective_ctx = min(ctx, prompt_tokens + max_tokens)

    live_base = limits.used_gb              # mirrors measure_one.run's exact call site
    reason = measure_one.safety_gate(info, limits, effective_ctx, margin_gb=margin_gb,
                                     overhead_gb=overhead_gb, live_base=live_base)
    if reason is not None:
        return _refused(ctx, reason)        # report the ceiling, not the effective ctx

    completion = _generate(hf_id, prompt, max_tokens)
    return {"context": ctx, "completion": completion}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe one-shot inference under the L4 veto.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    args = ap.parse_args(argv)
    prompt = sys.stdin.read()               # PROMPT comes from stdin, never argv
    result = run(args.hf_id, args.ctx, margin_gb=args.margin, overhead_gb=args.overhead,
                 max_tokens=args.max_tokens, prompt=prompt)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
