# Agent guide — Will's CUDA Suite

Orientation for any AI agent working in this repo. Read this before running anything.

## What this project is

A custom VRAM/stress bench for local CUDA inference on **NVIDIA GPUs**. It finds each
model's **safe context ceiling** so we never run a model+context that exhausts the card. It
does this by **extrapolating from safe measurements**, never by probing into the danger zone.

It is the CUDA sibling of [wmx-suite](https://github.com/willsarg/wmx-suite) (Apple Silicon)
and the engine [project-ara](https://github.com/willsarg/project-ara) installs on NVIDIA
hardware. Mirror wmx-suite's structure and conventions where they transfer.

Every machine-specific number (the VRAM wall, current usage, per-model fits) is **measured
live on whatever machine the suite runs on** — nothing is hardcoded to one card. The
**primary testbed is Will's Windows box with an RTX 2070 (8 GB)**.

**Design docs live outside this repo** — in the `project_ara` Obsidian vault, not under
`docs/`.

## RULE #1 — STAY UNDER THE VRAM WALL

Find safe ceilings; never probe into out-of-memory. Concretely:

- **The crash wall is the GPU's VRAM**, read live per machine (barebones: `nvidia-smi`),
  treated as an exact measured value — never rounded toward. The safe threshold is
  `wall − margin` (default margin 1 GB).
- **CUDA's failure mode is softer than Apple's but not free.** An over-allocation usually
  raises a recoverable OOM that kills the *workload*, not the OS. **But** on a GPU that also
  drives the display, VRAM exhaustion can freeze the desktop or trip a TDR driver reset — so
  the discipline still holds: extrapolate from below the wall, don't launch past the safe
  threshold "just to see."
- **Account for memory already in use** (display, other processes) — the budget for a *new*
  workload is headroom under the wall, not total VRAM.
- When in doubt, probe **smaller** first and let the extrapolation tell you the ceiling.

## Conventions (mirrored from wmx-suite)

- **uv only.** `uv sync`, `uv run wcx-suite ...`, `uv run pytest`.
- **Pure-Python where possible; minimal deps.** Barebones has zero runtime deps (nvidia-smi
  subprocess). CUDA/torch land only when a piece genuinely needs them.
- **Tests with pytest** for each piece (no enforced coverage gate, like wmx-suite).
- **Measured, not hardcoded.** Read facts live; never bake one card's numbers into logic.
- **The ARA-facing interface** (when we attach it) mirrors wmx-suite's: `system.read_limits()`,
  `config.margin_gb()`, and eventually `probe.calibrate()` / `models.describe()` / a `db` +
  `profiles` pair. Build toward that shape so ARA's backend adapter stays symmetric.

## License & AI agents (Apache 2.0)

wcx-suite is licensed **Apache-2.0** (`LICENSE`, `NOTICE`). Abide by it in both directions.

**Contributing here (inbound — Apache §5, inbound = outbound; no CLA/DCO):**
- All contributions are under Apache-2.0.
- Add only original code, or code under an Apache-2.0-compatible permissive license (MIT/BSD/ISC/Apache-2.0); preserve its copyright/license and record third-party components in `NOTICE`.
- Never introduce GPL/LGPL/AGPL, proprietary, or unknown-provenance code.
- Start every new source file with: `# SPDX-License-Identifier: Apache-2.0` and `# Copyright 2026 Will Sarg`.
- Do not alter or remove `LICENSE`, `NOTICE`, or existing SPDX headers.

**Cloning / forking / redistributing (outbound — Apache §4):**
- Keep `LICENSE` and `NOTICE` intact in any copy or fork.
- Retain all SPDX headers and copyright notices in files you carry.
- State significant changes you make.
- You may relicense *your own* additions; the Apache-2.0-covered files stay Apache-2.0.
