# CLAUDE.md

This project's agent guidance lives in **[AGENTS.md](./AGENTS.md)** — read it first.
It is the single source of truth for purpose, the safety rule, commands, and conventions.
This file only adds Claude Code specifics.

## The one rule that matters most

**STAY UNDER THE VRAM WALL.** The crash wall is the GPU's VRAM, read live per machine
(barebones: `nvidia-smi`) and treated as exact — never rounded toward. The safe threshold is
`wall − margin` (default 1 GB). A CUDA over-allocation usually OOMs the workload rather than
the OS, but on a display-driving GPU it can freeze the desktop or trip a driver reset — so
find ceilings by extrapolating from below the wall, never by probing into it.

## Working here

- **uv only:** `uv sync`, `uv run wcx-suite system`, `uv run pytest`.
- Mirror [wmx-suite](https://github.com/willsarg/wmx-suite)'s structure and conventions.
- Design notes go in the `project_ara` Obsidian vault, not this repo.
