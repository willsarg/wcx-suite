# Contributing to wcx-suite

Thanks for your interest! wcx-suite is the **CUDA capability/safety engine** for
[Project ARA](https://github.com/willsarg/project-ara) — it measures a model's safe context
ceiling on NVIDIA hardware and refuses to launch anything predicted to breach it. Please read
[AGENTS.md](./AGENTS.md) first; it's the source of truth for the project's purpose and rules. This
file covers the human workflow.

## Setup

Requires **Python 3.12+** and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync                 # install into .venv (incl. dev tools)
uv run pytest -q        # tests
```

## Landing a change

1. Branch off `main`; keep PRs focused on a single change.
2. Run the actual commands and paste the output — safety claims need evidence.
3. Add or update tests for every fix or behavior change.
4. Match the surrounding style; comments explain *why*, not *what*.

## License

wcx-suite is licensed under the **Apache License 2.0** (`LICENSE` / `NOTICE`). By submitting a
contribution you agree it is provided under Apache 2.0 — the "inbound = outbound" rule of Apache §5
(no separate CLA or DCO). Contribute only code you have the right to license this way: code you
wrote, or code under an Apache-2.0-**compatible** permissive license (MIT/BSD/ISC/Apache-2.0) with
its notice preserved in `NOTICE`. Never paste in GPL/LGPL/AGPL, proprietary, or unknown-provenance
code. Start every new source file with `# SPDX-License-Identifier: Apache-2.0` and
`# Copyright 2026 Will Sarg`.

**AI coding agents** are welcome under the same rules; the human who opens the PR is responsible
for the assurances above. See "License & AI agents" in [AGENTS.md](./AGENTS.md).
