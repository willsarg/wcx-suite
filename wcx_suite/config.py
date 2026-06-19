"""Runtime configuration shared by CLI and programmatic entry points."""
from __future__ import annotations

import math
import os

# VRAM cushion kept under the wall. Smaller than wmx's 2 GB because consumer NVIDIA
# cards are smaller (an 8 GB card can't spare 2 GB) — refined during build-out.
DEFAULT_MARGIN_GB = 1.0
MARGIN_ENV = "WCX_SUITE_MARGIN_GB"


def margin_gb(value: float | str | None = None) -> float:
    """Return a validated safety margin; an explicit value overrides the environment."""
    raw = os.environ.get(MARGIN_ENV, str(DEFAULT_MARGIN_GB)) if value is None else value
    try:
        margin = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{MARGIN_ENV} / --margin must be a number") from exc
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"{MARGIN_ENV} / --margin must be finite and non-negative")
    return margin
