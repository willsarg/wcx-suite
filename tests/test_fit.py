"""probe fit math — extrapolate the safe context ceiling from VRAM-vs-context points."""
from __future__ import annotations

from wcx_suite import probe


def test_fit_ceiling_linear_extrapolation():
    # VRAM grows 0.6 GB from 512→2048 tokens: slope 0.000390625 GB/tok, intercept 1.8 GB.
    # budget 7 GB → ceiling = (7 − 1.8) / 0.000390625 ≈ 13312 tokens (floored conservatively).
    points = [(512, 2.0), (2048, 2.6)]
    assert probe.fit_ceiling(points, budget_gb=7.0) in (13311, 13312)


def test_fit_ceiling_none_when_vram_flat():
    # No growth with context → not context-bound here; ceiling is undefined (None).
    assert probe.fit_ceiling([(512, 2.0), (2048, 2.0)], budget_gb=7.0) is None


def test_fit_ceiling_none_with_single_point():
    assert probe.fit_ceiling([(512, 2.0)], budget_gb=7.0) is None


def test_fit_ceiling_floors_to_int():
    # budget just above intercept → small positive ceiling, floored.
    points = [(0, 1.0), (1000, 2.0)]   # slope 0.001 GB/tok, intercept 1.0
    assert probe.fit_ceiling(points, budget_gb=1.5) == 500
