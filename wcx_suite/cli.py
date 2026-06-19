"""Command-line entry point for the suite.

    wcx-suite system        # show the GPU's VRAM wall + free, and the safe budget

More views (characterize / list / run) arrive as the suite is built out — mirroring
wmx-suite, but against CUDA VRAM instead of Apple's unified-memory working set.
"""
from __future__ import annotations

import argparse
import sys

from . import config, probe, system


def _ensure_utf8(stream) -> None:
    """Best-effort: make *stream* emit UTF-8 without crashing on a glyph.

    Windows consoles default to a legacy codepage (cp1252), where our output glyphs
    (−, —, ·) raise UnicodeEncodeError on a plain print. Reconfiguring to UTF-8 with
    errors='replace' makes output crash-proof; no-op on already-UTF-8 or plain streams.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    encoding = (getattr(stream, "encoding", "") or "").lower()
    if reconfigure is None or encoding.startswith("utf"):
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, LookupError):
        pass


def _print_system(lim: "system.GPULimits") -> None:
    margin = config.margin_gb()
    print(f"  GPU          {lim.device}")
    print(f"  VRAM total   {lim.total_gb:.1f} GB")
    print(f"  wall         {lim.wall_gb:.1f} GB   the VRAM ceiling — never cross")
    print(f"  safe budget  {lim.safe_threshold_gb(margin):.1f} GB   wall − {margin:.0f} GB margin")
    print(f"  free now     {lim.free_gb:.1f} GB   ({lim.used_gb:.1f} GB already in use)")


def cmd_system() -> int:
    lim = system.read_limits()
    if lim is None:
        print("  no NVIDIA GPU detected (nvidia-smi unavailable or no CUDA device)")
        return 1
    _print_system(lim)
    return 0


def cmd_calibrate() -> int:
    cal = probe.calibrate()
    if cal is None:
        print("  calibration failed — is CUDA available? (needs the [cuda] extra: torch)")
        return 1
    print(f"  device         {cal.device}")
    print(f"  CUDA overhead  {cal.overhead_gb:.2f} GB   fixed VRAM cost before any model weights")
    return 0


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8(sys.stdout)
    parser = argparse.ArgumentParser(
        prog="wcx-suite",
        description="VRAM/stress bench for local CUDA inference — safe context ceilings.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("system", help="show the GPU's VRAM wall, free memory, and safe budget")
    sub.add_parser("calibrate", help="measure the CUDA context's fixed VRAM overhead")
    args = parser.parse_args(argv)

    if args.command == "system":
        return cmd_system()
    if args.command == "calibrate":
        return cmd_calibrate()
    parser.print_help()
    return 1
