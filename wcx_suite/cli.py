"""Command-line entry point for the suite.

    wcx-suite system        # show the GPU's VRAM wall + free, and the safe budget

More views (characterize / list / run) arrive as the suite is built out — mirroring
wmx-suite, but against CUDA VRAM instead of Apple's unified-memory working set.
"""
from __future__ import annotations

import argparse

from . import config, system


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wcx-suite",
        description="VRAM/stress bench for local CUDA inference — safe context ceilings.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("system", help="show the GPU's VRAM wall, free memory, and safe budget")
    args = parser.parse_args(argv)

    if args.command == "system":
        return cmd_system()
    parser.print_help()
    return 1
