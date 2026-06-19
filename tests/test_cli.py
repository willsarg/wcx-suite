"""cli.py — the `wcx-suite` entry point (barebones: the `system` view)."""
from __future__ import annotations

from wcx_suite import cli


def test_main_system_prints_vram(monkeypatch, capsys):
    monkeypatch.setattr(cli.system, "read_limits",
                        lambda: cli.system.GPULimits("NVIDIA GeForce RTX 2070",
                                                     total_gb=8.0, wall_gb=8.0,
                                                     free_gb=7.0, used_gb=1.0))
    rc = cli.main(["system"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "RTX 2070" in out
    assert "8.0 GB" in out          # the VRAM wall
    assert "7.0 GB" in out          # free now


def test_main_system_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(cli.system, "read_limits", lambda: None)
    rc = cli.main(["system"])
    assert rc == 1
    assert "no NVIDIA GPU" in capsys.readouterr().out


def test_main_no_command_shows_help(capsys):
    rc = cli.main([])
    assert rc == 1
    assert "wcx-suite" in capsys.readouterr().out
